"""
run_tz_stage9_online_speed.py — TZ_stage9 §3.3-3.4 (18.09.2026): online-замер
задержки nrc_strong (обычный PyTorch MLP) vs NRCLmKANBaseline (fused-kernel
KAN) на ОДНОЙ и той же GPU (Kaggle T4).

ЗАПУСК: только на Kaggle T4x2 notebook, ПОСЛЕ Шага 0
(run_tz_stage9_step0_smoke_test.py — сборка lmkan) и ПОСЛЕ git push текущих
локальных изменений (models/kan_lmkan_baseline.py и этот файл не в
удалённом репозитории на момент написания, 18.09.2026).

Ячейка 1 (перед этим скриптом, в Kaggle Notebook):
    !git clone https://github.com/AlexanderRumyantcev/LNN-LowLight.git /kaggle/working/repo
    !cd /kaggle/working/lmkan && pip install .   # если ещё не установлен (Шаг 0)
    import sys; sys.path.insert(0, '/kaggle/working/repo')
    sys.path.insert(0, '/kaggle/working/repo/blender')
    %cd /kaggle/working/repo

ЧТО ИЗМЕРЯЕТСЯ (§3.3 ТЗ, ИСПРАВЛЕНО 18.09.2026 — см. чат): batch_size и
n_steps по умолчанию — НЕ произвольная оценка "на глаз", а верифицированный
протокол реального NRC (Müller et al. 2021, arXiv 2106.12372, раздел
"Amortization in a Real-time Path Tracer", проверено напрямую по тексту
статьи, не по памяти): "we select s = 4 training batches and l = 16384
records per bat[ch]" — т.е. НАСТОЯЩИЙ NRC на каждый кадр делает 4
последовательных forward+backward шага по 16384 сэмплов каждый, в рамках
бюджета 16.6мс/кадр (60fps), из которых на весь кэш (query+update) уходит
~2.6мс. STRONG_S_BATCHES=4, STRONG_L_RECORDS=16384 ниже — эти числа, не
"реалистичное приближение". Первая версия этого скрипта (до исправления)
использовала batch_size=1024 без проверки по первоисточнику — тот же
класс ошибки, что был с CfC (казалось, что тестируем сопоставимую
конфигурацию, на деле НЕТ, см. EXPERIMENT_LOG.md "ПОПРАВКА 2026-09-12") —
исправлено ДО первого прогона на GPU, не постфактум.

НЕ batched offline-инференс (протокол TZ_stage6-9 до TZ_stage9) — малый
батч на шаг, forward+backward КАЖДЫЙ шаг, s=4 шага на "кадр", много
последовательных "кадров" подряд. Задержка на шаг И на кадр (сумма 4
шагов) — последнее прямо сопоставимо по масштабу с ~2.6мс бюджетом
настоящего NRC (не заявляем паритет, только даём одну и ту же единицу
измерения для честного взгляда на масштаб разрыва).

ЧТО НЕ ИЗМЕРЯЕТСЯ (§3.4/§5 ТЗ): точность моделей (переоценка точности вне
scope — вывод TZ_stage8 переиспользуется как есть), сравнение с реальным
продакшн NRC (tiny-cuda-nn, fused half-precision CUDA, ~2.6мс/кадр — другой,
гораздо более высокий бар, см. mempalace-drawer про fused-kernel NRC).

ЯВНОЕ ДОПУЩЕНИЕ (не в TZ_stage9, зафиксировано здесь при реализации): "малый
батч на шаг" = batch_size ПИКСЕЛЕЙ (пространственная ось), каждый со своей
полной временной последовательностью T (как во всём остальном pipeline
проекта — build_batch_dense не поддерживает частичные по времени
последовательности без переписывания dataset_adapter). Настоящий per-frame
online-профиль NRC (один шаг = один новый кадр рендера, T=1) потребовал бы
отдельной перестройки temporal-пайплайна — вне бюджета этого этапа. Батч из
batch_size независимых пикселей с полным T — практичное приближение того же
вопроса (малый батч, много последовательных forward+backward шагов), не
идентичное оригиналу 1:1.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset_adapter_zeroday import load_fully_valid_pixel_sequences_zeroday  # noqa: E402
from run_experiment_dense import build_batch_dense, _get_spatial, SPATIAL_DIM  # noqa: E402
from models.baselines import NRCStyleBaseline  # noqa: E402
from models.kan_lmkan_baseline import NRCLmKANBaseline  # noqa: E402

DEFAULT_DATASET = Path(__file__).parent / "generate_dataset_zeroday_result_nostep_v1.npz"
STRONG_N_HIDDEN_LAYERS = 5   # TZ_stage6 §2.1 / TZ_stage7 §9, патент-верный конфиг
STRONG_BIAS = False          # TZ_stage6 §2.1 / TZ_stage7 §9, патент-верный конфиг
CAPACITY_LEVELS = [16, 64]   # TZ_stage9 §3.2, переиспользованы точки TZ_stage8
STRONG_S_BATCHES = 4         # Müller et al. 2021, arXiv 2106.12372, "Amortization..."
STRONG_L_RECORDS = 16384     # Müller et al. 2021, arXiv 2106.12372, "Amortization..."


def _build_step_batch(seqs, idx, device):
    """Собирает вход/таргет для ОДНОГО online-шага (малый батч пикселей,
    §3.3 ТЗ) — тот же build_batch_dense, что offline-протокол TZ_stage6-9,
    но вызывается заново на КАЖДЫЙ шаг с новым случайным idx, а не один раз
    на train/eval сплит."""
    batch = build_batch_dense(seqs, idx)
    obs = batch["obs"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    true = batch["true"].to(device)
    spatial = _get_spatial(batch, device)
    return obs, cold, conf, disocc, mismatch, true, spatial


def _timed_online_run(model, build_input_fn, seqs, obs_dim, batch_size, n_frames,
                       s_batches, warmup_frames, device, rng):
    """§3.3-3.4 ТЗ (числа протокола — Müller et al. 2021, см. докстринг
    модуля): n_frames "кадров", каждый — s_batches последовательных
    forward+backward шагов на СЛУЧАЙНОМ батче из batch_size пикселей (не
    train/eval сплит — online-профиль). warmup_frames прогоняются и
    отбрасываются (CUDA JIT/аллокатор warm-up — честность замера, которой
    сознательно не было в Шаге 0 §3.0, там это была ГРУБАЯ прикидка, здесь
    — протокольный замер).
    Возвращает: per-step forward-only [N], per-step forward+backward [N],
    per-frame forward+backward (сумма s_batches подряд идущих шагов) [n_frames].
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    fwd_times, fwdbwd_times = [], []
    frame_totals = []

    for frame in range(warmup_frames + n_frames):
        frame_total = 0.0
        for _ in range(s_batches):
            idx = rng.choice(len(seqs), size=min(batch_size, len(seqs)), replace=False)
            obs, cold, conf, disocc, mismatch, true, spatial = _build_step_batch(seqs, idx, device)
            u = build_input_fn(obs, cold, conf, use_staleness=True,
                                extra_staleness=[disocc, mismatch], spatial=spatial)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(u)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss = loss_fn(pred, true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            if frame >= warmup_frames:
                fwd_times.append((t1 - t0) * 1000)
                fwdbwd_times.append((t2 - t0) * 1000)
                frame_total += (t2 - t0) * 1000
        if frame >= warmup_frames:
            frame_totals.append(frame_total)

    return np.array(fwd_times), np.array(fwdbwd_times), np.array(frame_totals)


def run(dataset_path=DEFAULT_DATASET, batch_size=STRONG_L_RECORDS, n_frames=25,
        s_batches=STRONG_S_BATCHES, warmup_frames=5, capacity_levels=CAPACITY_LEVELS, seed=0):
    assert torch.cuda.is_available(), "TZ_stage9 требует CUDA (fused-kernel lmKAN) — CPU не подходит"
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    seqs = load_fully_valid_pixel_sequences_zeroday(dataset_path)
    obs_dim = 3  # проверено напрямую по датасету 18.09.2026, см. models/kan_lmkan_baseline.py
    print(f"device={device}, batch_size={batch_size} (Müller et al. 2021: l=16384), "
          f"s_batches={s_batches} (Müller et al. 2021: s=4), n_frames={n_frames} "
          f"(+{warmup_frames} warmup), n_pixels_total={len(seqs)}, "
          f"capacity_levels={capacity_levels}", flush=True)

    results = {}
    for hdim in capacity_levels:
        print(f"\n--- hidden_dim={hdim} ---", flush=True)

        nrc = NRCStyleBaseline(
            obs_dim=obs_dim, hidden_dim=hdim, use_staleness=True, staleness_dim=4,
            spatial_dim=SPATIAL_DIM, n_hidden_layers=STRONG_N_HIDDEN_LAYERS, bias=STRONG_BIAS,
        ).to(device)
        fwd_nrc, fwdbwd_nrc, frames_nrc = _timed_online_run(
            nrc, NRCStyleBaseline.build_input, seqs, obs_dim, batch_size, n_frames,
            s_batches, warmup_frames, device, rng)
        print(f"  nrc_strong:  шаг(fwd+bwd)={fwdbwd_nrc.mean():.3f}мс "
              f"(p50={np.median(fwdbwd_nrc):.3f}, p95={np.percentile(fwdbwd_nrc,95):.3f})  "
              f"КАДР(s={s_batches} шагов)={frames_nrc.mean():.3f}мс "
              f"(p50={np.median(frames_nrc):.3f}, p95={np.percentile(frames_nrc,95):.3f})", flush=True)

        kan = NRCLmKANBaseline(
            obs_dim=obs_dim, hidden_dim=hdim, use_staleness=True, staleness_dim=4,
            spatial_dim=SPATIAL_DIM, n_hidden_layers=STRONG_N_HIDDEN_LAYERS,
        ).to(device)
        fwd_kan, fwdbwd_kan, frames_kan = _timed_online_run(
            kan, NRCLmKANBaseline.build_input, seqs, obs_dim, batch_size, n_frames,
            s_batches, warmup_frames, device, rng)
        print(f"  kan(lmkan):  шаг(fwd+bwd)={fwdbwd_kan.mean():.3f}мс "
              f"(p50={np.median(fwdbwd_kan):.3f}, p95={np.percentile(fwdbwd_kan,95):.3f})  "
              f"КАДР(s={s_batches} шагов)={frames_kan.mean():.3f}мс "
              f"(p50={np.median(frames_kan):.3f}, p95={np.percentile(frames_kan,95):.3f})", flush=True)

        ratio_fwdbwd = fwdbwd_kan.mean() / fwdbwd_nrc.mean()
        ratio_frame = frames_kan.mean() / frames_nrc.mean()
        print(f"  RATIO (шаг, kan/nrc_strong): {ratio_fwdbwd:.2f}x   "
              f"RATIO (кадр, kan/nrc_strong): {ratio_frame:.2f}x", flush=True)
        print(f"  для справки: бюджет настоящего NRC на весь кэш ~2.6мс/кадр "
              f"(Müller et al. 2021, fused-kernel — НЕ прямое сравнение, другой "
              f"класс реализации, см. TZ_stage9 §3.4/§5)", flush=True)

        results[hdim] = {
            "nrc_strong": {"fwdbwd_ms": fwdbwd_nrc.tolist(), "frame_ms": frames_nrc.tolist()},
            "kan_lmkan": {"fwdbwd_ms": fwdbwd_kan.tolist(), "frame_ms": frames_kan.tolist()},
            "ratio_fwdbwd": ratio_fwdbwd,
            "ratio_frame": ratio_frame,
        }

    print("\n=== §4 TZ_stage9 decision rule (предварительно, по обеим точкам) ===")
    ratios = [results[h]["ratio_fwdbwd"] for h in capacity_levels]
    max_ratio = max(ratios)
    if max_ratio <= 5.0:
        verdict = "ДА (предварительно) — разрыв закрылся до разумных пределов на обеих точках"
    elif max_ratio >= 10.0:
        verdict = "НЕТ (предварительно) — разрыв остаётся на порядок и больше даже с fused-kernel"
    else:
        verdict = "ПРОМЕЖУТОЧНО — требует решения (PolyKAN как 2й кандидат, либо явный бюджет/кадр)"
    print(f"  max ratio (fwd+bwd, по {capacity_levels}) = {max_ratio:.2f}x -> {verdict}")

    out_path = Path(dataset_path).parent / "run_tz_stage9_online_speed_results.json"
    with open(out_path, "w") as f:
        json.dump({"batch_size": batch_size, "s_batches": s_batches, "n_frames": n_frames,
                   "results": results, "max_ratio_fwdbwd": max_ratio, "verdict": verdict},
                  f, indent=2)
    print(f"\nСырые результаты сохранены в {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--batch_size", type=int, default=STRONG_L_RECORDS,
                         help="l в терминах Müller et al. 2021 (по умолчанию 16384, как в статье)")
    parser.add_argument("--n_frames", type=int, default=25)
    parser.add_argument("--s_batches", type=int, default=STRONG_S_BATCHES,
                         help="s в терминах Müller et al. 2021 (по умолчанию 4, как в статье)")
    parser.add_argument("--warmup_frames", type=int, default=5)
    parser.add_argument("--capacity_levels", type=int, nargs="+", default=CAPACITY_LEVELS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(dataset_path=args.dataset, batch_size=args.batch_size, n_frames=args.n_frames,
        s_batches=args.s_batches, warmup_frames=args.warmup_frames,
        capacity_levels=args.capacity_levels, seed=args.seed)
