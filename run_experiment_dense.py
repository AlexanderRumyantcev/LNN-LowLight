"""
Единый скрипт прогона pipeline dense-пути (per-pixel,
TZ_stage1b_per_pixel_dense_fallback.md) на РЕАЛЬНЫХ Blender-данных
(blender/generate_dataset_dense_result_full.npz) — аналог
run_experiment_blender.py (probe-путь), максимально переиспользует
существующий код:

БЕЗ ИЗМЕНЕНИЙ переиспользуется: build_batch (обрезка по мин. длине +
вывод obs_dim из формы данных), _select_device, _summarize (per-
segment-type/early/floor метрики §6.1-6.3, канально усредняемые),
_reconstruct_light_schedule — всё из run_experiment.py/
run_experiment_blender.py; NRCRelativeL2Loss (models/losses.py),
calibrate_time_gate_init (models/temporal/cfc_probe_module.py) — без
изменений; CfCProbeModule/NRCStyleBaseline/GRUProbeBaseline/
NRDStyleBaseline — их staleness_dim=4/use_disocclusion_reset/
use_mismatch_honest расширения (§2.6/§4.2), реализованные ранее в
этой же сессии.

ЧТО ДЕЙСТВИТЕЛЬНО ОТЛИЧАЕТСЯ ОТ PROBE-ПУТИ (не может быть
переиспользовано как есть):
- Источник последовательностей: build_all_pixel_sequences
  (blender/dataset_adapter_dense.py) — одна последовательность на
  ЭКРАННЫЙ ПИКСЕЛЬ, не world-space пробу (§2.3).
- train_model_dense/predict_dense: honest-версии CfC/NRC/GRU получают
  staleness_dim=4/extra_staleness_dim=2 с extra_staleness=
  [disocclusion_flag, geometric_mismatch_score] (§2.6/§4.2) вместо
  старого 2-/3-элементного вектора. NRD-style использует НОВЫЕ режимы
  (use_disocclusion_reset для faithful, use_mismatch_honest для
  honest, §4.2) вместо старого use_honest_dt — Δt-awareness для
  NRD-style в dense-пути неактуальна тем же образом, что в probe-пути.
- Метрики: ДОБАВЛЯЕТСЯ §6 delta (disocclusion_vs_stable_mse,
  error_vs_warp_age_curve, evaluation/metrics.py) поверх старых
  per_segment_type/early_zone_floor/error_vs_offset (те остаются без
  изменений — §6 ТЗ явно требует этого для static/step/drift среза;
  dense-датасет тоже использует light_schedule.py, тот же сигнал).
- Интерполяция между пробами (models/probe_interpolation.py) НЕ
  ПЕРЕНОСИТСЯ — per-pixel dense не имеет разреженных проб, которые
  нужно интерполировать между собой (§2.3: сама warp/reprojection —
  это и есть механизм переноса между КАДРАМИ, не между пикселями).
- "Seeds" здесь ЕЩЁ СЛАБЕЕ, чем в run_experiment_blender.py (см. её
  докстринг): там вариация — per-probe subsampling + train/eval split
  ПРОБ поверх одного рендера. Здесь Δt РЕГУЛЯРНЫЙ (§2.3) — subsampling
  не при чём вообще; единственная вариация между "сидами" — train/eval
  РАЗБИЕНИЕ ПИКСЕЛЕЙ поверх одного и того же рендера (одна геометрия,
  одна камера-траектория, одно световое расписание). Значимость
  измеряет устойчивость ТОЛЬКО к разбиению пикселей — печатается явно.

МОЁ ИНЖЕНЕРНОЕ РЕШЕНИЕ (не задано ТЗ явно): для этой первой версии
скрипта используются только пиксели, валидные НА ВСЕХ кадрах
(seq["valid"].all()) — не "валиден хотя бы раз", как в
dataset_adapter_dense.py. NRCRelativeL2Loss (models/losses.py) не
поддерживает per-timestep маскирование (только continuous
confidence_weight, не boolean mask) — обучение/оценка на пикселях с
невалидными (sky) кадрами внутри последовательности без маскирования
означало бы тренировку на бессмысленных значениях. Полное маскирование
loss — отдельная доработка на будущее, не в этой сессии; текущий
фильтр — простой и честный способ не завязывать sanity-check на
нерешённой проблеме. На полном датасете (generate_dataset_dense_result_
full.npz) это оставляет 29 из 64 пикселей (см. mempalace) — достаточно
для sanity-прогона, недостаточно для финальных выводов о качестве.

ГЛАВНЫЙ КРИТЕРИЙ §5 stage1b — три возможных исхода, определяемые
ИМЕННО disocclusion-срезом (§5/§6 delta), не агрегатом по кадру —
печатается отдельным блоком в конце run_multi_seed().
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "blender"))
from dataset_adapter_dense import build_all_pixel_sequences  # noqa: E402

from run_experiment import build_batch as _build_batch_base
from run_experiment_blender import _select_device, _reconstruct_light_schedule, _summarize
from models.temporal.cfc_probe_module import CfCProbeModule, calibrate_time_gate_init, full_gate_diagnostics
from models.baselines import NRDStyleBaseline, NRCStyleBaseline, GRUProbeBaseline
from models.losses import NRCRelativeL2Loss
from evaluation.metrics import (
    label_samples, paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
    disocclusion_vs_stable_mse, error_vs_warp_age_curve,
)

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())

DEFAULT_DATASET = Path(__file__).parent / "blender" / "generate_dataset_dense_result_full.npz"

MODEL_KINDS = ["cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
               "gru_faithful", "gru_honest"]


def load_fully_valid_pixel_sequences(npz_path):
    """build_all_pixel_sequences (blender/dataset_adapter_dense.py) без
    изменений + фильтр на "валиден на ВСЕХ кадрах" (см. докстринг
    модуля про причину)."""
    seqs = build_all_pixel_sequences(npz_path)
    filtered = [s for s in seqs if s["valid"].all()]
    print(f"пикселей всего с последовательностью: {len(seqs)}, "
          f"валидных на всех кадрах: {len(filtered)}")
    return filtered


def build_batch_dense(seqs, indices):
    """build_batch (run_experiment.py) БЕЗ ИЗМЕНЕНИЙ для obs/dt/cold/
    conf/true/t (ключи совпадают — dataset_adapter_dense.py намеренно
    использует ту же номенклатуру, что probe_subsampling.py/
    dataset_adapter.py) + доп. dense-специфичные каналы
    (disocclusion_flag/geometric_mismatch_score/valid), обрезанные до
    той же длины T, что вычислил build_batch."""
    batch = _build_batch_base(seqs, indices)
    T = batch["dt"].shape[1]
    batch["disocclusion_flag"] = torch.tensor(
        np.stack([seqs[i]["disocclusion_flag"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["geometric_mismatch_score"] = torch.tensor(
        np.stack([seqs[i]["geometric_mismatch_score"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["valid"] = torch.tensor(
        np.stack([seqs[i]["valid"][:T] for i in indices]), dtype=torch.bool,
    )
    return batch


def train_model_dense(kind: str, batch, device, epochs: int, lr: float, hidden_dim: int,
                       gate_lr: float | None = None):
    """Аналог train_model (run_experiment.py) — honest-версии получают
    staleness_dim=4/extra_staleness_dim=2 с extra_staleness=
    [disocclusion_flag, geometric_mismatch_score] (§2.6/§4.2) вместо
    старого 2-/3-элементного вектора. nrd_faithful/nrd_honest сюда не
    попадают — у NRDStyleBaseline нет обучаемых параметров (как и в
    train_model для probe-пути).

    gate_lr (только для kind='cfc', МОЙ инженерный эксперимент, не из
    ТЗ, 2026-08-09): отдельный learning rate для model.cell.W_a/W_b
    (time-gate readout) поверх общего lr для всей остальной сети.
    Мотивация — эмпирическая, не догадка: gate-диагностика (см.
    mempalace) показала, что backbone z РЕАЛЬНО различает disocclusion/
    stable контексты (cos_sim=0.886, не ~1.0), но W_a/W_b не
    конвертируют это различие в разный sigma_tau — а их градиент на
    порядок меньше остального backbone (0.0038/0.0018 vs 0.063/0.023).
    None (по умолчанию) = старое поведение, единый lr на все параметры,
    один Adam без param groups.
    """
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf, true = batch["cold"].to(device), batch["conf"].to(device), batch["true"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    if kind == "cfc":
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt)[0]
    elif kind == "nrc_honest":
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                  staleness_dim=4).to(device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch])
        forward = lambda: model(u)
    elif kind == "nrc_faithful":
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=False).to(device)
        u = NRCStyleBaseline.build_input(obs, use_staleness=False)
        forward = lambda: model(u)
    elif kind == "gru_honest":
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=True,
                                  extra_staleness_dim=2).to(device)
        u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                          extra_staleness=[disocc, mismatch])
        forward = lambda: model(u)[0]
    elif kind == "gru_faithful":
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=False).to(device)
        u = GRUProbeBaseline.build_input(obs, use_dt_staleness=False)
        forward = lambda: model(u)[0]
    else:
        raise ValueError(kind)

    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    if kind == "cfc" and gate_lr is not None:
        gate_params = list(model.cell.W_a.parameters()) + list(model.cell.W_b.parameters())
        gate_param_ids = {id(p) for p in gate_params}
        rest_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
        opt = torch.optim.Adam([
            {"params": rest_params, "lr": lr},
            {"params": gate_params, "lr": gate_lr},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = forward()
        loss = loss_fn(pred, true)
        loss.backward()
        opt.step()
    return model


def predict_dense(kind: str, model_or_none, batch, device, tau_nrd: float = 3.0, alpha_nrd: float = 0.3):
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        if kind == "cfc":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrc_honest":
            u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch])
            return model_or_none(u).cpu().numpy()
        if kind == "nrc_faithful":
            u = NRCStyleBaseline.build_input(obs, use_staleness=False)
            return model_or_none(u).cpu().numpy()
        if kind == "gru_honest":
            u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                              extra_staleness=[disocc, mismatch])
            return model_or_none(u)[0].cpu().numpy()
        if kind == "gru_faithful":
            u = GRUProbeBaseline.build_input(obs, use_dt_staleness=False)
            return model_or_none(u)[0].cpu().numpy()
        if kind == "nrd_faithful":
            m = NRDStyleBaseline(alpha=alpha_nrd, use_disocclusion_reset=True).to(device)
            return m(obs, dt, disocclusion_flag_seq=disocc).cpu().numpy()
        if kind == "nrd_honest":
            m = NRDStyleBaseline(tau=tau_nrd, use_mismatch_honest=True).to(device)
            return m(obs, dt, geometric_mismatch_seq=mismatch).cpu().numpy()
    raise ValueError(kind)


def _summarize_disocclusion(model_kinds, preds, true, disocc_flags, n_eval, verbose=True):
    """§6 delta — MSE disocclusion vs stable + error-vs-warp-age, тем же
    способом канального усреднения (obs_dim каналов -> одно число на
    пиксель -> одно число на весь eval), что _summarize (run_experiment_
    blender.py) делает для static/step/drift."""
    results = {}
    for kind in model_kinds:
        disocc_vals, stable_vals = [], []
        curve_accum = {label: [] for label in BIN_LABELS}
        for p in range(n_eval):
            pred_p = preds[kind][p].astype(np.float64)  # [T, C]
            true_p = true[p].astype(np.float64)
            flag_p = disocc_flags[p].astype(bool)

            d_c, s_c = [], []
            curve_c = {label: [] for label in BIN_LABELS}
            for c in range(pred_p.shape[-1]):
                split = disocclusion_vs_stable_mse(pred_p[:, c], true_p[:, c], flag_p)
                if not np.isnan(split["disocclusion"]):
                    d_c.append(split["disocclusion"])
                if not np.isnan(split["stable"]):
                    s_c.append(split["stable"])
                curve = error_vs_warp_age_curve(pred_p[:, c], true_p[:, c], flag_p, bin_edges=BIN_EDGES)
                for label, val in curve.items():
                    if not np.isnan(val):
                        curve_c[label].append(val)
            if d_c:
                disocc_vals.append(float(np.mean(d_c)))
            if s_c:
                stable_vals.append(float(np.mean(s_c)))
            for label in BIN_LABELS:
                if curve_c[label]:
                    curve_accum[label].append(float(np.mean(curve_c[label])))

        results[kind] = dict(
            disocclusion=float(np.mean(disocc_vals)) if disocc_vals else float("nan"),
            stable=float(np.mean(stable_vals)) if stable_vals else float("nan"),
            warp_age_curve={label: (float(np.mean(vals)) if vals else float("nan"))
                             for label, vals in curve_accum.items()},
        )

    if verbose:
        print("\n=== §6 delta: disocclusion vs stable MSE ===")
        for kind in model_kinds:
            r = results[kind]
            print(f"{kind:14s}  disocclusion={r['disocclusion']:.4f}  stable={r['stable']:.4f}")

    return results


def _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr, hidden_dim, device,
                   verbose=True, gate_lr=None):
    """Один 'сид' = train/eval разбиение ПИКСЕЛЕЙ поверх ОДНОГО и того
    же рендера (см. предупреждение в докстринге модуля — Δt регулярный,
    значит здесь нет второй оси вариации, которая была в probe-пути
    через per-probe subsampling)."""
    n_pixels = len(seqs)
    if n_train_pixels >= n_pixels:
        raise ValueError(f"n_train_pixels={n_train_pixels} >= доступно валидных пикселей={n_pixels}")

    perm = np.random.default_rng(seed).permutation(n_pixels)
    train_idx = list(perm[:n_train_pixels])
    eval_idx = list(perm[n_train_pixels:])

    train_batch = build_batch_dense(seqs, train_idx)
    eval_batch = build_batch_dense(seqs, eval_idx)

    cfc_model = train_model_dense("cfc", train_batch, device, epochs, lr, hidden_dim, gate_lr=gate_lr)
    nrc_f_model = train_model_dense("nrc_faithful", train_batch, device, epochs, lr, hidden_dim)
    nrc_h_model = train_model_dense("nrc_honest", train_batch, device, epochs, lr, hidden_dim)
    gru_f_model = train_model_dense("gru_faithful", train_batch, device, epochs, lr, hidden_dim)
    gru_h_model = train_model_dense("gru_honest", train_batch, device, epochs, lr, hidden_dim)

    # ПОЛНАЯ gate-диагностика (перенесена из run_experiment.py::run, ранее не была
    # подключена в _run_one_seed для dense-пути — см. обсуждение с пользователем
    # 2026-08-09 про то, реагирует ли гейт CfC на disocclusion_flag/geometric_mismatch
    # вообще, до объяснений архитектурными аргументами). На TRAIN-данных (не eval) —
    # градиентная проверка должна идти на том, что модель реально видела при обучении.
    train_seg_type = np.stack([
        label_samples(train_batch["t"][i].astype(np.float64), light_schedule)[0]
        for i in range(len(train_idx))
    ])
    u_train = cfc_model.build_input(
        train_batch["obs"].to(device), train_batch["cold"].to(device),
        train_batch["conf"].to(device), use_staleness=True,
        extra_staleness=[train_batch["disocclusion_flag"].to(device),
                          train_batch["geometric_mismatch_score"].to(device)],
    )
    full_gate_diagnostics(
        cfc_model, u_train, train_batch["dt"].to(device), true=train_batch["true"].to(device),
        seg_type=train_seg_type, segment_names=SEGMENT_NAMES,
        label=f"seed={seed} (train, dense)",
    )

    # ДОПОЛНИТЕЛЬНО (не входит в full_gate_diagnostics — та группирует только по
    # static/step/drift): реагирует ли sigma_tau (сам гейт, доля переноса СТАРОГО
    # состояния) конкретно на disocclusion_flag? full_gate_diagnostics уже сделал
    # forward(record_gates=True) внутри себя — переиспользуем сохранённый
    # model.last_diag_log, не гоняем модель ещё раз.
    diag = cfc_model.last_diag_log
    disocc_mask = train_batch["disocclusion_flag"].to(device).bool()
    stable_mask = ~disocc_mask
    if disocc_mask.any() and stable_mask.any():
        sig_tau, pre_sig, t_a = diag["sigma_tau"], diag["pre_sigmoid"], diag["t_a"]
        z = diag["z"]
        print(f"    [disocclusion gate reaction check] seed={seed}")
        print(f"      disocclusion=1 (n={int(disocc_mask.sum())}): "
              f"sigma_tau_mean={sig_tau[disocc_mask].mean():.4f} "
              f"pre_sigmoid_mean={pre_sig[disocc_mask].mean():.4f} "
              f"t_a_mean={t_a[disocc_mask].mean():.4f}")
        print(f"      disocclusion=0 (n={int(stable_mask.sum())}): "
              f"sigma_tau_mean={sig_tau[stable_mask].mean():.4f} "
              f"pre_sigmoid_mean={pre_sig[stable_mask].mean():.4f} "
              f"t_a_mean={t_a[stable_mask].mean():.4f}")
        # ЗАПЛАНИРОВАННЫЙ, НО НЕ ВЫПОЛНЕННЫЙ РАНЬШЕ ШАГ (см. mempalace,
        # запись 2026-08-01 про probe-путь): различается ли САМ backbone-
        # выход z (вход в W_a/W_b) между контекстами, ДО того как смотреть
        # на W_a/W_b/lr? Если z сам не различается — проблема выше по
        # потоку (backbone), не в W_a/W_b и не в их learning rate.
        z_disocc_norm = z[disocc_mask].norm(dim=-1).mean()
        z_stable_norm = z[stable_mask].norm(dim=-1).mean()
        z_disocc_mean = z[disocc_mask].mean(dim=0)
        z_stable_mean = z[stable_mask].mean(dim=0)
        z_mean_diff_norm = (z_disocc_mean - z_stable_mean).norm()
        cos_sim = torch.nn.functional.cosine_similarity(
            z_disocc_mean.unsqueeze(0), z_stable_mean.unsqueeze(0)
        ).item()
        print(f"      [backbone z reaction check] "
              f"||z||_disocc={z_disocc_norm:.4f} ||z||_stable={z_stable_norm:.4f} "
              f"||mean(z_disocc)-mean(z_stable)||={z_mean_diff_norm:.4f} "
              f"cos_sim(mean_z_disocc, mean_z_stable)={cos_sim:.4f}")

    preds = {
        "cfc": predict_dense("cfc", cfc_model, eval_batch, device),
        "nrd_faithful": predict_dense("nrd_faithful", None, eval_batch, device),
        "nrd_honest": predict_dense("nrd_honest", None, eval_batch, device),
        "nrc_faithful": predict_dense("nrc_faithful", nrc_f_model, eval_batch, device),
        "nrc_honest": predict_dense("nrc_honest", nrc_h_model, eval_batch, device),
        "gru_faithful": predict_dense("gru_faithful", gru_f_model, eval_batch, device),
        "gru_honest": predict_dense("gru_honest", gru_h_model, eval_batch, device),
    }
    true = eval_batch["true"].numpy()
    t_arr = eval_batch["t"]
    disocc_flags = eval_batch["disocclusion_flag"].numpy()

    seg_offset_per_pixel = [
        label_samples(t_arr[p].astype(np.float64), light_schedule)
        for p in range(len(eval_idx))
    ]
    results = _summarize(MODEL_KINDS, preds, true, seg_offset_per_pixel, len(eval_idx), verbose=verbose)
    disocc_results = _summarize_disocclusion(MODEL_KINDS, preds, true, disocc_flags, len(eval_idx),
                                              verbose=verbose)
    for kind in MODEL_KINDS:
        results[kind]["disocclusion"] = disocc_results[kind]["disocclusion"]
        results[kind]["stable"] = disocc_results[kind]["stable"]
        results[kind]["warp_age_curve"] = disocc_results[kind]["warp_age_curve"]

    return results


def run(dataset_path=DEFAULT_DATASET, n_train_pixels=20, epochs=200, lr=1e-3, hidden_dim=32,
        seed=0, light_schedule_total_duration=400.0, light_schedule_seed=0, gate_lr=None):
    """Однопроходная версия (без значимости) — быстрая отладка на одном 'сиде'."""
    seqs = load_fully_valid_pixel_sequences(dataset_path)
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = _select_device()
    print(f"device: {device}")
    return _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr, hidden_dim, device,
                          gate_lr=gate_lr)


def run_multi_seed(dataset_path=DEFAULT_DATASET, n_seeds=8, n_train_pixels=20, epochs=200,
                    lr=1e-3, hidden_dim=32, light_schedule_total_duration=400.0,
                    light_schedule_seed=0, verbose_per_seed=True, gate_lr=None):
    """§6.4-аналог — n_seeds прогонов _run_one_seed поверх ОДНОГО рендера
    (см. предупреждение в докстринге модуля) + paired bootstrap значимость
    CfC vs честные baseline'ы: агрегатно (early-zone), по типу сегмента
    (§6.3) И — ГЛАВНОЕ ДЛЯ §5 STAGE1B — отдельно в disocclusion-срезе
    (§6 delta), который и определяет исход (не агрегат по кадру)."""
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(f"n_seeds={n_seeds} < {MIN_N_SEEDS} (§6.4)")
    seqs = load_fully_valid_pixel_sequences(dataset_path)
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = _select_device()
    print(f"device: {device}")
    print("\n!!! ВАЖНО: все n_seeds ниже используют ОДИН И ТОТ ЖЕ Blender-рендер (одна")
    print("геометрия, одна траектория камеры, одно световое расписание) — варьируется только")
    print("train/eval разбиение ПИКСЕЛЕЙ. Δt регулярный (§2.3), поэтому здесь нет второй оси")
    print("вариации (per-probe subsampling), которая была в run_experiment_blender.py.")
    print("Значимость измеряет устойчивость ТОЛЬКО к разбиению пикселей. См. докстринг модуля.\n")

    early_by_kind = {k: [] for k in MODEL_KINDS}
    floor_by_kind = {k: [] for k in MODEL_KINDS}
    segtype_by_kind = {k: {name: [] for name in SEGMENT_TYPE_NAMES} for k in MODEL_KINDS}
    disocc_by_kind = {k: [] for k in MODEL_KINDS}
    stable_by_kind = {k: [] for k in MODEL_KINDS}
    warp_age_by_kind = {k: {label: [] for label in BIN_LABELS} for k in MODEL_KINDS}

    for seed in range(n_seeds):
        results = _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr,
                                 hidden_dim, device, verbose=verbose_per_seed, gate_lr=gate_lr)
        for kind in MODEL_KINDS:
            r = results[kind]
            early_by_kind[kind].append(r["early"])
            floor_by_kind[kind].append(r["floor"])
            for name in SEGMENT_TYPE_NAMES:
                segtype_by_kind[kind][name].append(r["per_segment_type"][name])
            disocc_by_kind[kind].append(r["disocclusion"])
            stable_by_kind[kind].append(r["stable"])
            for label in BIN_LABELS:
                warp_age_by_kind[kind][label].append(r["warp_age_curve"][label])
        print(f"seed {seed}: disocclusion(cfc)={disocc_by_kind['cfc'][-1]:.4f} "
              f"stable(cfc)={stable_by_kind['cfc'][-1]:.4f}")

    print("\n=== summary (mean over seeds): early / floor ===")
    for kind in MODEL_KINDS:
        print(f"{kind:14s}  early={np.nanmean(early_by_kind[kind]):.4f}  "
              f"floor={np.nanmean(floor_by_kind[kind]):.4f}")

    print("\n=== summary (mean over seeds): §6 delta — disocclusion / stable ===")
    for kind in MODEL_KINDS:
        print(f"{kind:14s}  disocclusion={np.nanmean(disocc_by_kind[kind]):.4f}  "
              f"stable={np.nanmean(stable_by_kind[kind]):.4f}")

    print("\n=== summary (mean over seeds): error-vs-warp-age (§6 delta) ===")
    header = "kind".ljust(14) + "".join(lbl.rjust(11) for lbl in BIN_LABELS)
    print(header)
    for kind in MODEL_KINDS:
        row = kind.ljust(14)
        for label in BIN_LABELS:
            val = np.nanmean(warp_age_by_kind[kind][label])
            row += f"{val:11.4f}" if not np.isnan(val) else f"{'nan':>11s}"
        print(row)

    print("\n=== §5 stage1b КРИТЕРИЙ: CfC-B vs честные baseline'ы В DISOCCLUSION-СРЕЗЕ ===")
    print("(именно этот блок, не агрегат по кадру, определяет исход §5 — три сценария в ТЗ)")
    disocc_significance = {}
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        a = np.array(disocc_by_kind["cfc"])
        b = np.array(disocc_by_kind[honest_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            print(f"cfc vs {honest_kind} (disocclusion): пропущено — NaN хотя бы в одном сиде")
            continue
        res = paired_bootstrap_significance(a, b)
        disocc_significance[honest_kind] = res
        print(f"cfc vs {honest_kind} (disocclusion): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== для контраста: тот же paired bootstrap в stable-срезе (non-disocclusion) ===")
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        a = np.array(stable_by_kind["cfc"])
        b = np.array(stable_by_kind[honest_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            print(f"cfc vs {honest_kind} (stable): пропущено — NaN хотя бы в одном сиде")
            continue
        res = paired_bootstrap_significance(a, b)
        print(f"cfc vs {honest_kind} (stable): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== для сравнения с probe-путём: агрегат по early-zone (§5.2 старого формата) ===")
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        res = paired_bootstrap_significance(
            np.array(early_by_kind["cfc"]), np.array(early_by_kind[honest_kind]),
        )
        print(f"cfc vs {honest_kind} (early-zone): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    return dict(
        early_by_kind=early_by_kind, floor_by_kind=floor_by_kind,
        segtype_by_kind=segtype_by_kind, disocc_by_kind=disocc_by_kind,
        stable_by_kind=stable_by_kind, warp_age_by_kind=warp_age_by_kind,
        disocc_significance=disocc_significance,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--n_train_pixels", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1,
                         help="1 = однопроходный run() (быстрая отладка); >=8 = run_multi_seed() "
                              "с paired bootstrap значимостью (§6.4-аналог, см. докстринг модуля)")
    parser.add_argument("--gate_lr", type=float, default=None,
                         help="отдельный lr для W_a/W_b гейта CfC (эксперимент 2026-08-09, "
                              "см. docstring train_model_dense); None = единый lr на всё")
    args = parser.parse_args()
    if args.n_seeds == 1:
        run(
            dataset_path=args.dataset, n_train_pixels=args.n_train_pixels, epochs=args.epochs,
            lr=args.lr, hidden_dim=args.hidden_dim, seed=args.seed, gate_lr=args.gate_lr,
        )
    else:
        run_multi_seed(
            dataset_path=args.dataset, n_seeds=args.n_seeds, n_train_pixels=args.n_train_pixels,
            epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim, gate_lr=args.gate_lr,
        )
