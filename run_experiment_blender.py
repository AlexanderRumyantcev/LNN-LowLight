"""
Единый скрипт прогона пайплайна этапа-1 на РЕАЛЬНЫХ данных из Blender
(blender/generate_dataset_result.json) — в отличие от run_experiment.py
(synthetic-путь, data/synthetic_probe_scene.py). Переиспользует build_batch/
train_model/predict оттуда БЕЗ ИЗМЕНЕНИЙ — единственное отличие источника
данных (blender/dataset_adapter.py вместо synthetic-генератора), не
архитектуры/обучения/метрик.

ВАЖНОЕ ОГРАНИЧЕНИЕ (уточнено 2026-08-06, см. run_multi_seed ниже): "seeds" на
реальных данных — это НЕ n_seeds>=8 независимых Blender-сцен (§6.4
synthetic-пути подразумевает именно это). У нас пока ОДИН реальный рендер
(одна геометрия, одно световое расписание). run_multi_seed() варьирует между
"сидами" ДВЕ вещи поверх этого одного рендера: (1) per-probe async subsampling
поток (ProbeSubsampleConfig(seed=seed) — какие именно кадры пула "видит"
каждая проба), (2) train/eval разбиение проб (случайная перестановка вместо
фиксированного first-N/last-M). Это ЧЕСТНО СЛАБЕЕ, чем независимость реальных
synthetic-сцен — геометрия/свет одни и те же для всех "сидов", поэтому
paired_bootstrap_significance здесь измеряет устойчивость к субсэмплингу и
разбиению проб, а НЕ устойчивость к независимым сценам. Это явно печатается
в выводе run_multi_seed(), чтобы результат не читался как более сильное
заявление, чем он есть. run() (однопроходная версия) остаётся как есть —
для быстрой отладки без значимости.

obs_dim=3 (RGB, дыра №3 РЕШЕНА) — метрики (early/floor/per-segment/curve),
рассчитанные в evaluation/metrics.py на скалярных pred/true, здесь
применяются ПОКАНАЛЬНО и усредняются по каналам в одно число на пробу —
сами функции метрик не переписывались под векторный вход, увеличивать их
поверхность (§6.1-6.4 уже зафиксированы под синтетику) не было целью этого
шага.

ИНТЕРПОЛЯЦИЯ МЕЖДУ ПРОБАМИ (§2.3, models/probe_interpolation.py) — ВПЕРВЫЕ
подключена здесь к предсказаниям CfC (а не только к сырому ground truth, как
в tests/test_probe_interpolation_real_data.py). Ключевая сложность: у каждой
пробы СВОЙ асинхронный поток моментов наблюдения (per-probe subsampling, §3.3)
— поэтому предсказания разных проб нельзя просто взять "с одного и того же
индекса шага" и подать в интерполятор, они относятся к разным реальным t.
Решение: для общего "момента запроса" t* каждая проба обрезается по своей
истории ДО t* (её собственные асинхронные наблюдения ≤ t*), CfC прогоняется
на этом обрезке, последнее предсказание в последовательности — "текущее
убеждение" пробы о irradiance к моменту t* (причинно, проба не видит будущего
своей же истории). Эти per-probe "убеждения" на одном t* и интерполируются
между собой (leave-one-out, тот же принцип, что в test_probe_interpolation_
real_data.py, но теперь на выходах CfC, не на raw irradiance) и сравниваются
с реально отрендеренным irradiance в этой точке на ближайшем по t кадре.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "blender"))
from dataset_adapter import load_dataset, build_all_probe_sequences  # noqa: E402
from probe_subsampling import ProbeSubsampleConfig  # noqa: E402
from light_schedule import LightScheduleConfig, build_light_schedule  # noqa: E402

from run_experiment import build_batch, train_model, predict
from models.probe_interpolation import ProbeInterpolator, compute_interpolation_weights
from models.temporal.cfc_probe_module import CfCProbeModule
from evaluation.metrics import (
    label_samples, early_zone_floor_split, per_segment_type_mse, error_vs_offset_curve,
    paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
)

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())

DEFAULT_DATASET = Path(__file__).parent / "blender" / "generate_dataset_result.json"


def _predict_probe_belief_at(model, seq, query_t, device):
    """"Текущее убеждение" CfC о irradiance пробы к моменту query_t: прогоняет
    модель ПРИЧИННО только по тем наблюдениям пробы, что уже случились (t <= query_t),
    берёт последнее предсказание. None, если проба ещё ни разу не обновлялась к query_t."""
    t = seq["t"]
    k = int(np.searchsorted(t, query_t, side="right"))
    if k == 0:
        return None
    obs = torch.tensor(seq["obs"][:k], dtype=torch.float32, device=device).unsqueeze(0)
    cold = torch.tensor(seq["cold_start"][:k], dtype=torch.float32, device=device).unsqueeze(0)
    spp = torch.tensor(seq["spp"][:k], dtype=torch.float32, device=device).unsqueeze(0)
    conf = torch.log1p(spp) / np.log1p(64.0)
    dt = torch.tensor(seq["dt"][:k], dtype=torch.float32, device=device).unsqueeze(0)
    u = CfCProbeModule.build_input(obs, cold, conf, use_staleness=True)
    with torch.no_grad():
        pred_seq, _ = model(u, dt)
    return pred_seq[0, -1].cpu().numpy()


def evaluate_interpolation(dataset, seqs, model, eval_idx, query_times, device):
    """Leave-one-out интерполяция МЕЖДУ eval-пробами НА ПРЕДСКАЗАНИЯХ CfC (не на raw
    irradiance, см. докстринг модуля) в нескольких общих "моментах запроса" query_times.
    Сравнивает результат с реально отрендеренным irradiance на ближайшем по t кадре."""
    frames = dataset["frames"]
    frame_ts = np.array([f["t"] for f in frames])
    positions = np.array(
        [frames[0]["probes"][p]["position"] for p in range(dataset["num_probes"])], dtype=np.float32
    )
    normals = np.array(
        [frames[0]["probes"][p]["normal"] for p in range(dataset["num_probes"])], dtype=np.float32
    )
    vis_matrix = np.array(dataset["visibility_matrix"], dtype=np.float32)

    rel_errors = []
    for qt in query_times:
        beliefs = {p: _predict_probe_belief_at(model, seqs[p], qt, device) for p in eval_idx}
        beliefs = {p: b for p, b in beliefs.items() if b is not None}
        if len(beliefs) < 2:
            continue  # некого интерполировать друг по другу на этом query_t
        frame_idx = int(np.argmin(np.abs(frame_ts - qt)))
        for i in beliefs:
            others = [k for k in beliefs if k != i]
            if not others:
                continue
            query_pos = torch.tensor(positions[i:i + 1], dtype=torch.float32, device=device)
            query_normal = torch.tensor(normals[i:i + 1], dtype=torch.float32, device=device)
            probe_pos = torch.tensor(positions[others], dtype=torch.float32, device=device).unsqueeze(0)
            vis = torch.tensor(vis_matrix[i, others], dtype=torch.float32, device=device).unsqueeze(0)
            cand_irr = torch.tensor(
                np.stack([beliefs[k] for k in others]), dtype=torch.float32, device=device
            ).unsqueeze(0)

            weights = compute_interpolation_weights(query_pos, query_normal, probe_pos, vis)
            pred = torch.einsum("bk,bkc->bc", weights, cand_irr)[0].cpu().numpy()
            true = np.array(frames[frame_idx]["probes"][i]["irradiance"], dtype=np.float32)
            rel_err = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-6))
            rel_errors.append(rel_err)

    return dict(
        mean_rel_err=float(np.mean(rel_errors)) if rel_errors else float("nan"),
        median_rel_err=float(np.median(rel_errors)) if rel_errors else float("nan"),
        n=len(rel_errors),
    )


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reconstruct_light_schedule(total_duration=400.0, seed=0):
    """РЕКОНСТРУКЦИЯ light_schedule: JSON датасета пока не хранит параметры расписания
    (ИЗВЕСТНЫЙ ПРОБЕЛ — стоит когда-нибудь научить generate_dataset.py сохранять их
    в вывод, чтобы не полагаться на память о дефолтах). Используем те же значения,
    что были в generate_dataset.py на момент генерации ТЕКУЩЕГО датасета: total_duration=
    400.0 (TOTAL_DURATION), LightScheduleConfig.seed=0 (дефолт, CLI не переопределялся).
    Подтверждено сверкой: max(t) по факту в датасете = 399.75 < 400.0 — согласуется."""
    return build_light_schedule(LightScheduleConfig(total_duration=total_duration, seed=seed))


def _run_one_seed(dataset, light_schedule, total_duration, seed, n_train_probes, epochs, lr,
                   hidden_dim, device, verbose=True):
    """Один 'сид' на РЕАЛЬНЫХ данных = один subsample-поток + одно train/eval разбиение
    проб поверх ОДНОГО И ТОГО ЖЕ Blender-рендера (см. ограничение в докстринге модуля —
    это НЕ независимая сцена). seed управляет ОБОИМИ: ProbeSubsampleConfig(seed=seed) и
    перестановкой проб для train/eval split (не фиксированный first-N/last-M — иначе
    все 'сиды' делили бы пробы ОДИНАКОВО, и единственным источником вариации был бы
    только subsampling, а не оба заявленных источника)."""
    num_probes = dataset["num_probes"]
    cfg = ProbeSubsampleConfig(seed=seed)
    seqs = build_all_probe_sequences(dataset, cfg, base_seed=seed * 1000)

    perm = np.random.default_rng(seed).permutation(num_probes)
    train_idx = list(perm[:n_train_probes])
    eval_idx = list(perm[n_train_probes:])

    train_batch = build_batch(seqs, train_idx)
    eval_batch = build_batch(seqs, eval_idx)

    model_kinds = ["cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
                   "gru_faithful", "gru_honest"]

    cfc_model = train_model("cfc", train_batch, device, epochs, lr, hidden_dim)
    nrc_f_model = train_model("nrc_faithful", train_batch, device, epochs, lr, hidden_dim)
    nrc_h_model = train_model("nrc_honest", train_batch, device, epochs, lr, hidden_dim)
    gru_f_model = train_model("gru_faithful", train_batch, device, epochs, lr, hidden_dim)
    gru_h_model = train_model("gru_honest", train_batch, device, epochs, lr, hidden_dim)

    preds = {
        "cfc": predict("cfc", cfc_model, eval_batch, device),
        "nrd_faithful": predict("nrd_faithful", None, eval_batch, device),
        "nrd_honest": predict("nrd_honest", None, eval_batch, device),
        "nrc_faithful": predict("nrc_faithful", nrc_f_model, eval_batch, device),
        "nrc_honest": predict("nrc_honest", nrc_h_model, eval_batch, device),
        "gru_faithful": predict("gru_faithful", gru_f_model, eval_batch, device),
        "gru_honest": predict("gru_honest", gru_h_model, eval_batch, device),
    }
    true = eval_batch["true"].numpy()
    t_arr = eval_batch["t"]

    seg_offset_per_probe = [
        label_samples(t_arr[p].astype(np.float64), light_schedule)
        for p in range(len(eval_idx))
    ]
    results = _summarize(model_kinds, preds, true, seg_offset_per_probe, len(eval_idx), verbose=verbose)

    # интерполяция между eval-пробами на предсказаниях CfC (§2.3, см. докстринг модуля)
    query_times = np.linspace(30.0, total_duration - 30.0, 8)
    interp = evaluate_interpolation(dataset, seqs, cfc_model, eval_idx, query_times, device)
    if verbose:
        print(f"\n=== [seed={seed}] интерполяция между пробами (CfC-предсказания, {interp['n']} "
              f"сравнений на {len(query_times)} query_t) ===")
        print(f"mean_rel_err={interp['mean_rel_err']:.4f}  median_rel_err={interp['median_rel_err']:.4f}")
    results["_interpolation"] = interp

    return results


def run(dataset_path=DEFAULT_DATASET, n_train_probes=48, epochs=200, lr=1e-3, hidden_dim=32,
        subsample_seed=0, light_schedule_total_duration=400.0, light_schedule_seed=0):
    """Однопроходная версия (без значимости) — для быстрой отладки пайплайна на одном 'сиде'."""
    dataset = load_dataset(dataset_path)
    if n_train_probes >= dataset["num_probes"]:
        raise ValueError(f"n_train_probes={n_train_probes} >= num_probes={dataset['num_probes']}")
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = _select_device()
    print(f"device: {device}")
    return _run_one_seed(dataset, light_schedule, light_schedule_total_duration, subsample_seed,
                          n_train_probes, epochs, lr, hidden_dim, device)


def run_multi_seed(dataset_path=DEFAULT_DATASET, n_seeds=8, n_train_probes=48, epochs=200,
                    lr=1e-3, hidden_dim=32, light_schedule_total_duration=400.0,
                    light_schedule_seed=0, verbose_per_seed=True):
    """§6.4-аналог для РЕАЛЬНЫХ данных — n_seeds прогонов _run_one_seed поверх ОДНОГО
    Blender-рендера (см. предупреждение в докстринге модуля про то, что именно варьируется
    между 'сидами' здесь) + paired bootstrap значимость CfC vs честные baseline'ы, агрегатно
    и по типу сегмента — тот же формат отчёта, что run_experiment.py (§5.2/§6.3), чтобы
    результаты были сравнимы между synthetic- и Blender-путями."""
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(f"n_seeds={n_seeds} < {MIN_N_SEEDS} (§6.4)")
    dataset = load_dataset(dataset_path)
    if n_train_probes >= dataset["num_probes"]:
        raise ValueError(f"n_train_probes={n_train_probes} >= num_probes={dataset['num_probes']}")
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = _select_device()
    print(f"device: {device}")
    print("\n!!! ВАЖНО: все n_seeds ниже используют ОДИН И ТОТ ЖЕ Blender-рендер (одна геометрия,")
    print("одно световое расписание) — варьируются только per-probe subsampling-поток и")
    print("train/eval разбиение проб. Значимость измеряет устойчивость К ЭТОМУ, НЕ к независимым")
    print("сценам (в отличие от synthetic-пути run_experiment.py). См. докстринг модуля.\n")

    model_kinds = ["cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
                   "gru_faithful", "gru_honest"]
    early_by_kind = {k: [] for k in model_kinds}
    floor_by_kind = {k: [] for k in model_kinds}
    segtype_by_kind = {k: {name: [] for name in SEGMENT_TYPE_NAMES} for k in model_kinds}
    curve_by_kind = {k: {label: [] for label in BIN_LABELS} for k in model_kinds}
    interp_by_seed = []

    for seed in range(n_seeds):
        results = _run_one_seed(dataset, light_schedule, light_schedule_total_duration, seed,
                                 n_train_probes, epochs, lr, hidden_dim, device,
                                 verbose=verbose_per_seed)
        for kind in model_kinds:
            r = results[kind]
            early_by_kind[kind].append(r["early"])
            floor_by_kind[kind].append(r["floor"])
            for name in SEGMENT_TYPE_NAMES:
                segtype_by_kind[kind][name].append(r["per_segment_type"][name])
            for label in BIN_LABELS:
                curve_by_kind[kind][label].append(r["curve"][label])
        interp_by_seed.append(results["_interpolation"])
        print(f"seed {seed}: early(cfc)={early_by_kind['cfc'][-1]:.4f} "
              f"floor(cfc)={floor_by_kind['cfc'][-1]:.4f} "
              f"interp_rel_err={interp_by_seed[-1]['mean_rel_err']:.4f}")

    print("\n=== summary (mean over seeds): early / floor ===")
    for kind in model_kinds:
        print(f"{kind:14s}  early={np.nanmean(early_by_kind[kind]):.4f}  "
              f"floor={np.nanmean(floor_by_kind[kind]):.4f}")

    print("\n=== summary (mean over seeds): per-segment-type MSE (static / step / drift) ===")
    for kind in model_kinds:
        row = "  ".join(f"{name}={np.nanmean(segtype_by_kind[kind][name]):.4f}"
                         for name in SEGMENT_TYPE_NAMES)
        print(f"{kind:14s}  {row}")

    print("\n=== summary (mean over seeds): error-vs-offset-since-jump (STEP-сегменты) ===")
    header = "kind".ljust(14) + "".join(lbl.rjust(11) for lbl in BIN_LABELS)
    print(header)
    for kind in model_kinds:
        row = kind.ljust(14)
        for label in BIN_LABELS:
            val = np.nanmean(curve_by_kind[kind][label])
            row += f"{val:11.4f}" if not np.isnan(val) else f"{'nan':>11s}"
        print(row)

    interp_means = [d["mean_rel_err"] for d in interp_by_seed if not np.isnan(d["mean_rel_err"])]
    print(f"\n=== интерполяция между пробами: mean_rel_err по сидам = "
          f"{np.mean(interp_means):.4f} (n_seeds={len(interp_means)}) ===")

    print("\n=== CfC-B vs честные baseline'ы (paired bootstrap, early-zone, агрегат) ===")
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        res = paired_bootstrap_significance(
            np.array(early_by_kind["cfc"]), np.array(early_by_kind[honest_kind]),
        )
        print(f"cfc vs {honest_kind} (early-zone): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== CfC-B vs честные baseline'ы, ОТДЕЛЬНО ПО ТИПАМ СЕГМЕНТОВ ===")
    for seg_name in SEGMENT_TYPE_NAMES:
        for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
            a = np.array(segtype_by_kind["cfc"][seg_name])
            b = np.array(segtype_by_kind[honest_kind][seg_name])
            if np.isnan(a).any() or np.isnan(b).any():
                print(f"cfc vs {honest_kind} ({seg_name}): пропущено — сегмент не встретился "
                      f"хотя бы в одном сиде")
                continue
            res = paired_bootstrap_significance(a, b)
            print(f"cfc vs {honest_kind} ({seg_name}): mean_diff={res['mean_diff']:.4f} "
                  f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    return dict(early_by_kind=early_by_kind, floor_by_kind=floor_by_kind,
                segtype_by_kind=segtype_by_kind, curve_by_kind=curve_by_kind,
                interp_by_seed=interp_by_seed)


def _summarize(model_kinds, preds, true, seg_offset_per_probe, n_eval_probes, verbose=True):
    """Покадрово-канальные метрики -> одно число на пробу (усреднение по 3 RGB-каналам,
    тем же способом, каким synthetic-путь усредняет по кадрам) -> одно число на весь eval."""
    results = {}
    for kind in model_kinds:
        early_vals, floor_vals = [], []
        seg_mse_accum = {name: [] for name in SEGMENT_TYPE_NAMES}
        curve_accum = {label: [] for label in BIN_LABELS}

        for p in range(n_eval_probes):
            seg_type, offset = seg_offset_per_probe[p]
            pred_p = preds[kind][p].astype(np.float64)   # [T, 3]
            true_p = true[p].astype(np.float64)          # [T, 3]

            e_c, f_c = [], []
            seg_c = {name: [] for name in SEGMENT_TYPE_NAMES}
            curve_c = {label: [] for label in BIN_LABELS}
            for c in range(pred_p.shape[-1]):
                e, f = early_zone_floor_split(pred_p[:, c], true_p[:, c], seg_type, offset)
                if not np.isnan(e):
                    e_c.append(e)
                if not np.isnan(f):
                    f_c.append(f)
                seg_mse = per_segment_type_mse(pred_p[:, c], true_p[:, c], seg_type)
                for name, val in seg_mse.items():
                    if not np.isnan(val):
                        seg_c[name].append(val)
                curve = error_vs_offset_curve(pred_p[:, c], true_p[:, c], seg_type, offset, bin_edges=BIN_EDGES)
                for label, val in curve.items():
                    if not np.isnan(val):
                        curve_c[label].append(val)

            if e_c:
                early_vals.append(float(np.mean(e_c)))
            if f_c:
                floor_vals.append(float(np.mean(f_c)))
            for name in SEGMENT_TYPE_NAMES:
                if seg_c[name]:
                    seg_mse_accum[name].append(float(np.mean(seg_c[name])))
            for label in BIN_LABELS:
                if curve_c[label]:
                    curve_accum[label].append(float(np.mean(curve_c[label])))

        results[kind] = dict(
            early=float(np.mean(early_vals)) if early_vals else float("nan"),
            floor=float(np.mean(floor_vals)) if floor_vals else float("nan"),
            per_segment_type={name: (float(np.mean(vals)) if vals else float("nan"))
                               for name, vals in seg_mse_accum.items()},
            curve={label: (float(np.mean(vals)) if vals else float("nan"))
                   for label, vals in curve_accum.items()},
        )

    if verbose:
        print("\n=== single-seed sanity-check на РЕАЛЬНЫХ Blender-данных ===")
        for kind in model_kinds:
            r = results[kind]
            print(f"{kind:14s}  early={r['early']:.4f}  floor={r['floor']:.4f}")

    if verbose:
        print("\n=== per-segment-type MSE (static / step / drift) ===")
        for kind in model_kinds:
            row = "  ".join(f"{name}={results[kind]['per_segment_type'][name]:.4f}" for name in SEGMENT_TYPE_NAMES)
            print(f"{kind:14s}  {row}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--n_train_probes", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--subsample_seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1,
                         help="1 = однопроходный run() (быстрая отладка); >=8 = run_multi_seed() "
                              "с paired bootstrap значимостью (§6.4-аналог, см. докстринг модуля)")
    args = parser.parse_args()
    if args.n_seeds == 1:
        run(
            dataset_path=args.dataset, n_train_probes=args.n_train_probes, epochs=args.epochs,
            lr=args.lr, hidden_dim=args.hidden_dim, subsample_seed=args.subsample_seed,
        )
    else:
        run_multi_seed(
            dataset_path=args.dataset, n_seeds=args.n_seeds, n_train_probes=args.n_train_probes,
            epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
        )
