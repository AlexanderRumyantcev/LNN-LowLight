"""
Единый скрипт этапа-1 (ТЗ §1.3 sanity-check): генерация synthetic-сцен → обучение CfC-B +
обучаемых baseline'ов (NRC-style) + прямое применение необучаемых baseline'ов (NRD-style) →
evaluation по §6 (per-segment-type, error-vs-offset, early-zone/floor) → paired bootstrap
значимость (§5.2/§6.4) против ЧЕСТНЫХ версий baseline'ов — как по early-zone (агрегат),
так и по каждому типу сегмента отдельно (§6.3).

Рассчитан на запуск на Kaggle GPU (локальная машина — MacBook Air M1, без CUDA): пуш этого
репозитория на GitHub → в Kaggle-ноутбуке git clone → python run_experiment.py.
Устройство выбирается автоматически (cuda, если доступна, иначе cpu).

Дизайн эксперимента (см. §6.4 — n_seeds>=8, §2.1 — online per-scene адаптация):
для каждого из n_seeds НЕЗАВИСИМЫХ synthetic-сцен пробы сцены делятся на train/eval
(одна и та же сцена, разные пробы) — CfC-B и обучаемые baseline'ы (NRC-faithful/honest)
обучаются на train-пробах, оцениваются на eval-пробах той же сцены (проверка обобщения
внутри сцены, согласуется с философией online per-scene адаптации §2.1). NRD-style
(faithful/honest) обучаемых параметров не имеет — применяется напрямую к eval-пробам.

Loss для ВСЕХ обучаемых моделей — одинаковый, БЕЗ confidence-weighting (§5.1: confidence-
weighting — опциональное расширение, не часть базового loss; одинаковый loss для CfC и
NRC-baseline делает сравнение чистым, см. §5.1/§4).

РАСШИРЕНИЕ ОТЧЁТА (2026-07-31, по итогам первого sanity-прогона): агрегированные early/floor
маскируют структуру (тот же урок, что уже был на run_v3 spike-теста, см. §6.1) — агрегат не
даёт ответить на вопрос "ГДЕ именно CfC-B работает/не работает", только "лучше/хуже В СРЕДНЕМ".
Добавлены:
    - per_segment_type_mse (§6.3) — static/step/drift раздельно, а не одно число;
    - error_vs_offset_curve (§6.1) — ОСНОВНАЯ метрика по ТЗ, MSE по бинам offset-с-момента-
      скачка внутри STEP-сегментов; агрегат по early/floor — производная от неё, не наоборот;
    - paired bootstrap значимость (§5.2/§6.4) ПОВТОРЕНА для каждого типа сегмента отдельно
      (не только для early-zone), чтобы увидеть, отличается ли вывод "CfC не выигрывает"
      в зависимости от типа динамики (static/step/drift), а не только усреднённо.
"""

import argparse
import sys

import numpy as np
import torch

from data.synthetic_probe_scene import SceneGenConfig, generate_scene, sample_probe_sequence, SEG_STEP
from models.temporal.cfc_probe_module import CfCProbeModule, full_gate_diagnostics, calibrate_time_gate_init
from models.baselines import NRDStyleBaseline, NRCStyleBaseline
from models.losses import NRCRelativeL2Loss
from evaluation.metrics import (
    label_samples, early_zone_floor_split, per_segment_type_mse, error_vs_offset_curve,
    paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
)

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())  # ["static", "step", "drift"]


def _mean_by_segment_type(values: np.ndarray, segment_type: np.ndarray) -> dict:
    """Как per_segment_type_mse, но для произвольного ЗНАЧЕНИЯ (не квадрата ошибки) —
    здесь используется для среднего гейта σ_τ CfC-B, не для error."""
    out = {}
    for seg_id, name in SEGMENT_NAMES.items():
        mask = segment_type == seg_id
        out[name] = float(values[mask].mean()) if mask.any() else float("nan")
    return out


def _mean_vs_offset_curve(values: np.ndarray, segment_type: np.ndarray, offset: np.ndarray,
                           bin_edges=BIN_EDGES) -> dict:
    """Как error_vs_offset_curve, но усредняет произвольное ЗНАЧЕНИЕ (не квадрат ошибки)
    по бинам offset-с-момента-скачка внутри STEP-сегментов."""
    mask = (segment_type == SEG_STEP) & (offset >= 0) & ~np.isnan(offset)
    v = values[mask]
    off = offset[mask]
    bin_idx = np.digitize(off, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(v[b_mask].mean()) if b_mask.any() else float("nan")
    return curve


def build_batch(seqs, indices):
    """Собирает батч [n_probes, T, ...] по общей МИНИМАЛЬНОЙ длине среди выбранных проб
    (последовательности асинхронны и разной длины — см. data/synthetic_probe_scene.py)."""
    T = min(len(seqs[i]["t"]) for i in indices)
    obs = torch.tensor(np.stack([seqs[i]["obs"][:T] for i in indices]), dtype=torch.float32).unsqueeze(-1)
    dt = torch.tensor(np.stack([seqs[i]["dt"][:T] for i in indices]), dtype=torch.float32)
    cold = torch.tensor(np.stack([seqs[i]["cold_start"][:T] for i in indices]), dtype=torch.float32)
    spp = torch.tensor(np.stack([seqs[i]["spp"][:T] for i in indices]), dtype=torch.float32)
    conf = torch.log1p(spp) / np.log1p(64.0)
    true = torch.tensor(np.stack([seqs[i]["true_irradiance"][:T] for i in indices]), dtype=torch.float32).unsqueeze(-1)
    t_arr = np.stack([seqs[i]["t"][:T] for i in indices])
    return dict(obs=obs, dt=dt, cold=cold, conf=conf, true=true, t=t_arr)


def train_model(kind: str, batch, device, epochs: int, lr: float, hidden_dim: int):
    """kind: 'cfc' | 'nrc_faithful' | 'nrc_honest' — все три обучаемые."""
    obs, dt, cold, conf, true = (
        batch["obs"].to(device), batch["dt"].to(device),
        batch["cold"].to(device), batch["conf"].to(device), batch["true"].to(device),
    )
    if kind == "cfc":
        model = CfCProbeModule(hidden_dim=hidden_dim, use_staleness=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True)
        # Data-driven калибровка масштаба time-gate (2026-08-01, см. mempalace/LNN_LowLight/risks
        # — dt в synthetic_probe_scene.py условный, поэтому масштаб W_a калибруется по фактическим
        # данным на входе, а не хардкодится числом; см. calibrate_time_gate_init).
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt)[0]
    elif kind == "nrc_honest":
        model = NRCStyleBaseline(hidden_dim=hidden_dim, use_staleness=True).to(device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True)
        forward = lambda: model(u)
    elif kind == "nrc_faithful":
        model = NRCStyleBaseline(hidden_dim=hidden_dim, use_staleness=False).to(device)
        u = NRCStyleBaseline.build_input(obs, use_staleness=False)
        forward = lambda: model(u)
    else:
        raise ValueError(kind)

    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = forward()
        loss = loss_fn(pred, true)
        loss.backward()
        opt.step()
    return model


def predict(kind: str, model_or_none, batch, device, tau_nrd: float = 3.0, alpha_nrd: float = 0.3):
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    with torch.no_grad():
        if kind == "cfc":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True)
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrc_honest":
            u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True)
            return model_or_none(u).cpu().numpy()
        if kind == "nrc_faithful":
            u = NRCStyleBaseline.build_input(obs, use_staleness=False)
            return model_or_none(u).cpu().numpy()
        if kind == "nrd_faithful":
            m = NRDStyleBaseline(alpha=alpha_nrd, use_honest_dt=False).to(device)
            return m(obs, dt).cpu().numpy()
        if kind == "nrd_honest":
            m = NRDStyleBaseline(tau=tau_nrd, use_honest_dt=True).to(device)
            return m(obs, dt).cpu().numpy()
    raise ValueError(kind)


def predict_cfc_with_gates(model: CfCProbeModule, batch, device):
    """Как predict('cfc', ...), но дополнительно возвращает средний по hidden_dim гейт
    σ_τ (mixing weight между h_cand и g в CfCProbeCell.forward) на каждом шаге — диагностика
    ДЛЯ ПРОВЕРКИ ГИПОТЕЗЫ о несходимости CfC-B к низкому floor на стабильных участках
    (см. drawer_LNN_LowLight_results_fb8eb27e64cc4c2b24ddbdd9 в mempalace): σ_τ близко к 1 —
    состояние определяется h_cand-веткой (обновление), близко к 0 — g-веткой. НЕ предполагает
    сама по себе, что нужно добавлять event-детектор — это отдельный, уже отложенный вопрос
    (см. drawer про event-функции/ODE-event-B в mempalace, room=decisions, 2026-07-27)."""
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    with torch.no_grad():
        u = model.build_input(obs, cold, conf, use_staleness=True)
        pred, _ = model(u, dt, record_gates=True)
        gate_mean = model.last_gate_log.mean(dim=-1)  # [B, T] — среднее по hidden units
    return pred.cpu().numpy(), gate_mean.cpu().numpy()


def run(n_seeds: int, n_probes: int, n_train_probes: int, epochs: int, lr: float, hidden_dim: int):
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(f"n_seeds={n_seeds} < {MIN_N_SEEDS} (§6.4)")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon (локальная машина, M1)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    model_kinds = ["cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest"]
    early_by_kind = {k: [] for k in model_kinds}
    floor_by_kind = {k: [] for k in model_kinds}
    # §6.3 — MSE по типу сегмента (static/step/drift), отдельный список ПО СИДАМ на каждый kind/тип
    segtype_by_kind = {k: {name: [] for name in SEGMENT_TYPE_NAMES} for k in model_kinds}
    # §6.1 — error-vs-offset-since-jump кривая (только STEP), список ПО СИДАМ на каждый kind/бин
    curve_by_kind = {k: {label: [] for label in BIN_LABELS} for k in model_kinds}

    # ДИАГНОСТИКА (2026-07-31): средний гейт σ_τ CfC-B — только для cfc, не per-kind
    gate_segtype_by_seed = {name: [] for name in SEGMENT_TYPE_NAMES}
    gate_curve_by_seed = {label: [] for label in BIN_LABELS}
    gate_global_stats_by_seed = []  # (mean, std, min, max) по ВСЕМ сэмплам сида, без биннинга

    for seed in range(n_seeds):
        cfg = SceneGenConfig(n_probes=n_probes, seed=seed)
        rng = np.random.default_rng(seed)
        scene = generate_scene(rng, cfg)
        seqs = [sample_probe_sequence(rng, scene["probe_positions"][i], scene, cfg) for i in range(n_probes)]

        train_idx = list(range(n_train_probes))
        eval_idx = list(range(n_train_probes, n_probes))

        train_batch = build_batch(seqs, train_idx)
        eval_batch = build_batch(seqs, eval_idx)

        cfc_model = train_model("cfc", train_batch, device, epochs, lr, hidden_dim)
        nrc_f_model = train_model("nrc_faithful", train_batch, device, epochs, lr, hidden_dim)
        nrc_h_model = train_model("nrc_honest", train_batch, device, epochs, lr, hidden_dim)

        # ПОЛНАЯ gate-диагностика (2026-08-01, за ОДИН проход вместо проверки по одной метрике):
        # pre-sigmoid, W_a(z)/W_b(z) отдельно, z по seg_type, градиентные нормы W_a/W_b —
        # на TRAIN-данных (не eval), т.к. градиентная проверка должна идти на том, что модель
        # реально видела при обучении. См. models/temporal/cfc_probe_module.py::full_gate_diagnostics.
        train_seg_type = np.stack([
            label_samples(train_batch["t"][i].astype(np.float64), scene["light_schedule"])[0]
            for i in range(len(train_idx))
        ])
        u_train = cfc_model.build_input(
            train_batch["obs"].to(device), train_batch["cold"].to(device),
            train_batch["conf"].to(device), use_staleness=True,
        )
        full_gate_diagnostics(
            cfc_model, u_train, train_batch["dt"].to(device), true=train_batch["true"].to(device),
            seg_type=train_seg_type, segment_names=SEGMENT_NAMES,
            label=f"seed={seed} (train)",
        )


        cfc_pred, cfc_gate_mean = predict_cfc_with_gates(cfc_model, eval_batch, device)
        preds = {
            "cfc": cfc_pred,
            "nrd_faithful": predict("nrd_faithful", None, eval_batch, device),
            "nrd_honest": predict("nrd_honest", None, eval_batch, device),
            "nrc_faithful": predict("nrc_faithful", nrc_f_model, eval_batch, device),
            "nrc_honest": predict("nrc_honest", nrc_h_model, eval_batch, device),
        }
        true = eval_batch["true"].numpy()
        t_arr = eval_batch["t"]

        # seg_type/offset не зависят от kind — считаем один раз на пробу, не на каждый kind
        seg_offset_per_probe = [
            label_samples(t_arr[p].astype(np.float64), scene["light_schedule"])
            for p in range(len(eval_idx))
        ]

        # ДИАГНОСТИКА: агрегация среднего гейта σ_τ CfC-B по типу сегмента и по offset
        gate_segtype_accum = {name: [] for name in SEGMENT_TYPE_NAMES}
        gate_curve_accum = {label: [] for label in BIN_LABELS}
        for p in range(len(eval_idx)):
            seg_type, offset = seg_offset_per_probe[p]
            g_p = cfc_gate_mean[p].astype(np.float64)

            seg_stats = _mean_by_segment_type(g_p, seg_type)
            for name, val in seg_stats.items():
                if not np.isnan(val):
                    gate_segtype_accum[name].append(val)

            curve = _mean_vs_offset_curve(g_p, seg_type, offset)
            for label, val in curve.items():
                if not np.isnan(val):
                    gate_curve_accum[label].append(val)

        for name in SEGMENT_TYPE_NAMES:
            vals = gate_segtype_accum[name]
            gate_segtype_by_seed[name].append(float(np.mean(vals)) if vals else float("nan"))
        for label in BIN_LABELS:
            vals = gate_curve_accum[label]
            gate_curve_by_seed[label].append(float(np.mean(vals)) if vals else float("nan"))

        # глобальная статистика гейта по ВСЕМ сэмплам (все пробы, все шаги) этого сида —
        # отличить "гейт застрял ровно на одном значении" от "варьируется, но в среднем ~0.5"
        gate_flat = cfc_gate_mean.reshape(-1).astype(np.float64)
        gate_global_stats_by_seed.append((
            float(gate_flat.mean()), float(gate_flat.std()),
            float(gate_flat.min()), float(gate_flat.max()),
        ))

        for kind in model_kinds:
            early_vals, floor_vals = [], []
            seg_mse_accum = {name: [] for name in SEGMENT_TYPE_NAMES}
            curve_accum = {label: [] for label in BIN_LABELS}

            for p in range(len(eval_idx)):
                seg_type, offset = seg_offset_per_probe[p]
                pred_p = preds[kind][p, :, 0].astype(np.float64)
                true_p = true[p, :, 0].astype(np.float64)

                e, f = early_zone_floor_split(pred_p, true_p, seg_type, offset)
                if not np.isnan(e):
                    early_vals.append(e)
                if not np.isnan(f):
                    floor_vals.append(f)

                seg_mse = per_segment_type_mse(pred_p, true_p, seg_type)
                for name, val in seg_mse.items():
                    if not np.isnan(val):
                        seg_mse_accum[name].append(val)

                curve = error_vs_offset_curve(pred_p, true_p, seg_type, offset, bin_edges=BIN_EDGES)
                for label, val in curve.items():
                    if not np.isnan(val):
                        curve_accum[label].append(val)

            early_by_kind[kind].append(float(np.mean(early_vals)) if early_vals else float("nan"))
            floor_by_kind[kind].append(float(np.mean(floor_vals)) if floor_vals else float("nan"))
            for name in SEGMENT_TYPE_NAMES:
                vals = seg_mse_accum[name]
                segtype_by_kind[kind][name].append(float(np.mean(vals)) if vals else float("nan"))
            for label in BIN_LABELS:
                vals = curve_accum[label]
                curve_by_kind[kind][label].append(float(np.mean(vals)) if vals else float("nan"))

        print(f"seed {seed}: early(cfc)={early_by_kind['cfc'][-1]:.4f} floor(cfc)={floor_by_kind['cfc'][-1]:.4f}")

    print("\n=== summary (mean over seeds): early / floor ===")
    for kind in model_kinds:
        print(f"{kind:14s}  early={np.nanmean(early_by_kind[kind]):.4f}  floor={np.nanmean(floor_by_kind[kind]):.4f}")

    print("\n=== §6.3 summary (mean over seeds): per-segment-type MSE (static / step / drift) ===")
    for kind in model_kinds:
        vals = [np.nanmean(segtype_by_kind[kind][name]) for name in SEGMENT_TYPE_NAMES]
        row = "  ".join(f"{name}={v:.4f}" for name, v in zip(SEGMENT_TYPE_NAMES, vals))
        print(f"{kind:14s}  {row}")

    print("\n=== §6.1 summary (mean over seeds): error-vs-offset-since-jump (STEP-сегменты, основная метрика) ===")
    header = "kind".ljust(14) + "".join(lbl.rjust(11) for lbl in BIN_LABELS)
    print(header)
    for kind in model_kinds:
        row = kind.ljust(14)
        for label in BIN_LABELS:
            val = np.nanmean(curve_by_kind[kind][label])
            row += f"{val:11.4f}" if not np.isnan(val) else f"{'nan':>11s}"
        print(row)

    print("\n=== §5.2 критерий: CfC-B vs честные baseline'ы (paired bootstrap, early-zone, агрегат) ===")
    for honest_kind in ["nrd_honest", "nrc_honest"]:
        res = paired_bootstrap_significance(
            np.array(early_by_kind["cfc"]), np.array(early_by_kind[honest_kind]),
        )
        print(f"cfc vs {honest_kind} (early-zone): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== §6.3 доп. значимость: CfC-B vs честные baseline'ы, ОТДЕЛЬНО ПО ТИПАМ СЕГМЕНТОВ ===")
    print("(тот же paired bootstrap §6.4, но НЕ агрегат — чтобы увидеть, меняется ли вывод")
    print(" в зависимости от типа динамики; NaN у части сидов для сегмента -> пропуск с пометкой)")
    for seg_name in SEGMENT_TYPE_NAMES:
        for honest_kind in ["nrd_honest", "nrc_honest"]:
            a = np.array(segtype_by_kind["cfc"][seg_name])
            b = np.array(segtype_by_kind[honest_kind][seg_name])
            if np.isnan(a).any() or np.isnan(b).any():
                print(f"cfc vs {honest_kind} ({seg_name}): пропущено — сегмент не встретился хотя бы в одном сиде")
                continue
            res = paired_bootstrap_significance(a, b)
            print(f"cfc vs {honest_kind} ({seg_name}): mean_diff={res['mean_diff']:.4f} "
                  f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== ДИАГНОСТИКА: глобальная статистика гейта σ_τ CfC-B (по ВСЕМ сэмплам, все сиды) ===")
    print("(отличить 'гейт застрял на одном значении' от 'варьируется, но в среднем ~0.5')")
    means, stds, mins, maxs = zip(*gate_global_stats_by_seed)
    print(f"mean(mean)={np.mean(means):.4f}  mean(std)={np.mean(stds):.4f}  "
          f"mean(min)={np.mean(mins):.4f}  mean(max)={np.mean(maxs):.4f}")

    print("\n=== ДИАГНОСТИКА: средний гейт σ_τ CfC-B (mixing weight h_cand vs g), по типу сегмента ===")
    print("(σ_τ~1 -> состояние определяется h_cand-веткой (обновление), σ_τ~0 -> g-веткой;")
    print(" гипотеза: на static/drift, где floor должен быть низким, гейт может НЕ стабилизироваться)")
    vals = [np.nanmean(gate_segtype_by_seed[name]) for name in SEGMENT_TYPE_NAMES]
    print("  ".join(f"{name}={v:.4f}" for name, v in zip(SEGMENT_TYPE_NAMES, vals)))

    print("\n=== ДИАГНОСТИКА: средний гейт σ_τ CfC-B vs offset-с-момента-скачка (STEP-сегменты) ===")
    header = "".ljust(0) + "".join(lbl.rjust(11) for lbl in BIN_LABELS)
    print(header)
    row = ""
    for label in BIN_LABELS:
        val = np.nanmean(gate_curve_by_seed[label])
        row += f"{val:11.4f}" if not np.isnan(val) else f"{'nan':>11s}"
    print(row)

    return early_by_kind, floor_by_kind, segtype_by_kind, curve_by_kind, gate_segtype_by_seed, gate_curve_by_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--n_probes", type=int, default=48)
    parser.add_argument("--n_train_probes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    args = parser.parse_args()
    run(args.n_seeds, args.n_probes, args.n_train_probes, args.epochs, args.lr, args.hidden_dim)
