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

from data.synthetic_probe_scene import SceneGenConfig, generate_scene, sample_probe_sequence
from models.temporal.cfc_probe_module import CfCProbeModule
from models.baselines import NRDStyleBaseline, NRCStyleBaseline
from models.losses import NRCRelativeL2Loss
from evaluation.metrics import (
    label_samples, early_zone_floor_split, per_segment_type_mse, error_vs_offset_curve,
    paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
)

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())  # ["static", "step", "drift"]


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

        preds = {
            "cfc": predict("cfc", cfc_model, eval_batch, device),
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

    return early_by_kind, floor_by_kind, segtype_by_kind, curve_by_kind


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
