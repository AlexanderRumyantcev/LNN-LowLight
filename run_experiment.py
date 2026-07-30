"""
Единый скрипт этапа-1 (ТЗ §1.3 sanity-check): генерация synthetic-сцен → обучение CfC-B +
обучаемых baseline'ов (NRC-style) + прямое применение необучаемых baseline'ов (NRD-style) →
evaluation по §6 (per-segment-type, early-zone/floor) → paired bootstrap значимость (§5.2/§6.4)
против ЧЕСТНЫХ версий baseline'ов.

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
    label_samples, early_zone_floor_split, per_segment_type_mse,
    paired_bootstrap_significance, MIN_N_SEEDS,
)


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

        for kind in model_kinds:
            early_vals, floor_vals = [], []
            for p in range(len(eval_idx)):
                seg_type, offset = label_samples(t_arr[p].astype(np.float64), scene["light_schedule"])
                e, f = early_zone_floor_split(
                    preds[kind][p, :, 0].astype(np.float64), true[p, :, 0].astype(np.float64),
                    seg_type, offset,
                )
                if not np.isnan(e):
                    early_vals.append(e)
                if not np.isnan(f):
                    floor_vals.append(f)
            early_by_kind[kind].append(float(np.mean(early_vals)) if early_vals else float("nan"))
            floor_by_kind[kind].append(float(np.mean(floor_vals)) if floor_vals else float("nan"))

        print(f"seed {seed}: early(cfc)={early_by_kind['cfc'][-1]:.4f} floor(cfc)={floor_by_kind['cfc'][-1]:.4f}")

    print("\n=== summary (mean over seeds) ===")
    for kind in model_kinds:
        print(f"{kind:14s}  early={np.nanmean(early_by_kind[kind]):.4f}  floor={np.nanmean(floor_by_kind[kind]):.4f}")

    print("\n=== §5.2 критерий: CfC-B vs честные baseline'ы (paired bootstrap, early-zone) ===")
    for honest_kind in ["nrd_honest", "nrc_honest"]:
        res = paired_bootstrap_significance(
            np.array(early_by_kind["cfc"]), np.array(early_by_kind[honest_kind]),
        )
        print(f"cfc vs {honest_kind}: mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    return early_by_kind, floor_by_kind


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
