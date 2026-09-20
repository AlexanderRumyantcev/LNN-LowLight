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
  старого 2-/3-элементного вектора. NRD-style использует НОВЫЕ
  режимы (use_disocclusion_reset для faithful, use_mismatch_honest для
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
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "blender"))
from dataset_adapter_dense import build_all_pixel_sequences  # noqa: E402

from run_experiment import build_batch as _build_batch_base
from run_experiment_blender import _select_device, _reconstruct_light_schedule, _summarize
from models.temporal.cfc_probe_module import CfCProbeModule, calibrate_time_gate_init, full_gate_diagnostics
from models.baselines import NRDStyleBaseline, NRCStyleBaseline, GRUProbeBaseline, ODELSTMProbeBaseline
from models.losses import NRCRelativeL2Loss
from models.spatial_features import build_spatial_conditioning, spatial_dim as _spatial_dim_fn
from evaluation.metrics import (
    label_samples, paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
    disocclusion_vs_stable_mse, error_vs_warp_age_curve,
)
from determinism import set_determinism, capture_env_metadata  # 2026-08-28, methodology plan

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())

DEFAULT_DATASET = Path(__file__).parent / "blender" / "generate_dataset_dense_result_full.npz"

SPATIAL_DIM = _spatial_dim_fn(n_freqs_position=4, n_freqs_direction=2, include_roughness=False)

FLOOR_HALF_EXTENT = 4.0

MODEL_KINDS = ["cfc", "cfc_hardjump", "cfc_hardjump_skip", "cfc_mmrnn", "nrd_faithful",
               "nrd_honest", "nrc_faithful", "nrc_honest", "gru_faithful", "gru_honest",
               "ode_lstm_honest", "cfc_spatial", "nrc_honest_spatial", "gru_honest_spatial",
               "ode_lstm_spatial"]
SKIP_KINDS = set()


def load_fully_valid_pixel_sequences(npz_path):
    seqs = build_all_pixel_sequences(npz_path)
    filtered = [s for s in seqs if s["valid"].all()]
    print(f"пикселей всего с последовательностью: {len(seqs)}, "
          f"валидных на всех кадрах: {len(filtered)}")
    return filtered


def build_batch_dense(seqs, indices):
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
    batch["position"] = torch.tensor(
        np.stack([seqs[i]["position"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["normal"] = torch.tensor(
        np.stack([seqs[i]["normal"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["albedo"] = torch.tensor(
        np.stack([seqs[i]["albedo"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["direction"] = torch.tensor(
        np.stack([seqs[i]["direction"][:T] for i in indices]), dtype=torch.float32,
    )
    return batch


def _get_spatial(batch, device):
    return build_spatial_conditioning(
        batch["position"].to(device), batch["direction"].to(device),
        batch["normal"].to(device), batch["albedo"].to(device),
        position_scale=FLOOR_HALF_EXTENT,
    )


_FORCE_CPU_KINDS = {
    "nrc_honest", "nrc_faithful", "nrc_honest_spatial",
    "gru_honest", "gru_faithful", "gru_honest_spatial",
}


def train_model_dense(kind: str, batch, device, epochs: int, lr: float, hidden_dim: int,
                       gate_lr: float | None = None):
    if kind in _FORCE_CPU_KINDS:
        device = torch.device("cpu")
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
    elif kind == "cfc_hardjump":
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
    elif kind == "cfc_hardjump_skip":
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True,
                                use_jump_output_skip=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
    elif kind == "cfc_mmrnn":
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True,
                                use_jump_output_skip=True, use_mixed_memory=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
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
    elif kind == "ode_lstm_honest":
        model = ODELSTMProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                      staleness_dim=4).to(device)
        u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch])
        forward = lambda: model(u, dt)[0]
    elif kind == "cfc_spatial":
        spatial = _get_spatial(batch, device)
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
                               spatial=spatial)
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt)[0]
    elif kind == "nrc_honest_spatial":
        spatial = _get_spatial(batch, device)
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                  staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u)
    elif kind == "gru_honest_spatial":
        spatial = _get_spatial(batch, device)
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=True,
                                  extra_staleness_dim=2, spatial_dim=SPATIAL_DIM).to(device)
        u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u)[0]
    elif kind == "ode_lstm_spatial":
        spatial = _get_spatial(batch, device)
        model = ODELSTMProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                      staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u, dt)[0]
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
    if kind in _FORCE_CPU_KINDS:
        device = torch.device("cpu")
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        if kind == "cfc":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "cfc_hardjump":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
        if kind == "cfc_hardjump_skip":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
        if kind == "cfc_mmrnn":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
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
        if kind == "ode_lstm_honest":
            u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                                  extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "cfc_spatial":
            spatial = _get_spatial(batch, device)
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrc_honest_spatial":
            spatial = _get_spatial(batch, device)
            u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u).cpu().numpy()
        if kind == "gru_honest_spatial":
            spatial = _get_spatial(batch, device)
            u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u)[0].cpu().numpy()
        if kind == "ode_lstm_spatial":
            spatial = _get_spatial(batch, device)
            u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                                  extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrd_faithful":
            m = NRDStyleBaseline(alpha=alpha_nrd, use_disocclusion_reset=True).to(device)
            return m(obs, dt, disocclusion_flag_seq=disocc).cpu().numpy()
        if kind == "nrd_honest":
            m = NRDStyleBaseline(tau=tau_nrd, use_mismatch_honest=True).to(device)
            return m(obs, dt, geometric_mismatch_seq=mismatch).cpu().numpy()
    raise ValueError(kind)


def _summarize_disocclusion(model_kinds, preds, true, disocc_flags, n_eval, verbose=True):
    results = {}
    for kind in model_kinds:
        if kind not in preds:
            results[kind] = dict(
                disocclusion=float("nan"), stable=float("nan"),
                warp_age_curve={label: float("nan") for label in BIN_LABELS},
            )
            continue
        disocc_vals, stable_vals = [], []
        curve_accum = {label: [] for label in BIN_LABELS}
        for p in range(n_eval):
            pred_p = preds[kind][p].astype(np.float64)
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


def _aggregate_gate_diag(diag_list: list[dict]) -> dict:
    def _mean_field(get_group_dict, group_key, field):
        vals = [get_group_dict(d)[group_key][field] for d in diag_list
                 if group_key in get_group_dict(d)]
        return float(np.mean(vals)) if vals else float("nan")

    agg = dict(n_seeds=len(diag_list))

    agg["sigma_tau_mean"] = float(np.mean([d["sigma_tau"]["mean"] for d in diag_list]))
    agg["sat_low_frac"] = float(np.mean([d["sigma_tau"]["sat_low(<0.02)"] for d in diag_list]))
    agg["sat_high_frac"] = float(np.mean([d["sigma_tau"]["sat_high(>0.98)"] for d in diag_list]))
    agg["t_a_abs_mean"] = float(np.mean([d["t_a_dt_scale"]["t_a_abs_mean"] for d in diag_list]))
    agg["t_b_abs_mean"] = float(np.mean([abs(d["t_b=W_b(z)"]["mean"]) for d in diag_list]))

    for layer in ("backbone", "W_a", "W_b"):
        w = [d["grad"][layer]["weight_grad_norm"] for d in diag_list if d.get("grad")]
        b = [d["grad"][layer]["bias_grad_norm"] for d in diag_list if d.get("grad")]
        agg[f"grad_{layer}_weight_norm"] = float(np.mean(w)) if w else float("nan")
        agg[f"grad_{layer}_bias_norm"] = float(np.mean(b)) if b else float("nan")

    all_joint_keys = set()
    for d in diag_list:
        all_joint_keys.update(d.get("by_seg_type_x_disocclusion", {}).keys())
    by_joint = {}
    for key in sorted(all_joint_keys):
        by_joint[key] = dict(
            sigma_tau_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sigma_tau_mean"),
            sat_low_frac=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sat_low_frac"),
            sat_high_frac=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sat_high_frac"),
            t_a_abs_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "t_a_abs_mean"),
            t_b_abs_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "t_b_abs_mean"),
        )
    agg["by_seg_type_x_disocclusion"] = by_joint

    all_disocc_keys = set()
    for d in diag_list:
        all_disocc_keys.update(d.get("by_disocclusion", {}).keys())
    by_disocc = {}
    for key in sorted(all_disocc_keys):
        by_disocc[key] = dict(
            sigma_tau_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sigma_tau_mean"),
            sat_low_frac=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sat_low_frac"),
            sat_high_frac=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sat_high_frac"),
            t_a_abs_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "t_a_abs_mean"),
            t_b_abs_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "t_b_abs_mean"),
        )
    agg["by_disocclusion"] = by_disocc

    return agg


def _train_with_progress(kind, seed, n_seeds, train_batch, device, epochs, lr, hidden_dim,
                          gate_lr=None, verbose=True):
    if kind in SKIP_KINDS:
        if verbose:
            print(f"  [seed {seed+1}/{n_seeds}] {kind} — пропущено (SKIP_KINDS)", flush=True)
        return None
    if verbose:
        print(f"  [seed {seed+1}/{n_seeds}] обучение {kind} ({epochs} эпох)...",
              end="", flush=True)
    t0 = time.perf_counter()
    model = train_model_dense(kind, train_batch, device, epochs, lr, hidden_dim, gate_lr=gate_lr)
    if verbose:
        print(f" готово за {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr, hidden_dim, device,
                   verbose=True, gate_lr=None, n_seeds=1):
    n_pixels = len(seqs)
    if n_train_pixels >= n_pixels:
        raise ValueError(f"n_train_pixels={n_train_pixels} >= доступно валидных пикселей={n_pixels}")

    set_determinism(seed)

    perm = np.random.default_rng(seed).permutation(n_pixels)
    train_idx = list(perm[:n_train_pixels])
    eval_idx = list(perm[n_train_pixels:])

    train_batch = build_batch_dense(seqs, train_idx)
    eval_batch = build_batch_dense(seqs, eval_idx)

    cfc_model = _train_with_progress("cfc", seed, n_seeds, train_batch, device, epochs, lr,
                                      hidden_dim, gate_lr=gate_lr, verbose=verbose)
    cfc_hardjump_model = _train_with_progress("cfc_hardjump", seed, n_seeds, train_batch, device,
                                               epochs, lr, hidden_dim, verbose=verbose)
    cfc_hardjump_skip_model = _train_with_progress("cfc_hardjump_skip", seed, n_seeds, train_batch,
                                                    device, epochs, lr, hidden_dim, verbose=verbose)
    nrc_f_model = _train_with_progress("nrc_faithful", seed, n_seeds, train_batch, device,
                                        epochs, lr, hidden_dim, verbose=verbose)
    nrc_h_model = _train_with_progress("nrc_honest", seed, n_seeds, train_batch, device,
                                        epochs, lr, hidden_dim, verbose=verbose)
    gru_f_model = _train_with_progress("gru_faithful", seed, n_seeds, train_batch, device,
                                        epochs, lr, hidden_dim, verbose=verbose)
    gru_h_model = _train_with_progress("gru_honest", seed, n_seeds, train_batch, device,
                                        epochs, lr, hidden_dim, verbose=verbose)
    ode_lstm_model = _train_with_progress("ode_lstm_honest", seed, n_seeds, train_batch, device,
                                           epochs, lr, hidden_dim, verbose=verbose)
    cfc_spatial_model = _train_with_progress("cfc_spatial", seed, n_seeds, train_batch, device,
                                              epochs, lr, hidden_dim, verbose=verbose)
    nrc_spatial_model = _train_with_progress("nrc_honest_spatial", seed, n_seeds, train_batch,
                                              device, epochs, lr, hidden_dim, verbose=verbose)
    gru_spatial_model = _train_with_progress("gru_honest_spatial", seed, n_seeds, train_batch,
                                              device, epochs, lr, hidden_dim, verbose=verbose)
    ode_lstm_spatial_model = _train_with_progress("ode_lstm_spatial", seed, n_seeds, train_batch,
                                                   device, epochs, lr, hidden_dim, verbose=verbose)

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
    gate_diag = full_gate_diagnostics(
        cfc_model, u_train, train_batch["dt"].to(device), true=train_batch["true"].to(device),
        seg_type=train_seg_type, segment_names=SEGMENT_NAMES,
        disocclusion_flag=train_batch["disocclusion_flag"].to(device),
        label=f"seed={seed} (train, dense)",
    )

    preds = {
        "cfc": predict_dense("cfc", cfc_model, eval_batch, device),
        "cfc_hardjump": predict_dense("cfc_hardjump", cfc_hardjump_model, eval_batch, device),
        "cfc_hardjump_skip": predict_dense("cfc_hardjump_skip", cfc_hardjump_skip_model,
                                            eval_batch, device),
        "nrd_faithful": predict_dense("nrd_faithful", None, eval_batch, device),
        "nrd_honest": predict_dense("nrd_honest", None, eval_batch, device),
        "nrc_faithful": predict_dense("nrc_faithful", nrc_f_model, eval_batch, device),
        "nrc_honest": predict_dense("nrc_honest", nrc_h_model, eval_batch, device),
        "gru_faithful": predict_dense("gru_faithful", gru_f_model, eval_batch, device),
        "gru_honest": predict_dense("gru_honest", gru_h_model, eval_batch, device),
        "nrc_honest_spatial": predict_dense("nrc_honest_spatial", nrc_spatial_model, eval_batch, device),
    }
    for _kind, _model in (
        ("ode_lstm_honest", ode_lstm_model),
        ("ode_lstm_spatial", ode_lstm_spatial_model),
        ("cfc_spatial", cfc_spatial_model),
        ("gru_honest_spatial", gru_spatial_model),
    ):
        if _model is not None:
            preds[_kind] = predict_dense(_kind, _model, eval_batch, device)
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

    results["_gate_diag"] = gate_diag

    return results


def run(dataset_path=DEFAULT_DATASET, n_train_pixels=20, epochs=200, lr=1e-3, hidden_dim=32,
        seed=0, light_schedule_total_duration=400.0, light_schedule_seed=0, gate_lr=None):
    seqs = load_fully_valid_pixel_sequences(dataset_path)
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = _select_device()
    print(f"device: {device}")
    return _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr, hidden_dim, device,
                          gate_lr=gate_lr, n_seeds=1)


def run_multi_seed(dataset_path=DEFAULT_DATASET, n_seeds=8, n_train_pixels=20, epochs=200,
                    lr=1e-3, hidden_dim=32, light_schedule_total_duration=400.0,
                    light_schedule_seed=0, verbose_per_seed=True, gate_lr=None,
                    force_all_cpu=False):
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(f"n_seeds={n_seeds} < {MIN_N_SEEDS} (§6.4)")
    seqs = load_fully_valid_pixel_sequences(dataset_path)
    light_schedule = _reconstruct_light_schedule(light_schedule_total_duration, light_schedule_seed)
    device = torch.device("cpu") if force_all_cpu else _select_device()
    env_meta = capture_env_metadata(device)
    print(f"device: {device}  (backend={env_meta['backend']}, force_all_cpu={force_all_cpu})")

    early_by_kind = {k: [] for k in MODEL_KINDS}
    floor_by_kind = {k: [] for k in MODEL_KINDS}
    segtype_by_kind = {k: {name: [] for name in SEGMENT_TYPE_NAMES} for k in MODEL_KINDS}
    disocc_by_kind = {k: [] for k in MODEL_KINDS}
    stable_by_kind = {k: [] for k in MODEL_KINDS}
    warp_age_by_kind = {k: {label: [] for label in BIN_LABELS} for k in MODEL_KINDS}
    gate_diag_by_seed = []

    for seed in range(n_seeds):
        print(f"\n--- seed {seed+1}/{n_seeds} ---", flush=True)
        results = _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr,
                                 hidden_dim, device, verbose=verbose_per_seed, gate_lr=gate_lr,
                                 n_seeds=n_seeds)
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
        gate_diag_by_seed.append(results["_gate_diag"])
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

    warp_age_pairs = [
        ("ode_lstm_honest", "nrc_honest"),
        ("cfc_hardjump_skip", "nrc_honest"),
        ("cfc_hardjump_skip", "ode_lstm_honest"),
    ]
    for kind_a, kind_b in warp_age_pairs:
        for label in BIN_LABELS:
            a = np.array(warp_age_by_kind[kind_a][label])
            b = np.array(warp_age_by_kind[kind_b][label])
            if len(a) == 0 or len(b) == 0 or np.isnan(a).any() or np.isnan(b).any():
                continue
            res = paired_bootstrap_significance(a, b)
            print(f"{kind_a} vs {kind_b} (warp_age {label}): mean_diff={res['mean_diff']:.4f} "
                  f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    print("\n=== §5 stage1b КРИТЕРИЙ: CfC-B vs честные baseline'ы в DISOCCLUSION-СРЕЗЕ ===")
    disocc_significance = {}
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        a = np.array(disocc_by_kind["cfc"])
        b = np.array(disocc_by_kind[honest_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        res = paired_bootstrap_significance(a, b)
        disocc_significance[honest_kind] = res
        print(f"cfc vs {honest_kind} (disocclusion): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    hardjump_significance = {}
    for other_kind in ["cfc", "nrd_honest", "nrc_honest", "gru_honest", "ode_lstm_honest"]:
        a = np.array(disocc_by_kind["cfc_hardjump"])
        b = np.array(disocc_by_kind[other_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        res = paired_bootstrap_significance(a, b)
        hardjump_significance[other_kind] = res
        print(f"cfc_hardjump vs {other_kind:12s} (disocclusion): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    skip_significance = {}
    for other_kind in ["cfc_hardjump", "nrd_honest", "nrc_honest", "ode_lstm_honest"]:
        a = np.array(disocc_by_kind["cfc_hardjump_skip"])
        b = np.array(disocc_by_kind[other_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        res = paired_bootstrap_significance(a, b)
        skip_significance[other_kind] = res
        print(f"cfc_hardjump_skip vs {other_kind:12s} (disocclusion): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")
    a_stable = np.array(stable_by_kind["cfc_hardjump_skip"])
    b_stable = np.array(stable_by_kind["cfc_hardjump"])
    if not (np.isnan(a_stable).any() or np.isnan(b_stable).any()):
        res = paired_bootstrap_significance(a_stable, b_stable)
        print(f"cfc_hardjump_skip vs cfc_hardjump (stable, регрессия-чек): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    stable_significance = {}
    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        a = np.array(stable_by_kind["cfc"])
        b = np.array(stable_by_kind[honest_kind])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        res = paired_bootstrap_significance(a, b)
        stable_significance[honest_kind] = res
        print(f"cfc vs {honest_kind} (stable): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    for slice_name, by_kind in (("disocclusion", disocc_by_kind), ("stable", stable_by_kind)):
        a = np.array(by_kind["cfc_hardjump"])
        b = np.array(by_kind["ode_lstm_honest"])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        res = paired_bootstrap_significance(a, b)
        print(f"cfc_hardjump vs ode_lstm_honest ({slice_name}): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    for honest_kind in ["nrd_honest", "nrc_honest", "gru_honest"]:
        res = paired_bootstrap_significance(
            np.array(early_by_kind["cfc"]), np.array(early_by_kind[honest_kind]),
        )
        print(f"cfc vs {honest_kind} (early-zone): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    spatial_significance = {}
    for spatial_kind, base_kind in [
        ("cfc_spatial", "cfc"),
        ("nrc_honest_spatial", "nrc_honest"),
        ("gru_honest_spatial", "gru_honest"),
        ("ode_lstm_spatial", "ode_lstm_honest"),
    ]:
        for slice_name, by_kind in (("disocclusion", disocc_by_kind), ("stable", stable_by_kind),
                                     ("early", early_by_kind)):
            a = np.array(by_kind[spatial_kind])
            b = np.array(by_kind[base_kind])
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            res = paired_bootstrap_significance(a, b)
            spatial_significance[f"{spatial_kind}_vs_{base_kind}_{slice_name}"] = res
            print(f"{spatial_kind:20s} vs {base_kind:14s} ({slice_name:12s}): "
                  f"mean_diff={res['mean_diff']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
                  f"significant={res['significant']}")

    a = np.array(stable_by_kind["nrc_honest_spatial"])
    b = np.array(stable_by_kind["cfc"])
    if not (np.isnan(a).any() or np.isnan(b).any()):
        res = paired_bootstrap_significance(a, b)
        spatial_significance["nrc_honest_spatial_vs_cfc_stable"] = res
        print(f"nrc_honest_spatial vs cfc (stable): mean_diff={res['mean_diff']:.4f} "
              f"CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] significant={res['significant']}")

    return dict(
        early_by_kind=early_by_kind, floor_by_kind=floor_by_kind,
        segtype_by_kind=segtype_by_kind, disocc_by_kind=disocc_by_kind,
        stable_by_kind=stable_by_kind, warp_age_by_kind=warp_age_by_kind,
        disocc_significance=disocc_significance, stable_significance=stable_significance,
        hardjump_significance=hardjump_significance, spatial_significance=spatial_significance,
        gate_diag=_aggregate_gate_diag(gate_diag_by_seed),
        env_metadata=env_meta,
    )


def _compare_significance_dicts(name: str, sig_a: dict, sig_b: dict, verbose: bool = True) -> dict:
    verdicts = {}
    for key in sorted(set(sig_a) | set(sig_b)):
        if key not in sig_a or key not in sig_b:
            verdicts[key] = "MISSING_IN_ONE_RUN"
            continue
        a, b = sig_a[key], sig_b[key]
        sign_agree = (a["mean_diff"] > 0) == (b["mean_diff"] > 0)
        sig_agree = a["significant"] == b["significant"]
        if not a["significant"] and not b["significant"]:
            verdicts[key] = "CONFIRMED_TIE"
        elif sign_agree and sig_agree:
            verdicts[key] = "CONFIRMED"
        else:
            verdicts[key] = "UNCONFIRMED_escalate_n_seeds"
        if verbose:
            print(f"  [{name}] {key:22s}: run1(mean_diff={a['mean_diff']:+.4f}, sig={a['significant']}) "
                  f"vs run2(mean_diff={b['mean_diff']:+.4f}, sig={b['significant']}) -> {verdicts[key]}")
    return verdicts


def run_cross_env_check(dataset_path=DEFAULT_DATASET, n_seeds=8, n_train_pixels=20, epochs=200,
                         lr=1e-3, hidden_dim=32, light_schedule_total_duration=400.0,
                         light_schedule_seed=0, gate_lr=None, verbose_per_seed=False):
    print(f"\n{'='*70}\nCROSS-ENVIRONMENT CHECK: run 1/2 (обычный backend)\n{'='*70}")
    run1 = run_multi_seed(
        dataset_path=dataset_path, n_seeds=n_seeds, n_train_pixels=n_train_pixels,
        epochs=epochs, lr=lr, hidden_dim=hidden_dim,
        light_schedule_total_duration=light_schedule_total_duration,
        light_schedule_seed=light_schedule_seed, verbose_per_seed=verbose_per_seed,
        gate_lr=gate_lr, force_all_cpu=False,
    )
    print(f"\n{'='*70}\nCROSS-ENVIRONMENT CHECK: run 2/2 (forced-CPU)\n{'='*70}")
    run2 = run_multi_seed(
        dataset_path=dataset_path, n_seeds=n_seeds, n_train_pixels=n_train_pixels,
        epochs=epochs, lr=lr, hidden_dim=hidden_dim,
        light_schedule_total_duration=light_schedule_total_duration,
        light_schedule_seed=light_schedule_seed, verbose_per_seed=verbose_per_seed,
        gate_lr=gate_lr, force_all_cpu=True,
    )

    print(f"\n{'='*70}\nCROSS-ENVIRONMENT CHECK: сравнение "
          f"({run1['env_metadata']['backend']} vs {run2['env_metadata']['backend']})\n{'='*70}")
    verdicts = {}
    for name in ("disocc_significance", "stable_significance", "hardjump_significance",
                 "spatial_significance"):
        verdicts[name] = _compare_significance_dicts(name, run1[name], run2[name])

    n_unconfirmed = sum(1 for group in verdicts.values() for v in group.values()
                         if v == "UNCONFIRMED_escalate_n_seeds")
    print(f"\nИТОГ: {n_unconfirmed} сравнений UNCONFIRMED из "
          f"{sum(len(g) for g in verdicts.values())} всего.")

    return dict(run1=run1, run2=run2, verdicts=verdicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--n_train_pixels", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--gate_lr", type=float, default=None)
    parser.add_argument("--cross_env_check", action="store_true")
    parser.add_argument("--skip_kinds", type=str, default="")
    args = parser.parse_args()
    if args.skip_kinds:
        SKIP_KINDS.update(k.strip() for k in args.skip_kinds.split(",") if k.strip())
    if args.cross_env_check:
        run_cross_env_check(
            dataset_path=args.dataset, n_seeds=max(args.n_seeds, MIN_N_SEEDS),
            n_train_pixels=args.n_train_pixels, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.hidden_dim, gate_lr=args.gate_lr,
        )
    elif args.n_seeds == 1:
        run(
            dataset_path=args.dataset, n_train_pixels=args.n_train_pixels, epochs=args.epochs,
            lr=args.lr, hidden_dim=args.hidden_dim, seed=args.seed, gate_lr=args.gate_lr,
        )
    else:
        run_multi_seed(
            dataset_path=args.dataset, n_seeds=args.n_seeds, n_train_pixels=args.n_train_pixels,
            epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim, gate_lr=args.gate_lr,
        )
