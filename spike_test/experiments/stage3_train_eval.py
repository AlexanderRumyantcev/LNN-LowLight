"""
Этап 3: обучение всех вариантов + оценка по метрикам ТЗ (раздел 6-7).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

import numpy as np
import torch
import torch.nn as nn

from data.synthetic import generate_dataset, make_train_test_configs
from models.models import (
    CfCSequenceModel, GRUBaseline, NeuralODEEventModel, build_features,
    ema_baseline, precision_weighted_baseline, full_gate_diagnostics,
)

DEVICE = "cpu"


def make_batch(seqs, keys=("obs", "dt", "spp", "event_mask", "true_values", "seg_type", "dt_spike_mask")):
    return {k: np.stack([s[k] for s in seqs]) for k in keys}


def train_torch_model(model_ctor, feature_mode, train_seqs, seed, epochs=60, batch_size=16, lr=1e-2):
    torch.manual_seed(seed)
    batch = make_batch(train_seqs)
    feats, tau = build_features(batch, feature_mode, device=DEVICE)
    true = torch.as_tensor(batch["true_values"], dtype=torch.float32, device=DEVICE)

    model = model_ctor(feature_dim=feats.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = feats.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(feats[idx], tau[idx]).squeeze(-1)
            loss = ((pred - true[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1}/{epochs} loss={total_loss/N:.4f}")

    # ДИАГНОСТИКА (ТЗ Приложение А.3 + расширение 2026-08-01): saturation time-gate / event-prob
    # на трейне — отличить "архитектура плоха" от "гейт насытился и градиент не идёт".
    if isinstance(model, CfCSequenceModel):
        # CfC: полная диагностика за один проход — pre-sigmoid, W_a(z)/W_b(z) отдельно, z по
        # seg_type/dt-режиму, градиентные нормы W_a/W_b (нужен batch, а не весь train — иначе
        # backward по всей последовательности дорогой; используем полный train batch,
        # т.к. это диагностика, не шаг оптимизации, и opt.step() после неё не вызывается).
        full_gate_diagnostics(model, feats, tau, true=true, seg_type=batch["seg_type"],
                               dt_spike_mask=batch["dt_spike_mask"], label="sigma_tau")
    elif isinstance(model, NeuralODEEventModel):
        with torch.no_grad():
            model(feats, tau, record_gates=True)
            g = model.last_gate_log
            frac_low = (g < 0.02).float().mean().item()
            frac_high = (g > 0.98).float().mean().item()
            print(f"    [gate-diag] event_prob mean={g.mean().item():.3f} std={g.std().item():.3f} "
                  f"sat_low(<0.02)={frac_low:.1%} sat_high(>0.98)={frac_high:.1%}")
    return model


@torch.no_grad()
def predict_torch_model(model, feature_mode, seqs):
    batch = make_batch(seqs)
    feats, tau = build_features(batch, feature_mode, device=DEVICE)
    pred = model(feats, tau).squeeze(-1).cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# Метрики (ТЗ п.6): error-vs-time-since-event, per-segment-type MSE
# ---------------------------------------------------------------------------
def error_vs_time_since_event(pred: np.ndarray, true: np.ndarray, event_mask: np.ndarray, horizon: int = 25):
    """
    Для каждого события (скачка) собирает квадратичную ошибку на offset=0..horizon-1 шагов ПОСЛЕ
    события, усредняет по всем событиям во всех последовательностях. Возвращает массив (horizon,).
    """
    sq_err = (pred - true) ** 2
    B, T = pred.shape
    buckets = [[] for _ in range(horizon)]
    for b in range(B):
        events = np.where(event_mask[b] == 1)[0]
        for e in events:
            for off in range(horizon):
                t = e + off
                if t < T:
                    buckets[off].append(sq_err[b, t])
    return np.array([np.mean(v) if len(v) > 0 else np.nan for v in buckets])


def steady_state_floor(pred: np.ndarray, true: np.ndarray, event_mask: np.ndarray, seg_type: np.ndarray,
                        min_dist_from_event: int = 15):
    """Средняя ошибка вдали от любых скачков — справочный "пол" установившейся ошибки."""
    sq_err = (pred - true) ** 2
    B, T = pred.shape
    far_mask = np.ones_like(sq_err, dtype=bool)
    for b in range(B):
        events = np.where(event_mask[b] == 1)[0]
        for e in events:
            lo, hi = max(0, e - min_dist_from_event), min(T, e + min_dist_from_event)
            far_mask[b, lo:hi] = False
    return sq_err[far_mask].mean()


def early_zone_error_per_seq(pred: np.ndarray, true: np.ndarray, event_mask: np.ndarray, zone=(0, 5)):
    """
    ГЛАВНЫЙ критерий успеха (DESIGN_neuralODE_event.md): средняя quadratic error в первые
    zone[1]-zone[0] кадров ПОСЛЕ каждого события, агрегированная по всем событиям внутри
    последовательности -> (B,) массив (единица ресемплинга для significance-теста).
    """
    sq_err = (pred - true) ** 2
    B, T = pred.shape
    out = np.full(B, np.nan)
    for b in range(B):
        events = np.where(event_mask[b] == 1)[0]
        vals = []
        for e in events:
            for off in range(zone[0], zone[1]):
                t = e + off
                if t < T:
                    vals.append(sq_err[b, t])
        if vals:
            out[b] = np.mean(vals)
    return out


def mse_per_segment_type(pred: np.ndarray, true: np.ndarray, seg_type: np.ndarray):
    sq_err = (pred - true) ** 2
    out = {}
    for name, code in [("static", 0), ("step", 1), ("drift", 2)]:
        mask = seg_type == code
        out[name] = float(sq_err[mask].mean()) if mask.any() else float("nan")
    return out


def mse_by_dt_regime(pred: np.ndarray, true: np.ndarray, dt_spike_mask: np.ndarray, lookahead: int = 1):
    """
    НОВАЯ метрика (калибровка 2026-07-27): разбивает ошибку на шагах, СЛЕДУЮЩИХ сразу после
    тяжёлого Δt-выброса (модель только что получила скачок elapsed-time на вход), против
    ошибки на обычных шагах. Это прямая проверка того, ради чего вводился выброс — не деградирует
    ли предсказание сразу после нерегулярного скачка Δt, а не только после скачка ЗНАЧЕНИЯ (step).
    lookahead=1 — ошибка ИМЕННО на шаге t, для которого dt[t] был выбросом (elapsed time ДО t).
    """
    sq_err = (pred - true) ** 2
    spike_mask = dt_spike_mask.astype(bool)
    normal_mse = float(sq_err[~spike_mask].mean()) if (~spike_mask).any() else float("nan")
    spike_mse = float(sq_err[spike_mask].mean()) if spike_mask.any() else float("nan")
    return dict(normal=normal_mse, after_spike=spike_mse, n_spike_steps=int(spike_mask.sum()))


def paired_significance_test(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """
    Парное сравнение двух методов по squared error на СОВПАДАЮЩИХ точках (тот же тест-набор).
    Bootstrap по последовательностям (не по отдельным шагам — шаги внутри последовательности
    сильно коррелированы, ресемплинг должен идти по независимой единице — целой последовательности).

    err_a, err_b: (B, T) squared error каждого метода на одном и том же тестовом наборе.
    """
    B = err_a.shape[0]
    per_seq_diff = err_a.mean(axis=1) - err_b.mean(axis=1)  # (B,) разница по каждой последовательности
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(per_seq_diff, size=B, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return dict(
        mean_diff=float(per_seq_diff.mean()),
        ci95=(float(ci_lo), float(ci_hi)),
        significant=not (ci_lo < 0 < ci_hi),  # значимо, если 0 не внутри 95% CI
    )


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------
def run_all(n_train=64, n_test=32, seeds=(0, 1, 2), epochs=60):
    train_cfg, test_cfg = make_train_test_configs(base_seed=0)

    results = {}  # name -> list over seeds of dict(pred, metrics)

    for seed in seeds:
        print(f"=== SEED {seed} ===")
        rng_train = np.random.default_rng(1000 + seed)
        rng_test = np.random.default_rng(2000 + seed)
        train_cfg_s = train_cfg.__class__(**{**train_cfg.__dict__, "seed": 1000 + seed})
        test_cfg_s = test_cfg.__class__(**{**test_cfg.__dict__, "seed": 5000 + seed})

        train_seqs = [__import__("data.synthetic", fromlist=["generate_sequence"]).generate_sequence(rng_train, train_cfg_s)
                      for _ in range(n_train)]
        test_seqs = [__import__("data.synthetic", fromlist=["generate_sequence"]).generate_sequence(rng_test, test_cfg_s)
                     for _ in range(n_test)]
        test_batch = make_batch(test_seqs)

        configs = [
            ("CfC-A (goloy)", CfCSequenceModel, "A"),
            ("CfC-B (+staleness)", CfCSequenceModel, "B"),
            ("CfC-oracle_A", CfCSequenceModel, "oracle_A"),
            ("GRU-control (A-feats)", GRUBaseline, "A"),
            ("GRU-B (+staleness)", GRUBaseline, "B"),  # честный контроль для CfC-B (Приложение А.3)
            ("ODE-event-A", NeuralODEEventModel, "A"),
            ("ODE-event-B (+staleness)", NeuralODEEventModel, "B"),
        ]

        for name, ctor, mode in configs:
            print(f"  training {name} ...")
            model = train_torch_model(lambda feature_dim, ctor=ctor: ctor(feature_dim=feature_dim, hidden_dim=32),
                                       mode, train_seqs, seed=seed, epochs=epochs)
            pred = predict_torch_model(model, mode, test_seqs)
            results.setdefault(name, []).append(pred)

        # statistical baselines (без обучения, детерминированы)
        results.setdefault("EMA", []).append(ema_baseline(test_batch["obs"], test_batch["dt"]))
        results.setdefault("Precision-weighted", []).append(
            precision_weighted_baseline(test_batch["obs"], test_batch["spp"]))

        # сохраним тестовый батч последнего сида для метрик (у каждого сида свой test, метрики считаем per-seed)
        results.setdefault("_test_batches", []).append(test_batch)

    return results


def summarize(results):
    test_batches = results.pop("_test_batches")
    horizon = 25
    summary_curves = {}
    summary_table = []

    for name, preds_per_seed in results.items():
        curves = []
        seg_mses = []
        floors = []
        early_zones = []
        dt_regimes = []
        for pred, tb in zip(preds_per_seed, test_batches):
            true = tb["true_values"]
            ev = tb["event_mask"]
            seg = tb["seg_type"]
            curves.append(error_vs_time_since_event(pred, true, ev, horizon=horizon))
            seg_mses.append(mse_per_segment_type(pred, true, seg))
            floors.append(steady_state_floor(pred, true, ev, seg))
            early_zones.append(np.nanmean(early_zone_error_per_seq(pred, true, ev)))
            dt_regimes.append(mse_by_dt_regime(pred, true, tb["dt_spike_mask"]))
        mean_curve = np.nanmean(np.stack(curves), axis=0)
        summary_curves[name] = mean_curve
        mean_seg = {k: float(np.nanmean([d[k] for d in seg_mses])) for k in ["static", "step", "drift"]}
        mean_floor = float(np.mean(floors))
        mean_early = float(np.nanmean(early_zones))
        mean_dt_normal = float(np.nanmean([d["normal"] for d in dt_regimes]))
        mean_dt_spike = float(np.nanmean([d["after_spike"] for d in dt_regimes]))
        summary_table.append((name, mean_seg["static"], mean_seg["step"], mean_seg["drift"], mean_floor, mean_early,
                               mean_dt_normal, mean_dt_spike))

    # Статистическая значимость CfC-A vs GRU-control (ТЗ Приложение А.2) — парно, по сидам,
    # bootstrap по последовательностям, отдельно на step-сегментах (там разница интересна больше всего)
    sig_result = None
    if "CfC-A (goloy)" in results and "GRU-control (A-feats)" in results:
        err_a_all, err_b_all = [], []
        for pred_a, pred_b, tb in zip(results["CfC-A (goloy)"], results["GRU-control (A-feats)"], test_batches):
            step_mask = tb["seg_type"] == 1  # только step-сегменты
            true = tb["true_values"]
            sq_a = (pred_a - true) ** 2
            sq_b = (pred_b - true) ** 2
            # усредняем по step-точкам ВНУТРИ последовательности, чтобы сохранить B как единицу ресемплинга
            row_a = np.array([sq_a[i, step_mask[i]].mean() if step_mask[i].any() else np.nan
                               for i in range(sq_a.shape[0])])
            row_b = np.array([sq_b[i, step_mask[i]].mean() if step_mask[i].any() else np.nan
                               for i in range(sq_b.shape[0])])
            err_a_all.append(row_a)
            err_b_all.append(row_b)
        err_a_all = np.concatenate(err_a_all)
        err_b_all = np.concatenate(err_b_all)
        valid = ~np.isnan(err_a_all) & ~np.isnan(err_b_all)
        sig_result = paired_significance_test(err_a_all[valid].reshape(-1, 1), err_b_all[valid].reshape(-1, 1))

    # Приложение А.3: честная проверка CfC-B vs GRU-B (те же staleness-фичи у обоих) на step-сегментах
    sig_result_b = None
    if "CfC-B (+staleness)" in results and "GRU-B (+staleness)" in results:
        err_a_all, err_b_all = [], []
        for pred_a, pred_b, tb in zip(results["CfC-B (+staleness)"], results["GRU-B (+staleness)"], test_batches):
            step_mask = tb["seg_type"] == 1
            true = tb["true_values"]
            sq_a = (pred_a - true) ** 2
            sq_b = (pred_b - true) ** 2
            row_a = np.array([sq_a[i, step_mask[i]].mean() if step_mask[i].any() else np.nan
                               for i in range(sq_a.shape[0])])
            row_b = np.array([sq_b[i, step_mask[i]].mean() if step_mask[i].any() else np.nan
                               for i in range(sq_b.shape[0])])
            err_a_all.append(row_a)
            err_b_all.append(row_b)
        err_a_all = np.concatenate(err_a_all)
        err_b_all = np.concatenate(err_b_all)
        valid = ~np.isnan(err_a_all) & ~np.isnan(err_b_all)
        sig_result_b = paired_significance_test(err_a_all[valid].reshape(-1, 1), err_b_all[valid].reshape(-1, 1))

    # Главный критерий (DESIGN_neuralODE_event.md): significance на EARLY-ZONE (offset 0-4),
    # а не на усреднённой step-MSE — early-zone metric напрямую отражает скорость реакции
    # на событие, что и есть предмет сравнения hard-reset (ODE-event) vs soft-gate (CfC).
    def early_zone_significance(name_a, name_b):
        if name_a not in results or name_b not in results:
            return None
        err_a_all, err_b_all = [], []
        for pred_a, pred_b, tb in zip(results[name_a], results[name_b], test_batches):
            true = tb["true_values"]
            ev = tb["event_mask"]
            row_a = early_zone_error_per_seq(pred_a, true, ev)
            row_b = early_zone_error_per_seq(pred_b, true, ev)
            err_a_all.append(row_a)
            err_b_all.append(row_b)
        err_a_all = np.concatenate(err_a_all)
        err_b_all = np.concatenate(err_b_all)
        valid = ~np.isnan(err_a_all) & ~np.isnan(err_b_all)
        return paired_significance_test(err_a_all[valid].reshape(-1, 1), err_b_all[valid].reshape(-1, 1))

    sig_early_ode_vs_cfc = early_zone_significance("ODE-event-B (+staleness)", "CfC-B (+staleness)")
    sig_early_ode_vs_gru = early_zone_significance("ODE-event-B (+staleness)", "GRU-B (+staleness)")

    return summary_curves, summary_table, sig_result, sig_result_b, sig_early_ode_vs_cfc, sig_early_ode_vs_gru


if __name__ == "__main__":
    # n_seeds увеличен 3->8 (2026-07-27): при 3 сидах оценка значимости оказалась неустойчивой
    # от конкретной случайной выборки к выборке (см. mempalace: ODE-event-B vs CfC-B significant
    # менялось между прогонами без изменения изучаемого механизма) — больше сидов для устойчивости.
    results = run_all(n_train=48, n_test=32, seeds=(0, 1, 2, 3, 4, 5, 6, 7), epochs=40)
    curves, table, sig_result, sig_result_b, sig_early_ode_vs_cfc, sig_early_ode_vs_gru = summarize(results)

    print("\n=== MSE по типам сегментов + early-zone(offset0-4) + Δt-режим (усреднено по сидам) ===")
    print(f"{'method':26s} {'static':>10s} {'step':>10s} {'drift':>10s} {'floor(far)':>12s} {'early(0-4)':>12s} "
          f"{'dt_normal':>10s} {'dt_spike':>10s}")
    for row in table:
        print(f"{row[0]:26s} {row[1]:10.4f} {row[2]:10.4f} {row[3]:10.4f} {row[4]:12.4f} {row[5]:12.4f} "
              f"{row[6]:10.4f} {row[7]:10.4f}")
    print("(dt_spike = MSE на шагах СРАЗУ после тяжёлого Δt-выброса; dt_normal = MSE на остальных шагах;"
          " калибровка выбросов — 2026-07-27)")

    print("\n=== Значимость CfC-A vs GRU-control на step-сегментах (bootstrap по последовательностям) ===")
    if sig_result:
        print(f"mean_diff (CfC-A - GRU-control) = {sig_result['mean_diff']:.5f}")
        print(f"95% CI = [{sig_result['ci95'][0]:.5f}, {sig_result['ci95'][1]:.5f}]")
        print(f"значимо (0 вне CI): {sig_result['significant']}")

    print("\n=== Значимость CfC-B vs GRU-B на step-сегментах (честный контроль, Приложение А.3) ===")
    if sig_result_b:
        print(f"mean_diff (CfC-B - GRU-B) = {sig_result_b['mean_diff']:.5f}")
        print(f"95% CI = [{sig_result_b['ci95'][0]:.5f}, {sig_result_b['ci95'][1]:.5f}]")
        print(f"значимо (0 вне CI): {sig_result_b['significant']}")

    print("\n=== Значимость ODE-event-B vs CfC-B на EARLY-ZONE offset 0-4 (главный критерий) ===")
    if sig_early_ode_vs_cfc:
        print(f"mean_diff (ODE-event-B - CfC-B) = {sig_early_ode_vs_cfc['mean_diff']:.5f}")
        print(f"95% CI = [{sig_early_ode_vs_cfc['ci95'][0]:.5f}, {sig_early_ode_vs_cfc['ci95'][1]:.5f}]")
        print(f"значимо (0 вне CI): {sig_early_ode_vs_cfc['significant']}")

    print("\n=== Значимость ODE-event-B vs GRU-B на EARLY-ZONE offset 0-4 ===")
    if sig_early_ode_vs_gru:
        print(f"mean_diff (ODE-event-B - GRU-B) = {sig_early_ode_vs_gru['mean_diff']:.5f}")
        print(f"95% CI = [{sig_early_ode_vs_gru['ci95'][0]:.5f}, {sig_early_ode_vs_gru['ci95'][1]:.5f}]")
        print(f"значимо (0 вне CI): {sig_early_ode_vs_gru['significant']}")

    np.savez(os.path.join(OUT_DIR, "stage3_curves.npz"),
             **{k: v for k, v in curves.items()})
    import json
    with open(os.path.join(OUT_DIR, "stage3_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    with open(os.path.join(OUT_DIR, "stage3_significance.json"), "w") as f:
        json.dump({
            "CfC-A_vs_GRU-control": sig_result,
            "CfC-B_vs_GRU-B": sig_result_b,
            "ODE-event-B_vs_CfC-B_earlyzone": sig_early_ode_vs_cfc,
            "ODE-event-B_vs_GRU-B_earlyzone": sig_early_ode_vs_gru,
        }, f, indent=2)
    print("\nsaved curves + table + significance to outputs/")
