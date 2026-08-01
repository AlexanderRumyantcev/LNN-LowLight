"""
Модели для spike-теста (ТЗ разделы 4-5).

Обучаемые модели (PyTorch):
    CfCSequenceModel — голый CfC-гейт.
        mode='A'      — вход: только сырое наблюдение obs. Δt входит отдельно как elapsed-time
                          сигнал в гейт (не конкатенируется как обычный признак) — это и есть
                          "не как обычный вход", как решено в проекте.
        mode='B'      — вход: obs + staleness-вектор [cold_start_flag, confidence].
        mode='oracle' — вход: obs (+ staleness если oracle_mode='B') + explicit event_flag
                          (нечестный сигнал момента скачка, верхняя граница).
    GRUBaseline — капасити-контроль: обычная GRU той же скрытой размерности, БЕЗ адаптивного
                  time-gate. dt подаётся как обычный конкатенированный вход (не через elapsed-time
                  механизм) — специально, чтобы не давать GRU то архитектурное преимущество,
                  которое тестируется у CfC.

Ненаучаемые (statistical) baseline'ы — чистые numpy-функции, без обучения:
    ema_baseline(obs, dt, tau)
    precision_weighted_baseline(obs, spp)
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# CfC cell — closed-form update, как в разделе про CfC (Hasani et al. 2022):
#   z = backbone([u, h])
#   g = tanh(W_g z)          -- "новая" кандидат-оценка на основе текущего входа
#   h_cand = tanh(W_h z)     -- альтернативный кандидат (steady-state ветка)
#   sigma_tau = sigmoid(W_a z * tau + W_b z)   -- time-gate, tau = elapsed time (Δt)
#   h' = h_cand * sigma_tau + g * (1 - sigma_tau)
# ---------------------------------------------------------------------------
class CfCCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        backbone_in = input_dim + hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(backbone_in, hidden_dim),
            nn.Tanh(),
        )
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, hidden_dim)  # множитель на tau
        self.W_b = nn.Linear(hidden_dim, hidden_dim)  # смещение
        self.last_sigma_tau = None  # диагностика saturation гейта (ТЗ Приложение А.3)
        self.last_gate_diag = None  # расширенная диагностика: z, t_a, t_b, pre-sigmoid — см. full_gate_diagnostics()

    def forward(self, u_t: torch.Tensor, h_prev: torch.Tensor, tau_t: torch.Tensor):
        # u_t: (B, input_dim), h_prev: (B, hidden_dim), tau_t: (B, 1)
        z = self.backbone(torch.cat([u_t, h_prev], dim=-1))
        g = torch.tanh(self.W_g(z))
        h_cand = torch.tanh(self.W_h(z))
        t_a = self.W_a(z)
        t_b = self.W_b(z)
        pre_sigmoid = t_a * tau_t + t_b
        sigma_tau = torch.sigmoid(pre_sigmoid)
        self.last_sigma_tau = sigma_tau.detach()
        # ВАЖНО: сохраняем БЕЗ detach, чтобы диагностика градиента (full_gate_diagnostics) могла
        # backward-нуться через t_a/t_b и увидеть, доходит ли до W_a/W_b полезный градиент.
        self.last_gate_diag = dict(z=z, t_a=t_a, t_b=t_b, pre_sigmoid=pre_sigmoid, sigma_tau=sigma_tau)
        h_new = h_cand * sigma_tau + g * (1.0 - sigma_tau)
        return h_new


class CfCSequenceModel(nn.Module):
    """
    Оборачивает CfCCell по последовательности, выдаёт скалярное предсказание на каждый шаг.

    feature_dim — размерность "обычных" входных признаков в момент t (без tau, tau обрабатывается
                  отдельно через elapsed-time вход гейта).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.cell = CfCCell(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, tau: torch.Tensor, record_gates: bool = False):
        # features: (B, T, feature_dim), tau: (B, T, 1)
        B, T, _ = features.shape
        h = torch.zeros(B, self.hidden_dim, device=features.device)
        outputs = []
        gate_log = [] if record_gates else None
        diag_logs = {k: [] for k in ("z", "t_a", "t_b", "pre_sigmoid", "sigma_tau")} if record_gates else None
        for t in range(T):
            h = self.cell(features[:, t, :], h, tau[:, t, :])
            if record_gates:
                gate_log.append(self.cell.last_sigma_tau)
                for k in diag_logs:
                    diag_logs[k].append(self.cell.last_gate_diag[k])
            outputs.append(self.readout(h))
        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)  # (B,T,hidden) — для диагностики saturation
            # (B,T,hidden) на каждый ключ — держат граф, если вызывающий код не под no_grad()
            # (нужно для full_gate_diagnostics, которой требуется backward через t_a/t_b)
            self.last_diag_log = {k: torch.stack(v, dim=1) for k, v in diag_logs.items()}
        return torch.cat(outputs, dim=1)  # (B, T, 1) -> squeeze при использовании


# ---------------------------------------------------------------------------
# Комплексная диагностика time-gate CfC (2026-08-01) — ВСЕ метрики за один проход, вместо
# проверки по одной. Отвечает на вопрос: если гейт зажат в узком диапазоне, это (а) W_a/W_b не
# получают полезный градиент, или (б) z (выход backbone) сам не варьируется между
# staleness/dt-паттернами настолько, чтобы "пробить" W_a/W_b.
# ---------------------------------------------------------------------------
def full_gate_diagnostics(model: "CfCSequenceModel", feats: torch.Tensor, tau: torch.Tensor,
                           true: torch.Tensor = None, seg_type: np.ndarray = None,
                           dt_spike_mask: np.ndarray = None, label: str = "sigma_tau", verbose: bool = True):
    """
    Один проход даёт срез сразу по всем гипотезам:
      - saturation sigma_tau (как раньше)
      - pre_sigmoid = t_a*tau + t_b — ДО сжатия сигмоидой (видно, далеко ли от нуля вообще)
      - t_a = W_a(z) и t_b = W_b(z) — отдельно друг от друга и от tau
      - z (выход backbone) — общая статистика + разбивка по seg_type (static/step/drift) и по
        dt_spike_mask (обычный шаг / шаг сразу после Δt-выброса). Если z почти не отличается
        между группами — backbone не различает паттерны, через которые ДОЛЖЕН идти staleness-сигнал.
      - если передан `true` — градиентная норма W_a.weight/bias и W_b.weight/bias после ОДНОГО
        backward (не эпохи обучения) — прямая проверка "доходит ли туда вообще полезный градиент".

    Возвращает dict со всеми метриками; печатает компактную сводку, если verbose.
    """
    model.zero_grad(set_to_none=True)
    grad_info = {}
    if true is not None:
        pred = model(feats, tau, record_gates=True)
        diag = model.last_diag_log
        loss = ((pred.squeeze(-1) - true) ** 2).mean()
        loss.backward()
        for name in ("W_a", "W_b"):
            layer = getattr(model.cell, name)
            w_grad, b_grad = layer.weight.grad, layer.bias.grad
            grad_info[name] = dict(
                weight_grad_norm=float(w_grad.norm().item()) if w_grad is not None else None,
                bias_grad_norm=float(b_grad.norm().item()) if b_grad is not None else None,
            )
        model.zero_grad(set_to_none=True)
        diag = {k: v.detach() for k, v in diag.items()}
    else:
        with torch.no_grad():
            model(feats, tau, record_gates=True)
            diag = model.last_diag_log

    def stats(t):
        t = t.detach()
        return dict(mean=float(t.mean()), std=float(t.std()), min=float(t.min()), max=float(t.max()))

    z, t_a, t_b, pre_sig, sig_tau = diag["z"], diag["t_a"], diag["t_b"], diag["pre_sigmoid"], diag["sigma_tau"]

    result = {
        "sigma_tau": {**stats(sig_tau),
                      "sat_low(<0.02)": float((sig_tau < 0.02).float().mean()),
                      "sat_high(>0.98)": float((sig_tau > 0.98).float().mean())},
        "pre_sigmoid(t_a*tau+t_b)": stats(pre_sig),
        "t_a=W_a(z)": stats(t_a),
        "t_b=W_b(z)": stats(t_b),
        "z(backbone_out)": stats(z),
        "z_per_dim_std_mean": float(z.std(dim=(0, 1)).mean()),  # средний разброс на dim по батчу/времени
        "grad": grad_info,
    }

    if seg_type is not None:
        seg_t = torch.as_tensor(seg_type)
        by_seg = {}
        for name_code, code in [("static", 0), ("step", 1), ("drift", 2)]:
            mask = seg_t == code
            if mask.any():
                by_seg[name_code] = dict(z_norm_mean=float(z[mask].norm(dim=-1).mean()),
                                          t_a_mean=float(t_a[mask].mean()), t_b_mean=float(t_b[mask].mean()))
        result["by_seg_type"] = by_seg

    if dt_spike_mask is not None:
        spike_t = torch.as_tensor(dt_spike_mask).bool()
        by_dt = {}
        for name, mask in [("normal", ~spike_t), ("spike", spike_t)]:
            if mask.any():
                by_dt[name] = dict(z_norm_mean=float(z[mask].norm(dim=-1).mean()),
                                    t_a_mean=float(t_a[mask].mean()))
        result["by_dt_regime"] = by_dt

    if verbose:
        print(f"    [full-gate-diag] {label}")
        st = result["sigma_tau"]
        print(f"      sigma_tau:      mean={st['mean']:.3f} std={st['std']:.3f} "
              f"sat_low={st['sat_low(<0.02)']:.1%} sat_high={st['sat_high(>0.98)']:.1%}")
        ps = result["pre_sigmoid(t_a*tau+t_b)"]
        print(f"      pre-sigmoid:    mean={ps['mean']:.3f} std={ps['std']:.3f} range=[{ps['min']:.3f}, {ps['max']:.3f}]")
        ta, tb = result["t_a=W_a(z)"], result["t_b=W_b(z)"]
        print(f"      t_a=W_a(z):     mean={ta['mean']:.3f} std={ta['std']:.3f} range=[{ta['min']:.3f}, {ta['max']:.3f}]")
        print(f"      t_b=W_b(z):     mean={tb['mean']:.3f} std={tb['std']:.3f} range=[{tb['min']:.3f}, {tb['max']:.3f}]")
        zz = result["z(backbone_out)"]
        print(f"      z(backbone):    mean={zz['mean']:.3f} std={zz['std']:.3f} per_dim_std_mean={result['z_per_dim_std_mean']:.4f}")
        if grad_info:
            for name, gi in grad_info.items():
                wn = gi['weight_grad_norm']; bn = gi['bias_grad_norm']
                print(f"      grad[{name}]:    weight_norm={wn:.5f} bias_norm={bn:.5f}" if wn is not None
                      else f"      grad[{name}]:    None (нет градиента)")
        if "by_seg_type" in result:
            for seg, s in result["by_seg_type"].items():
                print(f"      seg={seg:7s}     z_norm={s['z_norm_mean']:.3f} t_a={s['t_a_mean']:.3f} t_b={s['t_b_mean']:.3f}")
        if "by_dt_regime" in result:
            for regime, s in result["by_dt_regime"].items():
                print(f"      dt={regime:7s}      z_norm={s['z_norm_mean']:.3f} t_a={s['t_a_mean']:.3f}")

    return result


class GRUBaseline(nn.Module):
    """Капасити-контроль: обычная GRUCell без адаптивного time-gate. dt — обычный вход."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        # dt конкатенируется как рядовой признак, а не через elapsed-time механизм
        self.cell = nn.GRUCell(input_size=feature_dim + 1, hidden_size=hidden_dim)
        self.hidden_dim = hidden_dim
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, tau: torch.Tensor):
        B, T, _ = features.shape
        h = torch.zeros(B, self.hidden_dim, device=features.device)
        outputs = []
        for t in range(T):
            inp = torch.cat([features[:, t, :], tau[:, t, :]], dim=-1)
            h = self.cell(inp, h)
            outputs.append(self.readout(h))
        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# Neural ODE + event-функция — третий arm сравнения (см. DESIGN_neuralODE_event.md).
# Упрощение относительно Chen et al. 2020: наблюдения только в дискретных точках сетки,
# поэтому решение "reset или нет" принимается ровно в момент прихода наблюдения, без
# root-finding точного τ_event внутри интервала.
# ---------------------------------------------------------------------------
class ODEEventCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_substeps: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_substeps = n_substeps
        # дрейф-функция ODE: dh/dt = f(h, u) — честная непрерывная интеграция по Δt (RK4)
        self.drift = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # reset-кандидат: зависит ТОЛЬКО от нового наблюдения, не от h_prev (настоящий "забыть")
        self.reset_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        # event-функция: обучаемый детектор скачка, НЕ зависит от tau explicitly (в отличие от CfC)
        self.event_fn = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.last_event_prob = None

    def _velocity(self, h: torch.Tensor, u: torch.Tensor):
        # dh/dt = tanh(raw_drift) - decay*h — ограниченная нелинейная часть (аналог того, что CfC
        # получает "бесплатно" через tanh-candidates) + диссипативный член, гарантирующий
        # устойчивую (не взрывающуюся) динамику при интегрировании по реальному Δt.
        raw = self.drift(torch.cat([h, u], dim=-1))
        return torch.tanh(raw) - 0.5 * h

    def _integrate(self, h: torch.Tensor, u: torch.Tensor, dt: torch.Tensor):
        # RK4 с n_substeps подшагами по интервалу dt. u считаем "замороженным" на интервале —
        # разумное приближение, т.к. между наблюдениями нет новой информации.
        # АДАПТИВНОЕ число подшагов (калибровка 2026-07-27): с тяжёлыми Δt-выбросами (до ~30,
        # см. data/synthetic.py dt_spike_multiplier_range) фиксированный n_substeps=4 даёт
        # sub_dt~7.5 на линейном затухающем члене (-0.5*h) — вне зоны устойчивости явного RK4
        # (порог |λ*sub_dt|≈2.785). Ограничиваем sub_dt сверху, увеличивая число подшагов на
        # выбросах, чтобы не путать архитектурный сигнал с артефактом численного интегрирования.
        max_sub_dt = 1.0
        dt_max = float(dt.max().item()) if dt.numel() > 0 else 0.0
        n_substeps = max(self.n_substeps, int(np.ceil(dt_max / max_sub_dt)))
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            k1 = self._velocity(h, u)
            k2 = self._velocity(h + 0.5 * sub_dt * k1, u)
            k3 = self._velocity(h + 0.5 * sub_dt * k2, u)
            k4 = self._velocity(h + sub_dt * k3, u)
            h = h + (sub_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h

    def forward(self, u_t: torch.Tensor, h_prev: torch.Tensor, tau_t: torch.Tensor):
        h_int = self._integrate(h_prev, u_t, tau_t)
        event_prob = torch.sigmoid(self.event_fn(torch.cat([u_t, h_prev], dim=-1)))
        self.last_event_prob = event_prob.detach()
        h_reset = self.reset_encoder(u_t)
        h_new = h_reset * event_prob + h_int * (1.0 - event_prob)
        return h_new


class NeuralODEEventModel(nn.Module):
    """Оборачивает ODEEventCell по последовательности — тот же интерфейс, что у CfC/GRU моделей."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32, n_substeps: int = 4):
        super().__init__()
        self.cell = ODEEventCell(input_dim=feature_dim, hidden_dim=hidden_dim, n_substeps=n_substeps)
        self.hidden_dim = hidden_dim
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, tau: torch.Tensor, record_gates: bool = False):
        B, T, _ = features.shape
        h = torch.zeros(B, self.hidden_dim, device=features.device)
        outputs = []
        gate_log = [] if record_gates else None
        for t in range(T):
            h = self.cell(features[:, t, :], h, tau[:, t, :])
            if record_gates:
                gate_log.append(self.cell.last_event_prob)
            outputs.append(self.readout(h))
        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)
        return torch.cat(outputs, dim=1)



# ---------------------------------------------------------------------------
# Сборка входных признаков под конкретную абляцию (ТЗ п.4) из батча данных генератора.
# ---------------------------------------------------------------------------
def build_features(batch: dict, mode: str, device="cpu"):
    """
    batch — словарь с ключами obs, dt, spp, event_mask, каждый (B, T) numpy/tensor.
    mode: 'A' | 'B' | 'oracle_A' | 'oracle_B'

    Возвращает (features, tau) — features: (B,T,F), tau: (B,T,1)
    """
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    dt = torch.as_tensor(batch["dt"], dtype=torch.float32, device=device)
    B, T = obs.shape

    feats = [obs.unsqueeze(-1)]

    if mode in ("B", "oracle_B"):
        spp = torch.as_tensor(batch["spp"], dtype=torch.float32, device=device)
        # confidence-сигнал: аналог того, что реальный рендерер знает spp своей оценки.
        # ИСПРАВЛЕНО (ТЗ Приложение А.1): нормализация была по max в батче — нестабильна между
        # батчами/запусками. Теперь фиксированная глобальная нормализация по известному
        # диапазону spp_range из GenConfig (max=64) — детерминированный, воспроизводимый масштаб.
        SPP_LOG_MAX = np.log1p(64.0)  # log1p(GenConfig.spp_range[1]), see data/synthetic.py
        confidence = torch.log1p(spp) / SPP_LOG_MAX
        cold_start = torch.zeros_like(obs)
        cold_start[:, 0] = 1.0
        staleness_vec = torch.stack([cold_start, confidence], dim=-1)  # (B,T,2)
        feats.append(staleness_vec)

    if mode in ("oracle_A", "oracle_B"):
        event = torch.as_tensor(batch["event_mask"], dtype=torch.float32, device=device)
        feats.append(event.unsqueeze(-1))

    features = torch.cat(feats, dim=-1)
    tau = dt.unsqueeze(-1)
    return features, tau


# ---------------------------------------------------------------------------
# Ненаучаемые statistical baseline'ы (ТЗ п.5, "naive-нижняя граница")
# ---------------------------------------------------------------------------
def ema_baseline(obs: np.ndarray, dt: np.ndarray, tau_const: float = 3.0):
    """Exponential moving average с фиксированной постоянной времени tau_const."""
    B, T = obs.shape
    out = np.zeros_like(obs)
    out[:, 0] = obs[:, 0]
    for t in range(1, T):
        alpha = 1.0 - np.exp(-dt[:, t] / tau_const)  # доля доверия новому наблюдению
        out[:, t] = out[:, t - 1] * (1 - alpha) + obs[:, t] * alpha
    return out


def precision_weighted_baseline(obs: np.ndarray, spp: np.ndarray):
    """
    Байесовское накопление независимых наблюдений, взвешенных по точности (~spp).
    Не учитывает Δt вообще (намеренно — это "оптимальная" оценка для статичной истины,
    baseline для сравнения с тем, что Δt-awareness CfC должен обыгрывать на step/drift).
    """
    B, T = obs.shape
    out = np.zeros_like(obs)
    cum_weight = np.zeros(B)
    cum_value = np.zeros(B)
    for t in range(T):
        w = spp[:, t]  # вес ~ точность (обратная дисперсия ~ spp)
        cum_value += obs[:, t] * w
        cum_weight += w
        out[:, t] = cum_value / np.maximum(cum_weight, 1e-6)
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.synthetic import generate_dataset, make_train_test_configs

    train_cfg, test_cfg = make_train_test_configs(0)
    seqs = generate_dataset(4, train_cfg)
    batch = {k: np.stack([s[k] for s in seqs]) for k in ["obs", "dt", "spp", "event_mask", "true_values"]}

    for mode in ["A", "B", "oracle_A", "oracle_B"]:
        feats, tau = build_features(batch, mode)
        model = CfCSequenceModel(feature_dim=feats.shape[-1], hidden_dim=16)
        pred = model(feats, tau)
        loss = ((pred.squeeze(-1) - torch.as_tensor(batch["true_values"])) ** 2).mean()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"mode={mode:10s} feats.shape={tuple(feats.shape)} pred.shape={tuple(pred.shape)} "
              f"loss={loss.item():.3f} grad_norm={grad_norm:.3f}")

    gru = GRUBaseline(feature_dim=1, hidden_dim=16)
    feats_gru, tau_gru = build_features(batch, "A")
    pred_gru = gru(feats_gru, tau_gru)
    print("GRU baseline pred.shape:", tuple(pred_gru.shape))

    ema = ema_baseline(batch["obs"], batch["dt"])
    pw = precision_weighted_baseline(batch["obs"], batch["spp"])
    print("EMA baseline shape:", ema.shape, "MSE vs true:",
          np.mean((ema - batch["true_values"]) ** 2))
    print("Precision-weighted baseline shape:", pw.shape, "MSE vs true:",
          np.mean((pw - batch["true_values"]) ** 2))
