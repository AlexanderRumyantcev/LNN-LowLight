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

    def forward(self, u_t: torch.Tensor, h_prev: torch.Tensor, tau_t: torch.Tensor):
        # u_t: (B, input_dim), h_prev: (B, hidden_dim), tau_t: (B, 1)
        z = self.backbone(torch.cat([u_t, h_prev], dim=-1))
        g = torch.tanh(self.W_g(z))
        h_cand = torch.tanh(self.W_h(z))
        t_a = self.W_a(z)
        t_b = self.W_b(z)
        sigma_tau = torch.sigmoid(t_a * tau_t + t_b)
        self.last_sigma_tau = sigma_tau.detach()
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
        for t in range(T):
            h = self.cell(features[:, t, :], h, tau[:, t, :])
            if record_gates:
                gate_log.append(self.cell.last_sigma_tau)
            outputs.append(self.readout(h))
        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)  # (B,T,hidden) — для диагностики saturation
        return torch.cat(outputs, dim=1)  # (B, T, 1) -> squeeze при использовании


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
