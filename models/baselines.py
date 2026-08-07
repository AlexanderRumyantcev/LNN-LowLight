"""
Baseline'ы для сравнения с CfC-B (ТЗ §4) — 4 конфигурации, по 2 версии на 2 семейства.

Faithful vs честная версия разводит два разных вопроса (см. ТЗ §4): (а) помогает ли вообще
доступ к Δt/staleness-информации, (б) даёт ли именно CfC-формулировка преимущество СВЕРХ
простого доступа к той же информации. Faithful воспроизводит то, как реально устроен
NRD/NRC в проде (без знания о Δt/staleness); честная версия — тот же механизм + равный
CfC-B доступ к информации.

Сознательно НЕ повторяется инженерная сложность продакшн-версий (variance-based clamping
в NRD, полный online-distillation цикл NRC на GPU) — не меняет алгоритмический вопрос,
который тестирует sanity-check этапа-1 (§1.3/§7.7).
"""

import torch
import torch.nn as nn


class NRDStyleBaseline(nn.Module):
    """
    Hand-crafted exponential accumulation (аналог NVIDIA NRD/ReBLUR/RELAX) — БЕЗ нейросети
    во temporal-части, поэтому нет обучаемых параметров: alpha/tau — фиксированные
    гиперпараметры (как и в реальном NRD, где decay задаётся вручную/эвристикой, не обучением).

    faithful (use_honest_dt=False): out_t = out_{t-1}*(1-alpha) + obs_t*alpha, alpha ФИКСИРОВАН,
        Δt игнорируется вообще (как в реальном NRD, рассчитанном на стабильный fps).
    честная версия (use_honest_dt=True): alpha_t = 1 - exp(-dt_t/tau) — continuous-time
        аналог EMA, даёт baseline'у ту же информацию о времени, что видит CfC.
    """

    def __init__(self, alpha: float = 0.3, tau: float = 3.0, use_honest_dt: bool = False):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.use_honest_dt = use_honest_dt

    def forward(self, obs_seq: torch.Tensor, dt_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B, T, obs_dim]
        dt_seq:  [B, T] или [B, T, 1] — используется только если use_honest_dt=True
        Returns: pred_seq [B, T, obs_dim]
        """
        if dt_seq.dim() == 2:
            dt_seq = dt_seq.unsqueeze(-1)  # [B, T, 1]

        B, T, _ = obs_seq.shape
        out = obs_seq[:, 0]  # cold-start: первое наблюдение как есть (нет истории)
        outputs = [out]
        for t in range(1, T):
            if self.use_honest_dt:
                a = 1.0 - torch.exp(-dt_seq[:, t] / self.tau)  # [B, 1]
            else:
                a = self.alpha
            out = out * (1.0 - a) + obs_seq[:, t] * a
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class NRCStyleBaseline(nn.Module):
    """
    Online per-scene per-frame MLP (аналог NVIDIA NRC / AMD FSR Radiance Cache) — БЕЗ
    рекуррентности: каждый шаг обрабатывается независимо тем же MLP (как в оригинальном
    NRC, где каждый query — независимый forward-pass, а обучение online происходит по
    накопленной статистике сцены, а не через carried-over hidden state между кадрами).

    faithful (use_staleness=False): вход = только obs, как в оригинальном NRC.
    честная версия (use_staleness=True): вход = obs + [cold_start, confidence], как у CfC-B
        (§2.2) — тот же staleness-вектор, тот же источник (log1p(spp) нормализованный).
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_staleness: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.use_staleness = use_staleness
        input_dim = obs_dim + (2 if use_staleness else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = False,
    ) -> torch.Tensor:
        """Тот же формат входа, что CfCProbeModule.build_input (§3.1/§3.2) — для честного
        сравнения оба baseline'а и CfC-B видят идентично собранный staleness-вектор."""
        if not use_staleness:
            return obs
        if cold_start is None or confidence is None:
            raise ValueError("use_staleness=True требует cold_start и confidence")
        staleness = torch.stack([cold_start, confidence], dim=-1)
        return torch.cat([obs, staleness], dim=-1)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        u_seq: [B, T, input_dim] (см. build_input)
        Returns: pred_seq [B, T, obs_dim] — каждый шаг обработан НЕЗАВИСИМО, без
                 рекуррентности (архитектурное отличие от CfC-B/GRU-baseline'ов).
        """
        return self.mlp(u_seq)


class GRUProbeBaseline(nn.Module):
    """
    Дискретный GRU-Δt baseline — 4-я конфигурация (доп. к NRD-style/NRC-style), закрывает
    вопрос "нужен честный baseline: RNN с Δt как доп. фичей, а не голая дискретная версия"
    (методология Vid-ODE Table 5; подтверждено в проекте ранее для видео-варианта до пивота).

    В отличие от NRD-style (без нейросети) и NRC-style (per-frame MLP, без рекуррентности),
    здесь ЕСТЬ и обучаемая нейросеть, И рекуррентность (carried-over hidden state между
    шагами) — но ДИСКРЕТНАЯ (nn.GRUCell), без closed-form continuous-time gate CfC-B.
    Разница honest-GRU vs CfC-B при РАВНОМ доступе к Δt/staleness изолирует именно
    архитектурное преимущество closed-form gate'а, а не сам факт наличия информации о
    времени (то самое разделение вопросов (а)/(б), которое задаёт faithful/честная схема).

    faithful (use_dt_staleness=False): голый discrete GRU по наблюдениям — один шаг сети
        на одно наблюдение, БЕЗ доступа к Δt/staleness вообще. Типичная наивная дискретная
        RNN, не "знающая" физического времени между шагами (не отличает частое обновление
        от редкого) — именно для такого случая литература документирует деградацию на
        нерегулярной последовательности.
    честная версия (use_dt_staleness=True): тот же GRUCell, но log1p(dt) и staleness
        [cold_start, confidence] подаются КАК ДОПОЛНИТЕЛЬНАЯ ФИЧА на входе (конкатенация
        с obs, не через архитектурный механизм) — та же информация, что видит CfC-B через
        свой gate, но использованная безо всякой continuous-time структуры.
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_dt_staleness: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_dt_staleness = use_dt_staleness
        input_dim = obs_dim + (3 if use_dt_staleness else 0)  # +[log1p(dt), cold_start, confidence]
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        dt: torch.Tensor | None = None,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_dt_staleness: bool = False,
    ) -> torch.Tensor:
        """
        obs: [B, T, obs_dim]; dt/cold_start/confidence: [B, T] (нужны только при
        use_dt_staleness=True — та же тройка сигналов, что видит CfC-B, но здесь просто
        конкатенируется с obs как обычная фича, а не подаётся в архитектурный gate).
        """
        if not use_dt_staleness:
            return obs
        if dt is None or cold_start is None or confidence is None:
            raise ValueError("use_dt_staleness=True требует dt, cold_start и confidence")
        dt_log = torch.log1p(dt)
        extra = torch.stack([dt_log, cold_start, confidence], dim=-1)
        return torch.cat([obs, extra], dim=-1)

    def forward(self, u_seq: torch.Tensor, h0: torch.Tensor | None = None):
        """
        u_seq: [B, T, input_dim] (см. build_input)
        h0:    [B, hidden_dim] или None (нулевое состояние — cold-start)
        Returns: (pred_seq, h_final) — тот же интерфейс, что CfCProbeModule.forward,
                 для единообразного вызова из train_model/predict.
        """
        B, T, _ = u_seq.shape
        h = (
            h0 if h0 is not None
            else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        )
        outputs = []
        for t in range(T):
            h = self.cell(u_seq[:, t], h)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1), h
