"""
CfC-B: closed-form continuous-time temporal-модель для world-space проб (ТЗ §2.2).

Отличие от models/temporal/cfc_module.py (старый видео-модуль, до пивота 24.07):
    там вход — spatial illumination feature map [B, C, H, W], здесь — скалярная/векторная
    временная последовательность НАБЛЮДЕНИЙ ОДНОЙ ПРОБЫ (world-space irradiance probe),
    без какой-либо spatial pooling/broadcast — проба уже точка в пространстве, а не карта.

Архитектура (ТЗ §2.2): staleness-вектор (cold-start флаг + confidence из spp, §2.2/§3.1)
    вплетён в тот же backbone z, из которого считаются и кандидаты (g, h_cand), и параметры
    time-gate (sigma_tau) — т.е. влияет на closed-form решение НЕ как последующая коррекция
    выхода, а через ту же представление z, что и обычный вход. Именно так это провалидировано
    в изолированном spike-тесте (TZ_spike_test_CfC_step_detection.md, mode='B') —
    см. spike_test/models/models.py: CfCCell/CfCSequenceModel, mode='B'. Код здесь —
    production-версия того же механизма для формата данных полного пайплайна (§3.1: obs,
    dt, cold_start, confidence), без синтетических oracle-режимов spike-теста.

dirty-флаг от движения сцены сюда НЕ входит (сознательно, §2.2/§7.3) — появится на этапе-2.

Интерполяция между пробами (visibility+normal weighting, §2.3) — забота ДРУГОГО модуля
    (деталь постпроцессинга ДО подачи в CfC-B): этот модуль получает на вход уже готовое
    per-probe наблюдение irradiance, не сырые буферы visibility/normal.
"""

import torch
import torch.nn as nn


class CfCProbeCell(nn.Module):
    """
    Один шаг closed-form update (Hasani et al. 2022), staleness вплетена в backbone z:
        z          = backbone([u_t, h_prev])
        g          = tanh(W_g z)                      -- "новая" кандидат-оценка
        h_cand     = tanh(W_h z)                       -- steady-state кандидат
        sigma_tau  = sigmoid(W_a z * dt_t + W_b z)     -- time-gate (dt = elapsed time)
        h'         = h_cand * sigma_tau + g * (1 - sigma_tau)

    u_t включает staleness-вектор (см. CfCProbeModule.build_input) — поэтому z, а через
    него и sigma_tau/g/h_cand, зависят от staleness не только от dt.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, hidden_dim)  # множитель на dt
        self.W_b = nn.Linear(hidden_dim, hidden_dim)  # смещение
        self.last_sigma_tau: torch.Tensor | None = None  # диагностика saturation гейта

    def forward(
        self,
        u_t: torch.Tensor,
        h_prev: torch.Tensor,
        dt_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        u_t:    [B, input_dim]  — obs (+ staleness-вектор), см. build_input
        h_prev: [B, hidden_dim]
        dt_t:   [B, 1]          — Δt с прошлого обновления этой пробы
        """
        z = self.backbone(torch.cat([u_t, h_prev], dim=-1))
        g = torch.tanh(self.W_g(z))
        h_cand = torch.tanh(self.W_h(z))
        t_a = self.W_a(z)
        t_b = self.W_b(z)
        sigma_tau = torch.sigmoid(t_a * dt_t + t_b)
        self.last_sigma_tau = sigma_tau.detach()
        return h_cand * sigma_tau + g * (1.0 - sigma_tau)


class CfCProbeModule(nn.Module):
    """
    Оборачивает CfCProbeCell по последовательности обновлений ОДНОЙ или БАТЧА проб.

    Батч-размерность — пробы (или последовательности проб), не пиксели/кадры видео:
    веса общие для всех проб (та же логика, что у per-scene MLP в NRC-style baseline, §4),
    сама модель ничего не знает о конкретной геометрии сцены — вся geometry-специфика
    (какая проба где) закодирована в наблюдениях (obs), не в весах.

    Args:
        obs_dim:      размерность самого наблюдения irradiance (обычно 1 — скаляр
                      яркости на канал; при RGB можно вызывать по одному на канал
                      или расширить obs_dim=3, зависит от финального решения по каналам)
        hidden_dim:   размер скрытого состояния CfC
        use_staleness: включать ли staleness-вектор [cold_start, confidence] во вход
                      (§2.2 — основная конфигурация CfC-B; False даёт "голый" CfC,
                      нужен как ablation-точка сравнения, см. spike-test mode='A')
    """

    def __init__(
        self,
        obs_dim: int = 1,
        hidden_dim: int = 32,
        use_staleness: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_staleness = use_staleness

        input_dim = obs_dim + (2 if use_staleness else 0)  # +[cold_start, confidence]
        self.cell = CfCProbeCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = True,
    ) -> torch.Tensor:
        """
        Собирает u_t из сырых буферов пробы (§3.1) в формат входа модели.

        obs:        [B, T, obs_dim]  — уже интерполированное (§2.3) наблюдение irradiance
        cold_start: [B, T]           — флаг "проба ни разу не обновлялась" (§2.2)
        confidence: [B, T]           — log1p(spp) сигнал, нормализованный по известному
                                        глобальному диапазону spp (см. spike-test
                                        models.py::build_features — фиксированная,
                                        не по-батчевая нормализация, важно для
                                        воспроизводимости между запусками)
        """
        if not use_staleness:
            return obs
        if cold_start is None or confidence is None:
            raise ValueError(
                "use_staleness=True требует cold_start и confidence "
                "(получить из spp-метаданных пробы, см. ТЗ §3.1/§3.2)"
            )
        staleness = torch.stack([cold_start, confidence], dim=-1)  # [B, T, 2]
        return torch.cat([obs, staleness], dim=-1)

    def forward(
        self,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        record_gates: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_seq:  [B, T, input_dim] — вход на каждый шаг (см. build_input)
            dt_seq: [B, T] или [B, T, 1] — Δt между последовательными обновлениями пробы
            h0:     [B, hidden_dim] или None (тогда нулевое состояние — cold-start)
            record_gates: сохранить ли sigma_tau по шагам (диагностика насыщения гейта)

        Returns:
            (pred_seq, h_final):
                pred_seq: [B, T, obs_dim] — предсказанная irradiance на каждом шаге
                          (причинно: шаг t не видит входов t+1..T-1)
                h_final:  [B, hidden_dim] — финальное скрытое состояние (для continuation
                          между вызовами — проба живёт дольше одного обучающего окна)
        """
        if dt_seq.dim() == 2:
            dt_seq = dt_seq.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        B, T, _ = u_seq.shape
        h = (
            h0
            if h0 is not None
            else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        )

        outputs = []
        gate_log = [] if record_gates else None
        for t in range(T):
            h = self.cell(u_seq[:, t], h, dt_seq[:, t])
            if record_gates:
                gate_log.append(self.cell.last_sigma_tau)
            outputs.append(self.readout(h))

        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)  # [B, T, hidden_dim]

        return torch.stack(outputs, dim=1), h
