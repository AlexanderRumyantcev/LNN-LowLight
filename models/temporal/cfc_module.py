"""
CfC (Closed-form Continuous-time) temporal module для моделирования
динамики illumination map между кадрами видео.

Архитектура:
  illumination_fea [B, C, H, W] (последовательность кадров)
  → spatial pooling → вектор [B, C']
  → CfC (ODE-based RNN) → обновлённый вектор
  → spatial broadcast → illum_fea_enhanced [B, C, H, W]

Использует ncps (neural circuit policies) от Ramin Hasani et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class SpatialPooling(nn.Module):
    """Сворачивает spatial illumination map в вектор для LNN."""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        # Лёгкий conv для адаптации каналов перед pooling
        self.adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # global avg pool → [B, C', 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, out_dim]"""
        x = self.adapt(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, out_dim]


class SpatialBroadcast(nn.Module):
    """Разворачивает вектор обратно в spatial feature map."""

    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_channels)

    def forward(self, vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        vec: [B, in_dim]
        ref: [B, C, H, W] — референс для H, W
        → [B, out_channels, H, W]
        """
        B, C, H, W = ref.shape
        out = self.proj(vec)           # [B, out_channels]
        out = out[:, :, None, None]    # [B, out_channels, 1, 1]
        out = out.expand(B, -1, H, W)  # broadcast по spatial
        return out


class CfCTemporalModule(nn.Module):
    """
    Темпоральный модуль на основе CfC для моделирования динамики освещения.

    Принимает последовательность illumination features из N кадров,
    возвращает обновлённую feature для текущего кадра.

    Args:
        in_channels:  число каналов входного illu_fea (из RetinexFormer estimator)
        hidden_dim:   размер скрытого пространства CfC
        n_neurons:    число нейронов в CfC сети
        window_size:  число кадров в скользящем окне (causal: текущий + прошлые)
    """

    def __init__(
        self,
        in_channels: int = 31,
        hidden_dim: int = 64,
        n_neurons: int = 32,
        window_size: int = 5,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Spatial → vector
        self.pooling = SpatialPooling(in_channels, hidden_dim)

        # CfC с AutoNCP wiring
        # AutoNCP требует: n_neurons > hidden_dim + 2
        # Гарантируем это автоматически
        effective_neurons = max(n_neurons, hidden_dim + 4)
        wiring = AutoNCP(effective_neurons, hidden_dim)
        self.cfc = CfC(hidden_dim, wiring, batch_first=True)

        # Vector → spatial
        self.broadcast = SpatialBroadcast(hidden_dim, in_channels)

        # Residual gate: сколько темпорального контекста добавить
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        illu_fea_seq: torch.Tensor,
        timespans: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            illu_fea_seq: [B, T, C, H, W] — последовательность illumination features
                          T = window_size, последний кадр — текущий
            timespans:    [B, T] — временные интервалы между кадрами (∆t = 1/fps).
                          Если None — используется равномерный ∆t=1.
        Returns:
            illu_fea_enhanced: [B, C, H, W] — enhanced illumination для текущего кадра
        """
        B, T, C, H, W = illu_fea_seq.shape

        # Свернуть каждый кадр в вектор: [B, T, hidden_dim]
        vecs = []
        for t in range(T):
            vecs.append(self.pooling(illu_fea_seq[:, t]))  # [B, hidden_dim]
        seq = torch.stack(vecs, dim=1)  # [B, T, hidden_dim]

        # Прогнать через CfC с корректными timespans.
        #
        # Баг в ncps (все версии на момент 2026-06): внутри CfC.forward делается
        #   ts = timespans[:, t].squeeze()
        # при B>1 это даёт [B] вместо [B,1], что ломает broadcast с t_a [B, H].
        #
        # Обход: реализуем пошаговый forward руками, передавая ts как [B, 1].
        # Используем rnn_cell и fc напрямую — публичные атрибуты CfC.
        h_state = torch.zeros(B, self.cfc.state_size, device=seq.device, dtype=seq.dtype)
        output_sequence = []
        for t in range(T):
            inputs_t = seq[:, t]          # [B, hidden_dim]
            if timespans is not None:
                ts_t = timespans[:, t].unsqueeze(-1)   # [B, 1] — корректный broadcast
            else:
                ts_t = torch.ones(B, 1, device=seq.device, dtype=seq.dtype)
            h_out, h_state = self.cfc.rnn_cell(inputs_t, h_state, ts_t)
            output_sequence.append(self.cfc.fc(h_out))   # [B, hidden_dim]
        out_seq = torch.stack(output_sequence, dim=1)     # [B, T, hidden_dim]

        # Берём выход для последнего (текущего) кадра
        out_vec = out_seq[:, -1, :]  # [B, hidden_dim]

        # Развернуть обратно в spatial
        temporal_fea = self.broadcast(out_vec, illu_fea_seq[:, -1])  # [B, C, H, W]

        # Residual gate: смешиваем оригинал и темпоральный контекст
        current_fea = illu_fea_seq[:, -1]  # [B, C, H, W]
        gate_input = torch.cat([current_fea, temporal_fea], dim=1)
        alpha = self.gate(gate_input)  # [B, C, H, W] in [0,1]
        out = alpha * temporal_fea + (1 - alpha) * current_fea

        return out
