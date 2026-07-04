"""
LSTM temporal module — baseline с тем же интерфейсом что CfCTemporalModule.
Используется для прямого сравнения: LNN vs LSTM.
"""

import torch
import torch.nn as nn


class SpatialPooling(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.adapt(x)).flatten(1)


class SpatialBroadcast(nn.Module):
    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_channels)

    def forward(self, vec, ref):
        B, C, H, W = ref.shape
        out = self.proj(vec)[:, :, None, None].expand(B, -1, H, W)
        return out


class LSTMTemporalModule(nn.Module):
    """
    LSTM temporal module — прямой baseline для сравнения с CfC.
    Интерфейс идентичен CfCTemporalModule: forward() возвращает
    (illu_fea_curr, illu_fea_prev_or_None) — см. cfc_module.py про то,
    зачем нужен prev.
    """

    def __init__(
        self,
        in_channels: int = 31,
        hidden_dim: int = 64,
        n_neurons: int = 32,
        window_size: int = 5,
        num_layers: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.pooling = SpatialPooling(in_channels, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_neurons,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Linear(n_neurons, hidden_dim)
        self.broadcast = SpatialBroadcast(hidden_dim, in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _finalize(self, out_vec, raw_fea):
        """Broadcast + residual-gate merge для одного временного шага."""
        temporal_fea = self.broadcast(out_vec, raw_fea)
        alpha = self.gate(torch.cat([raw_fea, temporal_fea], dim=1))
        return alpha * temporal_fea + (1 - alpha) * raw_fea

    def forward(self, illu_fea_seq, timespans=None):
        """
        illu_fea_seq: [B, T, C, H, W]
        timespans: ignored (LSTM не использует непрерывное время)
        → (illu_fea_curr [B, C, H, W], illu_fea_prev [B, C, H, W] или None)
        """
        B, T, C, H, W = illu_fea_seq.shape

        vecs = [self.pooling(illu_fea_seq[:, t]) for t in range(T)]
        seq = torch.stack(vecs, dim=1)  # [B, T, hidden_dim]

        # nn.LSTM уже считает выходы для ВСЕХ шагов в lstm_out — раньше
        # использовался только lstm_out[:, -1], шаг T-2 отбрасывался.
        lstm_out, _ = self.lstm(seq)

        out_curr = self._finalize(self.out_proj(lstm_out[:, -1]), illu_fea_seq[:, -1])
        out_prev = (
            self._finalize(self.out_proj(lstm_out[:, -2]), illu_fea_seq[:, -2])
            if T >= 2 else None
        )
        return out_curr, out_prev
