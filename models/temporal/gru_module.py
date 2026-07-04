"""GRU temporal module — лёгкий baseline."""

import torch
import torch.nn as nn
from .lstm_module import SpatialPooling, SpatialBroadcast


class GRUTemporalModule(nn.Module):
    """
    GRU baseline. Интерфейс идентичен CfCTemporalModule: forward()
    возвращает (illu_fea_curr, illu_fea_prev_or_None) — см. cfc_module.py.
    """

    def __init__(self, in_channels=31, hidden_dim=64, n_neurons=32, window_size=5, num_layers=1):
        super().__init__()
        self.window_size = window_size
        self.pooling = SpatialPooling(in_channels, hidden_dim)
        self.gru = nn.GRU(hidden_dim, n_neurons, num_layers=num_layers, batch_first=True)
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
        B, T, C, H, W = illu_fea_seq.shape
        seq = torch.stack([self.pooling(illu_fea_seq[:, t]) for t in range(T)], dim=1)

        # nn.GRU уже считает выходы для ВСЕХ шагов в out — раньше
        # использовался только out[:, -1], шаг T-2 отбрасывался.
        out, _ = self.gru(seq)

        out_curr = self._finalize(self.out_proj(out[:, -1]), illu_fea_seq[:, -1])
        out_prev = (
            self._finalize(self.out_proj(out[:, -2]), illu_fea_seq[:, -2])
            if T >= 2 else None
        )
        return out_curr, out_prev
