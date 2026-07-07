"""
Transformer temporal module — self-attention baseline для сравнения LNN vs
Transformer (см. ТЗ.md, раздел 11).

Причинность: используется causal-маска (нижнетреугольная) — модуль не видит
будущие относительно текущего шага кадры окна, чтобы сравнение с
CfC/LSTM/GRU (строго causal по построению — рекуррентность физически не
может видеть будущее) было честным. Без маски Transformer получил бы
информационное преимущество, не связанное с архитектурой как таковой.

Позиционное кодирование: learned positional embedding по индексу шага в
окне — attention, в отличие от RNN, не видит порядок кадров без явного
кодирования позиции (рекуррентность LSTM/GRU/CfC хранит порядок неявно,
через сам процесс пошагового обновления состояния).

Бюджет параметров: d_model=52 подобран так, чтобы дать 27 904 параметра —
в пределах 2% от CfCTemporalModule (28 305) при тех же n_feat/window_size.
Это первый tier сравнения из ТЗ.md, п.11.3 — тест "простая LNN конкурентна
с Transformer при равном бюджете параметров".
"""

import math
import torch
import torch.nn as nn
from .lstm_module import SpatialPooling, SpatialBroadcast


class TransformerTemporalModule(nn.Module):
    """
    Causal self-attention baseline. Интерфейс идентичен CfC/LSTM/GRU:
    forward() возвращает (illu_fea_curr, illu_fea_prev_or_None) — см.
    cfc_module.py про то, зачем нужен prev (нужен для TemporalConsistencyLoss
    в losses/combined_loss.py).

    Args:
        in_channels:  число каналов входного illu_fea (из RetinexFormer estimator)
        hidden_dim:   не используется здесь напрямую (оставлен для единообразия
                      сигнатуры с CfC/LSTM/GRU в TEMPORAL_MODULES registry)
        n_neurons:    не используется здесь напрямую (см. `hidden_dim` выше)
        window_size:  число кадров в скользящем окне (causal: текущий + прошлые)
        d_model:      размерность attention-пространства — подобрана под
                      бюджет параметров ~28k (см. докстринг модуля)
        n_heads:      число attention-голов (d_model должен делиться нацело)
        ff_ratio:     во сколько раз FFN-слой шире d_model
    """

    def __init__(
        self,
        in_channels: int = 31,
        hidden_dim: int = 64,
        n_neurons: int = 32,
        window_size: int = 5,
        d_model: int = 52,
        n_heads: int = 2,
        ff_ratio: int = 2,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model должен делиться нацело на n_heads"
        self.head_dim = d_model // n_heads

        # Spatial -> vector (переиспользуем существующий блок из lstm_module.py)
        self.pooling = SpatialPooling(in_channels, d_model)

        # Learned positional embedding по индексу шага в окне.
        # Инициализация trunc_normal(std=0.02) — стандартная практика для
        # positional embeddings в vision-трансформерах (ViT и т.п.).
        self.pos_emb = nn.Parameter(torch.zeros(window_size, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Q, K, V, output projections одного attention-блока
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Pre-LN трансформер-блок (attention + FFN, каждый с residual)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        ff_dim = d_model * ff_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

        # Vector -> spatial (переиспользуем существующий блок)
        self.broadcast = SpatialBroadcast(d_model, in_channels)

        # Residual gate: сколько темпорального контекста добавить —
        # идентично CfC/LSTM/GRU
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _causal_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model] -> [B, T, d_model], с causal-маской (шаг t не
        видит t+1..T-1). Pre-LN блок: attention+residual, затем FFN+residual.

        Реализовано вручную (без nn.MultiheadAttention/nn.TransformerEncoderLayer),
        чтобы явно контролировать маску причинности и подсчёт параметров —
        та же логика прозрачности, что и у ручного пошагового forward в
        CfCTemporalModule (см. cfc_module.py).
        """
        B, T, D = x.shape

        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_heads, T, head_dim]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # [B, nh, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = x + out

        x = x + self.ffn(self.norm2(x))
        return x

    def _finalize(self, out_vec: torch.Tensor, raw_fea: torch.Tensor) -> torch.Tensor:
        """
        Broadcast + residual-gate merge для ОДНОГО временного шага — та же
        логика, что в CfC/LSTM/GRU (см. cfc_module.py._finalize).
        """
        temporal_fea = self.broadcast(out_vec, raw_fea)
        gate_input = torch.cat([raw_fea, temporal_fea], dim=1)
        alpha = self.gate(gate_input)
        return alpha * temporal_fea + (1 - alpha) * raw_fea

    def forward(
        self,
        illu_fea_seq: torch.Tensor,
        timespans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            illu_fea_seq: [B, T, C, H, W] — последовательность illumination features
                          T = window_size, последний кадр — текущий
            timespans:    игнорируется (Transformer не использует непрерывное
                          время в отличие от CfC — параметр оставлен только
                          для единообразия интерфейса с TEMPORAL_MODULES)
        Returns:
            (illu_fea_curr, illu_fea_prev):
                illu_fea_curr: [B, C, H, W] — enhanced illumination для текущего (T-1) кадра
                illu_fea_prev: [B, C, H, W] или None — то же для предпоследнего (T-2)
                    кадра, None если T < 2 (нет "предыдущего" кадра в окне)
        """
        B, T, C, H, W = illu_fea_seq.shape

        vecs = [self.pooling(illu_fea_seq[:, t]) for t in range(T)]
        seq = torch.stack(vecs, dim=1)  # [B, T, d_model]
        seq = seq + self.pos_emb[:T].unsqueeze(0)

        # out_seq уже содержит выходы ДЛЯ ВСЕХ шагов 0..T-1 (causal-маска
        # гарантирует, что шаг T-2 не видел кадр T-1) — та же логика, что у
        # CfC/LSTM/GRU: T-2 не отбрасывается, а используется для prev.
        out_seq = self._causal_self_attention(seq)  # [B, T, d_model]

        illu_fea_curr = self._finalize(out_seq[:, -1], illu_fea_seq[:, -1])
        illu_fea_prev = (
            self._finalize(out_seq[:, -2], illu_fea_seq[:, -2])
            if T >= 2 else None
        )

        return illu_fea_curr, illu_fea_prev
