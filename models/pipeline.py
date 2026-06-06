"""
RetinexFormer + Temporal Module pipeline.

Ключевая идея: встраиваем темпоральный модуль (CfC/LSTM/GRU/none)
МЕЖДУ illumination estimator и damage restorer RetinexFormer.

Illumination estimator выдаёт illu_fea и illu_map.
Темпоральный модуль обновляет illu_fea с учётом прошлых кадров.
Обновлённая illu_fea поступает в denoiser (damage restorer).
"""

import sys
import os
import torch
import torch.nn as nn

# Загружаем arch-файл напрямую через importlib, минуя basicsr/__init__
# (который тянет lmdb и другие тяжёлые зависимости)
import importlib.util as _ilu

_ARCH_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'Retinexformer',
    'basicsr', 'models', 'archs', 'RetinexFormer_arch.py'
)
_spec = _ilu.spec_from_file_location('RetinexFormer_arch', _ARCH_PATH)
_arch_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_arch_mod)

Illumination_Estimator = _arch_mod.Illumination_Estimator
Denoiser = _arch_mod.Denoiser

from models.temporal import TEMPORAL_MODULES


class RetinexLNNPipeline(nn.Module):
    """
    Полный пайплайн: RetinexFormer + темпоральный модуль.

    Args:
        temporal_type: 'cfc' | 'lstm' | 'gru' | 'none'
        n_feat:        число feature-каналов в RetinexFormer (default 31)
        stage:         число стадий RetinexFormer
        num_blocks:    блоки на каждой стадии
        window_size:   размер скользящего окна (число кадров)
        hidden_dim:    скрытое пространство темпорального модуля
        n_neurons:     нейроны в CfC/LSTM/GRU
        fps:           FPS видео для вычисления ∆t (используется CfC)
    """

    def __init__(
        self,
        temporal_type: str = 'cfc',
        n_feat: int = 32,  # должен быть кратен 4 (groups=4 в Illumination_Estimator)
        stage: int = 1,
        num_blocks: list = None,
        window_size: int = 5,
        hidden_dim: int = 64,
        n_neurons: int = 32,
        fps: float = 30.0,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1]

        self.temporal_type = temporal_type
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps  # базовый временной шаг

        # RetinexFormer компоненты (одна стадия для ясности)
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=3, out_dim=3, dim=n_feat,
            level=2, num_blocks=num_blocks,
        )

        # Темпоральный модуль
        TemporalCls = TEMPORAL_MODULES.get(temporal_type)
        if TemporalCls is not None:
            self.temporal = TemporalCls(
                in_channels=n_feat,
                hidden_dim=hidden_dim,
                n_neurons=n_neurons,
                window_size=window_size,
            )
        else:
            self.temporal = None  # per-frame baseline

    def forward_single(self, img: torch.Tensor) -> tuple:
        """Прогнать один кадр через estimator. Возвращает (illu_fea, illu_map)."""
        return self.estimator(img)

    def forward(
        self,
        frames: torch.Tensor,
        timespans: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            frames:    [B, T, 3, H, W] — окно из T кадров.
                       Последний кадр (T-1) — текущий, остальные — контекст.
            timespans: [B, T] — ∆t между кадрами. Если None — используется 1/fps.
        Returns:
            enhanced:  [B, 3, H, W] — enhanced текущий кадр
        """
        B, T, C, H, W = frames.shape
        current = frames[:, -1]  # [B, 3, H, W]

        if self.temporal is None:
            # Baseline: per-frame, без темпорального контекста
            illu_fea, illu_map = self.estimator(current)
        else:
            # Прогнать все кадры через estimator
            illu_feas = []
            for t in range(T):
                fea, _ = self.estimator(frames[:, t])
                illu_feas.append(fea)

            illu_fea_seq = torch.stack(illu_feas, dim=1)  # [B, T, C, H, W]

            # Вычислить timespans если не переданы
            if timespans is None and self.temporal_type == 'cfc':
                timespans = torch.full(
                    (B, T), self.dt,
                    device=frames.device, dtype=frames.dtype,
                )

            # Темпоральный модуль обновляет illu_fea текущего кадра
            illu_fea = self.temporal(illu_fea_seq, timespans)

            # illu_map берём от текущего кадра (для input_img)
            _, illu_map = self.estimator(current)

        # Стандартный Retinex forward: enhance + denoise
        input_img = current * illu_map + current
        output_img = self.denoiser(input_img, illu_fea)

        return output_img

    def get_num_params(self) -> dict:
        """Подсчёт параметров по компонентам."""
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        result = {
            'estimator': count(self.estimator),
            'denoiser': count(self.denoiser),
            'temporal': count(self.temporal) if self.temporal else 0,
        }
        result['total'] = sum(result.values())
        return result
