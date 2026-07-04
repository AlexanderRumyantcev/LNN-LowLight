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
        return_prev: bool = False,
    ):
        """
        Args:
            frames:      [B, T, 3, H, W] — окно из T кадров.
                         Последний кадр (T-1) — текущий, остальные — контекст.
            timespans:   [B, T] — ∆t между кадрами. Если None — используется 1/fps.
            return_prev: если True — дополнительно вернуть enhanced-выход для
                         предпоследнего (T-2) кадра окна. Нужен для
                         TemporalConsistencyLoss (losses/combined_loss.py),
                         который без него всегда получает temporal=0.0.
                         Вычисляется почти бесплатно: illu_fea/illu_map для
                         шага T-2 уже посчитаны temporal-модулем и estimator'ом
                         ниже — единственная доп. работа — второй проход
                         denoiser'а. Не включён по умолчанию, чтобы не менять
                         поведение существующих вызовов (eval.py, inference.py,
                         smoke_test.py), которые ожидают один тензор на выходе.
        Returns:
            Если return_prev=False (по умолчанию):
                enhanced: [B, 3, H, W] — enhanced текущий кадр
            Если return_prev=True:
                (enhanced, enhanced_prev): enhanced_prev — [B, 3, H, W] или
                None, если T < 2 (нет предыдущего кадра в окне) или
                temporal_type='none' (per-frame baseline не даёт prev —
                для него temporal loss не применим по построению).
        """
        B, T, C, H, W = frames.shape
        current = frames[:, -1]  # [B, 3, H, W]

        illu_fea_prev = None
        illu_map_prev = None

        if self.temporal is None:
            # Baseline: per-frame, без темпорального контекста.
            # У baseline нет понятия "предыдущий enhanced через temporal
            # модуль" — temporal loss для него не применим по построению,
            # поэтому enhanced_prev всегда None независимо от return_prev.
            illu_fea, illu_map = self.estimator(current)
        else:
            # Прогнать все кадры через estimator. illu_maps[t] раньше
            # вычислялся, но выбрасывался (кроме t=T-1, для которого потом
            # оценивался ПОВТОРНО ниже) — теперь сохраняем всё за один проход.
            illu_feas = []
            illu_maps = []
            for t in range(T):
                fea, imap = self.estimator(frames[:, t])
                illu_feas.append(fea)
                illu_maps.append(imap)

            illu_fea_seq = torch.stack(illu_feas, dim=1)  # [B, T, C, H, W]

            # Вычислить timespans если не переданы
            if timespans is None and self.temporal_type == 'cfc':
                timespans = torch.full(
                    (B, T), self.dt,
                    device=frames.device, dtype=frames.dtype,
                )

            # Темпоральный модуль возвращает (illu_fea для T-1, illu_fea для
            # T-2 или None) — раньше отдавался только первый элемент.
            illu_fea, illu_fea_prev = self.temporal(illu_fea_seq, timespans)

            illu_map = illu_maps[-1]
            if T >= 2:
                illu_map_prev = illu_maps[-2]

        # Стандартный Retinex forward: enhance + denoise (текущий кадр)
        input_img = current * illu_map + current
        output_img = self.denoiser(input_img, illu_fea)

        if not return_prev:
            return output_img

        enhanced_prev = None
        if illu_fea_prev is not None and illu_map_prev is not None:
            prev_frame = frames[:, -2]
            input_img_prev = prev_frame * illu_map_prev + prev_frame
            enhanced_prev = self.denoiser(input_img_prev, illu_fea_prev)

        return output_img, enhanced_prev

    # ──────────────────────────────────────────────
    # Backbone freeze / unfreeze
    # ──────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Заморозить RetinexFormer (estimator + denoiser).
        Обучаться будет только темпоральный модуль (~24k params).
        Вызывать перед созданием оптимизатора.
        """
        for p in self.estimator.parameters():
            p.requires_grad = False
        for p in self.denoiser.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, lr_scale: float = 0.1) -> None:
        """
        Разморозить backbone для fine-tuning.

        Args:
            lr_scale: напоминание — backbone нужен меньший lr чем temporal.
                      Передай lr * lr_scale в param_groups оптимизатора.
        """
        for p in self.estimator.parameters():
            p.requires_grad = True
        for p in self.denoiser.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr: float, backbone_lr_scale: float = 0.1) -> list:
        """
        Вернуть param_groups для оптимизатора с разными lr.

        Использование:
            optimizer = AdamW(model.get_param_groups(lr=2e-4), ...)

        Args:
            lr:                базовый lr (для temporal модуля)
            backbone_lr_scale: множитель lr для backbone (default 0.1)
        Returns:
            список param_groups для torch.optim
        """
        temporal_params = list(self.temporal.parameters()) if self.temporal else []
        backbone_params = (
            list(self.estimator.parameters()) +
            list(self.denoiser.parameters())
        )
        groups = [
            {'params': temporal_params,  'lr': lr,                        'name': 'temporal'},
            {'params': backbone_params,  'lr': lr * backbone_lr_scale,    'name': 'backbone'},
        ]
        return [g for g in groups if len(g['params']) > 0]

    # ──────────────────────────────────────────────
    # Утилиты
    # ──────────────────────────────────────────────

    def get_num_params(self) -> dict:
        """Подсчёт параметров: всего / trainable / по компонентам."""
        def count_total(m):
            return sum(p.numel() for p in m.parameters())

        def count_trainable(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        result = {
            'estimator':          count_total(self.estimator),
            'denoiser':           count_total(self.denoiser),
            'temporal':           count_total(self.temporal) if self.temporal else 0,
            'trainable_temporal': count_trainable(self.temporal) if self.temporal else 0,
            'trainable_backbone': count_trainable(self.estimator) + count_trainable(self.denoiser),
        }
        result['total']     = result['estimator'] + result['denoiser'] + result['temporal']
        result['trainable'] = result['trainable_temporal'] + result['trainable_backbone']
        return result
