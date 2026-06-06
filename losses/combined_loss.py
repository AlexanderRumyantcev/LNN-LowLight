"""
Combined loss для low-light video enhancement.

L_total = λ1 * L_pixel + λ2 * L_ssim + λ3 * L_perceptual + λ4 * L_temporal

L_temporal (warping loss) — штраф за temporal inconsistency между кадрами.
Используется только при наличии предыдущего enhanced кадра.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """1 - SSIM как loss."""

    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        win = self.window.to(x.device)
        pad = self.window_size // 2
        mu_x = F.conv2d(x, win, padding=pad, groups=self.channel)
        mu_y = F.conv2d(y, win, padding=pad, groups=self.channel)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sigma_x = F.conv2d(x * x, win, padding=pad, groups=self.channel) - mu_x2
        sigma_y = F.conv2d(y * y, win, padding=pad, groups=self.channel) - mu_y2
        sigma_xy = F.conv2d(x * y, win, padding=pad, groups=self.channel) - mu_xy
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)


class TemporalConsistencyLoss(nn.Module):
    """
    Штраф за temporal inconsistency.
    Простая версия: L1 между соседними enhanced кадрами,
    взвешенная по оптической схожести входных кадров.
    """

    def forward(
        self,
        enhanced_prev: torch.Tensor,
        enhanced_curr: torch.Tensor,
        low_prev: torch.Tensor,
        low_curr: torch.Tensor,
    ) -> torch.Tensor:
        # Маска: пиксели где входные кадры похожи (нет движения/резких изменений)
        diff_input = (low_curr - low_prev).abs().mean(dim=1, keepdim=True)
        weight = torch.exp(-diff_input * 10.0)  # сильная разница → низкий вес
        loss = (weight * (enhanced_curr - enhanced_prev).abs()).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Итоговый loss для обучения.

    Args:
        lambda_pixel:    вес L1 pixel loss
        lambda_ssim:     вес SSIM loss
        lambda_temporal: вес temporal consistency loss
    """

    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_ssim: float = 0.1,
        lambda_temporal: float = 0.05,
    ):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_ssim = lambda_ssim
        self.lambda_temporal = lambda_temporal

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.temporal = TemporalConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        enhanced_prev: torch.Tensor | None = None,
        low_prev: torch.Tensor | None = None,
        low_curr: torch.Tensor | None = None,
    ) -> dict:
        """
        Returns dict с отдельными loss'ами и total.
        """
        l_pixel = self.l1(pred, target)
        l_ssim = self.ssim(pred, target)

        losses = {
            'pixel': l_pixel,
            'ssim': l_ssim,
        }

        total = self.lambda_pixel * l_pixel + self.lambda_ssim * l_ssim

        if enhanced_prev is not None and low_prev is not None and low_curr is not None:
            l_temp = self.temporal(enhanced_prev, pred, low_prev, low_curr)
            losses['temporal'] = l_temp
            total = total + self.lambda_temporal * l_temp
        else:
            losses['temporal'] = torch.tensor(0.0, device=pred.device)

        losses['total'] = total
        return losses
