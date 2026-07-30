"""
Функция потерь (ТЗ §5.1) — relative L2, формулировка из оригинальной статьи NVIDIA Neural
Radiance Caching (NRC), тот же NRC, что уже выбран reference-baseline'ом проекта (§4):
использование их же loss для CfC делает сравнение чистым (результат не спишется на разницу
в обучающих сигналах).

    L(Ls, L̂s) = (Ls - L̂s)^2 / (sg(L̂s)^2 + eps),   eps = 0.01

Нормализация ИМЕННО по предсказанию (L̂s), не по target (Ls), со stop-gradient на
нормализаторе. Причина (§5.1): target сам получен через Monte Carlo и зашумлён (переменный
spp) — нормализация по target даёт нестабильные/взрывающиеся градиенты в областях с низким
радиансом (где spp низкий и шум велик). Нормализация по detached-предсказанию сети устраняет
эту нестабильность.

Confidence-weighting (§5.1, расширение проекта, НЕ из статьи NRC) — опциональное: занижает
вес заведомо шумных low-spp сэмплов через тот же confidence-сигнал (log1p(spp)/norm), что
уже используется в staleness-gate CfC-B (models/temporal/cfc_probe_module.py) и в честных
версиях baseline'ов (models/baselines.py). Не обязательный компонент базового loss — явно
помечен как опция, включается отдельным флагом.
"""

import torch
import torch.nn as nn


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.01,
    confidence_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Args:
        pred:   [*]  — предсказание модели (L̂s)
        target: [*]  — обучающий сигнал (Ls), та же форма, что pred
        eps: числовая стабилизация нормализатора (0.01 — значение из статьи NRC)
        confidence_weight: [*] или None — опциональный вес per-element (§5.1, расширение,
            НЕ часть базового NRC-loss); ожидается нормализованный в [0, 1] сигнал
            (тот же log1p(spp)-based confidence, что в build_input CfC-B/baseline'ов)
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        скаляр (mean/sum) или тензор той же формы, что pred (none)
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred.shape {pred.shape} != target.shape {target.shape}")

    normalizer = pred.detach().pow(2) + eps  # stop-gradient — ключевая часть формулы
    per_element = (target - pred).pow(2) / normalizer

    if confidence_weight is not None:
        if confidence_weight.shape != pred.shape:
            raise ValueError(
                f"confidence_weight.shape {confidence_weight.shape} != pred.shape {pred.shape}"
            )
        per_element = per_element * confidence_weight

    if reduction == "none":
        return per_element
    if reduction == "sum":
        return per_element.sum()
    if reduction == "mean":
        return per_element.mean()
    raise ValueError(f"unknown reduction: {reduction}")


class NRCRelativeL2Loss(nn.Module):
    """
    nn.Module-обёртка вокруг relative_l2_loss — без обучаемых параметров (eps — фиксированный
    гиперпараметр из статьи NRC, не обучается), нужна только ради единообразного интерфейса
    (composability с остальными компонентами пайплайна, как ProbeInterpolator).
    """

    def __init__(self, eps: float = 0.01, use_confidence_weight: bool = False):
        super().__init__()
        self.eps = eps
        self.use_confidence_weight = use_confidence_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if self.use_confidence_weight and confidence_weight is None:
            raise ValueError(
                "use_confidence_weight=True требует передать confidence_weight в forward()"
            )
        return relative_l2_loss(
            pred,
            target,
            eps=self.eps,
            confidence_weight=confidence_weight if self.use_confidence_weight else None,
            reduction=reduction,
        )
