"""
Интерполяция между world-space пробами (ТЗ §2.3): visibility (raycast) + normal-weighting.

Ключевое архитектурное решение §2.3 — ДВЕ независимые оси:

(а) Источник visibility: точный precomputed raycast из Blender (§3.1), не runtime-
    аппроксимация (DDGI-style moment-based Chebyshev test) — на этапе-1 нет ограничения
    runtime-бюджета живого движка, трассировка честного луча один раз при генерации
    данных дешевле и точнее.

(б) Место интерполяции в пайплайне: ДЕТЕРМИНИРОВАННАЯ формула-постпроцессинг, а НЕ
    обучаемый компонент внутри CfC-B. CfC-B (см. models/temporal/cfc_probe_module.py)
    получает на входе уже готовое интерполированное значение irradiance, не сырые
    visibility/normal буферы. Причина: если бы формула была обучаемым компонентом,
    видящим точный raycast-visibility как фичу на этапе-1, на этапе-2 (где visibility
    заменяется DDGI-аппроксимацией — другая структура ошибки, включая некомпенсированный
    light leak) обучаемый компонент выучил бы паттерн именно под точный сигнал —
    distribution shift. При формуле-постпроцессинге этой проблемы нет: она не содержит
    обучаемых параметров вовсе, поэтому здесь ВСЕ константы (falloff power, eps) —
    зафиксированные гиперпараметры, не nn.Parameter.

Формула (классический подход для irradiance volumes/DDGI):
    weight_k = visibility_k * normal_term_k * distance_term_k
    normal_term_k   = clamp(dot(query_normal, direction_to_probe_k), min=0)
        -- решает вопрос ОРИЕНТАЦИИ: видимая проба может физически находиться "с
        обратной стороны" поверхности относительно её нормали (тонкая двусторонняя
        стена/пол под углом) — без этого члена такая проба даст нефизичный результат.
    visibility_k    -- решает вопрос ОККЛЮЗИИ (есть ли прямая видимость проба-точка).
    distance_term_k = 1 / (distance_k ** power + eps)
        -- ближние пробы важнее; power как гиперпараметр (не обучаемый).
Веса нормализуются по сумме K. Если ВСЕ пробы оказались occluded/behind-normal
(сумма весов ~ 0) — fallback на distance-only веса (игнорируя visibility/normal),
чтобы не делить на ноль и не возвращать неопределённый (NaN) результат; это
инженерный fallback, не часть архитектурного решения §2.3, и подлежит пересмотру,
если на реальных Blender-данных такой случай окажется частым/значимым.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_interpolation_weights(
    query_position: torch.Tensor,
    query_normal: torch.Tensor,
    probe_positions: torch.Tensor,
    visibility: torch.Tensor,
    distance_falloff_power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        query_position:  [B, 3]     — точка шейдинга
        query_normal:     [B, 3]     — нормаль поверхности в точке шейдинга (нормализована)
        probe_positions:  [B, K, 3]  — позиции K кандидатов-проб для этой точки
        visibility:       [B, K]     — precomputed raycast visibility (0..1, обычно {0,1})
        distance_falloff_power: степень затухания по расстоянию (не обучаемая)
        eps: численная стабильность (деление на distance**power и на sum весов)

    Returns:
        weights: [B, K] — нормализованные веса (сумма по K равна 1, кроме случая где
                  все входные веса были 0 и B-элемент тоже получит нули — см. forward
                  в ProbeInterpolator, где обрабатывается fallback).
    """
    direction = probe_positions - query_position.unsqueeze(1)  # [B, K, 3]
    distance = torch.linalg.norm(direction, dim=-1)  # [B, K]
    direction_normalized = direction / (distance.unsqueeze(-1) + eps)

    normal_term = torch.clamp(
        torch.einsum("bkc,bc->bk", direction_normalized, query_normal), min=0.0
    )  # [B, K]

    distance_term = 1.0 / (distance.pow(distance_falloff_power) + eps)  # [B, K]

    raw_weights = visibility * normal_term * distance_term  # [B, K]

    weight_sum = raw_weights.sum(dim=-1, keepdim=True)  # [B, 1]
    normalized = raw_weights / (weight_sum + eps)

    # Fallback: там, где сумма весов ~0 (все пробы occluded и/или "за спиной" по нормали),
    # используем distance-only веса — иначе normalized выше даёт ~0 везде вместо
    # содержательной интерполяции. Явно помечено в докстринге модуля как fallback,
    # не архитектурное решение §2.3.
    degenerate = (weight_sum.squeeze(-1) < eps)  # [B]
    if degenerate.any():
        fallback_raw = distance_term
        fallback_normalized = fallback_raw / (
            fallback_raw.sum(dim=-1, keepdim=True) + eps
        )
        normalized = torch.where(
            degenerate.unsqueeze(-1), fallback_normalized, normalized
        )

    return normalized


class ProbeInterpolator(nn.Module):
    """
    Без обучаемых параметров (сознательно, см. докстринг модуля) — nn.Module только
    ради единообразного интерфейса с остальными компонентами пайплайна (composability),
    не ради параметров.
    """

    def __init__(self, distance_falloff_power: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.distance_falloff_power = distance_falloff_power
        self.eps = eps

    def forward(
        self,
        query_position: torch.Tensor,
        query_normal: torch.Tensor,
        probe_positions: torch.Tensor,
        probe_irradiance: torch.Tensor,
        visibility: torch.Tensor,
        return_weights: bool = False,
    ):
        """
        Args:
            query_position:   [B, 3]
            query_normal:      [B, 3]
            probe_positions:   [B, K, 3]
            probe_irradiance:  [B, K, obs_dim] — предсказания CfC-B для каждой из K проб
                               (градиент течёт сюда — единственная дифференцируемая
                               по параметрам модели входная величина здесь)
            visibility:        [B, K]

        Returns:
            interpolated: [B, obs_dim]
            weights (если return_weights=True): [B, K]
        """
        weights = compute_interpolation_weights(
            query_position,
            query_normal,
            probe_positions,
            visibility,
            distance_falloff_power=self.distance_falloff_power,
            eps=self.eps,
        )  # [B, K]

        interpolated = torch.einsum("bk,bko->bo", weights, probe_irradiance)  # [B, obs_dim]

        if return_weights:
            return interpolated, weights
        return interpolated
