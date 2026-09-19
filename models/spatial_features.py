"""
Позиционное кондиционирование (решение из чата 2026-08-20, см. mempalace/lnn-lowlight.md
"НАСКОЛЬКО ДАЛЕКИ ОТ РЕАЛЬНОГО NRC" 2026-08-19) — единая входная точка для закрытия
найденного разрыва: настоящий NRC (Müller et al. 2021) обобщается по сцене через
кондиционирование на 3D-позицию + направление взгляда + normal + albedo (+ roughness) с
positional encoding, а не через доступ к соседним пикселям. Ни одна модель в этом проекте
раньше не получала пространственный вход вообще (2026-08-19 grep-проверка build_input у
NRDStyleBaseline/NRCStyleBaseline/CfCProbeModule — общая staleness-only сборка).

ПРИМЕНЯЕТСЯ ОДИНАКОВО ко всем honest-моделям с общими весами (nrc_honest, cfc/cfc_hardjump/
cfc_hardjump_skip, gru_honest, ode_lstm_honest) — решение из того же чата: разница
honest vs honest_spatial при равном доступе к позиции изолирует именно архитектурное
преимущество, а не сам факт наличия пространственной информации (то же разделение вопросов
(а)/(б), что уже применено к Δt/staleness в models/baselines.py). NRD-style сюда НЕ входит
(отдельное решение того же чата) — у него нет обучаемых параметров/MLP, кондиционировать
нечего.
"""
import torch


def positional_encoding(x: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """NeRF-style [x, sin(2^k*pi*x), cos(2^k*pi*x)] для k=0..n_freqs-1 (Mildenhall et al.
    2020) — практичная замена "one-blob"/frequency encoding оригинальной NRC-статьи (§5.2):
    тот же принцип (высокочастотные компоненты помогают MLP разрешать мелкие пространственные
    детали, Rahaman et al. 2019 spectral bias), сам найденный разрыв — ОТСУТСТВИЕ
    пространственного входа как такового, не конкретная формула encoding'а.

    x: [..., D] -> [..., D*(1+2*n_freqs)]
    """
    out = [x]
    for k in range(n_freqs):
        freq = (2.0 ** k) * torch.pi
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


def build_spatial_conditioning(
    position: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    albedo: torch.Tensor,
    roughness: torch.Tensor | None = None,
    n_freqs_position: int = 4,
    n_freqs_direction: int = 2,
    position_scale: float = 1.0,
) -> torch.Tensor:
    """
    Собирает пространственный входной вектор (позиция+направление+normal+albedo(+roughness),
    positional encoding на позиции/направлении) — единая точка, используемая одинаково во
    всех honest-моделях с общими весами.

    position, direction, normal, albedo: [B, T, 3]. direction ожидается УЖЕ нормализованным
    (вызывающий код: normalize(position - camera_pos)) — здесь НЕ перенормируется, чтобы не
    скрыть баг апстрим. roughness: [B, T, 1] или None — None, если буфер недоступен
    (см. докстринг модуля про текущую Blender-сцену: материал константный, Roughness=1.0 у пола
    и occluder'а, не рендерится отдельным буфером); не выдумываем данные, если их нет.

    position_scale (2026-08-20, БАГФИКС после эксперимента с реальными данными — см. чат):
    position ДЕЛИТСЯ на position_scale ПЕРЕД positional encoding, чтобы привести к масштабу
    ~[-1,1] (как в NeRF/оригинальной NRC-статье, сцена нормализуется в единичный куб перед
    encoding). Без этого шага (position_scale=1.0, значение по умолчанию для обратной
    совместимости API и юнит-тестов на произвольных тензорах) высокие частоты (до
    2^(n_freqs_position-1)*pi) применяются к сырым мировым координатам — на реальной Blender-сцене
    (пол size=8, координаты до ±4) это даёт аргумент sin/cos вплоть до ~100 радиан, синусоида
    оборачивается ~16 раз на протяжении сцены (алиасинг): физически близкие точки дают
    почти противоположный encoding, целевая функция становится негладкой, и обучение НЕ УЛУЧШАЕТСЯ
    ростом объёма данных (подтверждено экспериментально 2026-08-20: 4.5x больше train-пикселей
    почти не изменило степень деградации nrc_honest_spatial/gru_honest_spatial). ВЫЗЫВАЮЩИЙ КОД
    должен передать масштаб сцены явно (run_experiment_dense.py передаёт половину протяжённости
    пола, см. build_static_geometry: primitive_plane_add(size=8) -> координаты в [-4,4] ->
    position_scale=4.0) — здесь НЕ выводится автоматически из данных, чтобы не скрыть будущую
    смену сцены молча.

    Positional encoding применяется ТОЛЬКО к position/direction — normal/albedo/roughness уже
    bounded-величины, доп. encoding им не нужен (тот же принцип, что в оригинальной NRC-статье).
    direction уже unit-вектор (масштаб [-1,1] по построению) — position_scale на него НЕ влияет.

    Returns: [B, T, spatial_dim] — см. spatial_dim() ниже для точной формулы размерности.
    """
    pos_enc = positional_encoding(position / position_scale, n_freqs_position)
    dir_enc = positional_encoding(direction, n_freqs_direction)
    components = [pos_enc, dir_enc, normal, albedo]
    if roughness is not None:
        components.append(roughness)
    return torch.cat(components, dim=-1)


def spatial_dim(n_freqs_position: int = 4, n_freqs_direction: int = 2,
                include_roughness: bool = False) -> int:
    """Итоговая размерность build_spatial_conditioning БЕЗ вызова с реальными тензорами —
    нужно ЗАРАНЕЕ при инстанцировании моделей (__init__ должен знать input_dim до forward)."""
    d = 3 * (1 + 2 * n_freqs_position) + 3 * (1 + 2 * n_freqs_direction) + 3 + 3
    if include_roughness:
        d += 1
    return d
