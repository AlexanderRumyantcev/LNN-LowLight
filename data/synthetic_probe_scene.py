"""
Synthetic stand-in генератор данных под формат ТЗ §3.1 (буферы пробы) / §3.3 (Δt-механизм),
НЕ замена реального Blender headless пайплайна (§3.1) — тот ещё не реализован. Нужен, чтобы
протестировать models/temporal/cfc_probe_module.py + models/probe_interpolation.py вместе,
на пространственно-согласованных (не независимых по пробам) данных, до появления реальной
геометрии/рендеринга.

Сцена: N проб на случайных позициях + один точечный источник света на фиксированной позиции.
Интенсивность источника меняется во времени по кусочно-заданному расписанию (static/step/drift
сегменты — тот же тип динамики, что уже провалидирован в spike_test/data/synthetic.py, но
определённый через РЕАЛЬНУЮ длительность сегмента, не число шагов — т.к. §3.3 требует
параметрического семплирования по вещественному t, не по кадровой сетке).

Истинная irradiance пробы i в момент t: intensity(t) / (falloff * dist(probe_i, light)^2 + eps)
— так пробы физически согласованы между собой (ближе к свету — ярче, синхронно меняются
с интенсивностью источника), а не генерируются независимо. Каждая проба обновляется
асинхронно (свой Δt-поток, сэмплированный из уже откалиброванной в spike-тесте модели с
тяжёлыми выбросами, см. GenConfig.dt_spike_prob), поэтому пробы в любой момент имеют разные
"последние моменты обновления" — вещественные, не выровненные по общей кадровой сетке.

Visibility (§2.3/§3.1) — synthetic-заглушка ray-sphere intersection против набора occluder-
сфер (НЕ настоящий raycast по mesh из Blender), достаточная, чтобы протестировать формулу
интерполяции на неслучайных, геометрически осмысленных occlusion-паттернах.
"""

import numpy as np
from dataclasses import dataclass, field

SEG_STATIC, SEG_STEP, SEG_DRIFT = 0, 1, 2


@dataclass
class SceneGenConfig:
    n_probes: int = 32
    bbox_size: float = 10.0            # пробы и occluders внутри куба [-bbox_size/2, bbox_size/2]^3
    n_occluders: int = 4
    occluder_radius_range: tuple = (0.5, 1.5)

    # расписание интенсивности источника (в РЕАЛЬНОМ времени, не в шагах)
    total_duration: float = 400.0
    n_segments_range: tuple = (4, 7)
    segment_duration_range: tuple = (40.0, 90.0)
    intensity_range: tuple = (0.5, 20.0)
    step_magnitude_range: tuple = (0.3, 3.0)
    drift_rate_range: tuple = (-0.02, 0.02)   # относительное изменение за единицу t

    falloff: float = 1.0
    eps: float = 1e-3

    # Δt / шум — те же откалиброванные константы, что в spike_test/data/synthetic.py
    dt_range: tuple = (0.5, 2.0)
    dt_spike_prob: float = 0.07
    dt_spike_multiplier_range: tuple = (5.0, 15.0)
    spp_range: tuple = (1, 64)
    noise_c: float = 0.5

    seed: int = 0


def _build_light_schedule(rng: np.random.Generator, cfg: SceneGenConfig):
    """Кусочное расписание intensity(t) по РЕАЛЬНОЙ длительности сегментов (не по числу шагов)."""
    n_segments = rng.integers(cfg.n_segments_range[0], cfg.n_segments_range[1] + 1)
    durations = rng.uniform(cfg.segment_duration_range[0], cfg.segment_duration_range[1], size=n_segments)
    durations = durations * (cfg.total_duration / durations.sum())  # растянуть под total_duration
    boundaries = np.concatenate([[0.0], np.cumsum(durations)])
    types = rng.integers(0, 3, size=n_segments)

    cur_value = rng.uniform(*cfg.intensity_range)
    segments = []  # (t_start, t_end, type, params)
    for i in range(n_segments):
        t_start, t_end = boundaries[i], boundaries[i + 1]
        seg_type = int(types[i])
        if seg_type == SEG_STATIC:
            segments.append((t_start, t_end, SEG_STATIC, dict(value=cur_value)))
        elif seg_type == SEG_STEP:
            jump_frac = rng.uniform(0.1, 0.9)
            jump_t = t_start + jump_frac * (t_end - t_start)
            mag = rng.uniform(*cfg.step_magnitude_range)
            sign = rng.choice([-1.0, 1.0])
            new_value = float(np.clip(cur_value * (1.0 + sign * mag), *cfg.intensity_range))
            segments.append((t_start, t_end, SEG_STEP, dict(old=cur_value, new=new_value, jump_t=jump_t)))
            cur_value = new_value
        else:  # SEG_DRIFT
            rate = rng.uniform(*cfg.drift_rate_range)
            segments.append((t_start, t_end, SEG_DRIFT, dict(start_value=cur_value, rate=rate)))
            cur_value = float(np.clip(cur_value * np.exp(rate * (t_end - t_start)), *cfg.intensity_range))

    return segments


def _intensity_at(segments, t: np.ndarray) -> np.ndarray:
    """Векторизованная оценка intensity(t) по расписанию сегментов."""
    out = np.zeros_like(t, dtype=np.float64)
    for t_start, t_end, seg_type, p in segments:
        mask = (t >= t_start) & (t < t_end)
        if not mask.any():
            continue
        local_t = t[mask]
        if seg_type == SEG_STATIC:
            out[mask] = p["value"]
        elif seg_type == SEG_STEP:
            out[mask] = np.where(local_t < p["jump_t"], p["old"], p["new"])
        else:  # SEG_DRIFT
            out[mask] = p["start_value"] * np.exp(p["rate"] * (local_t - t_start))
    # t за пределами последнего сегмента (округление) — берём значение последнего сегмента
    last_end = segments[-1][1]
    tail = t >= last_end
    if tail.any():
        out[tail] = _intensity_at(segments, np.array([last_end - 1e-6]))[0]
    return out


def generate_scene(rng: np.random.Generator, cfg: SceneGenConfig):
    half = cfg.bbox_size / 2.0
    probe_positions = rng.uniform(-half, half, size=(cfg.n_probes, 3))
    light_position = rng.uniform(-half, half, size=3)
    occluder_centers = rng.uniform(-half, half, size=(cfg.n_occluders, 3))
    occluder_radii = rng.uniform(*cfg.occluder_radius_range, size=cfg.n_occluders)
    light_schedule = _build_light_schedule(rng, cfg)
    return dict(
        probe_positions=probe_positions.astype(np.float32),
        light_position=light_position.astype(np.float32),
        occluder_centers=occluder_centers.astype(np.float32),
        occluder_radii=occluder_radii.astype(np.float32),
        light_schedule=light_schedule,
    )


def sample_probe_sequence(rng: np.random.Generator, probe_pos: np.ndarray, scene: dict, cfg: SceneGenConfig):
    """
    Асинхронная последовательность обновлений ОДНОЙ пробы: свой Δt-поток (реализация §3.3 —
    параметрическое семплирование по вещественному t, здесь через накопление сэмплированных
    Δt), своя spp/шум (та же откалиброванная модель, что в spike_test/data/synthetic.py).
    """
    T = 0
    dt_list, t_list = [], []
    t_cur = 0.0
    while t_cur < cfg.total_duration:
        dt = rng.uniform(*cfg.dt_range)
        if rng.uniform(0.0, 1.0) < cfg.dt_spike_prob:
            dt *= rng.uniform(*cfg.dt_spike_multiplier_range)
        t_cur += dt
        dt_list.append(dt)
        t_list.append(t_cur)
        T += 1
    dt = np.array(dt_list[:-1] if t_list[-1] > cfg.total_duration else dt_list, dtype=np.float64)
    t = np.array(t_list[: len(dt)], dtype=np.float64)
    dt[0] = 0.0  # cold-start: первый шаг без предыдущего наблюдения

    intensity = _intensity_at(scene["light_schedule"], t)
    dist_sq = np.sum((probe_pos - scene["light_position"]) ** 2)
    true_irradiance = intensity / (cfg.falloff * dist_sq + cfg.eps)

    spp = rng.integers(cfg.spp_range[0], cfg.spp_range[1] + 1, size=len(t))
    noise_std = np.sqrt(cfg.noise_c * np.maximum(true_irradiance, 1e-3) / spp)
    obs = true_irradiance + rng.normal(0.0, 1.0, size=len(t)) * noise_std

    cold_start = np.zeros(len(t), dtype=np.float32)
    cold_start[0] = 1.0

    return dict(
        t=t.astype(np.float32),
        dt=dt.astype(np.float32),
        obs=obs.astype(np.float32),
        true_irradiance=true_irradiance.astype(np.float32),
        spp=spp.astype(np.float32),
        cold_start=cold_start,
    )


def ray_sphere_visibility(
    origins: np.ndarray, targets: np.ndarray, occluder_centers: np.ndarray, occluder_radii: np.ndarray
) -> np.ndarray:
    """
    Synthetic-заглушка visibility (§2.3/§3.1) — ray-sphere intersection, не настоящий raycast
    по mesh. origins/targets: [M, 3]. Возвращает [M] в {0,1}: 1 = прямая видимость свободна.
    """
    M = origins.shape[0]
    visible = np.ones(M, dtype=np.float32)
    direction = targets - origins
    seg_len = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_normalized = direction / (seg_len + 1e-8)

    for center, radius in zip(occluder_centers, occluder_radii):
        oc = origins - center  # [M, 3]
        b = np.einsum("mc,mc->m", oc, direction_normalized)
        c = np.einsum("mc,mc->m", oc, oc) - radius ** 2
        disc = b ** 2 - c
        hit = disc > 0.0
        t_hit = -b - np.sqrt(np.clip(disc, 0.0, None))
        # occluder блокирует луч, только если пересечение лежит МЕЖДУ origin и target
        blocking = hit & (t_hit > 1e-4) & (t_hit < seg_len.squeeze(-1) - 1e-4)
        visible = np.where(blocking, 0.0, visible)

    return visible
