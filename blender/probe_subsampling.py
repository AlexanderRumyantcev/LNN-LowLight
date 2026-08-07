"""
Пункт (c) плана адаптера: для каждой пробы — своя асинхронная
подпоследовательность моментов, ПОДВЫБРАННАЯ из уже отрендеренного
пула (b), без повторного рендера.

Работает НАД результатом generate_dataset.py (generate_dataset_result.json)
— чистый python/numpy, bpy не нужен, Blender для этого шага не
запускается.

Асинхронность: у каждой пробы свой независимый Δt-поток, сэмплированный
из ТОЙ ЖЕ откалиброванной модели с тяжёлыми выбросами, что в
data/synthetic_probe_scene.py (dt_range, dt_spike_prob,
dt_spike_multiplier_range) — переиспользуем откалиброванные константы
по духу, не изобретаем заново. per-probe dt_range здесь ЗАВЕДОМО
крупнее плотности пула (moment_pool.MomentPoolConfig.dt_range=(1.0,2.0)),
чтобы между последовательными выборками пробы гарантированно оставались
разные точки пула, без дублей.

КВАНТИЗАЦИЯ (см. докстринг moment_pool.py): желаемый непрерывный t
снэпится к БЛИЖАЙШЕЙ доступной точке пула справа от текущей позиции —
рендер физически ограничен дискретным пулом (b), это прямое следствие
того, что рендерится общий грид, а не произвольный t на пробу. dt в
выходе — РЕАЛЬНЫЙ dt между выбранными quantized t (не желаемый draw),
чтобы модель обучалась на честных интервалах, а не на номинальных.

ЧЕГО ЗДЕСЬ ЕЩЁ НЕТ: true_irradiance (высокий spp, эталон без шума) —
это пункт (d), ещё не реализован. Пока есть только "obs" — то, что
реально отрендерено в пуле (адаптивный Cycles-семплинг, ≤SAMPLES_MAX).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeSubsampleConfig:
    dt_range: tuple = (3.0, 8.0)
    dt_spike_prob: float = 0.07
    dt_spike_multiplier_range: tuple = (5.0, 15.0)
    seed: int = 0


def sample_probe_indices(pool_t, cfg: ProbeSubsampleConfig, rng=None):
    """Список ИНДЕКСОВ в pool_t — асинхронная подпоследовательность для
    одной пробы. pool_t должен быть отсортирован по возрастанию (как
    t_values из moment_pool.build_moment_pool)."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    pool_t = np.asarray(pool_t, dtype=np.float64)
    total_duration = pool_t[-1]

    indices = [0]
    t_cur = pool_t[0]
    last_idx = 0
    while True:
        dt = float(rng.uniform(*cfg.dt_range))
        if rng.uniform(0.0, 1.0) < cfg.dt_spike_prob:
            dt *= float(rng.uniform(*cfg.dt_spike_multiplier_range))
        target_t = t_cur + dt
        if target_t > total_duration:
            break
        tail = pool_t[last_idx + 1:]
        candidate = last_idx + 1 + int(np.searchsorted(tail, target_t))
        candidate = min(candidate, len(pool_t) - 1)
        if candidate <= last_idx:
            break  # пул исчерпан
        indices.append(candidate)
        t_cur = pool_t[candidate]
        last_idx = candidate

    return indices


def build_probe_sequence(pool_t, probe_series: dict, cfg: ProbeSubsampleConfig, rng=None):
    """probe_series: dict параллельных массивов ('obs', 'spp', ...), той
    же длины, что pool_t — по одному значению на пул-момент для ОДНОЙ
    пробы (извлекается из frames[i]['probes'][probe_idx] вызывающим
    кодом). Возвращает асинхронную подпоследовательность в формате,
    близком к data/synthetic_probe_scene.py:sample_probe_sequence —
    t/dt/cold_start/pool_index + переданные ключи probe_series,
    проиндексированные по выбранной подпоследовательности.
    """
    indices = sample_probe_indices(pool_t, cfg, rng)
    pool_t = np.asarray(pool_t, dtype=np.float64)
    t = pool_t[indices]
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[0] = 0.0  # cold-start: первый шаг без предыдущего наблюдения

    cold_start = np.zeros(len(indices), dtype=np.float32)
    cold_start[0] = 1.0

    out = dict(
        t=t.astype(np.float32),
        dt=dt,
        cold_start=cold_start,
        pool_index=np.array(indices, dtype=np.int64),
    )
    for key, values in probe_series.items():
        values = np.asarray(values)
        out[key] = values[indices]
    return out
