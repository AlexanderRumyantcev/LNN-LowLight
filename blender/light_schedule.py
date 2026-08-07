"""
Длинное расписание интенсивности источника света по РЕАЛЬНОМУ t (не по
кадровой сетке) — пункт (a) плана адаптера real Blender data -> формат
run_experiment.py/build_batch (см. drawer_LNN_LowLight_implementation
2026-08-03, "ТРИ КОНКРЕТНЫЕ ДЫРЫ").

Само по себе НЕ рендерит и не трогает bpy — чистый numpy, как и
data/synthetic_probe_scene.py, откуда сюда перенесена ровно логика
_build_light_schedule/_intensity_at (static/step/drift сегменты,
параметризованные длительностью, а не числом шагов). Здесь схема
УЖЕ провалидирована на синтетике и в spike-тесте — переносим её как
есть, а не изобретаем заново, чтобы адаптер (b)/(c)/(d) видел ту же
структуру сегментов, что и synthetic-путь.

Отличие от synthetic_probe_scene.py: там schedule масштабирует
independent-от-геометрии intensity(t), которая потом делится на
dist^2 аналитической формулой. Здесь intensity(t) идёт напрямую в
light.data.energy Blender-источника — дальнейшее падение яркости с
расстоянием и все GI-эффекты считает сам Cycles, аналитическая
формула падения не нужна.
"""
from dataclasses import dataclass

import numpy as np

SEG_STATIC, SEG_STEP, SEG_DRIFT = 0, 1, 2


@dataclass
class LightScheduleConfig:
    # Общая длительность расписания в РЕАЛЬНОМ t. Достаточно большая,
    # чтобы из неё можно было набрать "сотни моментов на пробу" (дыра
    # №1) — конкретное число проверяется отдельно на этапе (b)/(c), не
    # здесь; этот модуль только про форму intensity(t).
    total_duration: float = 400.0
    n_segments_range: tuple = (4, 7)
    segment_duration_range: tuple = (40.0, 90.0)

    # Диапазон в единицах Blender AREA light energy (Вт) — прежнее
    # константное значение в generate_dataset.py было energy=250.0,
    # берём его как середину диапазона, чтобы сохранить сопоставимую
    # общую яркость сцены со старым поведением.
    intensity_range: tuple = (60.0, 600.0)
    step_magnitude_range: tuple = (0.3, 3.0)
    drift_rate_range: tuple = (-0.02, 0.02)

    seed: int = 0


def build_light_schedule(cfg: LightScheduleConfig):
    """Список сегментов (t_start, t_end, type, params), суммарно = cfg.total_duration.

    Идентичная по структуре _build_light_schedule из
    data/synthetic_probe_scene.py, продублирована здесь намеренно —
    blender/ и data/ остаются независимыми модулями (первый требует
    bpy.app.binary_path в окружении Blender-python, второй тестируется
    системным python без Blender; общий импорт создал бы ненужную
    связку окружений ради шести строк логики).
    """
    rng = np.random.default_rng(cfg.seed)
    n_segments = rng.integers(cfg.n_segments_range[0], cfg.n_segments_range[1] + 1)
    durations = rng.uniform(cfg.segment_duration_range[0], cfg.segment_duration_range[1], size=n_segments)
    durations = durations * (cfg.total_duration / durations.sum())
    boundaries = np.concatenate([[0.0], np.cumsum(durations)])
    types = rng.integers(0, 3, size=n_segments)

    cur_value = rng.uniform(*cfg.intensity_range)
    segments = []
    for i in range(n_segments):
        t_start, t_end = float(boundaries[i]), float(boundaries[i + 1])
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


def intensity_at(segments, t: float) -> float:
    """Скалярная intensity(t) — версия _intensity_at для одного t (не векторизованная,
    т.к. Blender-цикл рендерит один t за раз, в отличие от synthetic, где вся
    последовательность t считается разом).
    """
    last_end = segments[-1][1]
    t_query = t if t < last_end else last_end - 1e-6

    for t_start, t_end, seg_type, p in segments:
        if not (t_start <= t_query < t_end):
            continue
        if seg_type == SEG_STATIC:
            return float(p["value"])
        elif seg_type == SEG_STEP:
            return float(p["old"] if t_query < p["jump_t"] else p["new"])
        else:  # SEG_DRIFT
            return float(p["start_value"] * np.exp(p["rate"] * (t_query - t_start)))

    raise AssertionError(f"t={t} not covered by any segment (segments span [0, {last_end}])")
