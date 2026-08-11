"""
Camera path — камера движется по круговой панораме вокруг сцены,
взгляд всегда направлен в центр (look-at). См. §2.1
TZ_stage1b_per_pixel_dense_fallback.md.

Как и light_schedule.py, модуль ЧИСТЫЙ numpy, без bpy — сэмплируется
по РЕАЛЬНОМУ t, не по кадровой сетке, тем же принципом, что и
intensity_at (bpy-зависимый код в generate_dataset.py потом строит
camera.location/rotation из того, что вернёт этот модуль).

В отличие от light_schedule.py траектория АНАЛИТИЧЕСКАЯ, не кусочная
(нет step/drift-сегментов) — не нужен объект "segments", достаточно
одной детерминированной функции camera_pose_at(cfg, t). Простая форма
(круг, постоянная угловая скорость) выбрана намеренно — см. §2.1 ТЗ:
цель не "интересная" траектория, а контролируемая, воспроизводимая
частота диссокклюзионных событий, регулируемая одним параметром
(n_revolutions).
"""
from dataclasses import dataclass

import numpy as np

@dataclass
class CameraPathConfig:
    # Должна совпадать с LightScheduleConfig.total_duration — свет и
    # движение камеры определены на одном и том же интервале t как
    # две ОРТОГОНАЛЬНЫЕ оси нерегулярности (§3.1 ТЗ этапа 1b), а не
    # заменяют друг друга.
    total_duration: float = 400.0

    # Число полных оборотов панорамы за total_duration — единственный
    # параметр, управляющий частотой диссокклюзионных событий.
    # Стартовое значение не откалибровано против реального распределения
    # диссокклюзий — по факту первого прогона генератора подлежит
    # эмпирической калибровке (§3.4 ТЗ этапа 1b, тот же принцип, что и
    # калибровка dt_spike_prob в spike-test).
    n_revolutions: float = 2.0

    radius: float = 8.0
    height: float = 3.0
    center: tuple = (0.0, 0.0, 0.0)
    start_angle_deg: float = 0.0

def camera_pose_at(cfg: CameraPathConfig, t: float):
    """Позиция камеры и точка look-at в момент t.

    Возвращает (position, look_at_target) — оба world-space (x, y, z)
    numpy-массивы формы (3,). Rotation здесь НЕ строится (модуль без
    bpy) — bpy-код применяет camera.location = position и наводит
    камеру на look_at_target любым эквивалентным способом ('Track To'
    constraint на empty-объект в центре сцены, либо прямой matrix_world
    look-at расчёт).
    """
    angular_speed = 2.0 * np.pi * cfg.n_revolutions / cfg.total_duration
    theta = np.deg2rad(cfg.start_angle_deg) + angular_speed * t

    cx, cy, cz = cfg.center
    position = np.array([
        cx + cfg.radius * np.cos(theta),
        cy + cfg.radius * np.sin(theta),
        cz + cfg.height,
    ])
    look_at_target = np.array([cx, cy, cz], dtype=float)
    return position, look_at_target
