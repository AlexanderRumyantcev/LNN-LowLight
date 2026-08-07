"""
Пул моментов t для рендера на ОБЪЕДИНЕНИИ (union) — пункт (b) плана
адаптера (drawer_LNN_LowLight_implementation, 2026-08-03, "ТРИ
КОНКРЕТНЫЕ ДЫРЫ", дыра №1).

Раньше generate_dataset.py рендерил 4 захардкоженных t на весь грид
сразу. Сотни моментов не нужно рендерить ПОРОЗНЬ на пробу — один
GRID_RES x GRID_RES рендер уже даёт буферы ВСЕХ проб разом за один
Cycles-проход. Пункт (b) — заменить эти 4 t на плотный пул моментов,
покрывающий весь light_schedule.total_duration; пункт (c) далее
подвыберет из НЕГО для каждой пробы свою асинхронную подпоследователь-
ность (без повторного рендера — переиспользуя уже посчитанные буферы).

Важное следствие: per-probe асинхронные Δt из (c) будут КВАНТОВАНЫ до
ближайшего t из этого пула — рендер физически не может производиться
на произвольный непрерывный t для каждой пробы отдельно, рендерится
общий грид. Поэтому плотность пула (dt_range) должна быть заметно
мельче per-probe dt_range из data/synthetic_probe_scene.py ((0.5, 2.0)),
иначе подвыборке в (c) будет не из чего выбирать.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MomentPoolConfig:
    total_duration: float = 400.0
    # Плотность пула — сознательно мельче per-probe dt_range=(0.5, 2.0)
    # из synthetic_probe_scene.py (см. докстринг модуля).
    dt_range: tuple = (1.0, 2.0)
    seed: int = 0


def build_moment_pool(cfg: MomentPoolConfig):
    """Список вещественных t в [0, cfg.total_duration), плотно и неравномерно —
    то же требование §3.3, что и для light_position_for_t: dt между соседними
    t не совпадает с целочисленным шагом кадровой сетки Blender.
    """
    rng = np.random.default_rng(cfg.seed)
    t_values = [0.0]
    t_cur = 0.0
    while True:
        dt = float(rng.uniform(*cfg.dt_range))
        t_cur += dt
        if t_cur >= cfg.total_duration:
            break
        t_values.append(t_cur)
    return t_values
