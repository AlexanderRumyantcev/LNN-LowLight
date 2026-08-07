"""
§3.2(Б) — проверочная партия РЕАЛЬНОГО мульти-spp рендеринга.

Цель: собрать эмпирическое распределение шума Cycles на нескольких
уровнях spp (не полагаясь на синтетическую формулу std=sqrt(C·value/spp),
которую нельзя валидировать сама на себе — circular validation).

Метод: та же статичная сцена (build_static_geometry + фиксированный t=0,
без анимации света), N_REPEATS независимых рендеров на каждый spp-уровень
из SPP_LEVELS, различающихся только cycles.seed. Adaptive sampling
ВЫКЛЮЧЕН здесь намеренно (в отличие от generate_dataset.py) — нужен
контролируемый фиксированный spp как независимая переменная, а не
эффективный spp, который сам подстраивается под шум (что и происходит
при adaptive sampling в основном пайплайне).

Плюс один high-spp референс (REFERENCE_SPP) как приближение к "истинному"
значению — для оценки bias, если он появится на низких spp.

Этот скрипт только СОБИРАЕТ сырые данные в JSON. Статистический анализ
(гауссовость, монотонность std(spp), пространственная/временная
корреляция — три пункта §3.2) — следующий шаг, отдельно.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/verify_noise_model.py
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import generate_dataset as gd  # noqa: E402

SPP_LEVELS = [1, 2, 3, 4, 8, 16, 64]
N_REPEATS = 32
REFERENCE_SPP = 2048


def render_components_only(scene, dirs):
    """Как render_frame_buffers, но НЕ схлопывает diffuse_direct/indirect в
    сумму — нужны раздельно, чтобы проверить, не смешивает ли комбинированный
    irradiance два компонента с разной шумовой природой (indirect обычно
    шумит на семпл сильнее direct — многократные отскоки)."""
    bpy.ops.render.render(write_still=False)
    diffdir = gd.load_exr_rgb(gd.find_exr(dirs["diffuse_direct"]))
    diffind = gd.load_exr_rgb(gd.find_exr(dirs["diffuse_indirect"]))
    alpha = gd.load_exr_scalar(gd.find_exr(dirs["alpha"]))
    return diffdir, diffind, alpha > 0.5

def main():
    gd.clear_all()
    floor, occluder = gd.build_static_geometry()
    scene = gd.bpy.context.scene
    dirs = gd.setup_render_and_passes(scene, gd.OUT_DIR)

    # Фиксированная статичная сцена, свет не двигается — расписание нужно
    # только чтобы получить energy(t=0.0), animation здесь не используется
    # (update_light теперь требует light_schedule, см. generate_dataset.py
    # пункт (a)).
    light_schedule = gd.build_light_schedule(gd.LightScheduleConfig())
    gd.update_light(0.0, light_schedule)

    # Контролируемый spp вместо adaptive — независимая переменная для теста.
    scene.cycles.use_adaptive_sampling = False

    print(f"Rendering reference at spp={REFERENCE_SPP}...")
    scene.cycles.samples = REFERENCE_SPP
    scene.cycles.seed = 0
    ref_direct, ref_indirect, reference_valid = render_components_only(scene, dirs)

    results_direct = {}
    results_indirect = {}
    valid_intersection = reference_valid.copy()
    for spp in SPP_LEVELS:
        print(f"Rendering spp={spp}, {N_REPEATS} repeats...")
        scene.cycles.samples = spp
        rep_direct, rep_indirect = [], []
        for r in range(N_REPEATS):
            scene.cycles.seed = 1000 * spp + r  # разные seed'ы, но детерминированно
            diffdir, diffind, valid = render_components_only(scene, dirs)
            valid_intersection = valid_intersection & valid
            rep_direct.append(diffdir.tolist())
            rep_indirect.append(diffind.tolist())
        results_direct[str(spp)] = rep_direct
        results_indirect[str(spp)] = rep_indirect
        print(f"  done: {len(rep_direct)} repeats collected")

    n_dropped = int(reference_valid.sum() - valid_intersection.sum())
    print(f"Probes dropped for edge instability at low spp: {n_dropped} "
          f"of {int(reference_valid.sum())}")


    out = {
        "grid_res": gd.GRID_RES,
        "spp_levels": SPP_LEVELS,
        "n_repeats": N_REPEATS,
        "reference_spp": REFERENCE_SPP,
        "reference_direct": ref_direct.tolist(),
        "reference_indirect": ref_indirect.tolist(),
        "valid": reference_valid.tolist(),
        "stable_valid": valid_intersection.tolist(),
        "n_dropped_for_instability": n_dropped,
        "repeats_by_spp_direct": results_direct,
        "repeats_by_spp_indirect": results_indirect,
    }
    out_path = Path(__file__).parent / "verify_noise_model_components_result.json"
    out_path.write_text(json.dumps(out))
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
