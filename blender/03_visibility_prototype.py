"""
Prototype §3.1, пункт 3: precomputed visibility (raycast проба -> точка).

Гипотеза: bpy scene.ray_cast(depsgraph, origin, direction, distance=...)
корректно детектирует окклюзию, если запрашивать луч на дистанцию чуть
короче полного расстояния до целевой точки (иначе луч попадёт в саму
целевую поверхность и это ложно посчитается как "препятствие").

Независимая проверка: ray-AABB intersection (slab method) на чистом
Python, без обращения к Blender API вообще — считает ожидаемую
видимость геометрически по тем же координатам occluder-бокса,
которые заданы в сцене. Если bpy-raycast и независимый расчёт
расходятся — тест проваливается.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/03_visibility_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path
from mathutils import Vector

EPS = 1e-4


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def build_scene():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    # Occluder: box centered at (2, 0, 1), half-extent 0.5 on each axis
    # -> AABB spans x:[1.5,2.5] y:[-0.5,0.5] z:[0.5,1.5]
    occluder_center = Vector((2.0, 0.0, 1.0))
    occluder_half = Vector((0.5, 0.5, 0.5))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(occluder_center))
    occluder = bpy.context.active_object
    occluder.name = "Occluder"

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 2))
    probe = bpy.context.active_object
    probe.name = "Probe_000"

    return probe, occluder_center, occluder_half


def ray_aabb_ground_truth(origin, target, box_center, box_half):
    """Slab-method ray-AABB intersection, полностью независимо от bpy.

    Возвращает True, если отрезок origin->target пересекает AABB
    строго между origin и target (то есть точка должна быть occluded).
    """
    d = [target[i] - origin[i] for i in range(3)]
    seg_len = sum(v * v for v in d) ** 0.5
    if seg_len < 1e-9:
        return False

    t_min, t_max = 0.0, 1.0  # параметр вдоль отрезка origin->target
    for i in range(3):
        lo = box_center[i] - box_half[i]
        hi = box_center[i] + box_half[i]
        if abs(d[i]) < 1e-12:
            if origin[i] < lo or origin[i] > hi:
                return False  # луч параллелен грани и вне слэба
            continue
        t1 = (lo - origin[i]) / d[i]
        t2 = (hi - origin[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    # Пересечение есть на параметре (t_min, t_max) внутри (0,1),
    # исключая сами концы отрезка (точку-цель на границе не считаем).
    return t_min < 1.0 - 1e-6 and t_max > 1e-6 and t_min < t_max


def bpy_is_visible(depsgraph, scene, origin, target):
    """True если между origin и target по мнению bpy.ray_cast нет препятствий.

    Луч запускается с небольшим отступом от origin и обрезается чуть
    короче полного расстояния до target, чтобы не засчитывать саму
    целевую поверхность (плоскость пола) как "препятствие".
    """
    origin_v = Vector(origin)
    target_v = Vector(target)
    direction = target_v - origin_v
    dist = direction.length
    direction.normalize()

    ray_origin = origin_v + direction * EPS
    max_dist = max(dist - 2 * EPS, 0.0)

    result, loc, normal, idx, obj, matrix = scene.ray_cast(
        depsgraph, ray_origin, direction, distance=max_dist
    )
    return not result


def main():
    clear_scene()
    probe, occ_center, occ_half = build_scene()
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    probe_pos = tuple(probe.matrix_world.translation)

    test_points = {
        "close_before_box": (0.5, 0.0, 0.0),
        "clearly_visible_side": (1.0, 3.0, 0.0),
        "through_box_center": (4.0, 0.0, 0.0),
        "far_beyond_but_offset_y": (4.0, 3.0, 0.0),
    }

    results = {}
    all_match = True
    for name, pt in test_points.items():
        expected_occluded = ray_aabb_ground_truth(probe_pos, pt, occ_center, occ_half)
        bpy_visible = bpy_is_visible(depsgraph, scene, probe_pos, pt)
        expected_visible = not expected_occluded
        match = bpy_visible == expected_visible
        all_match = all_match and match
        results[name] = {
            "point": pt,
            "independent_expected_visible": expected_visible,
            "bpy_ray_cast_visible": bpy_visible,
            "match": match,
        }

    out = {
        "probe_pos": probe_pos,
        "occluder_center": tuple(occ_center),
        "occluder_half_extent": tuple(occ_half),
        "results": results,
        "PASS": all_match,
    }

    out_path = Path(__file__).parent / "03_visibility_prototype_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("PROTOTYPE RESULT:", json.dumps(out, indent=2))

    if not all_match:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
