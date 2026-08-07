"""GUI-viewer сцены generate_dataset.py — строит геометрию и свет, оставляет Blender открытым.

Запуск (БЕЗ --background, чтобы открыть GUI):
  /Applications/Blender.app/Contents/MacOS/Blender --python blender/view_scene.py

Опционально — с конкретным t для позиции света:
  /Applications/Blender.app/Contents/MacOS/Blender --python blender/view_scene.py -- --t 50.0
"""
import bpy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from light_schedule import LightScheduleConfig, build_light_schedule, intensity_at

GRID_RES = 8


def light_position_for_t(t: float):
    radius = 2.5
    height = 3.0
    angle = t
    return (radius * math.cos(angle), radius * math.sin(angle), height)


def clear_all():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def build_static_geometry():
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new(name="FloorMat")
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    floor.data.materials.append(mat)

    bpy.context.view_layer.objects.active = floor
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=40)
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in floor.data.vertices:
        x, y = v.co.x, v.co.y
        v.co.z = (
            0.35 * math.sin(0.8 * x + 0.3) * math.cos(0.6 * y - 0.5)
            + 0.15 * math.sin(1.7 * x - y)
        )
    floor.data.update()
    bpy.ops.object.shade_smooth()

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.5, 1.0, 0.75))
    occluder = bpy.context.active_object
    occluder.name = "Occluder"
    occluder.scale = (1.0, 1.0, 1.5)
    occluder.data.materials.append(mat)

    bpy.ops.object.camera_add(location=(0, 0, 6))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 8.0
    bpy.context.scene.camera = cam

    return floor, occluder


def update_light(t: float, light_schedule):
    old = bpy.data.objects.get("Light")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    pos = light_position_for_t(t)
    bpy.ops.object.light_add(type="AREA", location=pos)
    light = bpy.context.active_object
    light.name = "Light"
    light.data.energy = intensity_at(light_schedule, t)
    light.data.size = 0.3
    direction = -1 * light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return pos


def main():
    t = 0.0
    for i, arg in enumerate(sys.argv):
        if arg == "--t" and i + 1 < len(sys.argv):
            t = float(sys.argv[i + 1])

    clear_all()
    build_static_geometry()

    cfg = LightScheduleConfig(seed=0)
    schedule = build_light_schedule(cfg)
    light_pos = update_light(t, schedule)
    intensity = intensity_at(schedule, t)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 1280

    seg_type_names = {0: "static", 1: "step", 2: "drift"}
    for t_start, t_end, seg_type, _ in schedule:
        if t_start <= t < t_end:
            seg_name = seg_type_names[seg_type]
            break
    else:
        seg_name = "?"

    print(f"\n=== Сцена готова ===")
    print(f"t = {t:.2f}")
    print(f"Сегмент: {seg_name}")
    print(f"Интенсивность света: {intensity:.1f} Вт")
    print(f"Позиция света: ({light_pos[0]:.2f}, {light_pos[1]:.2f}, {light_pos[2]:.2f})")
    print(f"Расписание: {len(schedule)} сегментов, total_duration={cfg.total_duration}")
    print(f"  Сегменты: {[(s[0], s[1], seg_type_names[s[2]]) for s in schedule]}")
    print(f"\nBlender открыт для интерактивного просмотра.")
    print(f"Нажмите Z -> Rendered для предпросмотра Cycles.")
    print(f"Для смены t: запустите с -- --t <значение>\n")


if __name__ == "__main__":
    main()
