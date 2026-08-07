"""
Prototype §3.1, пункт 1: позиция пробы (world-space).

Цель: проверить, что мы можем headless создать минимальную сцену,
разместить одну пробу на фиксированной world-space позиции и
детерминированно извлечь эту позицию — до того, как добавлять
irradiance/visibility/spp/albedo/normal буферы.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/01_probe_position_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path


def clear_scene():
    """Убрать дефолтные объекты (куб/камеру/свет), чтобы сцена была детерминирована."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def build_minimal_scene():
    """Пол (плоскость) + одна проба (Empty) на известной world-space позиции.

    Позиция пробы намеренно взята не в начале координат и со смещением
    по всем трём осям, чтобы прототип реально проверял, что matrix_world,
    а не локальные координаты объекта, читаются правильно.
    """
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    probe_world_pos = (1.25, -2.5, 0.8)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=probe_world_pos)
    probe = bpy.context.active_object
    probe.name = "Probe_000"

    return probe, probe_world_pos


def parent_probe_with_offset_transform(probe, floor):
    """Спарентить пробу к полу и сдвинуть/повернуть пол.

    Это специально усложняет тест: если бы мы читали
    probe.location (локальные координаты) вместо probe.matrix_world,
    результат оказался бы неверным после родительского трансформа.
    Так прототип реально проверяет извлечение world-space, а не
    просто эхо тех же чисел, что были заданы при создании.
    """
    floor.location = (5.0, 3.0, 0.0)
    floor.rotation_euler = (0.0, 0.0, 0.7853981634)  # 45 deg вокруг Z

    probe.parent = floor
    bpy.context.view_layer.update()


def extract_world_position(probe):
    m = probe.matrix_world
    return (m.translation.x, m.translation.y, m.translation.z)


def expected_world_position(probe, floor):
    """Независимый расчёт ожидаемой world-space позиции: floor.matrix_world @ probe.location.

    Считаем руками (не через matrix_world самой пробы), чтобы иметь
    независимую проверку, а не сравнение значения со самим собой.
    """
    local = probe.location.copy()
    world = floor.matrix_world @ local
    return (world.x, world.y, world.z)


def main():
    clear_scene()
    probe, probe_world_pos = build_minimal_scene()
    floor = bpy.data.objects["Floor"]

    parent_probe_with_offset_transform(probe, floor)

    got = extract_world_position(probe)
    expected = expected_world_position(probe, floor)

    ok = all(abs(a - b) < 1e-6 for a, b in zip(got, expected))

    result = {
        "probe_name": probe.name,
        "created_at_local": list(probe_world_pos),
        "extracted_world_position": list(got),
        "independently_expected_world_position": list(expected),
        "match": ok,
    }

    out_path = Path(__file__).parent / "01_probe_position_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("PROTOTYPE RESULT:", json.dumps(result, indent=2))
    if not ok:
        print("FAIL: extracted world position does not match independent expectation", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
