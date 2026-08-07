"""
Prototype §3.1, пункт 6: normal-буфер (world-space, для normal-weighting
в интерполяции §2.3 — независимо от visibility, которая решает окклюзию).

Гипотеза: Cycles-пасс "Normal" отдаёт world-space нормаль шейдируемой
поверхности. Проверяем на плоскости с намеренно "некруглым" поворотом
(не кратным 90°, сразу по двум осям), чтобы тест не проходил случайно
из-за симметрии. Ожидаемую world-space нормаль считаем независимо —
поворотом локальной нормали (0,0,1) матрицей object.matrix_world,
взятой ДО рендера, тем же способом, что и в прототипе 1 (позиция пробы).

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/06_normal_prototype.py
"""
import bpy
import json
import sys
import math
from pathlib import Path
from mathutils import Vector

OUT_DIR = Path(__file__).parent / "_normal_prototype_out"
RES = 32
SAMPLES = 64


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def build_scene():
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    receiver = bpy.context.active_object
    receiver.name = "Receiver"
    # Намеренно "некруглый" поворот по двум осям сразу — 17.2 град по X,
    # 11.5 град по Y — чтобы тест не мог случайно пройти из-за симметрии
    # или совпадения с осью камеры.
    receiver.rotation_euler = (math.radians(17.2), math.radians(11.5), 0.0)

    mat = bpy.data.materials.new(name="ReceiverMat")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0
    receiver.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    light = bpy.context.active_object
    light.data.energy = 200.0

    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.0
    bpy.context.scene.camera = cam

    bpy.context.view_layer.update()

    local_normal = Vector((0.0, 0.0, 1.0))
    # matrix_world для поворота направления берём без translation-компоненты:
    # to_3x3() даёт чисто линейную часть (rotation+scale).
    expected_world_normal = (receiver.matrix_world.to_3x3() @ local_normal).normalized()

    return receiver, expected_world_normal


def setup_render_and_passes(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False

    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    vl = scene.view_layers[0]
    vl.use_pass_normal = True

    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name
    print("DEBUG rl outputs:", [(o.name, o.type) for o in rl.outputs])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pass_dir = OUT_DIR / "normal"
    pass_dir.mkdir(parents=True, exist_ok=True)

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(pass_dir)
    fo.file_name = "buf"
    fo.format.media_type = "IMAGE"
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_output_items.new("VECTOR", "normal")
    tree.links.new(rl.outputs["Normal"], fo.inputs[0])

    return pass_dir


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def center_pixel_vec(exr_path):
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = img.pixels[:]
    half = 2
    cx, cy = w // 2, h // 2
    r_sum = g_sum = b_sum = 0.0
    n = 0
    for y in range(cy - half, cy + half):
        for x in range(cx - half, cx + half):
            idx = (y * w + x) * 4
            r_sum += px[idx]
            g_sum += px[idx + 1]
            b_sum += px[idx + 2]
            n += 1
    bpy.data.images.remove(img)
    return Vector((r_sum / n, g_sum / n, b_sum / n))


def main():
    clear_scene()
    receiver, expected_normal = build_scene()
    pass_dir = setup_render_and_passes(bpy.context.scene)

    bpy.ops.render.render(write_still=False)

    got_normal = center_pixel_vec(find_exr(pass_dir))
    got_normal_norm = got_normal.normalized() if got_normal.length > 1e-8 else got_normal

    dot = got_normal_norm.dot(expected_normal)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

    result = {
        "expected_world_normal": list(expected_normal),
        "rendered_normal_raw": list(got_normal),
        "rendered_normal_normalized": list(got_normal_norm),
        "angle_diff_degrees": angle_deg,
        "PASS": angle_deg < 2.0,
    }

    out_path = Path(__file__).parent / "06_normal_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("PROTOTYPE RESULT:", json.dumps(result, indent=2))

    if not result["PASS"]:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
