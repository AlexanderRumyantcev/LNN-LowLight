"""
Prototype §3.1, пункт 5: albedo-буфер (диагностический AOV, не вход модели).

Гипотеза: Cycles-пасс "Diffuse Color" отдаёт именно material base color
(без применения освещения) — то есть должен:
  (a) не меняться при изменении интенсивности источника света,
  (b) численно соответствовать заданному base_color материала
      (с точностью до цветового пространства/gamma — проверяем в linear).

Схема: один и тот же материал (albedo=0.5) рендерится дважды с разной
энергией света (100 и 400). Сравниваем Diffuse Color между рендерами
(ожидаем ~равенство) и с заданным base_color (ожидаем совпадение),
а Combined (beauty) — обязан отличаться между рендерами, иначе тест
ничего не проверяет.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/05_albedo_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "_albedo_prototype_out"
RES = 32
SAMPLES = 256
ALBEDO = 0.5


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def build_scene(light_energy):
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    receiver = bpy.context.active_object
    receiver.name = "Receiver"

    mat = bpy.data.materials.new(name="ReceiverMat")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (ALBEDO, ALBEDO, ALBEDO, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    receiver.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    light = bpy.context.active_object
    light.data.energy = light_energy
    light.data.size = 1.0

    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.0
    bpy.context.scene.camera = cam


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
    vl.use_pass_diffuse_color = True

    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def make_output(label, socket_name, socket_type):
        pass_dir = OUT_DIR / label
        pass_dir.mkdir(parents=True, exist_ok=True)
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.label = label
        fo.directory = str(pass_dir)
        fo.file_name = "buf"
        fo.format.media_type = "IMAGE"
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.file_output_items.new(socket_type, label)
        tree.links.new(rl.outputs[socket_name], fo.inputs[0])
        return pass_dir

    return {
        "albedo": make_output("albedo", "Diffuse Color", "RGBA"),
        "combined": make_output("combined", "Image", "RGBA"),
    }


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def center_pixel_mean(exr_path):
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
    return (r_sum / n, g_sum / n, b_sum / n)


def render_and_measure(light_energy, tag):
    clear_scene()
    build_scene(light_energy)
    dirs = setup_render_and_passes(bpy.context.scene)

    bpy.ops.render.render(write_still=False)

    albedo_px = center_pixel_mean(find_exr(dirs["albedo"]))
    combined_px = center_pixel_mean(find_exr(dirs["combined"]))

    return {"tag": tag, "light_energy": light_energy, "albedo_pixel": albedo_px, "combined_pixel": combined_px}


def main():
    r_dim = render_and_measure(100.0, "light_100")
    r_bright = render_and_measure(400.0, "light_400")

    a_dim = r_dim["albedo_pixel"]
    a_bright = r_bright["albedo_pixel"]
    albedo_rel_diff = max(
        abs(x - y) / max(abs(x), abs(y), 1e-8) for x, y in zip(a_dim, a_bright)
    )

    expected = (ALBEDO, ALBEDO, ALBEDO)
    albedo_abs_diff_from_expected = max(abs(x - e) for x, e in zip(a_dim, expected))

    c_dim = r_dim["combined_pixel"]
    c_bright = r_bright["combined_pixel"]
    combined_rel_diff = max(
        abs(x - y) / max(abs(x), abs(y), 1e-8) for x, y in zip(c_dim, c_bright)
    )

    albedo_invariant_to_light = albedo_rel_diff < 0.02
    albedo_matches_base_color = albedo_abs_diff_from_expected < 0.02
    combined_changed = combined_rel_diff > 0.5  # ~4x энергии должно заметно поднять яркость

    result = {
        "light_100": r_dim,
        "light_400": r_bright,
        "albedo_relative_diff_between_lights": albedo_rel_diff,
        "albedo_abs_diff_from_set_base_color": albedo_abs_diff_from_expected,
        "combined_relative_diff_between_lights": combined_rel_diff,
        "albedo_invariant_to_light": albedo_invariant_to_light,
        "albedo_matches_base_color": albedo_matches_base_color,
        "combined_changed_as_expected": combined_changed,
        "PASS": albedo_invariant_to_light and albedo_matches_base_color and combined_changed,
    }

    out_path = Path(__file__).parent / "05_albedo_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("PROTOTYPE RESULT:", json.dumps(result, indent=2))

    if not result["PASS"]:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
