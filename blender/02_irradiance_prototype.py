"""
Prototype §3.1, пункт 2: irradiance-буфер (НЕ radiance, независим от albedo).

Гипотеза для проверки: в Cycles пассы DiffDir + DiffInd (diffuse direct +
diffuse indirect, БЕЗ DiffCol) дают величину, пропорциональную irradiance
в точке — то есть НЕ должны меняться при смене albedo материала.
А итоговый composite (beauty) — обязан меняться, иначе тест ничего не
проверяет (могли просто забыть применить материал).

Схема: одна и та же минимальная сцена (плоскость-приёмник + area light)
рендерится дважды с разным base color (albedo 0.2 и 0.8). Сравниваем
DiffDir+DiffInd между рендерами (ожидаем ~равенство) и Combined
(ожидаем явное различие, пропорциональное отношению albedo).

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/02_irradiance_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "_irradiance_prototype_out"
RES = 32  # маленькое разрешение — нужен только центральный патч плоскости
SAMPLES = 512


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def build_scene(albedo):
    # Приёмник — сама "проба" в упрощённом виде: плоскость, на которую
    # смотрит камера в упор сверху.
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    receiver = bpy.context.active_object
    receiver.name = "Receiver"

    mat = bpy.data.materials.new(name="ReceiverMat")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (albedo, albedo, albedo, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0  # чисто диффузно
    bsdf.inputs["Metallic"].default_value = 0.0
    receiver.data.materials.append(mat)

    # Источник света — фиксированный, не зависит от albedo.
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    light = bpy.context.active_object
    light.data.energy = 200.0
    light.data.size = 1.0

    # Камера строго сверху, ортографическая — чтобы центральный пиксель
    # соответствовал одной и той же world-space точке на приёмнике.
    bpy.ops.object.camera_add(location=(0, 0, 5), rotation=(0, 0, 0))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.0
    bpy.context.scene.camera = cam

    return receiver, light, cam


def setup_render_and_passes(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False  # денойзер контент-зависим, ломает сравнение

    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    vl = scene.view_layers[0]
    vl.use_pass_diffuse_direct = True
    vl.use_pass_diffuse_indirect = True
    vl.use_pass_diffuse_color = True

    # Blender 5.x: scene.node_tree/use_nodes deprecated, компоузер теперь
    # отдельный NodeTree в bpy.data.node_groups, подключаемый через
    # scene.compositing_node_group.
    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def make_output(label, socket_name):
        # Blender 5.x: default fo.inputs[0] is a NodeSocketVirtual stub —
        # linking to it directly "succeeds" but produces no file. Real
        # input sockets must be created via file_output_items.new(), which
        # also decides the on-disk filename (directory + file_name +
        # item.name, concatenated with no separator — so each pass gets
        # its own subdirectory to avoid parsing that naming quirk).
        pass_dir = OUT_DIR / label
        pass_dir.mkdir(parents=True, exist_ok=True)

        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.label = label
        fo.directory = str(pass_dir)
        fo.file_name = "buf"
        fo.format.media_type = "IMAGE"
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"

        item = fo.file_output_items.new("RGBA", label)
        in_sock = fo.inputs[0]  # real socket is inputs[0]; inputs[-1] is the virtual stub
        tree.links.new(rl.outputs[socket_name], in_sock)
        return pass_dir

    make_output("diffdir", "Diffuse Direct")
    make_output("diffind", "Diffuse Indirect")
    make_output("combined", "Image")


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def center_pixel_mean(exr_path):
    """Среднее RGB по центральному 4x4 патчу EXR (снижает шум сэмплинга)."""
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = img.pixels[:]  # flat, row-major, bottom-to-top, RGBA
    channels = 4
    half = 2
    cx, cy = w // 2, h // 2
    r_sum = g_sum = b_sum = 0.0
    n = 0
    for y in range(cy - half, cy + half):
        for x in range(cx - half, cx + half):
            idx = (y * w + x) * channels
            r_sum += px[idx]
            g_sum += px[idx + 1]
            b_sum += px[idx + 2]
            n += 1
    bpy.data.images.remove(img)
    return (r_sum / n, g_sum / n, b_sum / n)


def render_and_measure(albedo, tag):
    scene = bpy.context.scene
    clear_scene()
    build_scene(albedo)
    setup_render_and_passes(scene)

    bpy.ops.render.render(write_still=False)

    diffdir = center_pixel_mean(find_exr(OUT_DIR / "diffdir"))
    diffind = center_pixel_mean(find_exr(OUT_DIR / "diffind"))
    combined = center_pixel_mean(find_exr(OUT_DIR / "combined"))

    irradiance_proxy = tuple(d + i for d, i in zip(diffdir, diffind))

    return {
        "tag": tag,
        "albedo": albedo,
        "diffdir": diffdir,
        "diffind": diffind,
        "irradiance_proxy (diffdir+diffind)": irradiance_proxy,
        "combined_beauty": combined,
    }


def main():
    r_low = render_and_measure(0.2, "albedo_0.2")
    r_high = render_and_measure(0.8, "albedo_0.8")

    irr_low = r_low["irradiance_proxy (diffdir+diffind)"]
    irr_high = r_high["irradiance_proxy (diffdir+diffind)"]
    irr_diff_rel = max(
        abs(a - b) / max(abs(a), abs(b), 1e-8) for a, b in zip(irr_low, irr_high)
    )

    beauty_low = r_low["combined_beauty"]
    beauty_high = r_high["combined_beauty"]
    beauty_ratio = [
        (h / l if l > 1e-8 else None) for l, h in zip(beauty_low, beauty_high)
    ]

    # Ожидание: irradiance почти не меняется (albedo-independent),
    # а beauty меняется примерно в 0.8/0.2 = 4 раза (плюс шум сэмплинга).
    irradiance_invariant = irr_diff_rel < 0.02
    beauty_changed = all(r is not None and abs(r - 4.0) < 0.5 for r in beauty_ratio)

    result = {
        "albedo_0.2": r_low,
        "albedo_0.8": r_high,
        "irradiance_relative_diff": irr_diff_rel,
        "beauty_ratio_high_over_low": beauty_ratio,
        "irradiance_invariant_to_albedo": irradiance_invariant,
        "beauty_changed_as_expected": beauty_changed,
        "PASS": irradiance_invariant and beauty_changed,
    }

    out_json = Path(__file__).parent / "02_irradiance_prototype_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    print("PROTOTYPE RESULT:", json.dumps(result, indent=2))

    if not result["PASS"]:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
