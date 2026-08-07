"""
Prototype §3.1, пункт 4: spp-метаданные (confidence-сигнал для staleness-gate).

Идея: при adaptive sampling Cycles отдаёт per-pixel "Debug Sample Count" —
реальное число сэмплов, потраченных на пиксель. Гипотеза для проверки:
эта карта не константна и осмысленно коррелирует со сложностью
освещения — в зоне с резким краем тени (высокая дисперсия) adaptive
sampling должен тратить больше сэмплов, чем в ровной, плоско освещённой
зоне, и все значения должны укладываться в [adaptive_min_samples, samples].

Сцена: пол, one narrow "gobo"-блокер над серединой пола создаёт резкую
границу тень/свет ровно по диагонали кадра — так в одном рендере сразу
есть "шумная" зона (у края тени) и "гладкая" зона (далеко от края,
внутри однородной тени или однородного света).

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/04_spp_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "_spp_prototype_out"
RES = 96
SAMPLES_MAX = 128
ADAPTIVE_MIN = 4
ADAPTIVE_THRESHOLD = 0.01


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
    mat = bpy.data.materials.new(name="ReceiverMat")
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    receiver.data.materials.append(mat)

    # Узкий "gobo"-блокер над одним краем плоскости -> резкая граница
    # тени примерно по диагонали кадра при виде сверху.
    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(-0.3, 0.3, 1.0))
    gobo = bpy.context.active_object
    gobo.name = "Gobo"
    gobo.rotation_euler = (0.0, 0.0, 0.7853981634)  # 45 градусов по Z

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 3))
    light = bpy.context.active_object
    light.data.energy = 300.0
    light.data.size = 0.3  # небольшой источник -> резкая тень, не мягкая

    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.active_object
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.0
    bpy.context.scene.camera = cam


def setup_render(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES_MAX
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_min_samples = ADAPTIVE_MIN
    scene.cycles.adaptive_threshold = ADAPTIVE_THRESHOLD
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False

    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    vl = scene.view_layers[0]
    vl.cycles.pass_debug_sample_count = True

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
        in_sock = fo.inputs[0]
        tree.links.new(rl.outputs[socket_name], in_sock)
        return pass_dir

    dirs = {
        "combined": make_output("combined", "Image", "RGBA"),
        "samplecount": make_output("samplecount", "Debug Sample Count", "FLOAT"),
    }
    return dirs


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def load_grid(exr_path, channels):
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = img.pixels[:]
    grid = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 4
            if channels == "gray":
                grid[y][x] = (px[idx] + px[idx + 1] + px[idx + 2]) / 3.0
            else:  # single value channel, stored in R
                grid[y][x] = px[idx]
    bpy.data.images.remove(img)
    return grid, w, h


def gradient_magnitude(grid, w, h):
    grad = [[0.0] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = grid[y][x + 1] - grid[y][x - 1]
            gy = grid[y + 1][x] - grid[y - 1][x]
            grad[y][x] = (gx * gx + gy * gy) ** 0.5
    return grad


def main():
    clear_scene()
    build_scene()
    dirs = setup_render(bpy.context.scene)

    bpy.ops.render.render(write_still=False)

    beauty, w, h = load_grid(find_exr(dirs["combined"]), "gray")
    spp, w2, h2 = load_grid(find_exr(dirs["samplecount"]), "value")
    assert (w, h) == (w2, h2), "resolution mismatch between passes"

    grad = gradient_magnitude(beauty, w, h)

    interior = [
        (x, y)
        for y in range(1, h - 1)
        for x in range(1, w - 1)
    ]
    grad_vals = sorted(interior, key=lambda p: grad[p[1]][p[0]])
    n = len(grad_vals)
    low_group = grad_vals[: n // 2]                  # нижние 50% по градиенту -> "гладкие" зоны
    high_group = grad_vals[int(n * 0.9):]             # верхние 10% по градиенту -> край тени

    def mean_spp(group):
        return sum(spp[y][x] * SAMPLES_MAX for x, y in group) / len(group)

    low_mean = mean_spp(low_group)
    high_mean = mean_spp(high_group)

    # "Debug Sample Count" отдаётся нормированным на samples_max (в диапазоне
    # 0..1, доля потраченного бюджета), а не сырым целым числом сэмплов —
    # переводим обратно в реальные spp для сверки с adaptive_min_samples.
    all_spp_vals = [spp[y][x] * SAMPLES_MAX for x, y in interior]
    spp_min, spp_max = min(all_spp_vals), max(all_spp_vals)

    within_bounds = spp_min >= ADAPTIVE_MIN - 1e-6 and spp_max <= SAMPLES_MAX + 1e-6
    varies = (spp_max - spp_min) > 0  # не константа по всему кадру
    edge_gets_more_samples = high_mean > low_mean * 1.2

    result = {
        "resolution": [w, h],
        "adaptive_min_samples": ADAPTIVE_MIN,
        "samples_max": SAMPLES_MAX,
        "spp_min_observed": spp_min,
        "spp_max_observed": spp_max,
        "low_gradient_zone_mean_spp": low_mean,
        "high_gradient_zone_mean_spp": high_mean,
        "within_bounds": within_bounds,
        "spp_varies_across_image": varies,
        "high_gradient_zone_gets_more_samples": edge_gets_more_samples,
        "PASS": within_bounds and varies and edge_gets_more_samples,
    }

    out_path = Path(__file__).parent / "04_spp_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("PROTOTYPE RESULT:", json.dumps(result, indent=2))

    if not result["PASS"]:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
