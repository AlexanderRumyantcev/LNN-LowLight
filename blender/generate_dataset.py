"""
Генератор датасета §3.1: собирает все 6 буферов (позиция, irradiance,
precomputed visibility, spp, albedo, normal) в единый headless-проход
Blender/Cycles, для сетки проб на поверхности сцены и произвольного
вещественного момента времени t (параметризующего позицию источника
света — независимо от целочисленной кадровой сетки Blender, §3.3).

Дизайн, вытекающий из прототипов 01-06:
- Пробы = пиксели одного top-down ortho-рендера сетки на полу сцены
  (позиция "зафиксирована геометрией сцены" — §3.1 п.1). Один рендер
  разом даёт position/normal/albedo/irradiance/spp для всей сетки —
  все нужные Cycles-пассы сосуществуют в одном Render Layers node
  (проверено в _probe_api.py: Position, Normal, Diffuse Direct/
  Indirect/Color, Debug Sample Count все доступны одновременно).
- Precomputed visibility — отдельная величина, НЕ из рендер-пассов:
  NxN raycast-матрица между позициями проб (bpy scene.ray_cast).
  Зависит только от статической геометрии сцены (пол+occluder), не
  от освещения — поэтому считается один раз, а не на каждый кадр.
  Геометрия и камера строятся один раз (build_static_geometry),
  свет — отдельно и пересоздаётся на каждый t (update_light).

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/generate_dataset.py
"""
import bpy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from light_schedule import LightScheduleConfig, build_light_schedule, intensity_at
from moment_pool import MomentPoolConfig, build_moment_pool

OUT_DIR = Path(__file__).parent / "_dataset_out"
# Одна total_duration на оба модуля — пул (b) должен покрывать ровно тот
# же диапазон t, на котором определено расписание освещения (a).
TOTAL_DURATION = 400.0
GRID_RES = 8             # проб на сторону сетки -> GRID_RES**2 проб на кадр
                          # (демонстрационный масштаб: visibility O(N^2)
                          # raycast'ов, 64 пробы -> 4096 пар, секунды;
                          # для полного датасета поднять до желаемого
                          # разрешения — цикл не завязан на это число)
SAMPLES_MAX = 128
ADAPTIVE_MIN = 4
ADAPTIVE_THRESHOLD = 0.01
VISIBILITY_EPS = 1e-4
# Пункт (d): второй, high-spp рендер того же t как эталон без шума
# (true_irradiance). Значение и общий подход (фиксированный spp,
# adaptive OFF) — то же REFERENCE_SPP, что уже откалибровано и
# проверено в verify_noise_model.py §3.2(Б).
REFERENCE_SPP = 2048


def clear_all():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def light_position_for_t(t: float):
    """Непрерывная параметрическая траектория источника света.

    Вещественный t (любое значение, не обязано быть целым и не
    привязано к bpy.context.scene.frame_current) отображается в
    орбитальную позицию источника — так dt между двумя выборками
    определяется исключительно разницей t, а не шагом кадровой сетки.
    """
    radius = 2.5
    height = 3.0
    angle = t  # t в радианах вдоль орбиты
    return (radius * math.cos(angle), radius * math.sin(angle), height)


def build_static_geometry():
    """Пол + occluder + камера — строятся один раз, не зависят от t.

    Повторяет геометрию прототипов 3/4: occluder создаёт и резкую
    тень (для содержательного spp-варьирования), и частичную
    окклюзию (для содержательной visibility-матрицы).
    """
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new(name="FloorMat")
    # По ТИПУ узла, не по имени: имя узла "Principled BSDF" локализуется в
    # Blender 5.x под русским UI ("Принципиальный BSDF") — обращение по
    # имени падает с KeyError на этой машине, обращение по type — нет.
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    floor.data.materials.append(mat)

    # РЕШЕНИЕ (2026-08-03, по итогам теста leave-one-out на compute_interpolation_weights):
    # плоский пол делал все пробы копланарными с ОДНОЙ normal (0,0,1) -> normal_term
    # в формуле §2.3 вырождался в ~0 для всех пар -> fallback на 100% проб. Реальные
    # сцены почти никогда не идеально плоские, поэтому даём полу мягкий холмистый рельеф
    # (детерминированная сумма синусоид, seed фиксирован через сами координаты, без
    # noise-текстуры Cycles — воспроизводимо между запусками) — это даёт пробам
    # содержательно разные position.z И разные normal, не меняя саму формулу §2.3 и
    # не вводя отдельное понятие "проба отдельно от точки поверхности".
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
    """Пересоздать источник света на позиции light_position_for_t(t), с
    интенсивностью intensity_at(light_schedule, t) — длинное расписание
    static/step/drift по реальному t (пункт (a) плана адаптера, см.
    blender/light_schedule.py).

    Удаляет только сам light-объект (если существовал), геометрию и
    камеру не трогает — так visibility-матрица остаётся валидной
    между кадрами без пересчёта.
    """
    old = bpy.data.objects.get("Light")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    pos = light_position_for_t(t)
    bpy.ops.object.light_add(type="AREA", location=pos)
    light = bpy.context.active_object
    light.name = "Light"
    light.data.energy = intensity_at(light_schedule, t)
    light.data.size = 0.3

    # Направить свет на начало координат сцены, а не оставлять его
    # смотрящим по умолчанию вниз локальной -Z без учёта позиции.
    direction = -1 * light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    return pos


def setup_render_and_passes(scene, out_dir):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES_MAX
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_min_samples = ADAPTIVE_MIN
    scene.cycles.adaptive_threshold = ADAPTIVE_THRESHOLD
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False

    scene.render.resolution_x = GRID_RES
    scene.render.resolution_y = GRID_RES
    scene.render.resolution_percentage = 100

    vl = scene.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_normal = True
    vl.use_pass_diffuse_color = True
    vl.use_pass_diffuse_direct = True
    vl.use_pass_diffuse_indirect = True
    vl.cycles.pass_debug_sample_count = True

    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name


    def make_output(label, socket_name, socket_type):
        pass_dir = out_dir / label
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

    scene.render.film_transparent = True

    return {
        "position": make_output("position", "Position", "VECTOR"),
        "normal": make_output("normal", "Normal", "VECTOR"),
        "diffuse_color": make_output("diffuse_color", "Diffuse Color", "RGBA"),
        "diffuse_direct": make_output("diffuse_direct", "Diffuse Direct", "RGBA"),
        "diffuse_indirect": make_output("diffuse_indirect", "Diffuse Indirect", "RGBA"),
        "sample_count": make_output("sample_count", "Debug Sample Count", "FLOAT"),
        "alpha": make_output("alpha", "Alpha", "FLOAT"),
    }


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def load_exr_rgb(exr_path):
    """Загрузить EXR в numpy (H, W, 3) — только RGB, без alpha."""
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[:, :, :3]


def load_exr_scalar(exr_path):
    """Загрузить EXR в numpy (H, W) — скалярный (FLOAT) пасс, канал R."""
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[:, :, 0]


def render_frame_buffers(scene, dirs):
    """Отрендерить и вернуть все 6 буферов как numpy-массивы (H, W, ...)."""
    bpy.ops.render.render(write_still=False)

    position = load_exr_rgb(find_exr(dirs["position"]))
    normal = load_exr_rgb(find_exr(dirs["normal"]))
    albedo = load_exr_rgb(find_exr(dirs["diffuse_color"]))
    diffdir = load_exr_rgb(find_exr(dirs["diffuse_direct"]))
    diffind = load_exr_rgb(find_exr(dirs["diffuse_indirect"]))
    spp_norm = load_exr_scalar(find_exr(dirs["sample_count"]))
    alpha = load_exr_scalar(find_exr(dirs["alpha"]))

    irradiance = diffdir + diffind
    spp = spp_norm * SAMPLES_MAX  # Debug Sample Count нормирован на samples (см. прототип 4)

    return {
        "position": position,
        "normal": normal,
        "albedo": albedo,
        "irradiance": irradiance,
        "spp": spp,
        "valid": alpha > 0.5,  # маска: пиксель реально попал в геометрию сцены
    }


def render_diffuse_components(scene, dirs):
    """Отрендерить и вернуть diffuse_direct+indirect+alpha БЕЗ схлопывания
    в irradiance — используется дважды с разными настройками spp
    (render_true_irradiance ниже), позиция/normal/albedo не нужны
    повторно (геометрия и материал не зависят от spp)."""
    bpy.ops.render.render(write_still=False)
    diffdir = load_exr_rgb(find_exr(dirs["diffuse_direct"]))
    diffind = load_exr_rgb(find_exr(dirs["diffuse_indirect"]))
    alpha = load_exr_scalar(find_exr(dirs["alpha"]))
    return diffdir + diffind, alpha > 0.5


def render_true_irradiance(scene, dirs):
    """Пункт (d): high-spp эталон (REFERENCE_SPP, adaptive OFF — нужен
    контролируемый фиксированный spp, не эффективный adaptive-spp) для
    ТОГО ЖЕ t/света, что уже выставлен вызывающим кодом. Переключает
    cycles-настройки на high-spp, рендерит, и восстанавливает исходные
    (low/adaptive obs-) настройки перед возвратом — вызывающий код не
    должен заботиться о состоянии scene.cycles между вызовами.
    """
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.samples = REFERENCE_SPP
    scene.cycles.seed = 0
    true_irradiance, valid = render_diffuse_components(scene, dirs)

    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = SAMPLES_MAX
    scene.cycles.adaptive_min_samples = ADAPTIVE_MIN
    scene.cycles.adaptive_threshold = ADAPTIVE_THRESHOLD
    scene.cycles.seed = 42

    return true_irradiance, valid


def compute_visibility_matrix(scene, positions):
    """NxN булева матрица видимости между позициями проб.

    Техника из прототипа 3: луч с отступом VISIBILITY_EPS от истока,
    обрезанный чуть короче полного расстояния до цели, чтобы не
    засчитывать саму целевую поверхность как препятствие. Не зависит
    от освещения — считается один раз для геометрии сцены.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    n = len(positions)
    vis = np.zeros((n, n), dtype=bool)

    for i in range(n):
        origin = np.array(positions[i])
        for j in range(n):
            if i == j:
                vis[i, j] = True
                continue
            target = np.array(positions[j])
            d = target - origin
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                vis[i, j] = True
                continue
            direction = d / dist
            ray_origin = origin + direction * VISIBILITY_EPS
            max_dist = max(dist - 2 * VISIBILITY_EPS, 0.0)
            hit, *_ = scene.ray_cast(depsgraph, ray_origin.tolist(), direction.tolist(), distance=max_dist)
            vis[i, j] = not hit

    # РЕШЕНИЕ (2026-08-03): на холмистом полу (см. build_static_geometry) луч
    # с отступом VISIBILITY_EPS может задеть локальный изгиб поверхности у
    # одного конца отрезка и не задеть у другого -> vis[i,j] != vis[j,i], хотя
    # геометрическая видимость симметрична по определению. Это артефакт
    # эпсилон-отступа на кривизне, а не осмысленная асимметрия, поэтому
    # симметризуем через AND: взаимно видимы только если оба направления
    # согласны, что препятствий нет (строже, чем OR — не маскирует, а
    # выкидывает именно те grazing-случаи, где расходятся ответы).
    vis = vis & vis.T
    return vis


def extract_probes(buffers):
    """Свернуть grid-буферы (H, W, ...) в плоский список валидных проб."""
    valid = buffers["valid"]
    h, w = valid.shape
    probes = []
    for y in range(h):
        for x in range(w):
            if not valid[y, x]:
                continue
            probes.append({
                "pixel": (x, y),
                "position": buffers["position"][y, x].tolist(),
                "normal": buffers["normal"][y, x].tolist(),
                "albedo": buffers["albedo"][y, x].tolist(),
                "irradiance": buffers["irradiance"][y, x].tolist(),
                "spp": float(buffers["spp"][y, x]),
                "true_irradiance": buffers["true_irradiance"][y, x].tolist(),
            })
    return probes


def main():
    clear_all()
    scene = bpy.context.scene
    build_static_geometry()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = setup_render_and_passes(scene, OUT_DIR)

    # (a) Длинное расписание освещения + (b) плотный пул моментов на её
    # объединении — заменяет прежние 4 захардкоженных t. duration_override
    # позволяет прогнать короткий тайминг-тест перед полным батчем на
    # TOTAL_DURATION (см. вызов main() ниже).
    duration = float(sys.argv[sys.argv.index("--") + 1]) if "--" in sys.argv else TOTAL_DURATION
    light_schedule = build_light_schedule(LightScheduleConfig(total_duration=duration))
    t_values = build_moment_pool(MomentPoolConfig(total_duration=duration))
    print(f"Moment pool: {len(t_values)} t-значений на duration={duration}")

    frames = []
    visibility = None
    probe_positions = None
    expected_probe_count = None

    for idx, t in enumerate(t_values):
        light_pos = update_light(t, light_schedule)
        buffers = render_frame_buffers(scene, dirs)
        # (d) второй рендер ТОГО ЖЕ t/света на REFERENCE_SPP — эталон без
        # шума. valid-маска должна совпадать с обычным рендером (та же
        # геометрия/камера, отличается только spp), иначе где-то разошлось
        # предположение "valid не зависит от spp".
        true_irradiance, true_valid = render_true_irradiance(scene, dirs)
        assert np.array_equal(true_valid, buffers["valid"]), (
            f"valid mask changed between obs-render and true_irradiance-render "
            f"at t={t:.3f} — spp should not affect which pixels hit geometry"
        )
        buffers["true_irradiance"] = true_irradiance
        probes = extract_probes(buffers)

        if expected_probe_count is None:
            expected_probe_count = len(probes)
        else:
            # Гео и камера статичны между кадрами -> valid-маска обязана
            # совпадать. Если нет — где-то нарушено предположение
            # "visibility матрица считается один раз для всех кадров".
            assert len(probes) == expected_probe_count, (
                f"probe count changed between frames ({expected_probe_count} -> "
                f"{len(probes)}) — static-geometry assumption violated"
            )

        if visibility is None:
            probe_positions = [p["position"] for p in probes]
            visibility = compute_visibility_matrix(scene, probe_positions)

        dt = None if idx == 0 else t - t_values[idx - 1]
        frames.append({
            "t": t,
            "dt_from_previous": dt,
            "light_position": list(light_pos),
            "num_probes": len(probes),
            "probes": probes,
        })
        print(f"frame t={t:.3f} dt={dt} -> {len(probes)} valid probes")

    dataset = {
        "grid_res": GRID_RES,
        "num_probes": expected_probe_count,
        "visibility_matrix": visibility.tolist(),
        "probe_positions": probe_positions,
        "frames": frames,
    }

    out_path = Path(__file__).parent / "generate_dataset_result.json"
    out_path.write_text(json.dumps(dataset))

    print("Saved dataset to", out_path)
    print("Frames:", len(frames), "Probes/frame:", expected_probe_count)
    print("Visibility matrix shape:", visibility.shape)
    print(f"Visibility density (fraction of pairs mutually visible): "
          f"{float(visibility.mean()):.3f}")


if __name__ == "__main__":
    main()
