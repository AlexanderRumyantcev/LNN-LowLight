"""
Генератор датасета §2.3/§3.2 TZ_stage1b_per_pixel_dense_fallback.md:
растровая (per-pixel dense) версия generate_dataset.py.

Ключевое структурное отличие от probe-пути (§2.3 ТЗ): не список из
GRID_RES**2 world-space проб с независимым асинхронным Δt на пробу
(moment_pool.py), а фиксированная экранная сетка GRID_RES×GRID_RES,
обновляемая РАЗОМ на каждом кадре с регулярным Δt — камера панорамирует
вокруг сцены (camera_path.py, §2.1), а не сидит неподвижно поверх
ортографической проекции пола.

Что переносится БЕЗ ИЗМЕНЕНИЙ (§3.1 ТЗ этапа 1b):
- Сама рендер-механика: один Cycles headless-проход даёт position/
  normal/albedo/irradiance/spp одновременно (что уже было в
  generate_dataset.py) — здесь просто ДОБАВЛЯЕТСЯ motion vector pass
  (§2.2) в тот же проход, не отдельным рендером.
- Двойной рендер per-кадр: обычный adaptive-spp (obs) + high-spp
  эталон (REFERENCE_SPP=2048, adaptive off) для true_irradiance —
  тот же принцип, что и в probe-пути.
- light_schedule.py (static/step/drift) — независимая, ортогональная
  ось нерегулярности освещения, движение камеры её НЕ заменяет.

Что убрано относительно generate_dataset.py:
- moment_pool.py (асинхронные per-пробу моменты) — заменён на
  простую регулярную сетку t_i = i * DT.
- compute_visibility_matrix / probe-based leave-one-out interpolation
  — не нужны per-pixel dense пути: сам warp по motion vector — это
  и есть механизм переноса истории между кадрами (§2.4), кросс-
  пиксельная visibility-интерполяция проб здесь не при чём.
- Fixed ortho top-down камера — заменена на перспективную камеру по
  траектории camera_path.py (без движения камеры дисокклюзии
  неоткуда взяться, см. §1.3 ТЗ).

Формат вывода (МОЯ инженерная реализация формата хранения, ТЗ §3.2
называет только СОСТАВ буферов на кадр, не формат файла): единый
.npz со стеком по оси кадров (frames, H, W, ...) для каждого буфера
— JSON, как в probe-пути, для full-frame растровых данных на порядок
менее подходящий формат (см. предупреждение об объёме в §3.3 ТЗ).
disocclusion_flag/geometric_mismatch_score (§2.5) сюда НЕ входят —
это задача адаптера, потребляющего пары соседних кадров (см.
blender/disocclusion.py), не самого генератора буферов.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/generate_dataset_dense.py -- [duration]
"""
import bpy
import json
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from light_schedule import LightScheduleConfig, build_light_schedule, intensity_at
from camera_path import CameraPathConfig, camera_pose_at

OUT_DIR = Path(__file__).parent / "_dataset_dense_out"
TOTAL_DURATION = 400.0
GRID_RES = 8  # см. §3.3 ТЗ: старт с малого разрешения, как и пробы
# DT=8: тот же шаг, что уже провалидирован в 07_motion_vector_prototype.py
# и 08_disocclusion_prototype.py (rel_error 2.47% на векторе, чёткое
# разделение disocc/stable geometric_mismatch). С n_revolutions=2.0 по
# умолчанию (camera_path.CameraPathConfig) это даёт целое число кадров
# на полный оборот — не откалибровано на частоту дисокклюзий по всему
# датасету (§3.4 ТЗ), только унаследовано из уже провалидированного
# прототипа как стартовая точка.
DT = 8.0
SAMPLES_MAX = 128
ADAPTIVE_MIN = 4
ADAPTIVE_THRESHOLD = 0.01
REFERENCE_SPP = 2048


def clear_all():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def look_at_quat(position, target):
    direction = Vector(target) - Vector(position)
    return direction.to_track_quat("-Z", "Y")


def build_static_geometry():
    """Пол (с холмистым рельефом, то же решение 2026-08-03, что и в
    generate_dataset.py — иначе normal вырождается в одну константу)
    + occluder + перспективная камера (без анимации, поза выставляется
    per-frame в main()). Без фиксированной ортографической камеры —
    в отличие от probe-пути, дисокклюзия здесь — рабочий сигнал, не
    артефакт, который нужно избегать.
    """
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
            0.35 * np.sin(0.8 * x + 0.3) * np.cos(0.6 * y - 0.5)
            + 0.15 * np.sin(1.7 * x - y)
        )
    floor.data.update()
    bpy.ops.object.shade_smooth()

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.5, 1.0, 0.75))
    occluder = bpy.context.active_object
    occluder.name = "Occluder"
    occluder.scale = (1.0, 1.0, 1.5)
    occluder.data.materials.append(mat)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = cam

    return floor, occluder, cam


def update_light(t, light_schedule):
    old = bpy.data.objects.get("Light")
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)

    # Та же орбитальная траектория света, что в generate_dataset.py —
    # независимая от движения камеры ось (§3.1 ТЗ), не трогается.
    radius, height = 2.5, 3.0
    angle = t
    pos = (radius * np.cos(angle), radius * np.sin(angle), height)

    bpy.ops.object.light_add(type="AREA", location=pos)
    light = bpy.context.active_object
    light.name = "Light"
    light.data.energy = intensity_at(light_schedule, t)
    light.data.size = 0.3
    direction = -1 * light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return pos


def setup_render_common(scene):
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
    scene.render.film_transparent = True

    # §2.7: vector pass несовместим с motion_blur включённым -
    # use_motion_blur=False, а окно смещения задаётся отдельными
    # motion_blur_shutter/position (провалидировано в 07/08).
    scene.render.use_motion_blur = False
    scene.render.motion_blur_shutter = 1.0
    scene.render.motion_blur_position = "START"


def make_outputs(scene, out_dir):
    vl = scene.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_normal = True
    vl.use_pass_diffuse_color = True
    vl.use_pass_diffuse_direct = True
    vl.use_pass_diffuse_indirect = True
    vl.cycles.pass_debug_sample_count = True
    vl.use_pass_vector = True

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

    return tree, {
        "position": make_output("position", "Position", "VECTOR"),
        "normal": make_output("normal", "Normal", "VECTOR"),
        "diffuse_color": make_output("diffuse_color", "Diffuse Color", "RGBA"),
        "diffuse_direct": make_output("diffuse_direct", "Diffuse Direct", "RGBA"),
        "diffuse_indirect": make_output("diffuse_indirect", "Diffuse Indirect", "RGBA"),
        "sample_count": make_output("sample_count", "Debug Sample Count", "FLOAT"),
        "alpha": make_output("alpha", "Alpha", "FLOAT"),
        "vector": make_output("vector", "Vector", "RGBA"),
    }


def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def load_exr(exr_path, n_channels):
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    return px[:, :, :n_channels]


def render_obs_buffers(scene, dirs):
    """Один рендер даёт position/normal/albedo/irradiance(obs)/spp/
    valid/motion_vector разом — камера уже выставлена вызывающим
    кодом (keyframes frame=1 текущая поза, frame=2 будущая, для
    vector pass), scene.frame_set(1) для рендера самого кадра t."""
    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)

    position = load_exr(find_exr(dirs["position"]), 3)
    normal = load_exr(find_exr(dirs["normal"]), 3)
    albedo = load_exr(find_exr(dirs["diffuse_color"]), 3)
    diffdir = load_exr(find_exr(dirs["diffuse_direct"]), 3)
    diffind = load_exr(find_exr(dirs["diffuse_indirect"]), 3)
    spp_norm = load_exr(find_exr(dirs["sample_count"]), 1)[:, :, 0]
    alpha = load_exr(find_exr(dirs["alpha"]), 1)[:, :, 0]
    vector_full = load_exr(find_exr(dirs["vector"]), 4)
    # Только Z,W — эмпирически подтверждённые каналы backward-вектора
    # (§2.2, 07_motion_vector_prototype.py), не R,G.
    motion_vector = vector_full[:, :, 2:4]

    return {
        "position": position,
        "normal": normal,
        "albedo": albedo,
        "irradiance": diffdir + diffind,
        "spp": spp_norm * SAMPLES_MAX,
        "valid": alpha > 0.5,
        "motion_vector": motion_vector,
    }


def render_true_irradiance(scene, dirs):
    """Тот же принцип, что в generate_dataset.py: high-spp эталон БЕЗ
    повторного рендера position/normal/albedo/vector (не зависят от
    spp) — только diffuse_direct/indirect+alpha на REFERENCE_SPP."""
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.samples = REFERENCE_SPP
    scene.cycles.seed = 0

    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)
    diffdir = load_exr(find_exr(dirs["diffuse_direct"]), 3)
    diffind = load_exr(find_exr(dirs["diffuse_indirect"]), 3)
    alpha = load_exr(find_exr(dirs["alpha"]), 1)[:, :, 0]

    scene.cycles.use_adaptive_sampling = True
    scene.cycles.samples = SAMPLES_MAX
    scene.cycles.adaptive_min_samples = ADAPTIVE_MIN
    scene.cycles.adaptive_threshold = ADAPTIVE_THRESHOLD
    scene.cycles.seed = 42

    return diffdir + diffind, alpha > 0.5


def set_camera_keyframes(cam, pos_curr, target_curr, pos_next, target_next):
    """Keyframe1=текущая поза кадра t_i (что и рендерим), keyframe2=
    поза t_{i+1} — forward-helper для vector pass (§2.2 рецепт,
    07_motion_vector_prototype.py). На регулярной сетке t_{i+1} — это
    просто СЛЕДУЮЩИЙ кадр датасета, отдельный "выброс" кадр не нужен,
    кроме как для последнего кадра последовательности (см. main())."""
    cam.animation_data_clear()

    cam.location = Vector(pos_curr)
    cam.rotation_quaternion = look_at_quat(pos_curr, target_curr)
    cam.keyframe_insert(data_path="location", frame=1)
    cam.keyframe_insert(data_path="rotation_quaternion", frame=1)

    cam.location = Vector(pos_next)
    cam.rotation_quaternion = look_at_quat(pos_next, target_next)
    cam.keyframe_insert(data_path="location", frame=2)
    cam.keyframe_insert(data_path="rotation_quaternion", frame=2)


def main():
    global GRID_RES
    cli_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    duration = float(cli_args[0]) if len(cli_args) >= 1 else TOTAL_DURATION
    # ВАЖНО: без тега все прогоны (в т.ч. короткие smoke-тесты из
    # pytest) пишут в ОДИН И ТОТ ЖЕ файл и молча затирают друг друга —
    # найдено на практике 2026-08-09: прогон tests/ поверх "боевого"
    # duration=400 датасета незаметно подменил его смоук-версией
    # (duration=24), из-за чего первый анализ частоты дисокклюзии на
    # "generate_dataset_dense_result.npz" оказался посчитан не на том
    # датасете. tag по умолчанию пуст (старое имя файла, обратная
    # совместимость с уже написанными тестами/анализом).
    tag = cli_args[1] if len(cli_args) >= 2 else ""
    suffix = f"_{tag}" if tag else ""
    # Третий аргумент — GRID_RES override (2026-08-09): проверка гипотезы,
    # что низкое разрешение сетки (мало обучающих пикселей + завышенная
    # частота дисокклюзии из-за большого footprint пикселя) искажает
    # сравнение CfC-B vs честные baseline'ы в run_experiment_dense.py —
    # см. mempalace. По умолчанию — старое значение (обратная совместимость).
    if len(cli_args) >= 3:
        GRID_RES = int(cli_args[2])

    clear_all()
    scene = bpy.context.scene
    build_static_geometry()
    cam = bpy.context.scene.camera
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_render_common(scene)
    tree, dirs = make_outputs(scene, OUT_DIR)

    light_schedule = build_light_schedule(LightScheduleConfig(total_duration=duration))
    cam_cfg = CameraPathConfig(total_duration=duration)

    n_frames = int(duration / DT) + 1
    t_values = [i * DT for i in range(n_frames)]
    print(f"Dense grid: {n_frames} кадров, DT={DT}, duration={duration}, GRID_RES={GRID_RES}")


    buffers_by_key = {
        k: [] for k in (
            "position", "normal", "albedo", "irradiance", "true_irradiance",
            "spp", "valid", "motion_vector",
        )
    }
    camera_positions = []
    light_positions = []
    total_edge_mismatch_px = 0

    for idx, t in enumerate(t_values):
        pos_curr, target_curr = camera_pose_at(cam_cfg, t)
        # Для последнего кадра последовательности нет t_{i+1} в самом
        # датасете — берём его ТОЛЬКО как forward-helper для vector
        # pass (не сохраняем как отдельный кадр), тем же принципом,
        # что t_future в 07/08 прототипах.
        t_next = t_values[idx + 1] if idx + 1 < n_frames else t + DT
        pos_next, target_next = camera_pose_at(cam_cfg, t_next)

        set_camera_keyframes(cam, tuple(pos_curr), tuple(target_curr), tuple(pos_next), tuple(target_next))
        light_pos = update_light(t, light_schedule)

        obs = render_obs_buffers(scene, dirs)
        true_irr, true_valid = render_true_irradiance(scene, dirs)
        # РЕШЕНИЕ (2026-08-09, при первом полном прогоне): при движущейся
        # перспективной камере (в отличие от прежней фиксированной ortho
        # top-down в generate_dataset.py) заметная доля пикселей на
        # GRID_RES=8 приходится на силуэтный край геометрии, где alpha —
        # сглаженная antialiasing-величина, не бинарная, и колеблется
        # вокруг порога 0.5 по-разному в зависимости от числа сэмплов
        # (обычный adaptive-рендер vs REFERENCE_SPP=2048) — это реальный
        # эффект антиалиасинга на низком разрешении, не баг геометрии.
        # Тот же принцип, что уже применён для симметризации visibility
        # в generate_dataset.py (AND вместо OR): валиден только пиксель,
        # однозначно валидный в ОБОИХ рендерах — не ослабление проверки,
        # а более строгий фильтр, выкидывающий именно эти edge-случаи.
        mismatch = obs["valid"] != true_valid
        if mismatch.any():
            total_edge_mismatch_px += int(mismatch.sum())
            print(f"  [edge-antialiasing] {int(mismatch.sum())} px valid-mismatch "
                  f"at t={t:.3f} (obs vs true_irradiance alpha) — excluded via AND")
        combined_valid = obs["valid"] & true_valid
        obs["valid"] = combined_valid

        for k in buffers_by_key:
            if k == "true_irradiance":
                buffers_by_key[k].append(true_irr)
            else:
                buffers_by_key[k].append(obs[k])
        camera_positions.append(list(pos_curr))
        light_positions.append(list(light_pos))
        print(f"frame {idx} t={t:.3f} cam={np.round(pos_curr, 2)} "
              f"valid_px={int(obs['valid'].sum())}/{GRID_RES * GRID_RES}")

    bpy.data.node_groups.remove(tree)

    stacked = {k: np.stack(v, axis=0) for k, v in buffers_by_key.items()}
    npz_path = Path(__file__).parent / f"generate_dataset_dense_result{suffix}.npz"
    np.savez_compressed(npz_path, **stacked,
                         camera_positions=np.array(camera_positions),
                         t_values=np.array(t_values))

    meta = {
        "n_frames": n_frames,
        "dt": DT,
        "grid_res": GRID_RES,
        "total_duration": duration,
        "samples_max": SAMPLES_MAX,
        "reference_spp": REFERENCE_SPP,
        "camera_config": {
            "n_revolutions": cam_cfg.n_revolutions,
            "radius": cam_cfg.radius,
            "height": cam_cfg.height,
            "center": list(cam_cfg.center),
        },
        "buffer_shapes": {k: list(v.shape) for k, v in stacked.items()},
        "total_edge_mismatch_px": total_edge_mismatch_px,
        "npz_path": str(npz_path),
    }
    meta_path = Path(__file__).parent / f"generate_dataset_dense_result_meta{suffix}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("Saved dense dataset to", npz_path)
    print("Metadata:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
