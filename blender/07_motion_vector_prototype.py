"""
Prototype §2.2 (TZ_stage1b_per_pixel_dense_fallback.md): motion vector
render-pass для camera-driven reprojection.

ЦЕЛЬ: эмпирически определить формат/семантику Cycles Vector pass в
Blender 5.2 при ДВИЖЕНИИ КАМЕРЫ (не объекта) между двумя кадрами —
по методологии проекта (§3.1 основного ТЗ: прототип-скрипт, эмпирика,
затем независимая проверка), а не предполагать формат заранее, т.к.
точная семантика каналов (forward/backward, единицы измерения) не
задокументирована однозначно для текущей версии Blender.

Сцена: статичная плоскость в начале координат, камера движется между
двумя позициями, взятыми из camera_path.camera_pose_at (тот же модуль,
что и в production-коде §2.1) — так прототип сразу проверяет реальную
траекторию, а не произвольную тестовую.

Известный нюанс (см. §2.7 ТЗ, источник: community-рецепты для Cycles
vector pass): Vector pass несовместим с scene.render.use_motion_blur
= True — здесь явно выставляется False.

ЭМПИРИЧЕСКАЯ НАХОДКА (первая итерация теста): при рендере на
frame=1 с scene.render.motion_blur_shutter=1.0,
motion_blur_position="START" (не относится к use_motion_blur, тот
остаётся False) — содержательные данные лежат в каналах Z,W вектор-
пасса (не R,G, которые нулевые при отсутствии "предыдущего" кадра
до frame=1), и (Z,W) численно совпадает (в пределах субпиксельной
погрешности) с ОБРАТНЫМ (frame1→frame2) экранным смещением, т.е.
уже готовый "backward"-вектор для формулы reprojection §2.4:
(x_prev, y_prev) = (x, y) + (vector.z, vector.w), без ручной
инверсии знака. Контрольная точка теста намеренно НЕ центр сцены
(= look-at target камеры) — его экранная позиция по конструкции не
меняется между кадрами независимо от движения камеры, что сделало
бы проверку тривиальной.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/07_motion_vector_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path

from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).parent))
from camera_path import CameraPathConfig, camera_pose_at

OUT_DIR = Path(__file__).parent / "_motion_vector_prototype_out"
RES = 64
SAMPLES = 32
# Небольшой шаг по t между двумя кадрами — не весь диапазон панорамы,
# просто чтобы получить измеримое, но не экстремальное смещение.
DT = 8.0

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def look_at_quat(position: Vector, target: Vector):
    """Кватернион поворота камеры: локальная -Z смотрит на target, +Y — вверх.
    Стандартный Blender-приём без constraint-объекта (mathutils.Vector.to_track_quat),
    что позволяет напрямую keyframe'ить rotation_quaternion на двух кадрах."""
    direction = target - position
    return direction.to_track_quat("-Z", "Y")


def build_scene():
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    receiver = bpy.context.active_object
    receiver.name = "Receiver"
    mat = bpy.data.materials.new(name="ReceiverMat")
    receiver.data.materials.append(mat)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4))
    light = bpy.context.active_object
    light.data.energy = 300.0

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = cam
    return receiver, cam

def animate_camera(cam):
    """Камера на двух кадрах — позиции из camera_path.camera_pose_at (§2.1),
    т.е. та же траектория, что и в production-генераторе, не искусственная."""
    cfg = CameraPathConfig()
    pos1, target1 = camera_pose_at(cfg, 0.0)
    pos2, target2 = camera_pose_at(cfg, DT)

    scene = bpy.context.scene
    scene.frame_set(1)
    cam.location = Vector(pos1)
    cam.rotation_quaternion = look_at_quat(Vector(pos1), Vector(target1))
    cam.keyframe_insert(data_path="location")
    cam.keyframe_insert(data_path="rotation_quaternion")

    scene.frame_set(2)
    cam.location = Vector(pos2)
    cam.rotation_quaternion = look_at_quat(Vector(pos2), Vector(target2))
    cam.keyframe_insert(data_path="location")
    cam.keyframe_insert(data_path="rotation_quaternion")

    return pos1, target1, pos2, target2

def setup_render_and_passes(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False
    # Известный нюанс community-рецептов (§2.7 ТЗ): Vector pass
    # несовместим с motion blur рендера — выключаем явно.
    scene.render.use_motion_blur = False
    # Shutter-окно контролирует, какой временной интервал Cycles
    # сэмплирует для Vector pass (даже при use_motion_blur=False —
    # экспериментально проверяем эту гипотезу, т.к. семантика не
    # задокументирована однозначно для 5.2, см. docstring модуля).
    scene.render.motion_blur_shutter = 1.0
    scene.render.motion_blur_position = "START"

    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100

    vl = scene.view_layers[0]
    vl.use_pass_vector = True

    tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name
    print("DEBUG rl outputs:", [(o.name, o.type) for o in rl.outputs])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pass_dir = OUT_DIR / "vector"
    pass_dir.mkdir(parents=True, exist_ok=True)

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(pass_dir)
    fo.file_name = "buf"
    fo.format.media_type = "IMAGE"
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_output_items.new("RGBA", "vector")
    tree.links.new(rl.outputs["Vector"], fo.inputs[0])

    return pass_dir

def find_exr(pass_dir):
    matches = list(Path(pass_dir).glob("*.exr"))
    assert len(matches) == 1, f"expected exactly one .exr in {pass_dir}, found {matches}"
    return matches[0]


def sample_pixel(exr_path, px, py):
    """Читает RGBA в заданном пикселе (px, py — целые пиксельные координаты,
    (0,0) = нижний левый угол, как в bpy.data.images)."""
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    data = img.pixels[:]
    idx = (py * w + px) * 4
    vals = tuple(data[idx:idx + 4])
    bpy.data.images.remove(img)
    return vals


def screen_pixel_at_frame(scene, cam, frame, world_point, res):
    """world_to_camera_view для world_point при камере в состоянии на frame.
    Возвращает пиксельные координаты (px, py), (0,0) = нижний левый угол,
    и признак того, что точка вообще видна (внутри [0,1]x[0,1], z>0)."""
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    ndc = world_to_camera_view(scene, cam, Vector(world_point))
    px = ndc.x * res
    py = ndc.y * res
    visible = (0.0 <= ndc.x <= 1.0) and (0.0 <= ndc.y <= 1.0) and (ndc.z > 0.0)
    return px, py, visible

def main():
    clear_scene()
    receiver, cam = build_scene()
    pos1, target1, pos2, target2 = animate_camera(cam)
    scene = bpy.context.scene
    pass_dir = setup_render_and_passes(scene)

    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)
    exr_path = find_exr(pass_dir)

    # НЕ центр плоскости (= look-at target, экранная позиция которого
    # по конструкции не меняется, т.к. камера всегда на него смотрит) —
    # берём угол плоскости, чтобы получить содержательное ожидаемое
    # смещение.
    world_point = (1.5, 1.5, 0.0)
    px2, py2, vis2 = screen_pixel_at_frame(scene, cam, 2, world_point, RES)
    px1, py1, vis1 = screen_pixel_at_frame(scene, cam, 1, world_point, RES)
    expected_disp = (px2 - px1, py2 - py1)

    sample_px, sample_py = int(round(px1)), int(round(py1))
    sample_px = max(0, min(RES - 1, sample_px))
    sample_py = max(0, min(RES - 1, sample_py))
    raw_vector = sample_pixel(exr_path, sample_px, sample_py)

    # ЭМПИРИЧЕСКАЯ НАХОДКА (см. docstring): каналы Z,W (не R,G) несут
    # содержательные данные при рендере на frame=1 с shutter=START,
    # покрывающим окно [1,2] — и (Z,W) численно совпадает с
    # -(expected_disp) (обратный знак), т.е. это уже готовый
    # "backward"-вектор в смысле §2.4 ТЗ: (x_prev,y_prev) = (x,y) +
    # (vector.z, vector.w) напрямую, без ручной инверсии знака.
    backward_vec = (raw_vector[2], raw_vector[3])
    warp_error = (
        (backward_vec[0] - (-expected_disp[0])) ** 2
        + (backward_vec[1] - (-expected_disp[1])) ** 2
    ) ** 0.5
    expected_mag = (expected_disp[0] ** 2 + expected_disp[1] ** 2) ** 0.5
    rel_error = warp_error / expected_mag if expected_mag > 1e-6 else float("inf")

    result = {
        "pos1": [float(x) for x in pos1], "pos2": [float(x) for x in pos2],
        "target1": [float(x) for x in target1], "target2": [float(x) for x in target2],
        "screen_pos_frame1_px": [px1, py1],
        "screen_pos_frame2_px": [px2, py2],
        "expected_pixel_displacement_frame1_to_2": list(expected_disp),
        "world_point_visible_frame1": vis1,
        "world_point_visible_frame2": vis2,
        "sample_pixel": [sample_px, sample_py],
        "raw_vector_rgba_at_sample_pixel": list(raw_vector),
        "backward_vec_zw": list(backward_vec),
        "rel_error_vs_expected_backward_disp": rel_error,
        "resolution": RES,
        "PASS": rel_error < 0.1,
    }

    out_path = Path(__file__).parent / "07_motion_vector_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("PROTOTYPE RESULT:")
    print(json.dumps(result, indent=2))

    if not result["PASS"]:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
