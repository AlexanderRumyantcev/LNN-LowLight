"""
Prototype §2.5 (TZ_stage1b_per_pixel_dense_fallback.md): disocclusion
detection на РЕАЛЬНЫХ Blender-буферах, а не синтетических.

Строит сцену: background-плоскость + occluder-куб, камера движется по
той же круговой траектории (camera_path.py, §2.1), что и в
07_motion_vector_prototype.py. Между t_prev=0 и t_curr=DT=8 (тот же
шаг, что уже провалидирован в §2.2) occluder должен закрывать одну
точку фона на t_prev и открывать её на t_curr (диссокклюзия) — и НЕ
закрывать другую, контрольную точку ни на t_prev, ни на t_curr
(стабильный, всегда видимый пиксель — негативный контроль).

Independent ground truth для выбора обеих точек — precomputed
raycast (тот же приём, что в 03_visibility_prototype.py): перебираем
кандидатов по Y на background-плоскости, для каждого проверяем
occluded_prev/occluded_curr через scene.ray_cast, ищем первый разрыв
(occluded_prev=True, occluded_curr=False) для disocc-точки и первую
пару (False, False) — для стабильной.

Дальше рендерим:
  - position+normal буферы на t_curr (текущий кадр) + vector pass
    (по рецепту 07: t_curr как "текущий" keyframe 1, t_future=2*DT
    как forward-helper keyframe 2, shutter=START)
  - position+normal буферы на t_prev отдельным статичным рендером
    (история, с которой сверяемся)

...и прогоняем реальные значения через blender/disocclusion.py
(reproject_and_check) — без переопределения логики внутри прототипа,
чтобы тест проверял именно тот код, который пойдёт в пайплайн.

Запуск:
  /Applications/Blender.app/Contents/MacOS/Blender --background \
      --python blender/08_disocclusion_prototype.py
"""
import bpy
import json
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).parent))
from camera_path import CameraPathConfig, camera_pose_at
from disocclusion import reproject_and_check

OUT_DIR = Path(__file__).parent / "_disocclusion_prototype_out"
RES = 64
SAMPLES = 32
DT = 8.0  # тот же шаг, что провалидирован в 07_motion_vector_prototype.py
EPS = 1e-4

OCCLUDER_CENTER = (1.0, 0.0, 1.5)
OCCLUDER_HALF = (0.75, 0.75, 0.75)
BACKGROUND_X = -3.0
BACKGROUND_Z = 1.5


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for ng in list(bpy.data.node_groups):
        bpy.data.node_groups.remove(ng)


def look_at_quat(position, target):
    direction = Vector(target) - Vector(position)
    return direction.to_track_quat("-Z", "Y")


def build_scene():
    bpy.ops.mesh.primitive_plane_add(
        size=8, location=(BACKGROUND_X, 0, BACKGROUND_Z),
        rotation=(0, np.deg2rad(90), 0),
    )
    background = bpy.context.active_object
    background.name = "Background"
    mat_bg = bpy.data.materials.new(name="BackgroundMat")
    background.data.materials.append(mat_bg)

    bpy.ops.mesh.primitive_cube_add(
        size=2 * OCCLUDER_HALF[0], location=OCCLUDER_CENTER,
    )
    occluder = bpy.context.active_object
    occluder.name = "Occluder"
    mat_occ = bpy.data.materials.new(name="OccluderMat")
    occluder.data.materials.append(mat_occ)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 6))
    light = bpy.context.active_object
    light.data.energy = 500.0
    light.data.size = 4.0

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = cam

    return background, occluder, cam


def ray_aabb_blocked(origin, target, box_center, box_half):
    """Slab-method ray-AABB, независимо от bpy (тот же метод, что в
    03_visibility_prototype.py) — используется только как sanity-
    проверка результата bpy raycast ниже, не как единственный
    источник истины (в отличие от 03, здесь сцена сложнее одного
    объекта, поэтому основной источник истины — сам bpy.ray_cast)."""
    d = [target[i] - origin[i] for i in range(3)]
    t_min, t_max = 0.0, 1.0
    for i in range(3):
        lo, hi = box_center[i] - box_half[i], box_center[i] + box_half[i]
        if abs(d[i]) < 1e-12:
            if origin[i] < lo or origin[i] > hi:
                return False
            continue
        t1, t2 = (lo - origin[i]) / d[i], (hi - origin[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min, t_max = max(t_min, t1), min(t_max, t2)
        if t_min > t_max:
            return False
    return t_min < 1.0 - 1e-6 and t_max > 1e-6 and t_min < t_max


def bpy_occluded(depsgraph, scene, origin, target):
    origin_v, target_v = Vector(origin), Vector(target)
    direction = target_v - origin_v
    dist = direction.length
    direction.normalize()
    ray_origin = origin_v + direction * EPS
    max_dist = max(dist - 2 * EPS, 0.0)
    result, *_ = scene.ray_cast(depsgraph, ray_origin, direction, distance=max_dist)
    return result


def well_framed(scene, cam, position, target, world_point, margin=0.15):
    """True если world_point проецируется заметно ВНУТРИ кадра (не
    у самого края) при камере в позе (position, target) — нужно,
    чтобы билинейная интерполяция буфера вокруг выбранного пикселя
    не задевала соседние "мимо-геометрии" пиксели (grazing angle у
    границы background-плоскости или края FOV камеры)."""
    scene.frame_set(1)
    cam.location = Vector(position)
    cam.rotation_quaternion = look_at_quat(position, target)
    bpy.context.view_layer.update()
    ndc = world_to_camera_view(scene, cam, Vector(world_point))
    return (
        margin <= ndc.x <= 1.0 - margin
        and margin <= ndc.y <= 1.0 - margin
        and ndc.z > 0.0
    )


def find_candidates(scene, cam, depsgraph, pos_prev, target_prev, pos_curr, target_curr):
    """Ищет по сетке Y на background-плоскости, от центра (y=0)
    наружу: первую точку с occluded_prev=True, occluded_curr=False
    (disocc) и первую с occluded_prev=False, occluded_curr=False
    (stable control) — обе дополнительно должны быть well_framed в
    ОБОИХ кадрах, чтобы избежать краевых артефактов интерполяции."""
    disocc_point, stable_point = None, None
    ys = sorted(np.arange(-3.0, 3.01, 0.1), key=abs)
    for y in ys:
        target = (BACKGROUND_X, float(y), BACKGROUND_Z)
        if not (
            well_framed(scene, cam, pos_prev, target_prev, target)
            and well_framed(scene, cam, pos_curr, target_curr, target)
        ):
            continue
        occ_prev = bpy_occluded(depsgraph, scene, pos_prev, target)
        occ_curr = bpy_occluded(depsgraph, scene, pos_curr, target)
        # независимая sanity-сверка ground truth для occluder-AABB
        gt_prev = ray_aabb_blocked(pos_prev, target, OCCLUDER_CENTER, OCCLUDER_HALF)
        gt_curr = ray_aabb_blocked(pos_curr, target, OCCLUDER_CENTER, OCCLUDER_HALF)
        assert occ_prev == gt_prev and occ_curr == gt_curr, (
            f"bpy raycast disagrees with independent AABB slab test at y={y}: "
            f"bpy=({occ_prev},{occ_curr}) aabb=({gt_prev},{gt_curr})"
        )
        if disocc_point is None and occ_prev and not occ_curr:
            disocc_point = target
        if stable_point is None and not occ_prev and not occ_curr:
            stable_point = target
        if disocc_point is not None and stable_point is not None:
            break
    return disocc_point, stable_point


def setup_render_common(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 42
    scene.cycles.use_denoising = False
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100


def render_position_normal(scene, cam, position, target, tag):
    """Статичный рендер position+normal буферов при заданной позе
    камеры (без анимации/vector pass) — используется для истории
    (t_prev) и как часть текущего кадра (t_curr)."""
    scene.frame_set(1)
    cam.location = Vector(position)
    cam.rotation_quaternion = look_at_quat(position, target)
    bpy.context.view_layer.update()

    vl = scene.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_normal = True

    tree = bpy.data.node_groups.new(f"Compositing_{tag}", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name

    pass_dir = OUT_DIR / tag
    pass_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for pass_name, socket in (("position", "Position"), ("normal", "Normal")):
        sub_dir = pass_dir / pass_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.directory = str(sub_dir)
        fo.file_name = "buf"
        fo.format.media_type = "IMAGE"
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.file_output_items.new("VECTOR", pass_name)
        tree.links.new(rl.outputs[socket], fo.inputs[0])
        paths[pass_name] = sub_dir

    bpy.ops.render.render(write_still=False)
    bpy.data.node_groups.remove(tree)

    result = {}
    for pass_name, sub_dir in paths.items():
        exr = list(sub_dir.glob("*.exr"))
        assert len(exr) == 1, f"expected 1 exr in {sub_dir}, found {exr}"
        img = bpy.data.images.load(str(exr[0]))
        w, h = img.size
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[:, :, :3]
        bpy.data.images.remove(img)
        result[pass_name] = arr
    return result


def render_current_with_vector(scene, cam, pos_curr, target_curr, pos_future, target_future):
    """t_curr буферы (position+normal+vector), тот же keyframe-рецепт,
    что в 07_motion_vector_prototype.py: keyframe1=t_curr (рендерим
    здесь), keyframe2=t_future — только чтобы Cycles посчитал forward
    motion, чьё (Z,W)-значение и есть искомый backward-вектор
    (см. disocclusion.warp_to_previous)."""
    scene.frame_set(1)
    cam.location = Vector(pos_curr)
    cam.rotation_quaternion = look_at_quat(pos_curr, target_curr)
    cam.keyframe_insert(data_path="location")
    cam.keyframe_insert(data_path="rotation_quaternion")

    scene.frame_set(2)
    cam.location = Vector(pos_future)
    cam.rotation_quaternion = look_at_quat(pos_future, target_future)
    cam.keyframe_insert(data_path="location")
    cam.keyframe_insert(data_path="rotation_quaternion")

    vl = scene.view_layers[0]
    vl.use_pass_position = True
    vl.use_pass_normal = True
    vl.use_pass_vector = True

    scene.render.use_motion_blur = False  # §2.7: несовместимо с vector pass
    scene.render.motion_blur_shutter = 1.0
    scene.render.motion_blur_position = "START"

    tree = bpy.data.node_groups.new("Compositing_curr", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name

    pass_dir = OUT_DIR / "curr"
    pass_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for pass_name, socket, item_type in (
        ("position", "Position", "VECTOR"),
        ("normal", "Normal", "VECTOR"),
        ("vector", "Vector", "RGBA"),
    ):
        sub_dir = pass_dir / pass_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.directory = str(sub_dir)
        fo.file_name = "buf"
        fo.format.media_type = "IMAGE"
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.file_output_items.new(item_type, pass_name)
        tree.links.new(rl.outputs[socket], fo.inputs[0])
        paths[pass_name] = sub_dir

    scene.frame_set(1)
    bpy.ops.render.render(write_still=False)
    bpy.data.node_groups.remove(tree)

    result = {}
    for pass_name, sub_dir in paths.items():
        exr = list(sub_dir.glob("*.exr"))
        assert len(exr) == 1, f"expected 1 exr in {sub_dir}, found {exr}"
        img = bpy.data.images.load(str(exr[0]))
        w, h = img.size
        n_ch = 4
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, n_ch)
        bpy.data.images.remove(img)
        result[pass_name] = arr[:, :, :3] if pass_name != "vector" else arr
    return result


def screen_pixel_for(scene, cam, position, target, world_point, res):
    scene.frame_set(1)
    cam.location = Vector(position)
    cam.rotation_quaternion = look_at_quat(position, target)
    bpy.context.view_layer.update()
    ndc = world_to_camera_view(scene, cam, Vector(world_point))
    return ndc.x * res, ndc.y * res


def main():
    clear_scene()
    background, occluder, cam = build_scene()
    scene = bpy.context.scene
    setup_render_common(scene)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    cfg = CameraPathConfig()
    pos_prev, target_prev = camera_pose_at(cfg, 0.0)
    pos_curr, target_curr = camera_pose_at(cfg, DT)
    pos_future, target_future = camera_pose_at(cfg, 2 * DT)

    disocc_point, stable_point = find_candidates(
        scene, cam, depsgraph,
        tuple(pos_prev), tuple(target_prev), tuple(pos_curr), tuple(target_curr),
    )
    assert disocc_point is not None, "no disocclusion candidate found in Y-sweep"
    assert stable_point is not None, "no stable control candidate found in Y-sweep"

    px_disocc_curr, py_disocc_curr = screen_pixel_for(
        scene, cam, pos_curr, target_curr, disocc_point, RES
    )
    px_stable_curr, py_stable_curr = screen_pixel_for(
        scene, cam, pos_curr, target_curr, stable_point, RES
    )

    buf_curr = render_current_with_vector(
        scene, cam, tuple(pos_curr), tuple(target_curr), tuple(pos_future), tuple(target_future)
    )
    buf_prev = render_position_normal(scene, cam, tuple(pos_prev), tuple(target_prev), "prev")

    from disocclusion import bilinear_sample

    def eval_point(px, py, label):
        pos_c, _ = bilinear_sample(buf_curr["position"], px, py)
        normal_c, _ = bilinear_sample(buf_curr["normal"], px, py)
        vec_zw, _ = bilinear_sample(buf_curr["vector"][:, :, 2:4], px, py)
        out = reproject_and_check(
            px, py, vec_zw, pos_c, normal_c,
            buf_prev["position"], buf_prev["normal"],
            camera_pos=pos_curr,
        )
        out["label"] = label
        out["screen_xy"] = (px, py)
        out["sampled_curr_position"] = list(pos_c)
        out["sampled_curr_normal"] = list(normal_c)
        out["motion_vector_zw"] = list(vec_zw)
        return out

    disocc_result = eval_point(px_disocc_curr, py_disocc_curr, "disocc")
    stable_result = eval_point(px_stable_curr, py_stable_curr, "stable")

    passed = (
        disocc_result["disocclusion_flag"] is True
        and stable_result["disocclusion_flag"] is False
    )

    result = {
        "dt": DT,
        "disocc_world_point": list(disocc_point),
        "stable_world_point": list(stable_point),
        "disocc_result": disocc_result,
        "stable_result": stable_result,
        "PASS": passed,
    }

    out_path = Path(__file__).parent / "08_disocclusion_prototype_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o))
    print("PROTOTYPE RESULT:", json.dumps(result, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else o))

    if not passed:
        print("FAIL", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
