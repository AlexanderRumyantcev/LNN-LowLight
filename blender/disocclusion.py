"""
Warp/reprojection + disocclusion detection — §2.4/§2.5
TZ_stage1b_per_pixel_dense_fallback.md.

Чистый numpy, без bpy (как camera_path.py/light_schedule.py) —
работает с уже отрендеренными position/normal/motion-vector буферами,
независимо от их источника (Blender EXR в 08_disocclusion_prototype.py
или синтетические буферы в tests/test_disocclusion.py).

Per §2.7 ТЗ: сама механика reprojection/disocclusion — общее место в
экранных денойзерах (NRD/TAA/SVGF), не специфика проекта. Порог и
структура confidence-сигнала здесь сверены с подходом открытого NRD
(github.com/NVIDIA-RTX/NRD, Shaders/Include/Common.hlsli,
GetDisocclusionThreshold/GetEncodingAwareNormalWeight):
  - NRD детектирует диссокклюзию через depth/plane-distance comparison
    + normal vector similarity + material ID + frustum checks; порог
    масштабируется как frustumSize * disocclusionThresholdBase /
    max(0.05, NoV) — т.е. НЕ фиксированное абсолютное число, а
    величина, зависящая от того, насколько "далеко" точка от камеры
    (через viewZ). Здесь, за неимением view-space depth (пайплайн
    хранит world-space position, не viewZ), используется тот же
    принцип через более простой прокси — расстояние world-space точки
    до камеры (distance_scale ниже).
  - normal mismatch у NRD — угол между нормалями (acos от dot),
    сглаженный smoothstep, не покомпонентная разница — здесь то же.
  - confidence — непрерывная величина (geometric_mismatch_score),
    аналог IN_DIFF_CONFIDENCE у NRD (§2.7 ТЗ прямо называет
    geometric_mismatch_score его концептуальным аналогом), а не
    только бинарный флаг.

NRD-специфичные вещи, которые НЕ портируются (другой стек — GPU/HLSL
у NRD против Python/PyTorch здесь, см. §2.7 ТЗ): material ID
matching, frustum boundary checks в clip-space, PSR (primary surface
replacement) — не применимы к текущему пайплайну world-space проб.

Пороги (distance_scale, normal_threshold_deg, mismatch_threshold) не
откалиброваны против реального распределения диссокклюзий в проекте
(тот же статус, что и disocclusionThreshold у самого NRD — калибруемый
параметр, не константа из первых принципов) — подлежат эмпирической
калибровке по факту первого прогона на реальном датасете, тем же
принципом, что n_revolutions в camera_path.py и dt_spike_prob в
spike-test (см. §2.1 ТЗ, §3.4 ТЗ).

distance_scale=0.2 — грубая первая прикидка по одному прогону
08_disocclusion_prototype.py (DT=8): валидный (не-диссоклюдированный)
геометрический разброс между текущим и warped-прошлым положением
оказался ~0.7 world-units на дистанции камеры ~8.5 (т.е. сама
экстраполяция motion vector §2.2 вносит эту ошибку, а не только
диссокклюзия) — что уже само по себе значимая доля от
distance_scale=0.05 (порог 0.43), которого не хватало на реальных
данных. Значение 0.2 (порог ~1.7) подобрано так, чтобы отделить этот
"нормальный" разброс warp-экстраполяции от заведомой диссокклюзии
(~4.8 в том же прогоне) — НЕ откалибровано на статистике по многим
кадрам/сценам.
"""
import numpy as np


def bilinear_sample(buffer, x, y):
    """Билинейно сэмплировать buffer (H, W, C) в дробных координатах
    (x, y); (0, 0) — нижний левый угол пикселя, та же конвенция, что
    и в остальных буферах проекта (см. sample_pixel в
    07_motion_vector_prototype.py).

    Возвращает (values, in_bounds). in_bounds=False — это и есть
    §2.5 out-of-bounds признак; values в этом случае — нули, не
    предназначены к использованию (см. is_disoccluded).
    """
    h, w = buffer.shape[0], buffer.shape[1]
    if x < 0.0 or y < 0.0 or x > w - 1 or y > h - 1:
        return np.zeros(buffer.shape[2:], dtype=float), False

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0

    v00 = buffer[y0, x0].astype(float)
    v10 = buffer[y0, x1].astype(float)
    v01 = buffer[y1, x0].astype(float)
    v11 = buffer[y1, x1].astype(float)
    top = v00 * (1 - fx) + v10 * fx
    bottom = v01 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bottom * fy, True


def warp_to_previous(x, y, motion_vector_zw):
    """§2.4 п.1: (x', y') — куда попадает содержимое пикселя (x, y)
    текущего кадра в кадре t-Δt.

    motion_vector_zw — уже сэмплированные (Z, W) каналы Cycles Vector
    pass в (x, y) (не R, G — см. 07_motion_vector_prototype.py).
    Эмпирически подтверждено (§2.2, 07_motion_vector_prototype.py,
    rel_error 2.47%), что эти каналы уже дают готовый backward-вектор:
    прибавляются напрямую, без ручной инверсии знака — в отличие от
    буквальной формулы §2.4 ("(x',y') = (x,y) - motion_vector"),
    которая описывает семантику результата, а не то, как именно
    инвертировать конкретно ЭТОТ канал Blender.
    """
    return x + motion_vector_zw[0], y + motion_vector_zw[1]


def geometric_mismatch_score(pos_curr, normal_curr, pos_prev, normal_prev,
                              camera_pos,
                              distance_scale=0.2,
                              normal_threshold_deg=45.0):
    """§2.5 geometric mismatch между точкой (x, y) кадра t и
    warped-семплом из (x', y') кадра t-Δt.

    Возвращает непрерывный score в [0, 1] (0 = полное совпадение,
    1 = полное расхождение по любому из двух независимых признаков)
    — аналог IN_DIFF_CONFIDENCE у NRD, см. docstring модуля.
    """
    pos_curr = np.asarray(pos_curr, dtype=float)
    pos_prev = np.asarray(pos_prev, dtype=float)
    camera_pos = np.asarray(camera_pos, dtype=float)

    dist_to_camera = np.linalg.norm(pos_curr - camera_pos)
    pos_threshold = max(distance_scale * dist_to_camera, 1e-4)
    pos_diff = np.linalg.norm(pos_curr - pos_prev)
    pos_score = float(np.clip(pos_diff / pos_threshold, 0.0, 1.0))

    nc = np.asarray(normal_curr, dtype=float)
    np_ = np.asarray(normal_prev, dtype=float)
    nc = nc / (np.linalg.norm(nc) + 1e-8)
    np_ = np_ / (np.linalg.norm(np_) + 1e-8)
    cos_angle = float(np.clip(np.dot(nc, np_), -1.0, 1.0))
    angle_deg = np.degrees(np.arccos(cos_angle))
    normal_score = float(np.clip(angle_deg / normal_threshold_deg, 0.0, 1.0))

    return max(pos_score, normal_score)


def is_disoccluded(in_bounds, mismatch_score, mismatch_threshold=0.5):
    """§2.5 итоговый бинарный disocclusion_flag для staleness-вектора
    (§2.6): out-of-bounds ИЛИ geometric_mismatch_score выше порога.
    """
    if not in_bounds:
        return True
    return mismatch_score > mismatch_threshold


def reproject_and_check(x, y, motion_vector_zw, pos_curr, normal_curr,
                         prev_position_buffer, prev_normal_buffer,
                         camera_pos,
                         distance_scale=0.2, normal_threshold_deg=45.0,
                         mismatch_threshold=0.5):
    """Полный конвейер §2.4+§2.5 для одного пикселя: warp -> сэмплинг
    истории -> geometric mismatch -> disocclusion_flag.

    Возвращает dict с промежуточными значениями (для диагностики в
    прототипе/тестах) и итоговыми disocclusion_flag,
    geometric_mismatch_score (§2.6 staleness-вектор).
    """
    x_prev, y_prev = warp_to_previous(x, y, motion_vector_zw)
    pos_prev, pos_in_bounds = bilinear_sample(prev_position_buffer, x_prev, y_prev)
    normal_prev, normal_in_bounds = bilinear_sample(prev_normal_buffer, x_prev, y_prev)
    in_bounds = pos_in_bounds and normal_in_bounds

    if in_bounds:
        score = geometric_mismatch_score(
            pos_curr, normal_curr, pos_prev, normal_prev, camera_pos,
            distance_scale=distance_scale,
            normal_threshold_deg=normal_threshold_deg,
        )
    else:
        score = 1.0  # out-of-bounds -> максимальное расхождение по построению

    flag = is_disoccluded(in_bounds, score, mismatch_threshold=mismatch_threshold)

    return {
        "warped_xy": (x_prev, y_prev),
        "in_bounds": in_bounds,
        "sampled_prev_position": list(pos_prev),
        "sampled_prev_normal": list(normal_prev),
        "geometric_mismatch_score": score,
        "disocclusion_flag": flag,
    }
