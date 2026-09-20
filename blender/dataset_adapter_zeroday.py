"""
Адаптер Zero Day dense-датасета (generate_dataset_zeroday.py) в тот же
формат per-pixel последовательностей, что build_all_pixel_sequences
(blender/dataset_adapter_dense.py) — drop-in замена для
load_fully_valid_pixel_sequences (run_experiment_dense.py) и всего, что
на неё опирается (в т.ч. run_hybrid_v0.py-подобные скрипты).

05.09.2026, чат, TZ_stage5 §5 (инференс hybrid_v0 на nostep_v1.npz).

ПЕРЕИСПОЛЬЗУЕТСЯ БЕЗ ИЗМЕНЕНИЙ: blender/disocclusion.py — сама
математика (warp_to_previous, geometric_mismatch_score, пороги
 distance_scale=0.2/normal_threshold_deg=45.0/mismatch_threshold=0.5,
включая их некалиброванный статус, см. докстринг disocclusion.py) не
меняется вообще. Единственное отличие — здесь она ВЕКТОРИЗОВАНА по
всей сетке H×W сразу за один кадровый переход, а не вызывается в python-цикле
по одному пикселю (как в dataset_adapter_dense.py).

ПОЧЕМУ векторизация обязательна именно здесь (МОЁ инженерное решение,
не из ТЗ): dt1.npz — сетка 16×16=256 пикселей, dataset_adapter_dense.py
использует чистый python-цикл (256 пикселей × 400 шагов = 102400
вызовов reproject_and_check — секунды). Zero Day — 320×180=57600
пикселей × 60 шагов = 3.456M вызовов на пиксель-кадр — тот же цикл
занял бы порядки времени больше (десятки минут — часы), это и есть
причина векторизации, а не общий принцип "векторизация лучше цикла".

КОРРЕКТНОСТЬ векторизации ОБЯЗАТЕЛЬНО проверяется численно против
    scalar disocclusion.reproject_and_check на случайной выборке
(_cross_validate, вызывается по умолчанию из build_all_pixel_sequences_
zeroday) — не принимается на веру только по совпадению формул на
бумаге, тем же принципом "empirical verification", что и весь проект.

Отличия входных данных zeroday.npz от dense.npz (см.
generate_dataset_zeroday.py):
- нет ключа "t_values" — здесь строится из "frames" (t = frames -
  frames[0]), та же регулярная сетка с dt=const=1.0, что dense-путь при
  DT=1 (dt1.npz).
- нет ключа "emissive_scales" аналога в dense-пути (там свет управляется
  тем же light_schedule.py, но иначе встроен) — не нужен для
disocclusion/hybrid_v0 инференса, здесь не используется.
- camera_positions — тот же формат [T, 3], что и dense.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from disocclusion import reproject_and_check as _scalar_reproject  # noqa: E402

# Та же фиксированная нормализация confidence, что во всех остальных путях проекта
# (run_experiment.py/run_experiment_blender.py/dataset_adapter_dense.py) — log1p(64.0),
# не пересчитывается по конкретному датасету, чтобы все пути оставались сопоставимы.
CONFIDENCE_LOG_MAX = float(np.log1p(64.0))


def _bilinear_sample_grid(buffer, x, y):
    """Векторизованный аналог disocclusion.bilinear_sample по ВСЕЙ сетке
    сразу (одна операция вместо H*W скалярных вызовов).

    buffer: (H, W, C). x, y: (H, W) — дробные координаты запроса, ОДНА
    query-точка на каждый выходной пиксель (не путать с buffer's own
    H,W — x/y это КУДА сэмплировать, свои для каждого пикселя).

    Возвращает (values (H,W,C), in_bounds (H,W) bool) — та же семантика
    границ (0 <= coord <= dim-1), что и disocclusion.bilinear_sample:
    строго `x > w - 1` (не `>=`) считается вне границ, x==w-1 — валиден.
    """
    h, w = buffer.shape[0], buffer.shape[1]
    in_bounds = (x >= 0.0) & (y >= 0.0) & (x <= w - 1) & (y <= h - 1)

    xc = np.clip(x, 0.0, w - 1)
    yc = np.clip(y, 0.0, h - 1)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xc - x0)[..., None]
    fy = (yc - y0)[..., None]

    v00 = buffer[y0, x0]
    v10 = buffer[y0, x1]
    v01 = buffer[y1, x0]
    v11 = buffer[y1, x1]
    top = v00 * (1 - fx) + v10 * fx
    bottom = v01 * (1 - fx) + v11 * fx
    values = top * (1 - fy) + bottom * fy
    values = np.where(in_bounds[..., None], values, 0.0)
    return values, in_bounds


def _disocclusion_grid(position_curr, normal_curr, position_prev, normal_prev,
                        motion_vector, camera_pos_curr,
                        distance_scale=0.2, normal_threshold_deg=45.0,
                        mismatch_threshold=0.5):
    """Векторизованный аналог disocclusion.reproject_and_check по ВСЕЙ
    сетке H×W за один кадровый переход (idx-1 -> idx) — построчная копия
    формул disocclusion.py, применённая к массиву (H,W,...) вместо скаляра
    (ничего не меняется по существу, см. docstring модуля про то, что
    disocclusion.py переиспользуется без изменений).

    ВАЖНО (найдено при первом _cross_validate, 05.09.2026): сам варп-
    сдвиг (x_prev = x + motion_vector_zw[0]) в scalar disocclusion.py
    вычисляется в float32, а НЕ float64 — motion_vector хранится как
    float32, и по правилам NumPy (NEP 50) `python_float + float32 ->
    float32`, т.е. scalar-путь ТЕРЯЕТ точность именно на этом шаге
    (пиксельная координата round'ится к float32 ДО floor/interpolation
    weights). Изначальная версия этой функции считала x_prev/y_prev в
    float64 (точнее оригинала!) и из-за этого расходилась со scalar-
    версией на интерполяционных весах (до ~2.5e-5 по score на некоторых
    сэмплах) — не потому что векторизация была неверна, а потому что
    она была ТОЧНЕЕ уже провалидированного scalar-пути. Чтобы остаться
    bit-совместимой с disocclusion.py (см. принцип "переиспользуется
    без изменений" выше), warp здесь тоже считается в float32: сетка
    индексов приводится к float32 ДО сложения с motion_vector (который
    остаётся в исходном float32), полностью воспроизводя потерю
    точности scalar-пути. Все остальные величины (position_curr/
    normal_curr/пороговые вычисления) приводятся к float64 ПОСЛЕ
    сэмплинга — ровно как это делает geometric_mismatch_score в
    disocclusion.py (np.asarray(..., dtype=float) на уже полученных
    значениях, а не на входных координатах).
    """
    h, w = position_curr.shape[0], position_curr.shape[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_prev = xx + motion_vector[..., 0]
    y_prev = yy + motion_vector[..., 1]

    pos_prev, pos_in_bounds = _bilinear_sample_grid(position_prev, x_prev, y_prev)
    normal_prev_s, normal_in_bounds = _bilinear_sample_grid(normal_prev, x_prev, y_prev)
    in_bounds = pos_in_bounds & normal_in_bounds

    position_curr = position_curr.astype(np.float64)
    normal_curr = normal_curr.astype(np.float64)
    camera_pos_curr = camera_pos_curr.astype(np.float64)

    dist_to_camera = np.linalg.norm(position_curr - camera_pos_curr[None, None, :], axis=-1)
    pos_threshold = np.maximum(distance_scale * dist_to_camera, 1e-4)
    pos_diff = np.linalg.norm(position_curr - pos_prev, axis=-1)
    pos_score = np.clip(pos_diff / pos_threshold, 0.0, 1.0)

    nc = normal_curr / (np.linalg.norm(normal_curr, axis=-1, keepdims=True) + 1e-8)
    npv = normal_prev_s / (np.linalg.norm(normal_prev_s, axis=-1, keepdims=True) + 1e-8)
    cos_angle = np.clip(np.sum(nc * npv, axis=-1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    normal_score = np.clip(angle_deg / normal_threshold_deg, 0.0, 1.0)

    score = np.maximum(pos_score, normal_score)
    score = np.where(in_bounds, score, 1.0)  # out-of-bounds -> максимальное расхождение, как в scalar-версии
    flag = (~in_bounds) | (score > mismatch_threshold)
    return flag.astype(np.float32), score.astype(np.float32)


def _cross_validate(position, normal, motion_vector, camera_positions,
                     disocclusion_flag, geometric_mismatch,
                     distance_scale, normal_threshold_deg, mismatch_threshold,
                     n_samples=200, seed=0):
    """Численно сверяет векторизованный результат со scalar
    disocclusion.reproject_and_check на случайной выборке
    (frame_idx, y, x). Бросает AssertionError, если хоть один сэмпл
    расходится — не должно проходить "почти совпадает"."""
    T, H, W = position.shape[0], position.shape[1], position.shape[2]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(1, T, size=n_samples)
    ys = rng.integers(0, H, size=n_samples)
    xs = rng.integers(0, W, size=n_samples)

    max_score_diff = 0.0
    n_flag_mismatch = 0
    for i in range(n_samples):
        idx, y, x = int(idxs[i]), int(ys[i]), int(xs[i])
        mv = motion_vector[idx, y, x]
        out = _scalar_reproject(
            x=float(x), y=float(y), motion_vector_zw=mv,
            pos_curr=position[idx, y, x], normal_curr=normal[idx, y, x],
            prev_position_buffer=position[idx - 1], prev_normal_buffer=normal[idx - 1],
            camera_pos=camera_positions[idx],
            distance_scale=distance_scale, normal_threshold_deg=normal_threshold_deg,
            mismatch_threshold=mismatch_threshold,
        )
        scalar_flag = 1.0 if out["disocclusion_flag"] else 0.0
        scalar_score = out["geometric_mismatch_score"]
        vec_flag = disocclusion_flag[idx, y, x]
        vec_score = geometric_mismatch[idx, y, x]
        max_score_diff = max(max_score_diff, abs(scalar_score - vec_score))
        if scalar_flag != vec_flag:
            n_flag_mismatch += 1

    assert max_score_diff < 1e-5, (
        f"vectorized/scalar geometric_mismatch_score расходились: max_diff={max_score_diff}"
    )
    assert n_flag_mismatch == 0, (
        f"vectorized/scalar disocclusion_flag расходились на {n_flag_mismatch}/{n_samples} сэмплах"
    )
    print(f"[cross-validate] {n_samples} сэмплов (idx,y,x): max_score_diff={max_score_diff:.2e}, "
          f"flag_mismatches=0 -- vectorized реализация подтверждена идентичной scalar disocclusion.py")


def build_all_pixel_sequences_zeroday(
    npz_path,
    distance_scale: float = 0.2,
    normal_threshold_deg: float = 45.0,
    mismatch_threshold: float = 0.5,
    validate_against_scalar: bool = True,
    n_validate_samples: int = 200,
    validate_seed: int = 0,
):
    """Список dict — та же схема ключей, что build_all_pixel_sequences
    (dataset_adapter_dense.py): pixel, t, dt, cold_start, obs,
    true_irradiance, spp, confidence, valid, disocclusion_flag,
    geometric_mismatch_score, position, normal, albedo, direction.

    distance_scale/normal_threshold_deg/mismatch_threshold — те же
    некалиброванные дефолты, что disocclusion.py (см. её докстринг),
    не переоткалиброваны отдельно здесь для сопоставимости с dt1.npz.
    """
    data = np.load(npz_path)
    position = data["position"]                # [T, H, W, 3]
    normal = data["normal"]                     # [T, H, W, 3]
    albedo = data["albedo"]                      # [T, H, W, 3]
    irradiance = data["irradiance"]              # [T, H, W, 3]
    true_irradiance = data["true_irradiance"]    # [T, H, W, 3]
    spp = data["spp"]                            # [T, H, W]
    valid = data["valid"]                        # [T, H, W] bool
    motion_vector = data["motion_vector"]        # [T, H, W, 2]
    camera_positions = data["camera_positions"]  # [T, 3]
    frames = data["frames"]                      # [T] int

    T, H, W = position.shape[0], position.shape[1], position.shape[2]
    assert T > 1, "нужно хотя бы 2 кадра для дисокклюзии"
    t_values = (frames.astype(np.float64) - frames[0]).astype(np.float32)
    dt_diffs = np.diff(t_values)
    assert np.allclose(dt_diffs, dt_diffs[0]), (
        f"ожидается регулярная сетка кадров (const step), получено {dt_diffs}"
    )
    dt_const = float(dt_diffs[0])

    disocclusion_flag = np.zeros((T, H, W), dtype=np.float32)
    geometric_mismatch = np.zeros((T, H, W), dtype=np.float32)
    # idx=0: нет предыдущего кадра для warp -> тривиальная дисокклюзия,
    # тем же принципом, что dataset_adapter_dense.py (out-of-bounds
    # семантика "history невалидна тривиально").
    disocclusion_flag[0] = 1.0
    geometric_mismatch[0] = 1.0

    for idx in range(1, T):
        flag, score = _disocclusion_grid(
            position[idx], normal[idx], position[idx - 1], normal[idx - 1],
            motion_vector[idx], camera_positions[idx],
            distance_scale=distance_scale, normal_threshold_deg=normal_threshold_deg,
            mismatch_threshold=mismatch_threshold,
        )
        disocclusion_flag[idx] = flag
        geometric_mismatch[idx] = score

    if validate_against_scalar:
        _cross_validate(position, normal, motion_vector, camera_positions,
                         disocclusion_flag, geometric_mismatch,
                         distance_scale, normal_threshold_deg, mismatch_threshold,
                         n_samples=n_validate_samples, seed=validate_seed)

    confidence = (np.log1p(spp) / CONFIDENCE_LOG_MAX).astype(np.float32)

    sequences = []
    for y in range(H):
        for x in range(W):
            if not valid[:, y, x].any():
                continue  # пиксель ни разу не попал в геометрию сцены

            dt = np.full(T, dt_const, dtype=np.float32)
            dt[0] = 0.0
            cold_start = np.zeros(T, dtype=np.float32)
            cold_start[0] = 1.0

            position_seq = position[:, y, x, :].astype(np.float32)
            diff = position_seq - camera_positions.astype(np.float32)
            direction_seq = (diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)

            sequences.append(dict(
                pixel=(x, y),
                t=t_values,
                dt=dt,
                cold_start=cold_start,
                obs=irradiance[:, y, x, :].astype(np.float32),
                true_irradiance=true_irradiance[:, y, x, :].astype(np.float32),
                spp=spp[:, y, x].astype(np.float32),
                confidence=confidence[:, y, x],
                valid=valid[:, y, x].astype(bool),
                disocclusion_flag=disocclusion_flag[:, y, x],
                geometric_mismatch_score=geometric_mismatch[:, y, x],
                position=position_seq,
                normal=normal[:, y, x, :].astype(np.float32),
                albedo=albedo[:, y, x, :].astype(np.float32),
                direction=direction_seq,
            ))
    return sequences


def load_fully_valid_pixel_sequences_zeroday(npz_path, **kwargs):
    """Аналог load_fully_valid_pixel_sequences (run_experiment_dense.py)
    для zeroday-датасета — build_all_pixel_sequences_zeroday +
    фильтр "валиден на ВСЕХ кадрах" (тот же принцип и то же
    обоснование, см. докстринг load_fully_valid_pixel_sequences)."""
    seqs = build_all_pixel_sequences_zeroday(npz_path, **kwargs)
    filtered = [s for s in seqs if s["valid"].all()]
    print(f"пикселей всего с последовательностью: {len(seqs)}, "
          f"валидных на всех кадрах: {len(filtered)}")
    return filtered
