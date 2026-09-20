"""
blender/dataset_adapter_dense.py — Stage 1b (per-pixel dense pipeline, §1.2/§2
TZ_stage1b_per_pixel_dense_fallback.md).

Адаптер между рендером Blender (motion-vector репроекция, real frame-
locked Δt) и форматом seqs[i], который потребляет run_experiment_dense.py
(build_batch_dense). В отличие от probe-варианта (dataset_adapter.py) — ЗДЕСЬ
КАЖДЫЙ пиксель экрана своя независимая последовательность, нет фиксированного
набора проб с заранее известной видимостью.

Spatial-вход (2026-08-20, models/spatial_features.py): каждая последовательность
теперь также несёт position/direction/normal/albedo — извлекаются из Blender
G-buffer пассов точно так же, как и основной irradiance-сигнал.
"""
import numpy as np

from disocclusion import compute_disocclusion_dense


def load_fully_valid_pixel_sequences(npz_path, max_pixels=None, rng=None):
    """Загружает dense per-pixel npz (см. generate_dataset_dense.py), возвращает
    список последовательностей-словарей, один на пиксель — тот же формат
    ключей, что и dataset_adapter_zeroday.py (оба должны оставаться взаимозаменяемыми
    для build_batch_dense).

    max_pixels/rng — опциональный случайный сабсэмплинг валидных пикселей для
    dev/smoke-запусков на полном разрешении без загрузки всех пикселей.
    """
    data = np.load(npz_path)
    valid = data["valid_mask"]  # [H, W, T] bool
    obs = data["radiance"]  # [H, W, T, obs_dim]
    position = data["position"]  # [H, W, T, 3]
    direction = data["direction"]  # [H, W, T, 3]
    normal = data["normal"]  # [H, W, T, 3]
    albedo = data["albedo"]  # [H, W, T, 3]
    motion_vector = data["motion_vector"]  # [H, W, T, 2]
    t = data["t"]  # [T]

    H, W, T = valid.shape[:3]
    fully_valid_mask = valid.all(axis=2)  # [H, W]
    ys, xs = np.where(fully_valid_mask)

    if max_pixels is not None and len(ys) > max_pixels:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(ys), size=max_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]

    disocclusion_flag, geometric_mismatch_score = compute_disocclusion_dense(
        position, normal, motion_vector, t
    )  # оба [H, W, T]

    seqs = []
    for y, x in zip(ys, xs):
        seqs.append(dict(
            t=t.copy(),
            obs=obs[y, x],
            cold=np.concatenate([[1.0], np.zeros(T - 1)]),
            conf=np.log1p(data["spp"][y, x]) if "spp" in data else np.ones(T),
            disocclusion_flag=disocclusion_flag[y, x],
            geometric_mismatch_score=geometric_mismatch_score[y, x],
            true=obs[y, x].copy(),
            position=position[y, x],
            direction=direction[y, x],
            normal=normal[y, x],
            albedo=albedo[y, x],
            pixel_coord=np.array([x, y], dtype=np.float64),
        ))
    return seqs
