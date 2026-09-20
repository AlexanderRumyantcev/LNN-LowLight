"""
blender/dataset_adapter_zeroday.py — TZ_stage5 (адаптация dataset_adapter_dense.py
под Zero Day, не synthetic-геометрию).

Отличия от dataset_adapter_dense.py (dense per-pixel, synthetic-сцена):
1. Источник данных — npz от generate_dataset_zeroday.py (Blender Cycles render,
   кадры 90-150 Zero Day scene), не procedural probe placement.
2. disocclusion вычисляется векторизованно (все пиксели кадра сразу,
   не по-пробному как в dataset_adapter_dense.py) — blender/disocclusion.py уже
   векторизован с 05.09.2026, см. докстринг там.
3. Только RGB irradiance (obs_dim=3) — нет depth/normal G-buffer как доп.
   каналов выхода (только как вход для spatial conditioning, см. models/
   spatial_features.py).

Формат возвращаемой последовательности (seqs[i]) ИДЕНТИЧЕН dataset_
аdapter_dense.py — всё нижестоящее (build_batch_dense, run_experiment_dense.py,
run_tz_stage9_online_speed.py) работает без изменений.
"""
import numpy as np

from disocclusion import compute_disocclusion_vectorized


def load_fully_valid_pixel_sequences_zeroday(npz_path):
    """Загружает generate_dataset_zeroday.py npz, возвращает список
    последовательностей-словарей в том же формате, что dataset_adapter_
    dense.py::load_fully_valid_pixel_sequences (t/obs/cold/conf/disocclusion_flag/
    geometric_mismatch_score/true + position/direction/normal/albedo для spatial).

    Фильтр "fully valid": пиксель берётся только если валиден во ВСЕХ
    кадрах сегмента (нет пропусков внутри последовательности — тот же
    принцип, что в synthetic-версии).
    """
    data = np.load(npz_path)
    valid = data["valid_mask"]  # [H, W, T] bool
    obs = data["irradiance"]  # [H, W, T, 3]
    position = data["position"]  # [H, W, T, 3]
    direction = data["direction"]  # [H, W, T, 3]
    normal = data["normal"]  # [H, W, T, 3]
    albedo = data["albedo"]  # [H, W, T, 3]
    t = data["t"]  # [T]

    H, W, T = valid.shape[:3]
    fully_valid_mask = valid.all(axis=2)  # [H, W]
    ys, xs = np.where(fully_valid_mask)

    # Векторизованный disocclusion по ВСЕМ пикселям сразу, см. blender/
    # disocclusion.py (05.09.2026)
    disocclusion_flag, geometric_mismatch_score = compute_disocclusion_vectorized(
        position, normal, t
    )  # оба [H, W, T]

    seqs = []
    for y, x in zip(ys, xs):
        seqs.append(dict(
            t=t.copy(),
            obs=obs[y, x],  # [T, 3]
            cold=np.concatenate([[1.0], np.zeros(T - 1)]),  # первый кадр — всегда cold-start
            conf=np.log1p(data["spp"][y, x]) if "spp" in data else np.ones(T),
            disocclusion_flag=disocclusion_flag[y, x],  # [T]
            geometric_mismatch_score=geometric_mismatch_score[y, x],  # [T]
            true=obs[y, x].copy(),  # цель — та же irradiance (denoising target)
            position=position[y, x],  # [T, 3]
            direction=direction[y, x],  # [T, 3]
            normal=normal[y, x],  # [T, 3]
            albedo=albedo[y, x],  # [T, 3]
            pixel_coord=np.array([x, y], dtype=np.float64),
        ))
    return seqs
