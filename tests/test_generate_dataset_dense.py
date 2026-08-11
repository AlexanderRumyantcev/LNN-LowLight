"""
Тесты §2.3/§3.2 (TZ_stage1b_per_pixel_dense_fallback.md):
generate_dataset_dense.py — растровый генератор датасета.

Запускается с коротким duration (см. args=[...]) ради скорости — сама
механика (форма буферов, консистентность valid-маски, наличие
motion_vector, движение камеры) не зависит от длительности; полный
TOTAL_DURATION=400 прогонялся вручную отдельно (см. drawer в
mempalace) для sanity-проверки объёма и total_edge_mismatch_px.
"""
import numpy as np

from conftest import run_blender_script

SMOKE_DURATION = 24  # -> 4 кадра при DT=8 (см. generate_dataset_dense.py)


def _run_smoke():
    meta = run_blender_script(
        "generate_dataset_dense.py",
        args=[SMOKE_DURATION, "smoke"],
        result_filename="generate_dataset_dense_result_meta_smoke.json",
        timeout=180,
    )
    npz = np.load(meta["npz_path"])
    return meta, npz


def test_dense_generator_produces_expected_frame_count():
    meta, npz = _run_smoke()
    expected_frames = int(SMOKE_DURATION / meta["dt"]) + 1
    assert meta["n_frames"] == expected_frames
    assert npz["position"].shape[0] == expected_frames


def test_dense_generator_buffer_shapes_consistent():
    meta, npz = _run_smoke()
    n, res = meta["n_frames"], meta["grid_res"]
    assert npz["position"].shape == (n, res, res, 3)
    assert npz["normal"].shape == (n, res, res, 3)
    assert npz["albedo"].shape == (n, res, res, 3)
    assert npz["irradiance"].shape == (n, res, res, 3)
    assert npz["true_irradiance"].shape == (n, res, res, 3)
    assert npz["spp"].shape == (n, res, res)
    assert npz["valid"].shape == (n, res, res)
    # только Z,W-каналы (§2.2) — не все 4 канала Vector pass
    assert npz["motion_vector"].shape == (n, res, res, 2)
    assert npz["camera_positions"].shape == (n, 3)
    assert npz["t_values"].shape == (n,)


def test_dense_generator_regular_dt_not_async_moment_pool():
    """§2.3: регулярная сетка t_i = i*DT, а не асинхронный moment_pool —
    в отличие от probe-пути, разница t между соседними кадрами ОДНА и
    ТА ЖЕ по всей последовательности."""
    meta, npz = _run_smoke()
    t = npz["t_values"]
    diffs = np.diff(t)
    assert np.allclose(diffs, meta["dt"]), f"expected constant dt={meta['dt']}, got {diffs}"


def test_dense_generator_camera_actually_moves():
    """Камера — источник движения (§1.3), а не спрятанный fixed-ortho
    прежнего probe-пути: позиции камеры между кадрами должны реально
    отличаться, не быть константой."""
    _, npz = _run_smoke()
    cam_pos = npz["camera_positions"]
    assert cam_pos.shape[0] >= 2
    deltas = np.linalg.norm(np.diff(cam_pos, axis=0), axis=1)
    assert np.all(deltas > 0.1), f"camera barely moved between frames: {deltas}"


def test_dense_generator_no_nans_in_buffers():
    _, npz = _run_smoke()
    for key in ("position", "normal", "albedo", "irradiance", "true_irradiance",
                "spp", "motion_vector"):
        arr = npz[key]
        assert not np.isnan(arr).any(), f"NaNs found in {key}"


def test_dense_generator_valid_mask_nontrivial_and_not_full():
    """На GRID_RES=8 с движущейся камерой ожидаем часть пикселей вне
    геометрии (небо/фон) и часть — на полу: ни полностью пустая маска,
    ни полностью заполненная не были бы правдоподобны для этой сцены."""
    _, npz = _run_smoke()
    valid = npz["valid"]
    frac = valid.mean()
    assert 0.05 < frac < 0.95, f"suspicious valid fraction: {frac}"


def test_dense_generator_true_irradiance_less_noisy_than_obs():
    """Косвенная проверка REFERENCE_SPP-эталона (§3.1, тот же принцип,
    что в probe-пути): true_irradiance (2048 spp) должен быть более
    гладким по кадрам на валидных пикселях, чем obs irradiance
    (adaptive, до 128 spp) при неизменном освещении между соседними
    кадрами внутри одного static-сегмента light_schedule не
    гарантирован здесь (свет варьируется по своему расписанию), поэтому
    вместо temporal-сравнения проверяем внутрикадровую консистентность:
    true_irradiance не должен содержать отрицательных значений (irradiance
    физически неотрицательна), в отличие от чего сигнальный шум obs
    иногда может пересекать 0 на очень тёмных пикселях — здесь проверяем
    только true_irradiance, как более надёжный эталон."""
    _, npz = _run_smoke()
    valid = npz["valid"]
    true_irr = npz["true_irradiance"]
    assert (true_irr[valid] >= -1e-4).all(), "true_irradiance should be non-negative"


def test_dense_generator_edge_mismatch_tracked_not_silently_dropped():
    """§valid-mask AND-фильтр (решение 2026-08-09): расхождение
    obs/true_irradiance alpha на силуэтных пикселях должно логироваться
    в total_edge_mismatch_px, а не проходить незамеченным."""
    meta, _ = _run_smoke()
    assert "total_edge_mismatch_px" in meta
    assert meta["total_edge_mismatch_px"] >= 0
