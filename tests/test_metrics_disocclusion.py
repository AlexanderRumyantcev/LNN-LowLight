"""
Тесты §6 delta (TZ_stage1b_per_pixel_dense_fallback.md): новый срез
метрик по disocclusion — disocclusion_vs_stable_mse,
compute_age_since_disocclusion, error_vs_warp_age_curve.

Первые тесты для evaluation/metrics.py вообще (модуль раньше не был
покрыт тестами) — заодно неявно фиксируют поведение уже существующих
paired_bootstrap_significance/per_segment_type_mse через переиспользование
в сквозном тесте ниже, не только новых функций.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (
    disocclusion_vs_stable_mse,
    compute_age_since_disocclusion,
    error_vs_warp_age_curve,
    paired_bootstrap_significance,
    MIN_N_SEEDS,
)


# --- disocclusion_vs_stable_mse ------------------------------------------


def test_disocclusion_vs_stable_mse_splits_correctly():
    pred = np.array([0.0, 5.0, 0.0, 5.0])
    true = np.array([0.0, 0.0, 0.0, 0.0])  # errors: [0, 25, 0, 25]
    flag = np.array([False, True, False, True])
    out = disocclusion_vs_stable_mse(pred, true, flag)
    assert out["disocclusion"] == 25.0
    assert out["stable"] == 0.0


def test_disocclusion_vs_stable_mse_shape_mismatch_raises():
    try:
        disocclusion_vs_stable_mse(np.zeros(4), np.zeros(4), np.zeros(3, dtype=bool))
        assert False
    except ValueError:
        pass


def test_disocclusion_vs_stable_mse_nan_when_group_empty():
    pred = np.array([1.0, 2.0])
    true = np.array([0.0, 0.0])
    flag = np.array([False, False])
    out = disocclusion_vs_stable_mse(pred, true, flag)
    assert np.isnan(out["disocclusion"])
    assert not np.isnan(out["stable"])


# --- compute_age_since_disocclusion ---------------------------------------


def test_age_since_disocclusion_basic_1d():
    flag = np.array([1, 0, 0, 1, 0, 0, 0], dtype=bool)
    age = compute_age_since_disocclusion(flag)
    assert np.array_equal(age, [0, 1, 2, 0, 1, 2, 3])


def test_age_since_disocclusion_nan_before_first_event():
    flag = np.array([0, 0, 1, 0], dtype=bool)
    age = compute_age_since_disocclusion(flag)
    assert np.isnan(age[0]) and np.isnan(age[1])
    assert age[2] == 0.0
    assert age[3] == 1.0


def test_age_since_disocclusion_batched_2d_independent_rows():
    flags = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=bool)
    age = compute_age_since_disocclusion(flags)
    assert age.shape == (2, 3)
    assert np.array_equal(age[0], [0, 1, 2])
    assert np.isnan(age[1, 0])
    assert age[1, 1] == 0.0
    assert age[1, 2] == 1.0


def test_age_since_disocclusion_rejects_bad_ndim():
    try:
        compute_age_since_disocclusion(np.zeros((2, 2, 2)))
        assert False
    except ValueError:
        pass


# --- error_vs_warp_age_curve -----------------------------------------------


def test_error_vs_warp_age_curve_bins_by_age_not_offset():
    # cold-start в t=0, ошибка растёт линейно с возрастом истории после warp
    flag = np.array([1, 0, 0, 0], dtype=bool)
    true = np.zeros(4)
    pred = np.array([0.0, 1.0, 2.0, 3.0])  # age=[0,1,2,3], err=[0,1,4,9]
    curve = error_vs_warp_age_curve(pred, true, flag, bin_edges=np.array([0, 1, 2, 3, np.inf]))
    assert curve["[0,1)"] == 0.0
    assert curve["[1,2)"] == 1.0
    assert curve["[2,3)"] == 4.0
    assert curve["[3,inf)"] == 9.0


def test_error_vs_warp_age_curve_accepts_2d_batch():
    flags = np.array([
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=bool)
    true = np.zeros((2, 3))
    pred = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    curve = error_vs_warp_age_curve(pred, true, flags, bin_edges=np.array([0, 1, 2, np.inf]))
    # age=0 у обеих строк на t=0 (err=0), age=1 на t=1 (err=1 у обеих), age=2 на t=2 (err=4)
    assert curve["[0,1)"] == 0.0
    assert curve["[1,2)"] == 1.0
    assert curve["[2,inf)"] == 4.0


def test_error_vs_warp_age_curve_default_bin_edges_match_offset_curve_convention():
    """Те же bin_edges по умолчанию, что error_vs_offset_curve (§6.1) —
    единообразие между static/step/drift и disocclusion срезами."""
    flag = np.array([1] + [0] * 25, dtype=bool)
    true = np.zeros(26)
    pred = np.zeros(26)
    curve = error_vs_warp_age_curve(pred, true, flag)
    assert set(curve.keys()) == {
        "[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,6)", "[6,8)", "[8,12)", "[12,20)", "[20,inf)",
    }


# --- Сквозная проверка: срез реально можно скормить paired_bootstrap_significance ---


def test_disocclusion_slice_feeds_into_paired_bootstrap():
    """§6 delta явно требует, чтобы именно disocclusion-срез (не агрегат
    по кадру) определял исход §5 — проверяем, что per-seed MSE из
    disocclusion_vs_stable_mse реально скармливается в существующий
    paired_bootstrap_significance без переделки API."""
    rng = np.random.default_rng(0)
    n_seeds = MIN_N_SEEDS
    errors_cfc = []
    errors_honest = []
    for seed in range(n_seeds):
        r = np.random.default_rng(seed)
        pred_cfc = r.normal(0, 0.5, size=50)
        pred_honest = r.normal(0, 1.0, size=50)
        true = np.zeros(50)
        flag = r.integers(0, 2, size=50).astype(bool)
        errors_cfc.append(disocclusion_vs_stable_mse(pred_cfc, true, flag)["disocclusion"])
        errors_honest.append(disocclusion_vs_stable_mse(pred_honest, true, flag)["disocclusion"])

    result = paired_bootstrap_significance(np.array(errors_honest), np.array(errors_cfc))
    assert "significant" in result
    assert result["n_seeds"] == n_seeds
