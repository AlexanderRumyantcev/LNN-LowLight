"""
Юнит-тесты §2.4/§2.5 (TZ_stage1b_per_pixel_dense_fallback.md):
warp/reprojection + disocclusion detection.

Два уровня проверки:
  1. Чистая логика (blender/disocclusion.py) на синтетических
     буферах — быстро, без запуска Blender, покрывает граничные
     случаи (out-of-bounds, точное совпадение, нормальный mismatch)
     явно заданными числами.
  2. Интеграционный прогон 08_disocclusion_prototype.py на РЕАЛЬНЫХ
     Blender-буферах (тот же паттерн, что test_motion_vector.py) —
     перепроверяет числа из результата независимо от внутреннего
     PASS самого прототипа.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from disocclusion import (
    bilinear_sample,
    warp_to_previous,
    geometric_mismatch_score,
    is_disoccluded,
    reproject_and_check,
)

from conftest import run_blender_script


# --- 1. Чистая логика на синтетических буферах ---------------------


def test_bilinear_sample_exact_pixel_center_matches_value():
    buf = np.zeros((4, 4, 3))
    buf[2, 1] = [1.0, 2.0, 3.0]
    val, in_bounds = bilinear_sample(buf, x=1.0, y=2.0)
    assert in_bounds
    assert np.allclose(val, [1.0, 2.0, 3.0])


def test_bilinear_sample_interpolates_between_four_neighbors():
    buf = np.zeros((2, 2, 1))
    buf[0, 0] = 0.0
    buf[0, 1] = 10.0
    buf[1, 0] = 0.0
    buf[1, 1] = 10.0
    val, in_bounds = bilinear_sample(buf, x=0.5, y=0.0)
    assert in_bounds
    assert np.allclose(val, [5.0])


def test_bilinear_sample_out_of_bounds_flagged():
    buf = np.ones((4, 4, 3))
    for x, y in [(-0.1, 1.0), (1.0, -0.1), (4.0, 1.0), (1.0, 4.0)]:
        val, in_bounds = bilinear_sample(buf, x, y)
        assert not in_bounds, f"({x},{y}) should be out of bounds"


def test_warp_to_previous_adds_vector_directly_no_sign_flip():
    """См. disocclusion.warp_to_previous docstring: (Z,W) канал
    Cycles Vector pass прибавляется напрямую (эмпирически
    провалидировано в 07_motion_vector_prototype.py), а не вычитается
    буквально по тексту формулы §2.4."""
    x_prev, y_prev = warp_to_previous(10.0, 20.0, (-3.0, 1.5))
    assert (x_prev, y_prev) == (7.0, 21.5)


def test_geometric_mismatch_zero_for_identical_position_and_normal():
    score = geometric_mismatch_score(
        pos_curr=(1.0, 2.0, 3.0), normal_curr=(0.0, 0.0, 1.0),
        pos_prev=(1.0, 2.0, 3.0), normal_prev=(0.0, 0.0, 1.0),
        camera_pos=(0.0, 0.0, 10.0),
    )
    assert score < 1e-3, f"expected ~0 for identical position/normal, got {score}"


def test_geometric_mismatch_high_for_large_position_divergence():
    """Позиции различаются на 5 world-units при камере в ~9 units —
    заведомо выше distance_scale=0.2 порога (~1.8) — должен быть
    насыщен в 1.0, как честная диссокклюзия в прототипе 08 (там же
    порядок величин: ~4.8 при пороге ~1.7)."""
    score = geometric_mismatch_score(
        pos_curr=(0.0, 0.0, 0.0), normal_curr=(0.0, 0.0, 1.0),
        pos_prev=(5.0, 0.0, 0.0), normal_prev=(0.0, 0.0, 1.0),
        camera_pos=(0.0, 0.0, 9.0),
    )
    assert score == 1.0


def test_geometric_mismatch_high_for_normal_flip():
    """Позиции совпадают, но нормали противоположны (окклюдер другой
    ориентации на том же месте) — тоже должно быть задетектировано,
    это второй независимый признак §2.5, не сводящийся к позиции."""
    score = geometric_mismatch_score(
        pos_curr=(0.0, 0.0, 0.0), normal_curr=(0.0, 0.0, 1.0),
        pos_prev=(0.0, 0.0, 0.0), normal_prev=(0.0, 0.0, -1.0),
        camera_pos=(0.0, 0.0, 9.0),
    )
    assert score == 1.0


def test_is_disoccluded_out_of_bounds_always_true_regardless_of_score():
    assert is_disoccluded(in_bounds=False, mismatch_score=0.0) is True


def test_is_disoccluded_respects_threshold():
    assert is_disoccluded(in_bounds=True, mismatch_score=0.4, mismatch_threshold=0.5) is False
    assert is_disoccluded(in_bounds=True, mismatch_score=0.6, mismatch_threshold=0.5) is True


def test_reproject_and_check_full_pipeline_valid_case():
    """История и текущий кадр показывают одну и ту же плоскую
    геометрию (position растёт по X построчно, normal константна) —
    валидная реконструкция, disocclusion_flag должен быть False."""
    h, w = 8, 8
    prev_pos = np.zeros((h, w, 3))
    prev_normal = np.zeros((h, w, 3))
    for yy in range(h):
        for xx in range(w):
            prev_pos[yy, xx] = [float(xx), float(yy), 5.0]
            prev_normal[yy, xx] = [0.0, 0.0, 1.0]

    out = reproject_and_check(
        x=4.0, y=4.0, motion_vector_zw=(-1.0, 0.0),
        pos_curr=(3.0, 4.0, 5.0), normal_curr=(0.0, 0.0, 1.0),
        prev_position_buffer=prev_pos, prev_normal_buffer=prev_normal,
        camera_pos=(0.0, 0.0, 20.0),
    )
    assert out["in_bounds"] is True
    assert out["disocclusion_flag"] is False
    assert out["geometric_mismatch_score"] < 0.1


def test_reproject_and_check_full_pipeline_out_of_bounds_case():
    h, w = 8, 8
    prev_pos = np.zeros((h, w, 3))
    prev_normal = np.zeros((h, w, 3))
    out = reproject_and_check(
        x=1.0, y=1.0, motion_vector_zw=(-50.0, 0.0),  # улетает далеко за границу
        pos_curr=(0.0, 0.0, 0.0), normal_curr=(0.0, 0.0, 1.0),
        prev_position_buffer=prev_pos, prev_normal_buffer=prev_normal,
        camera_pos=(0.0, 0.0, 20.0),
    )
    assert out["in_bounds"] is False
    assert out["disocclusion_flag"] is True
    assert out["geometric_mismatch_score"] == 1.0


# --- 2. Интеграционный прогон на реальных Blender-буферах -----------


def test_disocclusion_prototype_flags_genuine_disocclusion():
    result = run_blender_script("08_disocclusion_prototype.py")
    disocc = result["disocc_result"]
    assert disocc["in_bounds"] is True, "disocc point warp should stay in-frame"
    assert disocc["disocclusion_flag"] is True, (
        "point occluded at t-Δt and revealed at t should be flagged as disoccluded"
    )
    assert disocc["geometric_mismatch_score"] > 0.9, (
        f"expected near-saturated mismatch for genuine disocclusion, "
        f"got {disocc['geometric_mismatch_score']}"
    )


def test_disocclusion_prototype_does_not_flag_stable_point():
    result = run_blender_script("08_disocclusion_prototype.py")
    stable = result["stable_result"]
    assert stable["in_bounds"] is True
    assert stable["disocclusion_flag"] is False, (
        "point visible at both t-Δt and t (never occluded) should not be "
        "flagged as disoccluded"
    )


def test_disocclusion_prototype_disocc_mismatch_much_larger_than_stable():
    """Разделительная способность механизма важнее конкретного
    абсолютного порога (который сама ТЗ §2.7/§7 называет
    некалиброванным) — проверяем, что реальный разрыв между честной
    диссокклюзией и стабильной точкой большой (не просто по разные
    стороны от произвольного порога)."""
    result = run_blender_script("08_disocclusion_prototype.py")
    disocc_score = result["disocc_result"]["geometric_mismatch_score"]
    stable_score = result["stable_result"]["geometric_mismatch_score"]
    assert disocc_score > 2 * stable_score, (
        f"expected disocc score ({disocc_score}) to clearly separate from "
        f"stable score ({stable_score}), not just barely cross the threshold"
    )


def test_disocclusion_prototype_world_points_are_not_degenerate():
    """Санити: выбранные disocc/stable точки не совпадают и обе
    реально лежат на background-плоскости (X=-3), иначе find_candidates
    могла молча выбрать вырожденную пару."""
    result = run_blender_script("08_disocclusion_prototype.py")
    disocc_pt = result["disocc_world_point"]
    stable_pt = result["stable_world_point"]
    assert disocc_pt != stable_pt
    assert abs(disocc_pt[0] - (-3.0)) < 1e-6
    assert abs(stable_pt[0] - (-3.0)) < 1e-6
