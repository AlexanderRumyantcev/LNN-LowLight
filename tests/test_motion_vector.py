"""
Юнит-тест §2.2 (TZ_stage1b_per_pixel_dense_fallback.md): motion vector
render-pass для camera-driven reprojection.

Перепроверяет число из 07_motion_vector_prototype.py независимо от
внутреннего PASS-флага скрипта: backward-вектор (каналы Z,W вектор-
пасса) должен совпадать с независимо посчитанным обратным экранным
смещением (через bpy_extras.world_to_camera_view на двух кадрах
камеры) в пределах разумной субпиксельной погрешности.
"""
from conftest import run_blender_script


def test_backward_vector_matches_independent_screen_displacement():
    result = run_blender_script("07_motion_vector_prototype.py")

    rel_error = result["rel_error_vs_expected_backward_disp"]
    assert rel_error < 0.1, (
        f"backward vector (Z,W channels) diverges from independently computed "
        f"expected backward displacement by {rel_error:.3%} — expected < 10%. "
        f"backward_vec_zw={result['backward_vec_zw']} "
        f"expected_disp={result['expected_pixel_displacement_frame1_to_2']}"
    )

def test_control_point_is_not_the_degenerate_look_at_target():
    """Санити: контрольная точка (1.5, 1.5, 0) не должна совпадать с
    look-at целью камеры (0,0,0) — иначе тест мог бы тривиально пройти
    из-за того, что эта точка по конструкции всегда в центре экрана."""
    result = run_blender_script("07_motion_vector_prototype.py")
    disp = result["expected_pixel_displacement_frame1_to_2"]
    magnitude = (disp[0] ** 2 + disp[1] ** 2) ** 0.5
    assert magnitude > 1.0, (
        f"expected displacement magnitude {magnitude:.3f}px looks degenerate "
        f"(near zero) — control point may coincide with look-at target, "
        f"test would pass trivially"
    )
