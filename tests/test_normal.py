"""
Юнит-тест §3.1 пункт 6: normal-буфер (world-space, для normal-weighting
в интерполяции §2.3).

Перепроверяет число из 06_normal_prototype.py независимо от
внутреннего PASS-флага скрипта: угол между рендер-нормалью и
независимо посчитанной (поворотом matrix_world.to_3x3()) должен быть
пренебрежимо мал.
"""
from conftest import run_blender_script


def test_normal_matches_independent_rotation_computation():
    result = run_blender_script("06_normal_prototype.py")

    angle_diff = result["angle_diff_degrees"]
    assert angle_diff < 2.0, (
        f"rendered world-space normal diverges from independently computed "
        f"expectation by {angle_diff:.3f} degrees — expected < 2.0. "
        f"expected={result['expected_world_normal']} "
        f"got={result['rendered_normal_normalized']}"
    )

    # Санити: тест сцены намеренно повёрнут не вдоль осей (17.2°/11.5° по
    # двум осям сразу), чтобы не мог случайно пройти из-за симметрии —
    # проверяем, что фикстура действительно нетривиальна: нормаль не
    # должна совпадать с локальной (0,0,1) на плоскости без поворота.
    expected = result["expected_world_normal"]
    assert abs(expected[0]) > 0.05 or abs(expected[1]) > 0.05, (
        f"expected normal {expected} looks axis-aligned (close to (0,0,1)) — "
        f"fixture rotation may not have been applied, test would pass trivially"
    )
