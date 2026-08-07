"""
Юнит-тест §3.1 пункт 1: позиция пробы (world-space).

Перепроверяет числа из 01_probe_position_prototype.py независимо от
внутреннего PASS-флага скрипта: сама проба спарентена к объекту с
нетривиальным трансформом (rotation+translation), так что тест
реально ловит регресс, если кто-то в будущем случайно заменит
matrix_world.translation на локальные координаты object.location.
"""
import math

from conftest import run_blender_script


def test_probe_world_position_matches_independent_computation():
    result = run_blender_script("01_probe_position_prototype.py")

    got = result["extracted_world_position"]
    expected = result["independently_expected_world_position"]

    assert len(got) == 3 and len(expected) == 3

    max_abs_diff = max(abs(a - b) for a, b in zip(got, expected))
    assert max_abs_diff < 1e-5, (
        f"extracted world position {got} diverges from independently "
        f"computed expectation {expected} by {max_abs_diff}"
    )

    # Санити-проверка на вырожденный случай: тест не должен молча
    # проходить, если проба каким-то образом осталась в начале координат
    # (например, parent-трансформ не применился вообще).
    origin_dist = math.sqrt(sum(c * c for c in got))
    assert origin_dist > 1.0, (
        f"extracted position {got} suspiciously close to origin — "
        f"parent transform may not have been applied"
    )
