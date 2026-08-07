"""
Юнит-тест §3.1 пункт 3: precomputed visibility (raycast проба -> точка).

Перепроверяет числа из 03_visibility_prototype.py независимо от
внутреннего PASS-флага скрипта: для каждой тестовой точки сверяет
bpy scene.ray_cast с независимым ray-AABB intersection (slab method),
посчитанным в самом скрипте прототипа без обращения к Blender API.
"""
from conftest import run_blender_script


def test_visibility_matches_independent_ray_aabb_ground_truth():
    result = run_blender_script("03_visibility_prototype.py")

    points = result["results"]
    assert len(points) >= 4, "expected at least 4 test points in the prototype"

    mismatches = {
        name: data
        for name, data in points.items()
        if data["bpy_ray_cast_visible"] != data["independent_expected_visible"]
    }
    assert not mismatches, f"ray_cast disagreed with independent ground truth: {mismatches}"

    # Санити: тест должен реально покрывать оба класса (видимые И
    # перекрытые точки) — иначе он мог бы пройти случайно, если бы
    # occluder-куб вообще не участвовал в сцене.
    visible_count = sum(1 for d in points.values() if d["independent_expected_visible"])
    occluded_count = len(points) - visible_count
    assert visible_count > 0, "no visible test point in the fixture — test is not meaningful"
    assert occluded_count > 0, "no occluded test point in the fixture — test is not meaningful"
