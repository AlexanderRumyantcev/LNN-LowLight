"""
Юнит-тест §3.1 пункт 5: albedo-буфер (Diffuse Color) — диагностический
AOV, должен отражать material base color и не зависеть от освещения.

Перепроверяет числа из 05_albedo_prototype.py независимо от
внутреннего PASS-флага скрипта.
"""
from conftest import run_blender_script


def test_albedo_invariant_to_light_and_matches_base_color():
    result = run_blender_script("05_albedo_prototype.py")

    a_dim = result["light_100"]["albedo_pixel"]
    a_bright = result["light_400"]["albedo_pixel"]

    rel_diff = max(
        abs(x - y) / max(abs(x), abs(y), 1e-8) for x, y in zip(a_dim, a_bright)
    )
    assert rel_diff < 0.02, (
        f"albedo pixel changed by {rel_diff:.4f} relative when only light "
        f"energy changed (100->400) — should be invariant. "
        f"dim={a_dim} bright={a_bright}"
    )

    expected_base_color = 0.5  # ALBEDO constant in the prototype script
    abs_diff_from_expected = max(abs(x - expected_base_color) for x in a_dim)
    assert abs_diff_from_expected < 0.02, (
        f"albedo pixel {a_dim} diverges from the material's set base_color "
        f"{expected_base_color} by {abs_diff_from_expected}"
    )

    c_dim = result["light_100"]["combined_pixel"]
    c_bright = result["light_400"]["combined_pixel"]
    combined_rel_diff = max(
        abs(x - y) / max(abs(x), abs(y), 1e-8) for x, y in zip(c_dim, c_bright)
    )
    # Санити: если combined тоже не изменился, значит свет вообще не
    # повлиял на сцену и albedo-инвариантность выше не показательна.
    assert combined_rel_diff > 0.5, (
        f"beauty pixel barely changed ({combined_rel_diff:.4f}) between "
        f"light energies 100 and 400 — scene lighting setup is suspect"
    )
