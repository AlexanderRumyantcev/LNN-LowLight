"""
Юнит-тест §3.1 пункт 2: irradiance-буфер (Diffuse Direct + Diffuse
Indirect) должен быть инвариантен к albedo материала, а Combined
(beauty) — обязан меняться, иначе тест ничего не проверяет.

Перепроверяет числа из 02_irradiance_prototype.py независимо от
внутреннего PASS-флага скрипта.
"""
from conftest import run_blender_script


def test_irradiance_invariant_to_albedo_but_beauty_scales_with_it():
    result = run_blender_script("02_irradiance_prototype.py")

    irr_low = result["albedo_0.2"]["irradiance_proxy (diffdir+diffind)"]
    irr_high = result["albedo_0.8"]["irradiance_proxy (diffdir+diffind)"]

    rel_diff = max(
        abs(a - b) / max(abs(a), abs(b), 1e-8) for a, b in zip(irr_low, irr_high)
    )
    assert rel_diff < 0.02, (
        f"irradiance proxy changed by {rel_diff:.4f} relative when only "
        f"albedo changed (0.2->0.8) — should be invariant. "
        f"low={irr_low} high={irr_high}"
    )

    beauty_low = result["albedo_0.2"]["combined_beauty"]
    beauty_high = result["albedo_0.8"]["combined_beauty"]
    # Санити: если beauty НЕ изменился, значит материал вообще не
    # применился и тест выше прошёл случайно (сравнивал два нуля).
    beauty_ratio = [h / l for l, h in zip(beauty_low, beauty_high) if l > 1e-8]
    assert beauty_ratio, "beauty pixel too dark to compute a meaningful ratio"
    assert all(abs(r - 4.0) < 0.5 for r in beauty_ratio), (
        f"expected beauty to scale ~4x (albedo ratio 0.8/0.2), got ratios {beauty_ratio}"
    )
