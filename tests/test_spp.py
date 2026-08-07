"""
Юнит-тест §3.1 пункт 4: spp-метаданные (Debug Sample Count, confidence-
сигнал для staleness-gate).

Перепроверяет числа из 04_spp_prototype.py независимо от внутреннего
PASS-флага скрипта: реальный spp (уже переведённый из нормированной
доли в raw-число сэмплов) должен укладываться в
[adaptive_min_samples, samples], не быть константой по кадру, и зона
высокого градиента яркости должна получать заметно больше сэмплов,
чем гладкая зона.
"""
from conftest import run_blender_script


def test_spp_within_bounds_and_correlates_with_local_contrast():
    result = run_blender_script("04_spp_prototype.py")

    spp_min = result["spp_min_observed"]
    spp_max = result["spp_max_observed"]
    adaptive_min = result["adaptive_min_samples"]
    samples_max = result["samples_max"]

    assert spp_min >= adaptive_min - 1e-6, (
        f"observed min spp {spp_min} below configured adaptive_min_samples {adaptive_min}"
    )
    assert spp_max <= samples_max + 1e-6, (
        f"observed max spp {spp_max} exceeds configured samples budget {samples_max}"
    )
    assert spp_max > spp_min, (
        "spp is constant across the frame — adaptive sampling had no effect, "
        "or the sample-count pass was not correctly normalized back to raw counts"
    )

    low = result["low_gradient_zone_mean_spp"]
    high = result["high_gradient_zone_mean_spp"]
    assert high > low * 1.2, (
        f"high-gradient (shadow-edge) zone mean spp {high:.1f} not meaningfully "
        f"higher than low-gradient (flat) zone mean spp {low:.1f}"
    )
