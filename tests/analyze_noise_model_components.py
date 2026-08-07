"""
§3.2(Б) — раздельный анализ direct vs indirect компонент шума
(blender/verify_noise_model_components_result.json).

Гипотеза, которую проверяем: немонотонность std(spp) в
tests/analyze_noise_model.py (spp=4 шумнее spp=1 в комбинированном
irradiance = diffuse_direct + diffuse_indirect) — артефакт смешивания
двух компонент с разной шумовой природой (indirect требует нескольких
отскоков луча, обычно шумит на семпл сильнее direct), а не свойство
сцены/освещения в целом.
"""
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
RESULT_PATH = PROJECT_ROOT / "blender" / "verify_noise_model_components_result.json"


@pytest.fixture(scope="module")
def noise_data():
    assert RESULT_PATH.exists(), f"{RESULT_PATH} not found — run blender/verify_noise_model.py first"
    return json.loads(RESULT_PATH.read_text())


def _stable_probe_indices(stable_valid):
    arr = np.array(stable_valid, dtype=bool)
    return list(zip(*np.where(arr)))


def _luminance_series(repeats_by_spp, spp, coords):
    arr = np.array(repeats_by_spp[str(spp)])  # (n_repeats, H, W, 3)
    lum = arr.mean(axis=-1)
    return np.stack([lum[:, r, c] for r, c in coords], axis=1)  # (n_repeats, n_probes)


def _report_std_scaling(label, repeats_by_spp, spp_levels, coords):
    print(f"\n--- {label} ---")
    reference_mean_mag = None
    for spp in spp_levels:
        series = _luminance_series(repeats_by_spp, spp, coords)
        if reference_mean_mag is None:
            reference_mean_mag = float(series.mean())
    stds = {}
    for spp in spp_levels:
        series = _luminance_series(repeats_by_spp, spp, coords)
        stds[spp] = float(np.median(series.std(axis=0, ddof=1)))
    print(f"{label} median std per spp: {stds}")
    scaled = {spp: stds[spp] * (spp ** 0.5) for spp in spp_levels}
    cv = float(np.std(list(scaled.values())) / np.mean(list(scaled.values())))
    monotonic = all(
        stds[spp_levels[i]] > stds[spp_levels[i + 1]] for i in range(len(spp_levels) - 1)
    )
    print(f"{label} std*sqrt(spp): {scaled}  (CV={cv:.3f}, monotonic_decreasing={monotonic})")
    mean_val = {spp: float(np.median(_luminance_series(repeats_by_spp, spp, coords).mean(axis=0)))
                for spp in spp_levels}
    print(f"{label} median mean-value per spp (для контекста амплитуды): {mean_val}")
    return stds, cv, monotonic


def test_direct_vs_indirect_std_scaling(noise_data):
    spp_levels = noise_data["spp_levels"]
    coords = _stable_probe_indices(noise_data["stable_valid"])
    print(f"\nStable probes: {len(coords)}")

    direct_stds, direct_cv, direct_mono = _report_std_scaling(
        "DIRECT", noise_data["repeats_by_spp_direct"], spp_levels, coords
    )
    indirect_stds, indirect_cv, indirect_mono = _report_std_scaling(
        "INDIRECT", noise_data["repeats_by_spp_indirect"], spp_levels, coords
    )

    print(f"\nSUMMARY: direct monotonic={direct_mono} (CV={direct_cv:.3f}), "
          f"indirect monotonic={indirect_mono} (CV={indirect_cv:.3f})")
    if direct_mono and not indirect_mono:
        print("=> Немонотонность объясняется INDIRECT-компонентой, direct ведёт себя по учебнику.")
    elif not direct_mono and not indirect_mono:
        print("=> Обе компоненты немонотонны — гипотеза о 'смешивании' не подтверждается, "
              "проблема глубже (сцена/сэмплер в целом), а не просто indirect.")
    elif direct_mono and indirect_mono:
        print("=> Обе компоненты монотонны по отдельности, но сумма — нет?! Требует отдельного "
              "разбора (например противофазная корреляция direct/indirect по seed).")

    for arr_name, d in [("direct", noise_data["repeats_by_spp_direct"]),
                         ("indirect", noise_data["repeats_by_spp_indirect"])]:
        for spp in spp_levels:
            series = _luminance_series(d, spp, coords)
            assert not np.isnan(series).any(), f"NaN in {arr_name} at spp={spp}"
