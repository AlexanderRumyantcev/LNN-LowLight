"""
§3.2(Б) — статистический анализ собранной партии реального мульти-spp
рендеринга (blender/verify_noise_model_result.json, собран в предыдущем
шаге blender/verify_noise_model.py).

Проверяет три пункта, которые синтетическая формула std=sqrt(C·value/spp)
не может подтвердить сама на себе (§3.2 ТЗ):
  (1) гауссовость шума на каждом spp-уровне
  (2) монотонность std(spp) и согласие с масштабированием ~1/sqrt(spp)
  (3) пространственная независимость шума соседних проб

НЕ проверяется (данные для этого не собирались в этом батче, см. находку
ниже): временная корреляция между кадрами — вся партия снята на ОДНОМ
статичном кадре (t=0, свет не двигается), поэтому "между кадрами"
здесь просто нет. Это ограничение самого batch'а, а не результат "temporal
корреляции нет" — вопрос остаётся открытым, если понадобится когда-либо.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
RESULT_PATH = PROJECT_ROOT / "blender" / "verify_noise_model_result.json"


@pytest.fixture(scope="module")
def noise_data():
    assert RESULT_PATH.exists(), f"{RESULT_PATH} not found — run blender/verify_noise_model.py first"
    return json.loads(RESULT_PATH.read_text())


def _stable_probe_indices(stable_valid):
    """(row, col) координаты проб, стабильно валидных на всех spp/повторах."""
    arr = np.array(stable_valid, dtype=bool)
    return list(zip(*np.where(arr)))


def _luminance_series(repeats_by_spp, spp, coords):
    """(n_repeats, n_probes) — luminance (среднее по RGB) каждой стабильной
    пробы по всем повторам данного spp-уровня."""
    arr = np.array(repeats_by_spp[str(spp)])  # (n_repeats, H, W, 3)
    lum = arr.mean(axis=-1)  # (n_repeats, H, W)
    return np.stack([lum[:, r, c] for r, c in coords], axis=1)  # (n_repeats, n_probes)


def test_gaussianity_and_std_scaling(noise_data):
    spp_levels = noise_data["spp_levels"]
    coords = _stable_probe_indices(noise_data["stable_valid"])
    n_probes = len(coords)
    print(f"\nStable probes for analysis: {n_probes} / {len(coords) + noise_data['n_dropped_for_instability']}")

    # --- (1) Гауссовость: Shapiro-Wilk per probe per spp-уровень ---
    reject_fraction = {}
    skew_median = {}
    kurt_median = {}
    for spp in spp_levels:
        series = _luminance_series(noise_data["repeats_by_spp"], spp, coords)
        assert not np.isnan(series).any(), f"NaN in samples at spp={spp}"
        pvals = [stats.shapiro(series[:, p])[1] for p in range(n_probes)]
        skews = [stats.skew(series[:, p]) for p in range(n_probes)]
        kurts = [stats.kurtosis(series[:, p]) for p in range(n_probes)]  # excess (0=gauss)
        reject_fraction[spp] = float(np.mean(np.array(pvals) < 0.05))
        skew_median[spp] = float(np.median(skews))
        kurt_median[spp] = float(np.median(kurts))
        print(
            f"spp={spp:>3}: Shapiro reject@0.05 = {reject_fraction[spp]:.2f} of probes, "
            f"median skew={skew_median[spp]:+.3f}, median excess kurtosis={kurt_median[spp]:+.3f}"
        )

    # --- (2) Монотонность std(spp) и согласие с ~1/sqrt(spp) ---
    stds = {}
    for spp in spp_levels:
        series = _luminance_series(noise_data["repeats_by_spp"], spp, coords)
        stds[spp] = series.std(axis=0, ddof=1)  # (n_probes,) — std по повторам, per probe

    median_std = {spp: float(np.median(stds[spp])) for spp in spp_levels}
    print(f"\nmedian std per spp: {median_std}")
    monotonic_decreasing = all(
        median_std[spp_levels[i]] > median_std[spp_levels[i + 1]]
        for i in range(len(spp_levels) - 1)
    )
    print(f"std(spp) monotonically decreasing: {monotonic_decreasing}")

    # std * sqrt(spp) должно быть ~constant, если шум масштабируется как 1/sqrt(spp)
    # (стандартное поведение Monte Carlo при отсутствии adaptive/denoise-искажений)
    scaled = {spp: median_std[spp] * (spp ** 0.5) for spp in spp_levels}
    print(f"median std * sqrt(spp) per spp (должно быть ~constant): {scaled}")
    scaled_values = list(scaled.values())
    cv = float(np.std(scaled_values) / np.mean(scaled_values))
    print(f"coefficient of variation of std*sqrt(spp) across levels: {cv:.3f}")

    # --- (3) Пространственная корреляция шума соседних проб ---
    residuals = {}
    for spp in spp_levels:
        series = _luminance_series(noise_data["repeats_by_spp"], spp, coords)
        residuals[spp] = series - series.mean(axis=0, keepdims=True)  # (n_repeats, n_probes)

    coord_to_idx = {c: i for i, c in enumerate(coords)}
    adjacent_pairs = []
    for (r, c) in coords:
        for nr, nc in [(r + 1, c), (r, c + 1)]:
            if (nr, nc) in coord_to_idx:
                adjacent_pairs.append((coord_to_idx[(r, c)], coord_to_idx[(nr, nc)]))
    print(f"\nAdjacent probe pairs found: {len(adjacent_pairs)}")

    rng = np.random.default_rng(0)
    for spp in spp_levels:
        res = residuals[spp]  # (n_repeats, n_probes)
        adj_corrs = [np.corrcoef(res[:, i], res[:, j])[0, 1] for i, j in adjacent_pairs]
        # контроль: столько же случайных (не обязательно смежных) пар
        idx_a = rng.integers(0, n_probes, size=len(adjacent_pairs))
        idx_b = rng.integers(0, n_probes, size=len(adjacent_pairs))
        random_corrs = [
            np.corrcoef(res[:, i], res[:, j])[0, 1] for i, j in zip(idx_a, idx_b) if i != j
        ]
        print(
            f"spp={spp:>3}: median |corr| adjacent={np.median(np.abs(adj_corrs)):.3f}, "
            f"median |corr| random={np.median(np.abs(random_corrs)):.3f}"
        )
