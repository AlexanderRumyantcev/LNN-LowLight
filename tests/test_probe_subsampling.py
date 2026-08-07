"""
Юнит-тест на blender/probe_subsampling.py — чистый python/numpy, без
Blender (как test_light_schedule.py и test_moment_pool.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from moment_pool import MomentPoolConfig, build_moment_pool
from probe_subsampling import (
    ProbeSubsampleConfig, sample_probe_indices, build_probe_sequence,
)


def _pool(total_duration=100.0, seed=1):
    return build_moment_pool(MomentPoolConfig(total_duration=total_duration, seed=seed))


def test_indices_start_at_zero_and_strictly_increase():
    pool_t = _pool()
    cfg = ProbeSubsampleConfig(seed=1)
    idx = sample_probe_indices(pool_t, cfg)
    assert idx[0] == 0
    assert all(b > a for a, b in zip(idx, idx[1:]))


def test_indices_within_pool_bounds():
    pool_t = _pool()
    cfg = ProbeSubsampleConfig(seed=2)
    idx = sample_probe_indices(pool_t, cfg)
    assert min(idx) >= 0
    assert max(idx) < len(pool_t)


def test_different_probes_get_different_async_streams():
    """Ключевое свойство (c): разные пробы -> разные независимые Δt-потоки,
    не совпадающие индексы пула (не единая общая последовательность)."""
    pool_t = _pool()
    idx_a = sample_probe_indices(pool_t, ProbeSubsampleConfig(seed=10))
    idx_b = sample_probe_indices(pool_t, ProbeSubsampleConfig(seed=11))
    assert idx_a != idx_b


def test_quantized_dt_within_reasonable_bound_of_configured_range():
    """dt между выбранными quantized t не обязан точно попадать в
    cfg.dt_range (снэппинг к пулу может слегка сдвинуть), но не должен
    систематически уходить далеко за верхнюю границу + плотность пула."""
    pool_t = _pool(total_duration=300.0, seed=1)
    cfg = ProbeSubsampleConfig(dt_range=(3.0, 8.0), seed=3)
    idx = sample_probe_indices(pool_t, cfg)
    t = np.asarray(pool_t)[idx]
    dt = np.diff(t)
    pool_max_gap = 2.0  # MomentPoolConfig default dt_range upper bound
    assert (dt <= cfg.dt_range[1] * cfg.dt_spike_multiplier_range[1] + pool_max_gap).all()


def test_build_probe_sequence_shapes_and_cold_start():
    pool_t = _pool(total_duration=100.0, seed=1)
    n = len(pool_t)
    probe_series = dict(
        obs=np.linspace(1.0, 2.0, n).astype(np.float32),
        spp=np.full(n, 64.0, dtype=np.float32),
    )
    cfg = ProbeSubsampleConfig(seed=4)
    seq = build_probe_sequence(pool_t, probe_series, cfg)

    L = len(seq["t"])
    assert seq["dt"].shape == (L,)
    assert seq["cold_start"].shape == (L,)
    assert seq["obs"].shape == (L,)
    assert seq["spp"].shape == (L,)
    assert seq["dt"][0] == 0.0
    assert seq["cold_start"][0] == 1.0
    assert seq["cold_start"][1:].sum() == 0.0
    assert seq["t"][0] == pytest.approx(pool_t[0])


def test_build_probe_sequence_values_match_selected_indices():
    pool_t = _pool(total_duration=100.0, seed=1)
    n = len(pool_t)
    probe_series = dict(obs=np.arange(n, dtype=np.float32))
    cfg = ProbeSubsampleConfig(seed=5)
    seq = build_probe_sequence(pool_t, probe_series, cfg)

    # obs[k] должен совпадать со значением исходного массива по pool_index[k]
    for k, pool_idx in enumerate(seq["pool_index"]):
        assert seq["obs"][k] == probe_series["obs"][pool_idx]


def test_deterministic_given_same_seed():
    pool_t = _pool()
    cfg = ProbeSubsampleConfig(seed=7)
    a = sample_probe_indices(pool_t, cfg)
    b = sample_probe_indices(pool_t, cfg)
    assert a == b


def test_build_probe_sequence_handles_rgb_vector_obs():
    """РЕШЕНИЕ (дыра №3): irradiance/true_irradiance остаются RGB-тройками,
    не схлопываются в скаляр — probe_series['obs'] здесь [n_pool, 3], как
    реально приходит из generate_dataset.py."""
    pool_t = _pool(total_duration=100.0, seed=1)
    n = len(pool_t)
    rng = np.random.default_rng(0)
    probe_series = dict(
        obs=rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32),
        true_irradiance=rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32),
    )
    cfg = ProbeSubsampleConfig(seed=6)
    seq = build_probe_sequence(pool_t, probe_series, cfg)

    L = len(seq["t"])
    assert seq["obs"].shape == (L, 3)
    assert seq["true_irradiance"].shape == (L, 3)
    # значения по-прежнему должны совпадать с исходным массивом по pool_index
    for k, pool_idx in enumerate(seq["pool_index"]):
        assert np.array_equal(seq["obs"][k], probe_series["obs"][pool_idx])
