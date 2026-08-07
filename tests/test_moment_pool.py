"""
Юнит-тест на blender/moment_pool.py — чистый python/numpy, без Blender
(как и tests/test_light_schedule.py).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from moment_pool import MomentPoolConfig, build_moment_pool


def test_pool_starts_at_zero_and_stays_within_duration():
    cfg = MomentPoolConfig(total_duration=50.0, seed=1)
    pool = build_moment_pool(cfg)
    assert pool[0] == 0.0
    assert all(0.0 <= t < cfg.total_duration for t in pool)


def test_pool_strictly_increasing():
    cfg = MomentPoolConfig(total_duration=50.0, seed=1)
    pool = build_moment_pool(cfg)
    assert all(b > a for a, b in zip(pool, pool[1:]))


def test_pool_gaps_within_configured_dt_range():
    cfg = MomentPoolConfig(total_duration=100.0, dt_range=(1.0, 2.0), seed=2)
    pool = build_moment_pool(cfg)
    gaps = np.diff(pool)
    assert (gaps >= cfg.dt_range[0] - 1e-9).all()
    assert (gaps <= cfg.dt_range[1] + 1e-9).all()


def test_pool_denser_than_per_probe_dt_range():
    """Ключевое требование докстринга: пул должен быть плотнее per-probe
    dt_range=(0.5, 2.0) из synthetic_probe_scene.py, иначе (c) нечего
    подвыбирать."""
    cfg = MomentPoolConfig()
    per_probe_dt_range = (0.5, 2.0)
    assert cfg.dt_range[1] <= per_probe_dt_range[1]


def test_pool_count_matches_total_duration_and_density():
    cfg = MomentPoolConfig(total_duration=100.0, dt_range=(1.0, 1.0), seed=3)
    pool = build_moment_pool(cfg)
    # dt фиксирован в 1.0 -> ровно 100 моментов (0.0, 1.0, ..., 99.0)
    assert len(pool) == 100


def test_deterministic_given_same_seed():
    cfg = MomentPoolConfig(total_duration=50.0, seed=5)
    a = build_moment_pool(cfg)
    b = build_moment_pool(cfg)
    assert a == b
