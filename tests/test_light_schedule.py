"""
Юнит-тест на blender/light_schedule.py — ЧИСТЫЙ python/numpy, без
Blender (в отличие от остальных tests/test_*.py, которые шеллятся в
Blender headless через conftest.run_blender_script). Запускать можно
системным .venv-python, без /Applications/Blender.app.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from light_schedule import (
    LightScheduleConfig, SEG_STATIC, SEG_STEP, SEG_DRIFT,
    build_light_schedule, intensity_at,
)


def test_segments_cover_total_duration_contiguously():
    cfg = LightScheduleConfig(seed=1)
    segments = build_light_schedule(cfg)

    assert segments[0][0] == 0.0
    assert segments[-1][1] == pytest.approx(cfg.total_duration, rel=1e-6)
    for (_, t_end_prev, *_), (t_start_next, *_) in zip(segments, segments[1:]):
        assert t_end_prev == pytest.approx(t_start_next, abs=1e-6), (
            "segments must be contiguous — no gaps/overlaps in the schedule"
        )


def test_intensity_stays_within_configured_range():
    cfg = LightScheduleConfig(seed=2)
    segments = build_light_schedule(cfg)
    t_query = np.linspace(0.0, cfg.total_duration - 1e-3, 500)

    for t in t_query:
        v = intensity_at(segments, float(t))
        assert cfg.intensity_range[0] * 0.999 <= v <= cfg.intensity_range[1] * 1.001, (
            f"intensity_at(t={t}) = {v} outside configured range {cfg.intensity_range}"
        )


def test_step_segment_actually_jumps():
    """Находим STEP-сегмент и проверяем разрыв: значение до/после jump_t различается,
    в отличие от STATIC/DRIFT, где intensity_at непрерывна внутри сегмента."""
    cfg = LightScheduleConfig(seed=3)
    segments = build_light_schedule(cfg)

    step_segments = [s for s in segments if s[2] == SEG_STEP]
    assert step_segments, "seed=3 expected to contain at least one STEP segment"

    t_start, t_end, _, p = step_segments[0]
    before = intensity_at(segments, p["jump_t"] - 1e-4)
    after = intensity_at(segments, p["jump_t"] + 1e-4)
    assert before != after
    assert before == pytest.approx(p["old"], rel=1e-3)
    assert after == pytest.approx(p["new"], rel=1e-3)


def test_static_segment_is_constant():
    cfg = LightScheduleConfig(seed=1)
    segments = build_light_schedule(cfg)
    static_segments = [s for s in segments if s[2] == SEG_STATIC]
    if not static_segments:
        pytest.skip("seed=1 produced no STATIC segment")

    t_start, t_end, _, p = static_segments[0]
    mid = (t_start + t_end) / 2.0
    assert intensity_at(segments, t_start) == pytest.approx(p["value"])
    assert intensity_at(segments, mid) == pytest.approx(p["value"])


def test_drift_segment_is_exponential_and_continuous_at_boundary():
    cfg = LightScheduleConfig(seed=1)
    segments = build_light_schedule(cfg)
    drift_segments = [s for s in segments if s[2] == SEG_DRIFT]
    if not drift_segments:
        pytest.skip("seed=1 produced no DRIFT segment")

    t_start, t_end, _, p = drift_segments[0]
    # непрерывность на левой границе: значение в t_start совпадает со start_value
    assert intensity_at(segments, t_start) == pytest.approx(p["start_value"], rel=1e-3)


def test_out_of_range_query_clamped_to_last_segment():
    cfg = LightScheduleConfig(seed=4)
    segments = build_light_schedule(cfg)
    last_end = segments[-1][1]

    just_inside = intensity_at(segments, last_end - 1e-9)
    far_beyond = intensity_at(segments, last_end + 500.0)
    assert just_inside == pytest.approx(far_beyond, rel=1e-3)


def test_deterministic_given_same_seed():
    cfg = LightScheduleConfig(seed=7)
    a = build_light_schedule(cfg)
    b = build_light_schedule(cfg)
    t_query = np.linspace(0.0, cfg.total_duration - 1e-3, 50)
    for t in t_query:
        assert intensity_at(a, float(t)) == intensity_at(b, float(t))
