"""
Юнит-тест на blender/camera_path.py — ЧИСТЫЙ python/numpy, без Blender
(как и test_light_schedule.py). Запускается системным .venv-python.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from camera_path import CameraPathConfig, camera_pose_at


def test_camera_stays_on_circle_of_configured_radius():
    cfg = CameraPathConfig()
    t_query = np.linspace(0.0, cfg.total_duration, 200)
    for t in t_query:
        pos, target = camera_pose_at(cfg, float(t))
        cx, cy, cz = cfg.center
        planar_dist = np.hypot(pos[0] - cx, pos[1] - cy)
        assert planar_dist == pytest.approx(cfg.radius, rel=1e-6)
        assert pos[2] == pytest.approx(cz + cfg.height, rel=1e-6)

def test_look_at_target_is_always_scene_center():
    cfg = CameraPathConfig()
    for t in [0.0, 50.0, 123.4, cfg.total_duration]:
        _, target = camera_pose_at(cfg, t)
        assert np.allclose(target, np.array(cfg.center))


def test_completes_configured_number_of_revolutions():
    """theta(total_duration) - theta(0) должно быть ровно n_revolutions полных оборотов."""
    cfg = CameraPathConfig(n_revolutions=3.0, start_angle_deg=0.0)
    pos_start, _ = camera_pose_at(cfg, 0.0)
    pos_end, _ = camera_pose_at(cfg, cfg.total_duration)
    # после ровно N целых оборотов позиция должна вернуться в исходную точку
    assert np.allclose(pos_start, pos_end, atol=1e-6)


def test_deterministic_given_same_config():
    cfg = CameraPathConfig()
    for t in [0.0, 77.7, 400.0]:
        pos_a, target_a = camera_pose_at(cfg, t)
        pos_b, target_b = camera_pose_at(cfg, t)
        assert np.allclose(pos_a, pos_b)
        assert np.allclose(target_a, target_b)

def test_start_angle_offsets_initial_position():
    cfg_a = CameraPathConfig(start_angle_deg=0.0)
    cfg_b = CameraPathConfig(start_angle_deg=90.0)
    pos_a, _ = camera_pose_at(cfg_a, 0.0)
    pos_b, _ = camera_pose_at(cfg_b, 0.0)
    assert not np.allclose(pos_a, pos_b)


def test_position_is_continuous_over_t():
    """Малое приращение t не должно давать скачок позиции (гладкая траектория,
    в отличие от light_schedule со STEP-сегментами)."""
    cfg = CameraPathConfig()
    t0 = 200.0
    pos0, _ = camera_pose_at(cfg, t0)
    pos1, _ = camera_pose_at(cfg, t0 + 1e-3)
    assert np.allclose(pos0, pos1, atol=1e-2)
