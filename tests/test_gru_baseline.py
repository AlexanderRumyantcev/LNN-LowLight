"""
Тесты на GRUProbeBaseline (models/baselines.py) — 4-й baseline, discrete GRU-Δt.
По конвенции проекта (см. mempalace, реализация NRD/NRC): формы корректны,
честная версия даёт градиенты по параметрам, faithful vs honest действительно
разные (иначе staleness-фича не используется).
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines import GRUProbeBaseline  # noqa: E402


def _toy_batch(B=3, T=10, obs_dim=1):
    obs = torch.randn(B, T, obs_dim)
    dt = torch.rand(B, T).abs() + 0.1
    cold = torch.zeros(B, T)
    conf = torch.rand(B, T)
    return obs, dt, cold, conf


def test_faithful_shapes_and_no_dt_dependence():
    obs, dt, cold, conf = _toy_batch()
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=False)
    u = GRUProbeBaseline.build_input(obs, use_dt_staleness=False)
    assert u.shape == obs.shape  # faithful: вход = только obs, без doп. фич
    pred, h = model(u)
    assert pred.shape == obs.shape
    assert h.shape == (obs.shape[0], 8)


def test_honest_shapes_and_input_dim():
    obs, dt, cold, conf = _toy_batch(obs_dim=3)  # RGB-путь тоже должен работать
    model = GRUProbeBaseline(obs_dim=3, hidden_dim=8, use_dt_staleness=True)
    u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True)
    assert u.shape == (obs.shape[0], obs.shape[1], 3 + 3)  # obs + [log1p(dt), cold, conf]
    pred, h = model(u)
    assert pred.shape == obs.shape


def test_honest_requires_dt_staleness():
    obs, dt, cold, conf = _toy_batch()
    with pytest.raises(ValueError):
        GRUProbeBaseline.build_input(obs, use_dt_staleness=True)  # dt/cold/conf не переданы


def test_honest_gives_gradients():
    obs, dt, cold, conf = _toy_batch()
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True)
    u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True)
    pred, _ = model(u)
    pred.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_faithful_vs_honest_differ_when_dt_varies():
    """Если honest-версия действительно использует Δt, при разных Δt-потоках
    (одинаковый obs, разные dt) её выход должен отличаться — иначе staleness-
    фича на входе фактически игнорируется сетью (не архитектурная гарантия,
    но sanity-check, что вход реально доходит до выхода на случайной инициализации)."""
    obs, dt1, cold, conf = _toy_batch()
    dt2 = dt1 * 5.0 + 1.0
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True)
    u1 = GRUProbeBaseline.build_input(obs, dt1, cold, conf, use_dt_staleness=True)
    u2 = GRUProbeBaseline.build_input(obs, dt2, cold, conf, use_dt_staleness=True)
    with torch.no_grad():
        pred1, _ = model(u1)
        pred2, _ = model(u2)
    assert not torch.allclose(pred1, pred2)
