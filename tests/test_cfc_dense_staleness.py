"""
Тесты §2.6 (TZ_stage1b_per_pixel_dense_fallback.md): расширение
staleness-вектора CfCProbeModule под per-pixel dense путь —
[cold_start, confidence, disocclusion_flag, geometric_mismatch_score]
вместо probe-пути [cold_start, confidence].

Ключевая проверка: расширение НЕ ломает probe-путь (staleness_dim=2
по умолчанию, extra_staleness=None) — тот же код, что уже используют
run_experiment.py/run_experiment_blender.py, должен вести себя
побайтово так же, как до этого изменения (см. полный прогон tests/
72/72 после правки как первичное подтверждение, здесь — целевые
тесты именно на новую функциональность).
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from models.temporal.cfc_probe_module import CfCProbeModule  # noqa: E402


def _toy_batch(B=3, T=6, obs_dim=1):
    obs = torch.randn(B, T, obs_dim)
    dt = torch.rand(B, T).abs() + 0.1
    cold = torch.zeros(B, T)
    conf = torch.rand(B, T)
    return obs, dt, cold, conf


def test_probe_path_default_staleness_dim_unchanged():
    """staleness_dim по умолчанию всё ещё 2 — старые вызовы без
    extra_staleness дают ТОТ ЖЕ input_dim, что и до расширения."""
    model = CfCProbeModule(obs_dim=1, hidden_dim=8, use_staleness=True)
    assert model.staleness_dim == 2
    assert model.cell.backbone[0].in_features == 1 + 2 + 8  # obs+staleness+hidden


def test_probe_path_build_input_without_extra_staleness_matches_old_shape():
    obs, dt, cold, conf = _toy_batch()
    u = CfCProbeModule.build_input(obs, cold, conf, use_staleness=True)
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 2)


def test_dense_path_staleness_dim_four():
    obs, dt, cold, conf = _toy_batch()
    disocc_flag = torch.randint(0, 2, (obs.shape[0], obs.shape[1])).float()
    mismatch_score = torch.rand(obs.shape[0], obs.shape[1])

    model = CfCProbeModule(obs_dim=1, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True,
        extra_staleness=[disocc_flag, mismatch_score],
    )
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 4)
    assert model.cell.backbone[0].in_features == 1 + 4 + 8

    pred, h = model(u, dt)
    assert pred.shape == obs.shape
    assert h.shape == (obs.shape[0], 8)


def test_dense_path_channel_order_cold_conf_disocc_mismatch():
    """Порядок каналов в собранном u должен быть [obs, cold_start,
    confidence, disocclusion_flag, geometric_mismatch_score] — извлекаем
    obs_dim=1, значит staleness-часть — это u[..., 1:]."""
    obs = torch.zeros(1, 1, 1)
    cold = torch.tensor([[7.0]])
    conf = torch.tensor([[8.0]])
    disocc = torch.tensor([[9.0]])
    mismatch = torch.tensor([[10.0]])

    u = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
    )
    staleness_part = u[0, 0, 1:]
    assert torch.equal(staleness_part, torch.tensor([7.0, 8.0, 9.0, 10.0]))


def test_dense_path_gives_gradients_through_extended_staleness():
    """Sanity-check в том же духе, что test_gru_baseline.py: значения
    disocclusion_flag/mismatch_score реально доходят до backbone и
    получают полезный градиент, а не игнорируются архитектурой."""
    obs, dt, cold, conf = _toy_batch()
    disocc_flag = torch.randint(0, 2, (obs.shape[0], obs.shape[1])).float()
    mismatch_score = torch.rand(obs.shape[0], obs.shape[1])

    model = CfCProbeModule(obs_dim=1, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True,
        extra_staleness=[disocc_flag, mismatch_score],
    )
    pred, _ = model(u, dt)
    pred.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_dense_path_output_differs_when_disocclusion_flag_flips():
    """Если disocclusion_flag реально используется сетью, одинаковый obs
    с разным флагом должен давать разный выход (случайная инициализация
    — sanity-check доходимости сигнала, не архитектурная гарантия, тот
    же приём, что test_faithful_vs_honest_differ_when_dt_varies в
    test_gru_baseline.py)."""
    obs, dt, cold, conf = _toy_batch()
    mismatch_score = torch.rand(obs.shape[0], obs.shape[1])
    flag_off = torch.zeros(obs.shape[0], obs.shape[1])
    flag_on = torch.ones(obs.shape[0], obs.shape[1])

    model = CfCProbeModule(obs_dim=1, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u_off = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[flag_off, mismatch_score],
    )
    u_on = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[flag_on, mismatch_score],
    )
    with torch.no_grad():
        pred_off, _ = model(u_off, dt)
        pred_on, _ = model(u_on, dt)
    assert not torch.allclose(pred_off, pred_on)


def test_extra_staleness_none_is_backward_compatible_with_old_two_dim_call():
    """Явная регрессионная проверка: вызов ровно в том виде, в каком его
    делают run_experiment.py/run_experiment_blender.py (без extra_staleness
    вообще, как позиционные/keyword аргументы), продолжает работать."""
    obs, dt, cold, conf = _toy_batch()
    model = CfCProbeModule(obs_dim=1, hidden_dim=16, use_staleness=True)
    u = model.build_input(obs, cold, conf, use_staleness=True)
    pred, h = model(u, dt)
    assert pred.shape == obs.shape
    assert h.shape == (obs.shape[0], 16)
