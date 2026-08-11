"""
Тесты §4.2/§4.3 (TZ_stage1b_per_pixel_dense_fallback.md): адаптация
baseline'ов (models/baselines.py) под симметричный 4-мерный
staleness-вектор dense-пути.

Как и test_cfc_dense_staleness.py — ключевая проверка нерегрессии:
дефолты новых параметров воспроизводят СТАРОЕ поведение probe-пути
(полный набор tests/ 88/88 после правки — первичное подтверждение,
здесь — целевые тесты на новую функциональность).
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines import NRDStyleBaseline, NRCStyleBaseline, GRUProbeBaseline  # noqa: E402


def _toy(B=3, T=6, obs_dim=1):
    obs = torch.randn(B, T, obs_dim)
    dt = torch.rand(B, T).abs() + 0.1
    cold = torch.zeros(B, T)
    conf = torch.rand(B, T)
    disocc = torch.randint(0, 2, (B, T)).float()
    mismatch = torch.rand(B, T)
    return obs, dt, cold, conf, disocc, mismatch


# --- NRDStyleBaseline -------------------------------------------------


def test_nrd_faithful_dense_default_off_matches_old_behavior():
    """Без use_disocclusion_reset/use_mismatch_honest — тот же forward,
    что и до правки (регрессия проверяется полным прогоном tests/, тут
    просто явная проверка формы и отсутствия падения)."""
    obs, dt, *_ = _toy()
    model = NRDStyleBaseline(alpha=0.3, use_honest_dt=False)
    out = model(obs, dt)
    assert out.shape == obs.shape


def test_nrd_faithful_dense_hard_resets_on_disocclusion():
    """§4.2: faithful dense — жёсткий сброс out_t=obs_t на disocclusion_flag=1,
    без подмешивания continuous score."""
    obs, dt, *_ = _toy(B=1, T=4, obs_dim=1)
    disocc = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    model = NRDStyleBaseline(alpha=0.3, use_honest_dt=False, use_disocclusion_reset=True)
    out = model(obs, dt, disocclusion_flag_seq=disocc)
    assert torch.allclose(out[0, 2], obs[0, 2]), "hard reset expected exactly at t=2"


def test_nrd_faithful_dense_requires_disocclusion_seq():
    obs, dt, *_ = _toy()
    model = NRDStyleBaseline(use_disocclusion_reset=True)
    try:
        model(obs, dt)
        assert False, "expected ValueError without disocclusion_flag_seq"
    except ValueError:
        pass


def test_nrd_honest_dense_blends_continuous_mismatch_not_hard_reset():
    """§4.2 honest: geometric_mismatch подмешивается в alpha МЯГКО (max),
    а не жёстким reset — при score=1.0 фактически совпадает с полным
    reset (alpha=1), при малом score почти не влияет."""
    obs, dt, *_ = _toy(B=1, T=4, obs_dim=1)
    mismatch = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    model = NRDStyleBaseline(alpha=0.3, use_honest_dt=False, use_mismatch_honest=True)
    out = model(obs, dt, geometric_mismatch_seq=mismatch)
    # при mismatch=1.0 (t=2): alpha=max(0.3,1.0)=1.0 -> out=obs ровно как full reset
    assert torch.allclose(out[0, 2], obs[0, 2])


def test_nrd_honest_and_faithful_agree_at_score_one_but_differ_elsewhere():
    """При score=1.0 оба режима дают одинаковый (полный reset) результат
    на этом шаге, но НЕ идентичны в остальных точках последовательности,
    т.к. honest продолжает continuous-влиять на alpha, а faithful — нет."""
    obs, dt, *_ = _toy(B=1, T=5, obs_dim=1)
    disocc = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    mismatch = torch.tensor([[0.0, 1.0, 0.6, 0.0, 0.0]])
    faithful = NRDStyleBaseline(alpha=0.3, use_disocclusion_reset=True)
    honest = NRDStyleBaseline(alpha=0.3, use_mismatch_honest=True)
    out_f = faithful(obs, dt, disocclusion_flag_seq=disocc)
    out_h = honest(obs, dt, geometric_mismatch_seq=mismatch)
    assert torch.allclose(out_f[0, 1], obs[0, 1])
    assert torch.allclose(out_h[0, 1], obs[0, 1])
    assert not torch.allclose(out_f[0, 2], out_h[0, 2]), (
        "faithful (no reset at t=2) and honest (mismatch=0.6 still pulls alpha up) "
        "should diverge once the hard trigger is gone"
    )


# --- NRCStyleBaseline ---------------------------------------------------


def test_nrc_default_staleness_dim_unchanged():
    model = NRCStyleBaseline(obs_dim=1, hidden_dim=8, use_staleness=True)
    assert model.staleness_dim == 2
    assert model.mlp[0].in_features == 1 + 2


def test_nrc_build_input_without_extra_matches_old_shape():
    obs, dt, cold, conf, *_ = _toy()
    u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True)
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 2)


def test_nrc_dense_staleness_dim_four():
    obs, dt, cold, conf, disocc, mismatch = _toy()
    model = NRCStyleBaseline(obs_dim=1, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u = NRCStyleBaseline.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
    )
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 4)
    out = model(u)
    assert out.shape == obs.shape


# --- GRUProbeBaseline ----------------------------------------------------


def test_gru_default_extra_staleness_dim_unchanged():
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True)
    assert model.extra_staleness_dim == 0
    assert model.cell.weight_ih.shape[1] == 1 + 3  # obs + [dt,cold,conf]


def test_gru_build_input_without_extra_matches_old_shape():
    obs, dt, cold, conf, *_ = _toy()
    u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True)
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 3)


def test_gru_dense_extra_staleness_dim_two():
    obs, dt, cold, conf, disocc, mismatch = _toy()
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True, extra_staleness_dim=2)
    u = GRUProbeBaseline.build_input(
        obs, dt, cold, conf, use_dt_staleness=True, extra_staleness=[disocc, mismatch],
    )
    assert u.shape == (obs.shape[0], obs.shape[1], 1 + 5)  # obs+[dt,cold,conf,disocc,mismatch]
    pred, h = model(u)
    assert pred.shape == obs.shape
    assert h.shape == (obs.shape[0], 8)


def test_gru_dense_gradients_flow_through_extra_staleness():
    obs, dt, cold, conf, disocc, mismatch = _toy()
    model = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True, extra_staleness_dim=2)
    u = GRUProbeBaseline.build_input(
        obs, dt, cold, conf, use_dt_staleness=True, extra_staleness=[disocc, mismatch],
    )
    pred, _ = model(u)
    pred.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


# --- Симметрия staleness-доступа между всеми тремя честными baseline'ами и CfC-B ----


def test_all_honest_baselines_accept_same_four_channel_vector_shape():
    """§4.2: смысл честного сравнения — все получают ОДИН И ТОТ ЖЕ
    состав staleness-вектора. Явная проверка, что все три (NRC-style,
    GRU-Δt, и NRD-style через свой forward) реально принимают один и
    тот же набор сигналов без ошибок формы."""
    obs, dt, cold, conf, disocc, mismatch = _toy(obs_dim=1)

    nrc = NRCStyleBaseline(obs_dim=1, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u_nrc = NRCStyleBaseline.build_input(obs, cold, conf, True, extra_staleness=[disocc, mismatch])
    assert nrc(u_nrc).shape == obs.shape

    gru = GRUProbeBaseline(obs_dim=1, hidden_dim=8, use_dt_staleness=True, extra_staleness_dim=2)
    u_gru = GRUProbeBaseline.build_input(obs, dt, cold, conf, True, extra_staleness=[disocc, mismatch])
    pred, _ = gru(u_gru)
    assert pred.shape == obs.shape

    nrd = NRDStyleBaseline(use_mismatch_honest=True)
    out_nrd = nrd(obs, dt, geometric_mismatch_seq=mismatch)
    assert out_nrd.shape == obs.shape
