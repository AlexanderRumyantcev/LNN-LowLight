"""
Smoke test — проверяет что весь стек поднимается и forward pass проходит.

Запуск:
  python smoke_test.py

Тестирует:
  1. Импорт всех модулей
  2. Forward pass pipeline для всех temporal_type (cfc, lstm, gru, none)
  3. Loss backward (градиенты текут)
  4. Параметры модели
  5. MPS / CPU device
"""

import sys
import traceback
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")

passed = 0
failed = 0

def check(name, fn):
    global passed, failed
    print(f"\n[{name}]")
    try:
        fn()
        passed += 1
    except Exception as e:
        fail(str(e))
        traceback.print_exc()
        failed += 1

# ──────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────

def test_imports():
    import models
    ok("models")
    from models.pipeline import RetinexLNNPipeline
    ok("RetinexLNNPipeline")
    from models.temporal import CfCTemporalModule, LSTMTemporalModule, GRUTemporalModule
    ok("CfCTemporalModule, LSTMTemporalModule, GRUTemporalModule")
    from losses.combined_loss import CombinedLoss
    ok("CombinedLoss")

check("1. Imports", test_imports)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"\n  Device: {DEVICE}")

# ──────────────────────────────────────────────
# Synthetic batch
# ──────────────────────────────────────────────

B, T, C, H, W = 2, 5, 3, 128, 128
N_FEAT = 32  # должен быть кратен 4

def make_batch():
    frames = torch.rand(B, T, C, H, W, device=DEVICE)
    target = torch.rand(B, C, H, W, device=DEVICE)
    timespans = torch.full((B, T), 1.0 / 30.0, device=DEVICE)
    return frames, target, timespans

# ──────────────────────────────────────────────
# 2. Forward pass — все temporal types
# ──────────────────────────────────────────────

from models.pipeline import RetinexLNNPipeline
from losses.combined_loss import CombinedLoss

def test_forward(temporal_type):
    model = RetinexLNNPipeline(
        temporal_type=temporal_type,
        n_feat=N_FEAT,
        stage=1,
        num_blocks=[1, 1, 1],
        window_size=T,
        hidden_dim=64,
        n_neurons=32,
        fps=30.0,
    ).to(DEVICE)

    frames, target, timespans = make_batch()

    ts = timespans if temporal_type == 'cfc' else None
    with torch.no_grad():
        out = model(frames, ts)

    assert out.shape == (B, C, H, W), f"Expected {(B,C,H,W)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"

    params = model.get_num_params()
    ok(f"temporal_type='{temporal_type}' | shape={tuple(out.shape)} | params={params}")

check("2a. Forward [cfc]",      lambda: test_forward('cfc'))
check("2b. Forward [lstm]",     lambda: test_forward('lstm'))
check("2c. Forward [gru]",      lambda: test_forward('gru'))
check("2d. Forward [none]",     lambda: test_forward('none'))

# ──────────────────────────────────────────────
# 3. Backward pass (градиенты текут)
# ──────────────────────────────────────────────

def test_backward():
    model = RetinexLNNPipeline(
        temporal_type='cfc',
        n_feat=N_FEAT,
        window_size=T,
        hidden_dim=64,
        n_neurons=32,
    ).to(DEVICE)
    criterion = CombinedLoss(lambda_pixel=1.0, lambda_ssim=0.1, lambda_temporal=0.0)

    frames, target, timespans = make_batch()
    frames.requires_grad_(False)

    pred = model(frames, timespans)
    losses = criterion(pred, target)
    losses['total'].backward()

    # Проверяем что у части параметров есть градиенты
    grad_params = [p for p in model.parameters() if p.grad is not None]
    assert len(grad_params) > 0, "No gradients at all!"

    # Проверяем нет NaN градиентов
    nan_grads = [p for p in grad_params if torch.isnan(p.grad).any()]
    assert len(nan_grads) == 0, f"{len(nan_grads)} params have NaN gradients!"

    ok(f"loss={losses['total'].item():.4f} | params with grad: {len(grad_params)}")

check("3. Backward [cfc]", test_backward)

# ──────────────────────────────────────────────
# 4. Loss components
# ──────────────────────────────────────────────

def test_loss():
    criterion = CombinedLoss(lambda_pixel=1.0, lambda_ssim=0.1, lambda_temporal=0.05)
    pred   = torch.rand(B, C, H, W, device=DEVICE)
    target = torch.rand(B, C, H, W, device=DEVICE)

    losses = criterion(pred, target)
    for k, v in losses.items():
        assert not torch.isnan(v), f"NaN in loss[{k}]"
    ok(f"pixel={losses['pixel'].item():.4f} | ssim={losses['ssim'].item():.4f} | total={losses['total'].item():.4f}")

    # С temporal loss
    prev  = torch.rand(B, C, H, W, device=DEVICE)
    lprev = torch.rand(B, C, H, W, device=DEVICE)
    lcurr = torch.rand(B, C, H, W, device=DEVICE)
    losses2 = criterion(pred, target, prev, lprev, lcurr)
    ok(f"temporal={losses2['temporal'].item():.4f}")

check("4. Loss components", test_loss)

# ──────────────────────────────────────────────
# 5. Temporal modules в изоляции
# ──────────────────────────────────────────────

def test_temporal_module(cls, name):
    from models.temporal import CfCTemporalModule, LSTMTemporalModule, GRUTemporalModule
    mod = cls(in_channels=N_FEAT, hidden_dim=64, n_neurons=32, window_size=T).to(DEVICE)
    # [B, T, C, H, W]
    seq = torch.rand(B, T, N_FEAT, H // 4, W // 4, device=DEVICE)  # меньше для скорости
    ts  = torch.full((B, T), 1/30., device=DEVICE) if name == 'CfC' else None
    out = mod(seq, ts)
    assert out.shape == (B, N_FEAT, H // 4, W // 4), f"Bad shape: {out.shape}"
    assert not torch.isnan(out).any()
    ok(f"{name}: in={tuple(seq.shape)} → out={tuple(out.shape)}")

from models.temporal import CfCTemporalModule, LSTMTemporalModule, GRUTemporalModule
check("5a. CfCTemporalModule isolated",  lambda: test_temporal_module(CfCTemporalModule,  'CfC'))
check("5b. LSTMTemporalModule isolated", lambda: test_temporal_module(LSTMTemporalModule, 'LSTM'))
check("5c. GRUTemporalModule isolated",  lambda: test_temporal_module(GRUTemporalModule,  'GRU'))

# ──────────────────────────────────────────────
# 6. Freeze / unfreeze backbone
# ──────────────────────────────────────────────

def test_freeze():
    model = RetinexLNNPipeline(
        temporal_type='cfc', n_feat=N_FEAT,
        window_size=T, hidden_dim=64, n_neurons=32,
    ).to(DEVICE)

    # До заморозки — backbone должен быть trainable
    params_before = model.get_num_params()
    assert params_before['trainable_backbone'] > 0, \
        "До freeze backbone должен быть trainable"
    ok(f"before freeze: trainable_backbone={params_before['trainable_backbone']}, "
       f"trainable_temporal={params_before['trainable_temporal']}")

    # Заморозить backbone
    model.freeze_backbone()
    params_frozen = model.get_num_params()

    assert params_frozen['trainable_backbone'] == 0, \
        "После freeze backbone не должно быть trainable backbone параметров"
    assert params_frozen['trainable_temporal'] > 0, \
        "Temporal параметры должны оставаться trainable"
    ok(f"frozen: trainable={params_frozen['trainable']} "
       f"(temporal={params_frozen['trainable_temporal']}, "
       f"backbone={params_frozen['trainable_backbone']})")

    # Проверяем что backward работает только через temporal
    frames, target, timespans = make_batch()
    pred = model(frames, timespans)
    pred.mean().backward()

    # У backbone не должно быть градиентов
    backbone_grads = [p for p in list(model.estimator.parameters()) +
                      list(model.denoiser.parameters()) if p.grad is not None]
    assert len(backbone_grads) == 0, \
        f"У замороженного backbone не должно быть градиентов, нашли {len(backbone_grads)}"
    ok("backbone grad=None после freeze ✓")

    # Разморозить
    model.unfreeze_backbone()
    params_unfrozen = model.get_num_params()
    assert params_unfrozen['trainable_backbone'] == params_before['trainable_backbone'], \
        "После unfreeze backbone должен вернуться к исходному состоянию"
    ok(f"unfrozen: trainable={params_unfrozen['trainable']} ✓")

    # Проверяем get_param_groups
    groups = model.get_param_groups(lr=2e-4, backbone_lr_scale=0.1)
    assert len(groups) == 2, f"Должно быть 2 param groups, получили {len(groups)}"
    temporal_group = next(g for g in groups if g['name'] == 'temporal')
    backbone_group = next(g for g in groups if g['name'] == 'backbone')
    assert abs(temporal_group['lr'] - 2e-4) < 1e-9
    assert abs(backbone_group['lr'] - 2e-5) < 1e-9
    ok(f"param_groups: temporal lr={temporal_group['lr']:.1e}, "
       f"backbone lr={backbone_group['lr']:.1e} ✓")

check("6. Freeze / unfreeze backbone", test_freeze)

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

total = passed + failed
print(f"\n{'─'*50}")
print(f"  {GREEN}{passed}/{total} passed{RESET}", end="")
if failed:
    print(f"  {RED}{failed} FAILED{RESET}")
else:
    print(f"  {GREEN}  All good ✓{RESET}")
print(f"{'─'*50}\n")

sys.exit(0 if failed == 0 else 1)
