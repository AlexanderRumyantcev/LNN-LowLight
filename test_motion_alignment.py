"""
Unit-тест Этапа 3 ТЗ motion alignment (data/motion_alignment.py) — синтетический
сдвиг с известным ответом, как требует acceptance criteria этапа.

Запуск:
  python test_motion_alignment.py
"""

import sys
import torch

from data.motion_alignment import compose_flow_chain, splat_invert_flow, get_flow_to_anchor

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        ok(name)
    else:
        failed += 1
        fail(name)


def test_compose_constant_shift():
    info("Тест 1: forward-композиция при постоянном сдвиге (dx=2, dy=1 за шаг, k=3)")
    H, W = 32, 32
    flows = [torch.stack([torch.full((H, W), 2.0), torch.full((H, W), 1.0)]) for _ in range(3)]
    composed = compose_flow_chain(flows)
    mean = composed.mean(dim=(1, 2))
    check("composed mean dx≈6.0", abs(mean[0].item() - 6.0) < 1e-3)
    check("composed mean dy≈3.0", abs(mean[1].item() - 3.0) < 1e-3)
    check("нет NaN/Inf в composed flow", torch.isfinite(composed).all().item())


def test_splat_invert_recovers_shift():
    info("Тест 2: splat-инверсия восстанавливает -flow вдали от края (тот же сдвиг)")
    H, W = 32, 32
    flows = [torch.stack([torch.full((H, W), 2.0), torch.full((H, W), 1.0)]) for _ in range(3)]
    composed = compose_flow_chain(flows)
    backward, occlusion = splat_invert_flow(composed)

    interior = ~occlusion[0].bool()
    mean_bwd = backward[:, interior].mean(dim=1)
    check("backward mean dx≈-6.0 во внутренней области", abs(mean_bwd[0].item() + 6.0) < 1e-2)
    check("backward mean dy≈-3.0 во внутренней области", abs(mean_bwd[1].item() + 3.0) < 1e-2)

    # геометрически ожидаемая доля дыр для постоянного сдвига (6,3) на 32x32:
    # 1 - (1 - 6/32) * (1 - 3/32)
    expected_occl_frac = 1 - (1 - 6 / W) * (1 - 3 / H)
    actual_occl_frac = occlusion.float().mean().item()
    check(
        f"доля occluded пикселей ≈ ожидаемой геометрии ({actual_occl_frac:.3f} vs {expected_occl_frac:.3f})",
        abs(actual_occl_frac - expected_occl_frac) < 0.02,
    )


def test_zero_flow_no_motion():
    info("Тест 3: нулевой flow (нет движения) -> нулевая композиция, нет occlusion")
    H, W = 16, 16
    zero_flows = [torch.zeros(2, H, W) for _ in range(4)]
    composed = compose_flow_chain(zero_flows)
    check("composed flow == 0 при отсутствии движения", torch.allclose(composed, torch.zeros_like(composed)))

    backward, occlusion = splat_invert_flow(composed)
    check("backward flow == 0 при отсутствии движения", torch.allclose(backward, torch.zeros_like(backward)))
    check("occlusion mask пустая при отсутствии движения", occlusion.sum().item() == 0)


def test_get_flow_to_anchor_directions():
    info("Тест 4: get_flow_to_anchor выбирает верную ветку по направлению времени")
    H, W = 16, 16
    # anchor = 2, окно [0,1,2,3,4] -> нужны neighbor flows для t=0,1,2,3
    neighbor_flows = {t: torch.stack([torch.full((H, W), 1.0), torch.zeros(H, W)]) for t in range(4)}

    flow_same, occ_same = get_flow_to_anchor(neighbor_flows, frame_idx=2, anchor_idx=2)
    check("frame_idx == anchor_idx -> нулевой flow", torch.allclose(flow_same, torch.zeros_like(flow_same)))
    check("frame_idx == anchor_idx -> нулевая маска", occ_same.sum().item() == 0)

    # frame_idx < anchor_idx (прошлый кадр — ОСНОВНОЙ случай при текущей
    # каузальной схеме окна): цепочка flow[0],flow[1] стартует с flow[0],
    # которое живёт на сетке frame_idx=0 -> композиция получается на сетке
    # frame_idx, нужна splat-инверсия, чтобы попасть на сетку anchor_idx.
    # composed forward dx = 1+1 = 2 -> backward dx ≈ -2, occlusion не нулевая.
    flow_past, occ_past = get_flow_to_anchor(neighbor_flows, frame_idx=0, anchor_idx=2)
    interior_past = ~occ_past[0].bool()
    check("frame_idx < anchor_idx -> splat-инверсия (dx≈-2.0 во внутренней области)",
          abs(flow_past[:, interior_past].mean(dim=1)[0].item() + 2.0) < 1e-2)
    check("frame_idx < anchor_idx -> occlusion НЕ нулевая (splat даёт дыры на краю)",
          occ_past.sum().item() > 0)

    # frame_idx > anchor_idx (будущий кадр — при текущей каузальной схеме
    # окна не встречается, но функция общая): цепочка flow[2],flow[3]
    # стартует с flow[2], которое уже живёт на сетке anchor_idx=2 ->
    # композиция готова напрямую, инверсия не нужна, дыр не даёт.
    flow_future, occ_future = get_flow_to_anchor(neighbor_flows, frame_idx=4, anchor_idx=2)
    check("frame_idx > anchor_idx -> forward-композиция (dx≈2.0)",
          abs(flow_future.mean(dim=(1, 2))[0].item() - 2.0) < 1e-3)
    check("frame_idx > anchor_idx -> occlusion всегда нулевая (gather, дыр нет)",
          occ_future.sum().item() == 0)


if __name__ == "__main__":
    print("=" * 70)
    print("Motion alignment (Этап 3) — unit-тест на синтетике")
    print("=" * 70)
    test_compose_constant_shift()
    test_splat_invert_recovers_shift()
    test_zero_flow_no_motion()
    test_get_flow_to_anchor_directions()
    print("=" * 70)
    print(f"Итого: {GREEN}{passed} passed{RESET}, {RED if failed else GREEN}{failed} failed{RESET}")
    print("=" * 70)
    sys.exit(1 if failed else 0)
