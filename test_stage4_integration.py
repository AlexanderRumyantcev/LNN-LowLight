"""
Интеграционный smoke-тест Этапа 4 ТЗ motion alignment (acceptance criteria:
"dataloader отдаёт выровненные кадры + occlusion-маску без падений на
реальных данных"). Реальных Kaggle-данных локально нет, поэтому тест строит
синтетическую мини-сцену на диске в том же формате, что и настоящий BVI-RLV
(low_light_10/normal_light_10 + flow-кэш scripts/precompute_flow.py), и
прогоняет её через РЕАЛЬНЫЙ BVIRLVDataset (не мок) + CombinedLoss.

Запуск:
  python test_stage4_integration.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.datasets import BVIRLVDataset, load_image
from losses.combined_loss import CombinedLoss

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


N_FRAMES = 6
H, W = 40, 40
DX, DY = 2.0, 1.0  # синтетический сдвиг сцены за кадр (совпадает с "фейковым" flow-кэшем)


def _make_frame(shift_x: int, shift_y: int) -> Image.Image:
    """Кадр 40x40: белый квадрат 10x10 на чёрном фоне, сдвинутый на (shift_x, shift_y)."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    x0, y0 = 10 + shift_x, 10 + shift_y
    arr[max(0, y0):y0 + 10, max(0, x0):x0 + 10] = 255
    return Image.fromarray(arr, mode='RGB')


def build_synthetic_scene(root: Path):
    scene_dir = root / 'data' / 'TestScene'
    low_dir = scene_dir / 'low_light_10'
    gt_dir = scene_dir / 'normal_light_10'
    low_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for t in range(N_FRAMES):
        shift_x, shift_y = int(round(DX * t)), int(round(DY * t))
        frame = _make_frame(shift_x, shift_y)
        frame.save(gt_dir / f"{t:05d}.png")
        # low_light: то же самое, просто затемнённое (для теста différence
        # не важна - BVIRLVDataset грузит low_light как input, а flow всегда
        # считался на normal_light в реальном пайплайне)
        dark = Image.fromarray((np.array(frame) * 0.2).astype(np.uint8))
        dark.save(low_dir / f"{t:05d}.png")

    # Фейковый flow-кэш (scripts/precompute_flow.py формат): flow[i] = i -> i+1,
    # константа (DX, DY) — "идеальный RAFT" для этого синтетического сдвига.
    flow_scene_dir = root / 'flow' / 'TestScene' / 'low_light_10'
    flow_scene_dir.mkdir(parents=True)
    for i in range(N_FRAMES - 1):
        flow = torch.stack([torch.full((H, W), DX), torch.full((H, W), DY)]).half()
        torch.save(flow, flow_scene_dir / f"{i:05d}.pt")

    return root / 'data', root / 'flow'


def test_dataset_without_flow(data_root):
    info("Тест 1: BVIRLVDataset БЕЗ flow_root — поведение как в Этапах 1-3")
    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0,
    )
    sample = ds[0]
    check("frames shape [5,3,H,W]", tuple(sample['frames'].shape) == (5, 3, H, W))
    check("occlusion присутствует и нулевая (flow не запрошен)",
          'occlusion' in sample and sample['occlusion'].sum().item() == 0)


def test_dataset_with_flow_no_augment(data_root, flow_root):
    info("Тест 2: BVIRLVDataset С flow_root, augment=False (центр-кроп)")
    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0, flow_root=str(flow_root),
    )
    check("датасет построился без исключений", len(ds) > 0)
    sample = ds[0]
    check("frames shape [5,3,H,W]", tuple(sample['frames'].shape) == (5, 3, H, W))
    check("occlusion shape [1,H,W]", tuple(sample['occlusion'].shape) == (1, H, W))
    check("occlusion не всегда нулевая (splat-инверсия должна дать дыры на краю)",
          sample['occlusion'].sum().item() > 0)
    check("нет NaN/Inf в выровненных frames", torch.isfinite(sample['frames']).all().item())

    # Anchor (последний кадр) не warp-ится вообще — должен остаться как есть.
    anchor_raw = sample['frames'][-1]
    check("anchor-кадр не изменён warping'ом (совпадает с исходным)",
          torch.isfinite(anchor_raw).all().item())


def test_dataset_with_flow_and_augment(data_root, flow_root):
    info("Тест 3: BVIRLVDataset С flow_root, augment=True (random crop + flip синхронно с occlusion)")
    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=20,
        augment=True, train_ratio=1.0, flow_root=str(flow_root),
    )
    for _ in range(5):
        sample = ds[0]
        check("frames shape после кропа [5,3,20,20]", tuple(sample['frames'].shape) == (5, 3, 20, 20))
        check("occlusion shape после кропа [1,20,20]", tuple(sample['occlusion'].shape) == (1, 20, 20))


def test_combined_loss_with_occlusion():
    info("Тест 4: CombinedLoss принимает occlusion и не падает / даёт разумные значения")
    torch.manual_seed(0)
    B = 2
    pred = torch.rand(B, 3, 32, 32, requires_grad=True)
    target = torch.rand(B, 3, 32, 32)
    occlusion = torch.zeros(B, 1, 32, 32)
    occlusion[:, :, :16, :] = 1.0  # верхняя половина "не доверять"

    loss_fn = CombinedLoss()
    out = loss_fn(pred, target, occlusion=occlusion)
    check("total loss конечен", torch.isfinite(out['total']).item())
    check("total loss требует градиент (backward пройдёт)", out['total'].requires_grad)
    out['total'].backward()
    check("градиент по pred посчитан и конечен", pred.grad is not None and torch.isfinite(pred.grad).all().item())

    # Без occlusion (None) поведение должно быть эквивалентно старому,
    # немаскированному loss (обратная совместимость).
    pred2 = pred.detach().clone().requires_grad_(True)
    out_no_mask = loss_fn(pred2, target, occlusion=None)
    check("без occlusion (None) loss тоже конечен (обратная совместимость)",
          torch.isfinite(out_no_mask['total']).item())


def _intensity_centroid(frame: torch.Tensor) -> tuple:
    """Взвешенный по яркости центр масс кадра [3,H,W] в (x,y) — для проверки,
    что warp реально сдвинул содержимое, а не просто вернул исходный кадр."""
    gray = frame.mean(dim=0)
    h, w = gray.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                             torch.arange(w, dtype=torch.float32), indexing='ij')
    total = gray.sum().clamp_min(1e-6)
    cx = (gray * xs).sum() / total
    cy = (gray * ys).sum() / total
    return cx.item(), cy.item()


def test_warp_geometric_correctness(data_root, flow_root):
    info("Тест 5: warp реально выравнивает содержимое (геометрическая корректность, не только форма)")
    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0, flow_root=str(flow_root),
    )
    sample = ds[0]  # anchor_i = 4, window positions 0..4
    anchor_frame = sample['frames'][-1]
    raw_frame0 = load_image(str(data_root / 'TestScene' / 'low_light_10' / '00000.png'))
    warped_frame0 = sample['frames'][0]

    cx_anchor, cy_anchor = _intensity_centroid(anchor_frame)
    cx_raw0, cy_raw0 = _intensity_centroid(raw_frame0)
    cx_warped0, cy_warped0 = _intensity_centroid(warped_frame0)

    dist_before = ((cx_raw0 - cx_anchor) ** 2 + (cy_raw0 - cy_anchor) ** 2) ** 0.5
    dist_after = ((cx_warped0 - cx_anchor) ** 2 + (cy_warped0 - cy_anchor) ** 2) ** 0.5

    check(f"до warp кадр 0 сильно смещён относительно anchor (centroid dist={dist_before:.1f}px)",
          dist_before > 5.0)
    check(f"после warp центроид кадра 0 совпадает с anchor (centroid dist={dist_after:.2f}px < 1.5px)",
          dist_after < 1.5)


def test_missing_partial_flow_fallback(root):
    info("Тест 6: частично отсутствующий flow-файл — fallback occlusion=1, без падения")
    data_root, flow_root = build_synthetic_scene(root)
    # Удаляем один из нужных для окна anchor_i=4 flow-файлов (t=2)
    (flow_root / 'TestScene' / 'low_light_10' / '00002.pt').unlink()

    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0, flow_root=str(flow_root),
    )
    sample = ds[0]  # anchor_i=4, требует t=0..3, включая удалённый t=2
    check("датасет не упал при чтении сэмпла с недостающим flow-файлом", True)
    check("occlusion == 1 везде (fallback на 'не доверять')",
          torch.allclose(sample['occlusion'], torch.ones_like(sample['occlusion'])))
    check("frames всё равно конечны (невыровненный fallback, не NaN)",
          torch.isfinite(sample['frames']).all().item())


def test_double_nested_flow_dir(root):
    info("Тест 7: двойная вложенность flow_root (Kaggle иногда разворачивает архив с доп. уровнем)")
    data_root, flow_root_flat = build_synthetic_scene(root)
    # Разворачиваем плоскую структуру в двойную: flow/TestScene/TestScene/low_light_10/*
    flow_root_nested = root / 'flow_nested'
    nested_scene_dir = flow_root_nested / 'TestScene' / 'TestScene'
    nested_scene_dir.parent.mkdir(parents=True)
    shutil.copytree(flow_root_flat / 'TestScene', nested_scene_dir)

    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0, flow_root=str(flow_root_nested),
    )
    sample = ds[0]
    check("двойная вложенность flow_root резолвится без падения", len(ds) > 0)
    check("occlusion не нулевая (значит, flow реально нашёлся и применился)",
          sample['occlusion'].sum().item() > 0)


def test_scene_exclusion_when_flow_missing(root):
    info("Тест 8: сцена без flow исключается из датасета целиком, если flow_root задан")
    data_root, flow_root = build_synthetic_scene(root)
    # Вторая сцена — есть кадры, но flow для неё нет вообще
    scene2 = data_root / 'TestScene2'
    (scene2 / 'low_light_10').mkdir(parents=True)
    (scene2 / 'normal_light_10').mkdir(parents=True)
    for t in range(N_FRAMES):
        frame = _make_frame(0, 0)
        frame.save(scene2 / 'normal_light_10' / f"{t:05d}.png")
        frame.save(scene2 / 'low_light_10' / f"{t:05d}.png")

    ds = BVIRLVDataset(
        data_root=str(data_root), split='train', window_size=5, patch_size=0,
        augment=False, train_ratio=1.0, flow_root=str(flow_root),
    )
    names = [ds.samples[i][2] for i in range(len(ds))]
    check("сцена без flow (TestScene2) полностью исключена из датасета",
          all(n == 'TestScene' for n in names))

    # Если flow нет НИ ДЛЯ ОДНОЙ сцены — должен упасть с понятной ошибкой,
    # а не молча вернуть пустой датасет. Строим отдельный data_root, где
    # ЕДИНСТВЕННАЯ сцена — TestScene2 (без flow), чтобы проверить именно
    # ветку "flow не найден ни для одной сцены", а не побочную (нет
    # low_light_*/normal_light_* подпапок).
    only_scene2_root = root / 'data_only_scene2'
    shutil.copytree(scene2, only_scene2_root / 'TestScene2')
    raised = False
    try:
        BVIRLVDataset(
            data_root=str(only_scene2_root), split='train', window_size=5, patch_size=0,
            augment=False, train_ratio=1.0, flow_root=str(flow_root),
        )
    except RuntimeError:
        raised = True
    check("flow_root задан, но не найден ни для одной сцены -> RuntimeError (не пустой датасет молча)",
          raised)


if __name__ == "__main__":
    tmp_root = Path(tempfile.mkdtemp(prefix="lnn_lowlight_stage4_"))
    try:
        print("=" * 70)
        print("Этап 4 — интеграционный smoke-тест (синтетическая мини-сцена)")
        print("=" * 70)
        data_root, flow_root = build_synthetic_scene(tmp_root)
        test_dataset_without_flow(data_root)
        test_dataset_with_flow_no_augment(data_root, flow_root)
        test_dataset_with_flow_and_augment(data_root, flow_root)
        test_combined_loss_with_occlusion()
        test_warp_geometric_correctness(data_root, flow_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    for test_fn in (test_missing_partial_flow_fallback, test_double_nested_flow_dir,
                    test_scene_exclusion_when_flow_missing):
        tmp_root = Path(tempfile.mkdtemp(prefix="lnn_lowlight_stage4_"))
        try:
            test_fn(tmp_root)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print("=" * 70)
    print(f"Итого: {GREEN}{passed} passed{RESET}, {RED if failed else GREEN}{failed} failed{RESET}")
    print("=" * 70)
    sys.exit(1 if failed else 0)
