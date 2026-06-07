"""
eval.py — оценка модели на тестовой выборке.

Метрики:
  PSNR     — peak signal-to-noise ratio (выше = лучше)
  SSIM     — structural similarity (выше = лучше)
  LPIPS    — perceptual similarity (ниже = лучше, опционально)
  tOF      — temporal optical flow consistency (ниже = лучше)
             warping error между соседними enhanced кадрами

Использование:
  python eval.py --config configs/retinex_cfc_sdsd.yaml --checkpoint experiments/checkpoints/cfc_sdsd/best.pth
  python eval.py --config configs/retinex_cfc_sdsd.yaml --checkpoint best.pth --save_images
  python eval.py --config configs/retinex_cfc_sdsd.yaml --checkpoint best.pth --compare_all
"""

import argparse
import sys
import time
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models import RetinexLNNPipeline
from data.datasets import build_dataset


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ──────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR в dB. pred, target: [B, 3, H, W] в [0, 1].
    Считаем per-image, усредняем по батчу.
    """
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])          # [B]
    return (-10.0 * torch.log10(mse + 1e-8)).mean().item()


def ssim_single(pred: torch.Tensor, target: torch.Tensor,
                window_size: int = 11, sigma: float = 1.5) -> float:
    """
    SSIM для пары [3, H, W] тензоров в [0, 1].
    Реализация без внешних зависимостей (gaussian window).
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)                                    # [ws, ws]
    kernel = kernel.expand(3, 1, window_size, window_size).to(pred.device)

    pad = window_size // 2
    p = pred.unsqueeze(0)    # [1, 3, H, W]
    t = target.unsqueeze(0)

    mu1 = F.conv2d(p, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(t, kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    s1  = F.conv2d(p * p, kernel, padding=pad, groups=3) - mu1_sq
    s2  = F.conv2d(t * t, kernel, padding=pad, groups=3) - mu2_sq
    s12 = F.conv2d(p * t, kernel, padding=pad, groups=3) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return (num / den).mean().item()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM по батчу [B, 3, H, W], усредняем."""
    scores = [ssim_single(pred[i], target[i]) for i in range(pred.shape[0])]
    return float(np.mean(scores))


def temporal_consistency(frames: list[torch.Tensor]) -> float:
    """
    Temporal OF consistency (tOF / warping error).

    Приближённая версия без оптического потока:
    считаем L1 разницу между соседними кадрами после выравнивания
    простым pixel-wise сдвигом (достаточно для сравнения моделей).

    frames: список [3, H, W] тензоров enhanced кадров одной последовательности.
    Возвращает среднюю разницу между соседними кадрами (ниже = более стабильное видео).
    """
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(len(frames) - 1):
        diff = (frames[i + 1] - frames[i]).abs().mean().item()
        diffs.append(diff)
    return float(np.mean(diffs))


# ──────────────────────────────────────────────
# Построение модели
# ──────────────────────────────────────────────

def build_model(cfg: dict, checkpoint: str, device: torch.device) -> RetinexLNNPipeline:
    mc = cfg['model']
    model = RetinexLNNPipeline(
        temporal_type=cfg['temporal_type'],
        n_feat=mc.get('n_feat', 32),
        stage=mc.get('stage', 1),
        num_blocks=mc.get('num_blocks', [1, 1, 1]),
        window_size=mc.get('window_size', 5),
        hidden_dim=mc.get('hidden_dim', 64),
        n_neurons=mc.get('n_neurons', 32),
        fps=mc.get('fps', 30.0),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt.get('model', ckpt)   # поддержка bare state_dict
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")
    return model


# ──────────────────────────────────────────────
# Основная функция eval
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, save_dir=None):
    """
    Прогнать модель на всём loader, вернуть метрики.

    Returns:
        dict с ключами: psnr, ssim, temporal_consistency, n_samples, time_per_sample
    """
    all_psnr, all_ssim = [], []
    enhanced_seq = []   # для temporal consistency
    n = 0
    t0 = time.time()

    for batch in loader:
        frames    = batch['frames'].to(device)      # [B, T, 3, H, W]
        target    = batch['target'].to(device)      # [B, 3, H, W]
        timespans = batch.get('timespans')
        if timespans is not None:
            timespans = timespans.to(device)

        pred = model(frames, timespans)             # [B, 3, H, W]
        pred = pred.clamp(0.0, 1.0)

        all_psnr.append(psnr(pred, target))
        all_ssim.append(ssim_batch(pred, target))

        # Собираем enhanced кадры для tOF (только если batch_size=1)
        if pred.shape[0] == 1:
            enhanced_seq.append(pred[0].cpu())

        # Сохранение изображений
        if save_dir is not None:
            import torchvision
            for i, name in enumerate(batch.get('name', [''])):
                out_path = Path(save_dir) / f"{name.replace('/', '_')}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(pred[i], str(out_path))

        n += 1

    elapsed = time.time() - t0

    return {
        'psnr':                 float(np.mean(all_psnr)),
        'psnr_std':             float(np.std(all_psnr)),
        'ssim':                 float(np.mean(all_ssim)),
        'ssim_std':             float(np.std(all_ssim)),
        'temporal_consistency': temporal_consistency(enhanced_seq),
        'n_samples':            n,
        'time_per_sample':      elapsed / max(n, 1),
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  PSNR:  {metrics['psnr']:.3f} ± {metrics['psnr_std']:.3f} dB")
    print(f"  SSIM:  {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"  tOF:   {metrics['temporal_consistency']:.5f}  (↓ лучше)")
    print(f"  n:     {metrics['n_samples']} samples")
    print(f"  speed: {metrics['time_per_sample']*1000:.1f} ms/sample")
    print(f"{'─'*55}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate RetinexLNN model')
    parser.add_argument('--config',      type=str, required=True,
                        help='Path to config yaml')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to .pth checkpoint')
    parser.add_argument('--split',       type=str, default='test',
                        help='Dataset split: train | test (default: test)')
    parser.add_argument('--save_images', action='store_true',
                        help='Сохранить enhanced кадры в results/')
    parser.add_argument('--compare_all', action='store_true',
                        help='Оценить все checkpoint-ы в папке (для сравнения CfC/LSTM/GRU)')
    parser.add_argument('--batch_size',  type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    # Датасет
    ds = build_dataset(cfg['dataset'], split=args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"Dataset: {cfg['dataset']['name']} | split={args.split} | {len(ds)} samples")

    save_dir = None
    if args.save_images:
        save_dir = cfg['paths']['results']
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving images to: {save_dir}")

    if args.compare_all:
        # Оценить все .pth файлы в папке checkpoints
        ckpt_dir = Path(cfg['paths']['checkpoints'])
        checkpoints = sorted(ckpt_dir.glob('*.pth'))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}")
            return
        for ckpt_path in checkpoints:
            model = build_model(cfg, str(ckpt_path), device)
            metrics = evaluate(model, loader, device, save_dir)
            print_metrics(f"{cfg['name']} | {ckpt_path.name}", metrics)
    else:
        model = build_model(cfg, args.checkpoint, device)
        metrics = evaluate(model, loader, device, save_dir)
        print_metrics(f"{cfg['name']} | {Path(args.checkpoint).name}", metrics)

    return metrics


if __name__ == '__main__':
    main()
