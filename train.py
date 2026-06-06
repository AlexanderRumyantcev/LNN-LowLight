"""
train.py — основной скрипт обучения.

Использование:
  python train.py --config configs/retinex_cfc.yaml
  python train.py --config configs/retinex_lstm.yaml
  python train.py --config configs/retinex_baseline.yaml
"""

import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent))

from models import RetinexLNNPipeline
from data.datasets import build_dataset
from losses.combined_loss import CombinedLoss


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> RetinexLNNPipeline:
    mc = cfg['model']
    return RetinexLNNPipeline(
        temporal_type=cfg['temporal_type'],
        n_feat=mc.get('n_feat', 31),
        stage=mc.get('stage', 1),
        num_blocks=mc.get('num_blocks', [1, 1, 1]),
        window_size=mc.get('window_size', 5),
        hidden_dim=mc.get('hidden_dim', 64),
        n_neurons=mc.get('n_neurons', 32),
        fps=mc.get('fps', 30.0),
    )


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_losses = {'total': 0., 'pixel': 0., 'ssim': 0., 'temporal': 0.}
    n = 0

    for batch in loader:
        frames = batch['frames'].to(device)    # [B, T, 3, H, W]
        target = batch['target'].to(device)    # [B, 3, H, W]
        timespans = batch.get('timespans')
        if timespans is not None:
            timespans = timespans.to(device)

        # Forward
        pred = model(frames, timespans)

        # Loss (temporal loss не используем в первой фазе — нет предыдущего enhanced)
        losses = criterion(pred, target)

        optimizer.zero_grad()
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            if k in losses:
                total_losses[k] += losses[k].item()
        n += 1

    return {k: v / max(n, 1) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Валидация: PSNR + loss."""
    model.eval()
    total_psnr = 0.
    total_loss = 0.
    n = 0

    for batch in loader:
        frames = batch['frames'].to(device)
        target = batch['target'].to(device)
        timespans = batch.get('timespans')
        if timespans is not None:
            timespans = timespans.to(device)

        pred = model(frames, timespans)
        losses = criterion(pred, target)
        total_loss += losses['total'].item()

        # PSNR
        mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
        psnr = (-10 * torch.log10(mse + 1e-8)).mean().item()
        total_psnr += psnr
        n += 1

    return {
        'psnr': total_psnr / max(n, 1),
        'val_loss': total_loss / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    # Создать директории
    for p in [cfg['paths']['checkpoints'], cfg['paths']['logs'], cfg['paths']['results']]:
        os.makedirs(p, exist_ok=True)

    # Модель
    model = build_model(cfg).to(device)
    params = model.get_num_params()
    print(f"Parameters: {params}")

    # Данные
    train_ds = build_dataset(cfg['dataset'], split='train')
    val_ds = build_dataset(cfg['dataset'], split='eval')
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=(device.type != 'mps'),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Loss + Optimizer + Scheduler
    tc = cfg['train']
    lc = cfg['loss']
    criterion = CombinedLoss(
        lambda_pixel=lc['lambda_pixel'],
        lambda_ssim=lc['lambda_ssim'],
        lambda_temporal=lc['lambda_temporal'],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['epochs'] - tc['warmup_epochs'], eta_min=tc['lr_min']
    )

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(cfg['paths']['logs'])
    best_psnr = 0.0

    for epoch in range(start_epoch, tc['epochs']):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()

        # Логирование
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % tc['val_every'] == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            psnr = val_metrics['psnr']
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1}/{tc['epochs']} | "
                f"loss={train_metrics['total']:.4f} | "
                f"PSNR={psnr:.2f} dB | "
                f"lr={optimizer.param_groups[0]['lr']:.6f} | "
                f"t={elapsed:.1f}s"
            )

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr': psnr,
                    'config': cfg,
                }, os.path.join(cfg['paths']['checkpoints'], 'best.pth'))

        if (epoch + 1) % tc['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg,
            }, os.path.join(cfg['paths']['checkpoints'], f'epoch_{epoch+1:04d}.pth'))

    writer.close()
    print(f"Training done. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
