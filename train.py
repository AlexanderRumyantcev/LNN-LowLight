"""
train.py — основной скрипт обучения.

Двухфазная стратегия:
  Фаза 1 (эпохи 0..finetune_epoch-1): backbone заморожен, обучается только temporal
  Фаза 2 (эпохи finetune_epoch..epochs): backbone разморожен, lr*backbone_lr_scale

Использование:
  python train.py --config configs/retinex_cfc_sdsd.yaml
  python train.py --config configs/retinex_cfc_sdsd.yaml --resume experiments/checkpoints/cfc_sdsd/epoch_0050.pth
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

sys.path.insert(0, str(Path(__file__).parent))

from models import RetinexLNNPipeline
from data.datasets import build_dataset
from losses.combined_loss import CombinedLoss


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

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
    model = RetinexLNNPipeline(
        temporal_type=cfg['temporal_type'],
        n_feat=mc.get('n_feat', 32),
        stage=mc.get('stage', 1),
        num_blocks=mc.get('num_blocks', [1, 1, 1]),
        window_size=mc.get('window_size', 5),
        hidden_dim=mc.get('hidden_dim', 64),
        n_neurons=mc.get('n_neurons', 32),
        fps=mc.get('fps', 30.0),
    )
    if mc.get('freeze_backbone', False):
        model.freeze_backbone()
    return model


def build_optimizer(model: RetinexLNNPipeline, cfg: dict):
    tc = cfg['train']
    mc = cfg['model']
    lr = tc['lr']
    backbone_lr_scale = mc.get('backbone_lr_scale', 0.1)
    if mc.get('freeze_backbone', False):
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    else:
        groups = model.get_param_groups(lr=lr, backbone_lr_scale=backbone_lr_scale)
        return torch.optim.AdamW(groups, weight_decay=1e-4)


def build_scheduler(optimizer, cfg: dict, n_epochs: int):
    tc      = cfg['train']
    warmup  = tc.get('warmup_epochs', 5)
    T_cos   = max(1, n_epochs - warmup)
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (ep + 1) / warmup if ep < warmup else 1.0
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_cos, eta_min=tc['lr_min']
    )
    return warmup_sched, cosine_sched


# ──────────────────────────────────────────────
# Train / Validate
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    totals = {'total': 0., 'pixel': 0., 'ssim': 0., 'temporal': 0.}
    n = 0
    # non_blocking=True имеет эффект только если тензор лежит в pinned memory
    # (DataLoader создан с pin_memory=True) — иначе .to(..., non_blocking=True)
    # тихо деградирует до обычного синхронного переноса, без предупреждения.
    # Здесь это безопасно: train_loader строится с pin_memory=(device.type ==
    # 'cuda') в main() — то есть non_blocking реально не блокирует CPU-поток
    # только на CUDA; на CPU/MPS аргумент просто игнорируется PyTorch.
    for batch in loader:
        frames    = batch['frames'].to(device, non_blocking=True)
        target    = batch['target'].to(device, non_blocking=True)
        timespans = batch.get('timespans')
        if timespans is not None:
            timespans = timespans.to(device, non_blocking=True)
        pred   = model(frames, timespans)
        losses = criterion(pred, target)
        optimizer.zero_grad()
        losses['total'].backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_psnr, total_loss, n = 0., 0., 0
    for batch in loader:
        frames    = batch['frames'].to(device, non_blocking=True)
        target    = batch['target'].to(device, non_blocking=True)
        timespans = batch.get('timespans')
        if timespans is not None:
            timespans = timespans.to(device, non_blocking=True)
        pred   = model(frames, timespans).clamp(0., 1.)
        losses = criterion(pred, target)
        total_loss += losses['total'].item()
        mse  = ((pred - target) ** 2).mean(dim=[1, 2, 3])
        psnr = (-10 * torch.log10(mse + 1e-8)).mean().item()
        total_psnr += psnr
        n += 1
    return {'psnr': total_psnr / max(n, 1), 'val_loss': total_loss / max(n, 1)}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()
    tc     = cfg['train']
    mc     = cfg['model']

    print(f"\n{'='*55}")
    print(f"  Config:  {args.config}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {tc['epochs']}  (finetune @ {tc.get('finetune_epoch', 'none')})")
    print(f"{'='*55}\n")

    for p in [cfg['paths']['checkpoints'], cfg['paths']['logs'], cfg['paths']['results']]:
        os.makedirs(p, exist_ok=True)

    # Модель
    model  = build_model(cfg).to(device)
    params = model.get_num_params()
    print(f"Params: total={params['total']:,} | trainable={params['trainable']:,} "
          f"| temporal={params['temporal']:,}")

    # Данные
    train_ds = build_dataset(cfg['dataset'], split='train')
    val_ds   = build_dataset(cfg['dataset'], split='test')
    train_loader = DataLoader(
        train_ds, batch_size=tc['batch_size'], shuffle=True,
        num_workers=tc['num_workers'], pin_memory=(device.type == 'cuda'), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples\n")

    # Loss / Optimizer / Scheduler
    lc        = cfg['loss']
    criterion = CombinedLoss(
        lambda_pixel=lc['lambda_pixel'],
        lambda_ssim=lc['lambda_ssim'],
        lambda_temporal=lc['lambda_temporal'],
    )
    optimizer = build_optimizer(model, cfg)
    warmup_sched, cosine_sched = build_scheduler(optimizer, cfg, tc['epochs'])

    # Resume
    start_epoch    = 0
    best_psnr      = 0.0
    phase2_started = not mc.get('freeze_backbone', False)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch    = ckpt['epoch'] + 1
        best_psnr      = ckpt.get('best_psnr', 0.0)
        phase2_started = ckpt.get('phase2_started', phase2_started)
        print(f"Resumed from epoch {start_epoch} | best_psnr={best_psnr:.2f}")

    writer      = SummaryWriter(cfg['paths']['logs'])
    finetune_ep = tc.get('finetune_epoch')

    for epoch in range(start_epoch, tc['epochs']):

        # Переход в Фазу 2
        if finetune_ep is not None and epoch == finetune_ep and not phase2_started:
            print(f"\n>>> Epoch {epoch}: unfreeze backbone → Фаза 2")
            model.unfreeze_backbone()
            lr_now = optimizer.param_groups[0]['lr']
            groups = model.get_param_groups(
                lr=lr_now, backbone_lr_scale=mc.get('backbone_lr_scale', 0.1)
            )
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            remaining = tc['epochs'] - epoch
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining), eta_min=tc['lr_min']
            )
            phase2_started = True
            params = model.get_num_params()
            print(f"    trainable: {params['trainable']:,} params\n")
            writer.add_scalar('train/phase', 2, epoch)

        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        if epoch < tc.get('warmup_epochs', 5):
            warmup_sched.step()
        else:
            cosine_sched.step()

        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % tc['val_every'] == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            psnr    = val_metrics['psnr']
            elapsed = time.time() - t0
            phase   = '2' if phase2_started else '1'
            print(
                f"[Ph{phase}] Ep {epoch+1:4d}/{tc['epochs']} | "
                f"loss={train_metrics['total']:.4f} | "
                f"PSNR={psnr:.2f}dB | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'psnr': psnr, 'best_psnr': best_psnr,
                    'phase2_started': phase2_started, 'config': cfg,
                }, os.path.join(cfg['paths']['checkpoints'], 'best.pth'))
                print(f"    ✓ best.pth (PSNR={psnr:.2f}dB)")

        if (epoch + 1) % tc['save_every'] == 0:
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'phase2_started': phase2_started, 'config': cfg,
            }, os.path.join(cfg['paths']['checkpoints'], f'epoch_{epoch+1:04d}.pth'))

    writer.close()
    print(f"\nTraining done. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
