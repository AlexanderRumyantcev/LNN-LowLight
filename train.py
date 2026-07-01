"""
train.py — основной скрипт обучения.

Двухфазная стратегия:
  Фаза 1 (эпохи 0..finetune_epoch-1): backbone заморожен, обучается только temporal
  Фаза 2 (эпохи finetune_epoch..epochs): backbone разморожен, lr*backbone_lr_scale

Использование:
  python train.py --config configs/retinex_cfc_sdsd.yaml
  python train.py --config configs/retinex_cfc_sdsd.yaml --resume experiments/checkpoints/cfc_sdsd/epoch_0050.pth

Для длительных безнадзорных прогонов на Kaggle (Save and Run All / Commit,
без интерактивной сессии) — wall-clock лимит и периодический чекпоинт по
времени, не привязанные к завершённым эпохам:
  python train.py --config configs/retinex_cfc_bvirlv.yaml \
      --wall-clock-limit-sec 9000 --checkpoint-every-sec 600
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

def train_epoch(
    model, loader, optimizer, criterion, device, epoch,
    checkpoint_fn=None, checkpoint_every_sec=None, wall_clock_deadline=None,
):
    """
    checkpoint_fn: callable(extra_state: dict) -> None — сохраняет чекпоинт
        с доп. полями (epoch, batch_idx, partial_epoch, ...) сверху обычных
        model/optimizer state (которые checkpoint_fn берёт из замыкания в
        main(), не передаются здесь явно). Вызывается ИЗНУТРИ цикла по
        батчам, не дожидаясь конца эпохи — это и решает проблему "save_every
        считает эпохи, а до конца эпохи можем не дойти".
    checkpoint_every_sec: минимальный интервал между вызовами checkpoint_fn
        по таймеру (не на каждый батч — иначе сам torch.save() стал бы
        bottleneck при частом вызове).
    wall_clock_deadline: time.time()-совместимый timestamp. Проверяется
        ПОСЛЕ optimizer.step() текущего батча (не обрывает backward на
        середине) — если достигнут, сохраняется финальный чекпоинт
        (stopped_by_wall_clock=True) и цикл прерывается через break.

    Возвращает (metrics: dict, interrupted: bool).
    """
    model.train()
    totals = {'total': 0., 'pixel': 0., 'ssim': 0., 'temporal': 0.}
    n = 0
    last_checkpoint_t = time.time()
    interrupted = False

    # non_blocking=True имеет эффект только если тензор лежит в pinned memory
    # (DataLoader создан с pin_memory=True) — иначе .to(..., non_blocking=True)
    # тихо деградирует до обычного синхронного переноса, без предупреждения.
    # Здесь это безопасно: train_loader строится с pin_memory=(device.type ==
    # 'cuda') в main() — то есть non_blocking реально не блокирует CPU-поток
    # только на CUDA; на CPU/MPS аргумент просто игнорируется PyTorch.
    for batch_idx, batch in enumerate(loader):
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

        now = time.time()

        # Step-level чекпоинт по таймеру — независимо от того, дойдём ли до
        # конца эпохи. checkpoint_every_sec задаёт минимальный интервал,
        # чтобы torch.save() сам не стал заметным накладным расходом при
        # частом вызове (для CfC с frozen backbone чекпоинт мал — десятки МБ,
        # но всё равно лучше не писать на диск чаще, чем нужно).
        if checkpoint_fn is not None and checkpoint_every_sec is not None:
            if now - last_checkpoint_t >= checkpoint_every_sec:
                checkpoint_fn({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'partial_epoch': True,
                })
                last_checkpoint_t = now

        # Wall-clock hard stop. Проверяется здесь (после optimizer.step()),
        # поэтому никогда не обрывает обновление весов на середине батча —
        # следующий батч просто не начинается. Финальный чекпоинт сохраняется
        # явно (не полагаясь на checkpoint_every_sec, который мог не успеть
        # сработать в последний интервал) перед выходом, чтобы гарантированно
        # не потерять прогресс именно этого батча.
        if wall_clock_deadline is not None and now >= wall_clock_deadline:
            if checkpoint_fn is not None:
                checkpoint_fn({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'partial_epoch': True,
                    'stopped_by_wall_clock': True,
                })
            interrupted = True
            break

    metrics = {k: v / max(n, 1) for k, v in totals.items()}
    return metrics, interrupted


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
    parser.add_argument(
        '--wall-clock-limit-sec', type=float, default=None,
        help='Жёсткий лимит по времени (секунды от старта main()) — при '
             'достижении train_epoch прерывается контролируемо ПОСЛЕ '
             'текущего батча, чекпоинт сохраняется явно, скрипт завершается. '
             'Должен быть строго МЕНЬШЕ лимита самой Kaggle-сессии (9-12ч) '
             'с запасом на запись чекпоинта/логов перед принудительным '
             'обрывом — например, 9000 (2.5ч) для diagnostic-прогона.'
    )
    parser.add_argument(
        '--checkpoint-every-sec', type=float, default=600.0,
        help='Интервал step-level чекпоинтов внутри эпохи в секундах '
             '(по умолчанию 600 = 10 мин) — независимо от save_every/'
             'val_every из конфига, которые работают только на границах '
             'ЗАВЕРШЁННЫХ эпох и могут не сработать вообще, если скрипт не '
             'успевает пройти ни одной полной эпохи за отведённое время.'
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()
    tc     = cfg['train']
    mc     = cfg['model']

    run_start_t = time.time()
    wall_clock_deadline = (
        run_start_t + args.wall_clock_limit_sec
        if args.wall_clock_limit_sec is not None else None
    )

    print(f"\n{'='*55}")
    print(f"  Config:  {args.config}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {tc['epochs']}  (finetune @ {tc.get('finetune_epoch', 'none')})")
    if wall_clock_deadline is not None:
        print(f"  Wall-clock limit: {args.wall_clock_limit_sec/3600:.2f}h "
              f"(checkpoint every {args.checkpoint_every_sec/60:.1f} min)")
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
        persistent_workers=(tc['num_workers'] > 0),
        prefetch_factor=(4 if tc['num_workers'] > 0 else None),
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
        # Восстанавливаем scheduler state если присутствует в чекпоинте.
        # get() с fallback=None — backward-совместимость со старыми чекпоинтами
        # (до этого коммита), которые scheduler state не сохраняли.
        # Нюанс Фазы 2: если phase2_started=False и epoch==finetune_epoch,
        # cosine_sched будет пересоздан при unfreeze ниже в цикле —
        # восстановленный state будет перезаписан, что корректно (новый
        # cosine_sched для Фазы 2 стартует с нуля). Если phase2_started=True,
        # unfreeze не сработает и восстановленный cosine_sched.state_dict()
        # останется действительным.
        if ckpt.get('warmup_sched') is not None:
            warmup_sched.load_state_dict(ckpt['warmup_sched'])
        if ckpt.get('cosine_sched') is not None:
            cosine_sched.load_state_dict(ckpt['cosine_sched'])
        # partial_epoch чекпоинты (сохранённые внутри train_epoch по таймеру
        # или по wall-clock останову) хранят epoch текущей, НЕЗАВЕРШЁННОЙ
        # эпохи. При resume с такого чекпоинта эпоха ПЕРЕЗАПУСКАЕТСЯ С НАЧАЛА
        # (start_epoch = ckpt['epoch'], не +1) — НЕ с точного batch_idx:
        # DataLoader shuffle=True даёт новый порядок батчей на каждый запуск
        # процесса, так что точный resume "с того самого батча" не имеет
        # смысла без отдельной фиксации seed на каждую эпоху. Веса модели и
        # optimizer state восстанавливаются корректно в любом случае — при
        # partial-resume теряется только текущая неполная эпоха целиком (её
        # нужно пройти заново), а не весь накопленный прогресс обучения.
        if ckpt.get('partial_epoch', False):
            start_epoch = ckpt['epoch']
            print(f"Resumed from PARTIAL epoch {start_epoch} "
                  f"(was at batch {ckpt.get('batch_idx', '?')}) — "
                  f"эпоха перезапущена с начала")
        else:
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        best_psnr      = ckpt.get('best_psnr', 0.0)
        phase2_started = ckpt.get('phase2_started', phase2_started)
        print(f"best_psnr={best_psnr:.2f}")

    writer      = SummaryWriter(cfg['paths']['logs'])
    finetune_ep = tc.get('finetune_epoch')

    def save_checkpoint(path, extra=None):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # Оба scheduler state сохраняются всегда. В Фазе 1 активен
            # warmup_sched (эпохи < warmup_epochs), потом cosine_sched.
            # При unfreeze cosine_sched пересоздаётся с нуля — сохранённый
            # здесь pre-unfreeze state будет проигнорирован при resume
            # (перезаписан новым cosine_sched), что корректно.
            'warmup_sched': warmup_sched.state_dict(),
            'cosine_sched': cosine_sched.state_dict(),
            'best_psnr': best_psnr,
            'phase2_started': phase2_started,
            'config': cfg,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    stopped_early = False

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

        # checkpoint_fn — closure, захватывает model/optimizer/warmup_sched/
        # cosine_sched/best_psnr/phase2_started/cfg из окружающего main()
        # через save_checkpoint(). Всегда пишет В ОДИН И ТОТ ЖЕ файл
        # checkpoints/latest.pth (перезаписывается на каждый вызов), а не
        # растущий список — иначе чекпоинты каждые checkpoint_every_sec
        # секунд за 2-3ч diagnostic-прогона дали бы 12-18+ отдельных файлов
        # и сами стали бы заметным вкладом в размер output ноутбука.
        # best.pth и epoch_XXXX.pth (ниже) остаются отдельными именованными
        # файлами, но сохраняются только для ПОЛНЫХ (не partial) эпох.
        def checkpoint_fn(extra):
            save_checkpoint(
                os.path.join(cfg['paths']['checkpoints'], 'latest.pth'),
                extra=extra,
            )
            stop_note = ' (wall-clock stop)' if extra.get('stopped_by_wall_clock') else ''
            print(f"    [checkpoint] latest.pth saved | epoch={extra['epoch']} "
                  f"batch={extra.get('batch_idx', '?')}{stop_note}")

        train_metrics, interrupted = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            checkpoint_fn=checkpoint_fn,
            checkpoint_every_sec=args.checkpoint_every_sec,
            wall_clock_deadline=wall_clock_deadline,
        )

        if interrupted:
            elapsed_total = time.time() - run_start_t
            print(f"\n>>> Остановлено по wall-clock лимиту "
                  f"({elapsed_total/3600:.2f}h) внутри epoch {epoch}")
            stopped_early = True

        if epoch < tc.get('warmup_epochs', 5):
            warmup_sched.step()
        else:
            cosine_sched.step()

        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # val/best.pth/save_every пропускаются для прерванной (partial)
        # эпохи — design decision: частичная эпоха НЕ считается эквивалентной
        # полной для целей val/checkpoint-по-эпохам. train_metrics всё равно
        # усреднены по пройденным батчам и валидны для отслеживания тренда
        # loss даже без вызова validate().
        if not interrupted:
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
                    save_checkpoint(
                        os.path.join(cfg['paths']['checkpoints'], 'best.pth'),
                        extra={'epoch': epoch, 'psnr': psnr},
                    )
                    print(f"    ✓ best.pth (PSNR={psnr:.2f}dB)")

            if (epoch + 1) % tc['save_every'] == 0:
                save_checkpoint(
                    os.path.join(cfg['paths']['checkpoints'], f'epoch_{epoch+1:04d}.pth'),
                    extra={'epoch': epoch},
                )
        else:
            # Финальный partial-чекпоинт уже сохранён внутри train_epoch
            # (checkpoint_fn со stopped_by_wall_clock=True) перед break —
            # здесь просто печатаем итоговый train loss, накопленный до
            # момента остановки, чтобы тренд был виден даже без полной
            # эпохи/val, и выходим из цикла по эпохам.
            phase = '2' if phase2_started else '1'
            print(
                f"[Ph{phase}] Ep {epoch+1:4d} (partial) | "
                f"loss={train_metrics['total']:.4f} (avg по пройденным батчам)"
            )
            break

    writer.close()
    if stopped_early:
        print(f"\nОстановлено по wall-clock лимиту. Чекпоинт: "
              f"{os.path.join(cfg['paths']['checkpoints'], 'latest.pth')}")
    else:
        print(f"\nTraining done. Best PSNR: {best_psnr:.2f} dB")


if __name__ == '__main__':
    main()
