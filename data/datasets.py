"""
Датасеты для Low-Light Video/Image Enhancement.

Поддерживаемые датасеты:
  - LOL (Low-Light dataset): парные изображения low/normal
  - BVI-RLV: парные видеопоследовательности (основной)

Формат BVI-RLV:
  data_root/
    train/
      low/  seq_001/ 00001.png ...
      high/ seq_001/ 00001.png ...
    test/
      ...

Формат LOL:
  data_root/
    our485/  low/ ... high/ ...
    eval15/  low/ ... high/ ...
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


def load_image(path: str) -> torch.Tensor:
    """Загрузить изображение как тензор [3, H, W] float в [0, 1]."""
    img = Image.open(path).convert('RGB')
    return TF.to_tensor(img)


class LOLDataset(Dataset):
    """
    LOL dataset для image enhancement (не видео).
    Используется для Фазы 1 — проверки базового RetinexFormer.

    Возвращает пары (low, high) изображений.
    window_size=1: один кадр, обёрнутый в [1, 3, H, W] для совместимости с pipeline.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',  # 'train' | 'eval'
        patch_size: int = 128,
        augment: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')

        # LOL структура
        if split == 'train':
            low_dir = Path(data_root) / 'our485' / 'low'
            high_dir = Path(data_root) / 'our485' / 'high'
        else:
            low_dir = Path(data_root) / 'eval15' / 'low'
            high_dir = Path(data_root) / 'eval15' / 'high'

        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.low_paths = sorted([p for p in low_dir.iterdir() if p.suffix in exts])
        self.high_paths = sorted([p for p in high_dir.iterdir() if p.suffix in exts])
        assert len(self.low_paths) == len(self.high_paths), \
            f"LOL: mismatch low={len(self.low_paths)} high={len(self.high_paths)}"

    def __len__(self):
        return len(self.low_paths)

    def _random_crop(self, low, high):
        """Синхронный random crop обоих изображений."""
        _, H, W = low.shape
        ps = self.patch_size
        if H <= ps or W <= ps:
            low = TF.resize(low, [ps + 4, ps + 4])  # type: ignore
            high = TF.resize(high, [ps + 4, ps + 4])  # type: ignore
            _, H, W = low.shape
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        low = low[:, top:top + ps, left:left + ps]
        high = high[:, top:top + ps, left:left + ps]
        return low, high

    def _augment(self, low, high):
        if random.random() > 0.5:
            low = TF.hflip(low)
            high = TF.hflip(high)
        if random.random() > 0.5:
            low = TF.vflip(low)
            high = TF.vflip(high)
        return low, high

    def __getitem__(self, idx):
        low = load_image(str(self.low_paths[idx]))
        high = load_image(str(self.high_paths[idx]))

        if self.augment:
            low, high = self._random_crop(low, high)
            low, high = self._augment(low, high)

        # Оборачиваем в [T=1, 3, H, W] для совместимости с pipeline
        return {
            'frames': low.unsqueeze(0),   # [1, 3, H, W]
            'target': high,               # [3, H, W]
            'name': self.low_paths[idx].stem,
        }


class BVIRLVDataset(Dataset):
    """
    BVI-RLV dataset для video enhancement.
    Основной датасет проекта.

    Возвращает скользящее окно из window_size кадров (low) + target (high текущего кадра).
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        window_size: int = 5,
        patch_size: int = 256,
        augment: bool = True,
        fps: float = 30.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        self.fps = fps
        self.dt = 1.0 / fps

        low_root = Path(data_root) / split / 'low'
        high_root = Path(data_root) / split / 'high'

        self.samples = []  # list of (seq_low_paths, seq_high_paths, frame_idx)

        for seq_dir in sorted(low_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            high_dir = high_root / seq_dir.name
            if not high_dir.exists():
                continue

            exts = {'.png', '.jpg', '.jpeg'}
            low_frames = sorted([p for p in seq_dir.iterdir() if p.suffix in exts])
            high_frames = sorted([p for p in high_dir.iterdir() if p.suffix in exts])

            if len(low_frames) < window_size:
                continue

            # Каждый кадр начиная с window_size-1 может быть "текущим"
            for i in range(window_size - 1, len(low_frames)):
                self.samples.append((
                    [str(low_frames[i - window_size + 1 + j]) for j in range(window_size)],
                    str(high_frames[i]),
                ))

    def __len__(self):
        return len(self.samples)

    def _sync_crop(self, frames_list, high):
        """Синхронный crop для всей последовательности."""
        _, H, W = frames_list[0].shape
        ps = self.patch_size
        top = random.randint(0, max(0, H - ps))
        left = random.randint(0, max(0, W - ps))
        frames_list = [f[:, top:top + ps, left:left + ps] for f in frames_list]
        high = high[:, top:top + ps, left:left + ps]
        return frames_list, high

    def _sync_augment(self, frames_list, high):
        if random.random() > 0.5:
            frames_list = [TF.hflip(f) for f in frames_list]
            high = TF.hflip(high)
        if random.random() > 0.5:
            frames_list = [TF.vflip(f) for f in frames_list]
            high = TF.vflip(high)
        return frames_list, high

    def __getitem__(self, idx):
        low_paths, high_path = self.samples[idx]

        low_frames = [load_image(p) for p in low_paths]
        high = load_image(high_path)

        if self.augment:
            low_frames, high = self._sync_crop(low_frames, high)
            low_frames, high = self._sync_augment(low_frames, high)

        frames = torch.stack(low_frames, dim=0)  # [T, 3, H, W]

        # Равномерные timespans: ∆t = 1/fps для каждого шага
        timespans = torch.full((len(low_paths),), self.dt)

        return {
            'frames': frames,        # [T, 3, H, W]
            'target': high,          # [3, H, W]
            'timespans': timespans,  # [T]
            'name': Path(high_path).stem,
        }


class SDSDDataset(Dataset):
    """
    SDSD dataset (динамическая версия от dvlab-research) для video enhancement.

    Структура на диске (indoor_np / outdoor_np):
      data_root/
        GT/          ← normal-light кадры (.npy, float32, HWC, [0,1])
          pair1/
            00001.npy
            00002.npy
            ...
          pair2/ ...
        LQ/          ← low-light кадры (.npy, float32, HWC, [0,1])
          pair1/ ...
          pair2/ ...

    Каждый .npy файл: float32 array shape (H, W, 3), значения в [0, 1].
    Разрешение: 512 x 960 (H x W).

    Возвращает:
      frames:    [T, 3, H, W]  — окно из T low-light кадров
      target:    [3, H, W]     — GT текущего (последнего) кадра
      timespans: [T]           — ∆t между кадрами (1/fps)
      name:      str           — "pairN/XXXXX"
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',      # 'train' | 'test'
        window_size: int = 5,
        patch_size: int = 128,
        augment: bool = True,
        fps: float = 30.0,
        train_ratio: float = 0.9,  # доля пар для train (нет официального split)
        seed: int = 42,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')
        self.dt = 1.0 / fps

        lq_root = Path(data_root) / 'LQ'
        gt_root = Path(data_root) / 'GT'

        if not lq_root.exists():
            raise FileNotFoundError(f"SDSD LQ dir not found: {lq_root}")
        if not gt_root.exists():
            raise FileNotFoundError(f"SDSD GT dir not found: {gt_root}")

        # Собираем все пары (последовательности)
        all_pairs = sorted([d for d in lq_root.iterdir() if d.is_dir()])

        # Train/test split (детерминированный)
        rng = random.Random(seed)
        all_pairs_shuffled = all_pairs.copy()
        rng.shuffle(all_pairs_shuffled)
        n_train = max(1, int(len(all_pairs_shuffled) * train_ratio))

        if split == 'train':
            selected_pairs = all_pairs_shuffled[:n_train]
        else:
            selected_pairs = all_pairs_shuffled[n_train:]

        # Строим список сэмплов: (lq_frame_paths[T], gt_path)
        self.samples = []
        for pair_dir in sorted(selected_pairs):
            gt_dir = gt_root / pair_dir.name
            if not gt_dir.exists():
                continue

            lq_frames = sorted(pair_dir.glob('*.npy'))
            gt_frames = sorted(gt_dir.glob('*.npy'))

            # LQ и GT должны совпадать по числу кадров
            n = min(len(lq_frames), len(gt_frames))
            if n < window_size:
                continue

            for i in range(window_size - 1, n):
                window_lq = [lq_frames[i - window_size + 1 + j] for j in range(window_size)]
                target_gt = gt_frames[i]
                self.samples.append((window_lq, target_gt))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"SDSDDataset: no samples found in {data_root} (split={split}). "
                f"Проверь структуру: должны быть папки LQ/ и GT/ с подпапками pair*/."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_npy(path) -> torch.Tensor:
        """Загрузить .npy кадр → [3, H, W] float32 tensor в [0, 1]."""
        arr = np.load(str(path))          # (H, W, 3) float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:               # если вдруг [0, 255]
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W]

    def _sync_crop(self, frames: list, target: torch.Tensor):
        """Синхронный random crop всей последовательности + GT."""
        _, H, W = frames[0].shape
        ps = self.patch_size
        if H < ps or W < ps:
            return frames, target   # не кропаем если кадр меньше патча
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        frames = [f[:, top:top + ps, left:left + ps] for f in frames]
        target = target[:, top:top + ps, left:left + ps]
        return frames, target

    def _sync_augment(self, frames: list, target: torch.Tensor):
        """Случайные flip для всей последовательности + GT."""
        if random.random() > 0.5:
            frames = [TF.hflip(f) for f in frames]
            target = TF.hflip(target)
        if random.random() > 0.5:
            frames = [TF.vflip(f) for f in frames]
            target = TF.vflip(target)
        return frames, target

    def __getitem__(self, idx):
        lq_paths, gt_path = self.samples[idx]

        frames = [self._load_npy(p) for p in lq_paths]
        target = self._load_npy(gt_path)

        if self.augment:
            frames, target = self._sync_crop(frames, target)
            frames, target = self._sync_augment(frames, target)
        elif self.patch_size > 0:
            # Val/test: центральный кроп
            _, H, W = frames[0].shape
            ps = self.patch_size
            if H >= ps and W >= ps:
                top  = (H - ps) // 2
                left = (W - ps) // 2
                frames = [f[:, top:top + ps, left:left + ps] for f in frames]
                target = target[:, top:top + ps, left:left + ps]

        frames_tensor = torch.stack(frames, dim=0)         # [T, 3, H, W]
        timespans = torch.full((self.window_size,), self.dt)

        pair_name = lq_paths[-1].parent.name
        frame_name = lq_paths[-1].stem
        return {
            'frames':    frames_tensor,
            'target':    target,
            'timespans': timespans,
            'name':      f"{pair_name}/{frame_name}",
        }


def build_dataset(cfg: dict, split: str):
    """Фабрика датасетов по конфигу."""
    name = cfg.get('name', 'sdsd')
    if name == 'lol':
        return LOLDataset(
            data_root=cfg['data_root'],
            split=split,
            patch_size=cfg.get('patch_size', 128),
            augment=(split == 'train'),
        )
    elif name == 'bvi_rlv':
        return BVIRLVDataset(
            data_root=cfg['data_root'],
            split=split,
            window_size=cfg.get('window_size', 5),
            patch_size=cfg.get('patch_size', 256),
            fps=cfg.get('fps', 30.0),
        )
    elif name == 'sdsd':
        return SDSDDataset(
            data_root=cfg['data_root'],
            split=split,
            window_size=cfg.get('window_size', 5),
            patch_size=cfg.get('patch_size', 128),
            fps=cfg.get('fps', 30.0),
            train_ratio=cfg.get('train_ratio', 0.9),
            seed=cfg.get('seed', 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: lol, bvi_rlv, sdsd")
