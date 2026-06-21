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
    Основной датасет проекта (Lin et al., arXiv:2407.03535, 2024).

    Реальная структура на диске (как загружено на Kaggle):
      data_root/
        S02_animals1/
          S02_animals1/        ← двойная вложенность из-за архивирования (zip
                                  содержал папку сцены, Kaggle добавил ещё один
                                  уровень при распаковке) — поддерживается ниже
            low_light_10/      ← слабое освещение, уровень 10
              00001.png ...
            low_light_20/      ← слабое освещение, уровень 20
              00001.png ...
            normal_light_10/   ← GT для low_light_10 (в той же папке сцены!)
              00001.png ...
            normal_light_20/   ← GT для low_light_20
              00001.png ...
        S03_animals2/
          S03_animals2/ ...
        ...

    В отличие от многих датасетов, здесь НЕТ верхнеуровневого разделения на
    input/ и gt/ — low-light и normal-light (GT) кадры лежат рядом, внутри
    одной папки сцены, как подпапки low_light_* / normal_light_*.

    Класс терпим к обоим вариантам вложенности:
      data_root/SceneName/low_light_10/...              (без дублирования)
      data_root/SceneName/SceneName/low_light_10/...    (с дублированием, как сейчас на Kaggle)

    20 сцен × 2 уровня освещения = 40 пар последовательностей (в текущем подсете;
    официального train/test split нет — используем train_ratio).

    data_root может быть одним путём (str) или списком путей (list[str]) — второе
    используется, когда сцены разбиты по нескольким Kaggle-датасетам (например,
    подключены и -20-scene-subset, и -part-2 к одному ноутбуку) — все сцены со всех
    указанных каталогов объединяются в один пул до train/test сплита.

    Возвращает:
      frames:    [T, 3, H, W]  — окно из T low-light кадров
      target:    [3, H, W]     — GT текущего кадра
      timespans: [T]           — ∆t = 1/fps
      name:      str           — "S02_animals1/low_light_10/00005"
    """

    LIGHT_LEVELS = ['low_light_10', 'low_light_20']
    GT_LEVELS    = ['normal_light_10', 'normal_light_20']

    @staticmethod
    def _resolve_scene_content_dir(scene_dir: Path) -> Path:
        """
        Возвращает папку, где реально лежат low_light_*/normal_light_* подпапки.

        Если scene_dir сразу содержит low_light_10 и т.п. — возвращает scene_dir.
        Если внутри scene_dir есть ровно одна подпапка с тем же именем, в которой
        и лежат low_light_*/normal_light_* — это случай двойной вложенности
        (SceneName/SceneName/...), и нужно "спуститься" на уровень глубже.
        """
        if (scene_dir / 'low_light_10').exists() or (scene_dir / 'low_light_20').exists():
            return scene_dir

        nested = scene_dir / scene_dir.name
        if nested.is_dir() and (
            (nested / 'low_light_10').exists() or (nested / 'low_light_20').exists()
        ):
            return nested

        # Фоллбэк: одна-единственная подпапка-обёртка с любым именем
        subdirs = [d for d in scene_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and (
            (subdirs[0] / 'low_light_10').exists() or (subdirs[0] / 'low_light_20').exists()
        ):
            return subdirs[0]

        return scene_dir  # ничего не нашли — вызывающий код сам разберётся (пропустит сцену)

    def __init__(
        self,
        data_root,
        split: str = 'train',
        window_size: int = 5,
        patch_size: int = 256,
        augment: bool = True,
        fps: float = 30.0,
        train_ratio: float = 0.85,
        seed: int = 42,
    ):
        """
        data_root: путь к папке с сценами (str) ЛИБО список таких путей (list[str]).
        Список нужен, когда сцены распределены по разным Kaggle-датасетам
        (например /kaggle/input/bvi-rlv-...-20-scene-subset и
        /kaggle/input/bvi-rlv-...-part-2) — оба источника объединяются в один пул
        сцен до train/test сплита, без каких-либо копирований/symlink на диске.
        """
        super().__init__()
        self.window_size = window_size
        self.patch_size  = patch_size
        self.augment     = augment and (split == 'train')
        self.dt          = 1.0 / fps

        data_roots = [data_root] if isinstance(data_root, (str, Path)) else list(data_root)
        data_roots = [Path(r) for r in data_roots]
        for r in data_roots:
            if not r.exists():
                raise FileNotFoundError(f"BVI-RLV data_root not found: {r}")

        # Собираем все (сцена, уровень) пары со всех data_roots.
        # Каждая сцена — папка прямо в data_root (S02_animals1, S03_animals2, ...),
        # внутри которой (возможно, через один лишний уровень обёртки) лежат
        # low_light_10/low_light_20/normal_light_10/normal_light_20.
        all_seq_pairs = []
        skipped_scenes = []
        for root in data_roots:
            for scene_dir in sorted(root.iterdir()):
                if not scene_dir.is_dir():
                    continue
                content_dir = self._resolve_scene_content_dir(scene_dir)
                found_any = False
                for ll, nl in zip(self.LIGHT_LEVELS, self.GT_LEVELS):
                    low_dir = content_dir / ll
                    gt_dir  = content_dir / nl
                    if low_dir.exists() and gt_dir.exists():
                        # Сохраняем исходное имя сцены (scene_dir.name) явно — оно не
                        # зависит от того, есть ли двойная вложенность, и используется
                        # при построении поля имени в __getitem__.
                        all_seq_pairs.append((low_dir, gt_dir, scene_dir.name))
                        found_any = True
                if not found_any:
                    skipped_scenes.append(scene_dir.name)

        if skipped_scenes:
            print(
                f"[BVIRLVDataset] Пропущены сцены без low_light_*/normal_light_* "
                f"подпапок: {skipped_scenes}"
            )

        if not all_seq_pairs:
            raise RuntimeError(f"BVI-RLV: no sequences found in {data_roots}")

        # Детерминированный split
        rng = random.Random(seed)
        shuffled = all_seq_pairs.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio))
        selected = shuffled[:n_train] if split == 'train' else shuffled[n_train:]

        # Строим сэмплы (скользящее окно)
        exts = {'.png', '.jpg', '.jpeg'}
        self.samples = []
        for low_dir, gt_dir, scene_name in sorted(selected):
            low_frames = sorted([p for p in low_dir.iterdir() if p.suffix.lower() in exts])
            gt_frames  = sorted([p for p in gt_dir.iterdir()  if p.suffix.lower() in exts])
            n = min(len(low_frames), len(gt_frames))
            if n < window_size:
                continue
            for i in range(window_size - 1, n):
                window_lq = [low_frames[i - window_size + 1 + j] for j in range(window_size)]
                target_gt = gt_frames[i]
                self.samples.append((window_lq, target_gt, scene_name))

        if not self.samples:
            raise RuntimeError(
                f"BVIRLVDataset: no samples built (split={split}, "
                f"sequences={len(selected)}, window={window_size})"
            )

    def __len__(self):
        return len(self.samples)

    def _sync_crop(self, frames, target):
        _, H, W = frames[0].shape
        ps = self.patch_size
        if H < ps or W < ps:
            return frames, target
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        frames = [f[:, top:top + ps, left:left + ps] for f in frames]
        target = target[:, top:top + ps, left:left + ps]
        return frames, target

    def _sync_augment(self, frames, target):
        if random.random() > 0.5:
            frames = [TF.hflip(f) for f in frames]
            target = TF.hflip(target)
        if random.random() > 0.5:
            frames = [TF.vflip(f) for f in frames]
            target = TF.vflip(target)
        return frames, target

    def __getitem__(self, idx):
        lq_paths, gt_path, scene_name = self.samples[idx]
        frames = [load_image(str(p)) for p in lq_paths]
        target = load_image(str(gt_path))

        if self.augment:
            frames, target = self._sync_crop(frames, target)
            frames, target = self._sync_augment(frames, target)
        elif self.patch_size > 0:
            _, H, W = frames[0].shape
            ps = self.patch_size
            if H >= ps and W >= ps:
                top  = (H - ps) // 2
                left = (W - ps) // 2
                frames = [f[:, top:top + ps, left:left + ps] for f in frames]
                target = target[:, top:top + ps, left:left + ps]

        frames_t   = torch.stack(frames, dim=0)
        timespans  = torch.full((self.window_size,), self.dt)
        level      = lq_paths[-1].parent.name          # low_light_10
        frame_name = lq_paths[-1].stem                 # 00005
        return {
            'frames':    frames_t,
            'target':    target,
            'timespans': timespans,
            'name':      f"{scene_name}/{level}/{frame_name}",
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
