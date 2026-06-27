"""
Тестовый Dataset, читающий единый blob-файл сцены (см. scripts/pack_scene_to_blob.py)
через np.memmap, БЕЗ .copy() на чтении — для проверки гипотезы "memmap на один
файл быстрее, чем N отдельных np.load() вызовов" (см. mempalace wing=LNN_LowLight,
room=decisions, "ПРЕДЛОЖЕНИЯ ПО УСКОРЕНИЮ DATALOADER" для полного контекста).

ЭТО СЛУЖЕБНЫЙ ТЕСТОВЫЙ КОД ДЛЯ ШАГА 2 ИЗ 3 (упаковка -> Dataset -> бенчмарк),
не интеграция в основной BVIRLVDataset (data/datasets.py). Если гипотеза
подтвердится бенчмарком (шаг 3), интеграция в прод будет отдельным, осознанным
шагом — этот файл написан как самостоятельный, лёгко удаляемый эксперимент,
не трогающий существующий пайплайн.

КОНТРАКТ __getitem__ повторяет BVIRLVDataset настолько, насколько это нужно для
честного сравнения скорости в бенчмарке (shape тензоров, augment/crop логика
скопированы буквально) — НЕ потому что это финальный класс для train.py.

Использование (после того как scripts/pack_scene_to_blob.py создал
<scene>.blob + <scene>_index.npy + <scene>_meta.txt для одной сцены):

    ds = BlobSceneDataset(
        blob_dir="/kaggle/working/bvi_rlv_blob",
        scene_name="S02_animals1",
        window_size=5,
        patch_size=256,
        augment=True,
    )
    sample = ds[0]
    # sample['frames']: [5, 3, H, W] float32 в [0, 1]
    # sample['target']: [3, H, W] float32 в [0, 1]
    # sample['timespans']: [5]
    # sample['name']: str

ПРОВЕРЕНО: на синтетической сцене (4 levels x 8 кадров, 300x320x3), упакованной
через pack_scene_to_blob.py — базовая инициализация и len(), shape тензоров в
режимах augment=True (random crop+flip) и augment=False (center crop),
побитовое совпадение значений пикселей с исходными .npy (torch.allclose),
работа с реальным DataLoader (num_workers=2, без ошибок multiprocessing),
обработка ошибок (отсутствующие файлы, window_size больше числа кадров).
Отдельно проверено и подавлено: UserWarning про non-writable memmap-массив
(ложноположительный для путей, которые пишут в тензор только после
.permute().contiguous(), создающего новый writable буфер) — подтверждено
запуском с -W error::UserWarning, что warning не возникает.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


# np.memmap(..., mode='r') возвращает read-only массив. torch.from_numpy() на
# таком массиве выдаёт UserWarning при КАЖДОМ вызове (тысячи раз за эпоху),
# потому что в общем случае запись в такой тензор было бы undefined behavior.
# Поэтому warning — ложноположительный конкретно для пути чтения в этом файле,
# и подавляется здесь явно, а не глобально для всего процесса. Любой путь,
# который реально пишет в тензор (аугментации, crop), делает это после
# .permute().contiguous() — а .contiguous() на non-contiguous view материализует
# НОВЫЙ обычный (writable) буфер, не трогая память memmap (проверено вручную:
# data_ptr меняется, исходные байты в .blob не меняются).
warnings.filterwarnings(
    'ignore',
    message='The given NumPy array is not writable',
    category=UserWarning,
)

LIGHT_LEVELS = ['low_light_10', 'low_light_20']
GT_LEVELS    = ['normal_light_10', 'normal_light_20']


def _read_meta(meta_path: Path) -> dict:
    """
    Парсит <scene>_meta.txt (см. scripts/pack_scene_to_blob.py) — простой
    key=value формат, не JSON/YAML, чтобы не тащить лишнюю зависимость для
    четырёх строк служебных метаданных.
    """
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            meta[key] = value
    required = {'frame_shape', 'frame_nbytes', 'total_frames', 'dtype'}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"В {meta_path} отсутствуют поля: {missing}")
    return meta


class BlobSceneDataset(Dataset):
    """
    Датасет для ОДНОЙ сцены BVI-RLV, читающий кадры из единого blob-файла через
    np.memmap вместо отдельного np.load() на каждый кадр.

    В отличие от BVIRLVDataset (data/datasets.py), здесь нет train/test сплита
    по сценам и нет поддержки нескольких data_root — это намеренное упрощение
    для теста гипотезы на одной сцене (S02), а не недостаток функциональности,
    который нужно "доделать" здесь. Многосценовая версия — отдельный вопрос,
    если результат бенчмарка оправдает интеграцию в прод.

    blob_dir: директория, где лежат <scene_name>.blob, <scene_name>_index.npy,
              <scene_name>_meta.txt (результат pack_scene_to_blob.py).
    """

    def __init__(
        self,
        blob_dir: str,
        scene_name: str,
        window_size: int = 5,
        patch_size: int = 256,
        augment: bool = True,
        fps: float = 30.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_size  = patch_size
        self.augment     = augment
        self.dt          = 1.0 / fps

        blob_dir = Path(blob_dir)
        self.blob_path  = blob_dir / f"{scene_name}.blob"
        index_path      = blob_dir / f"{scene_name}_index.npy"
        meta_path       = blob_dir / f"{scene_name}_meta.txt"

        for p in (self.blob_path, index_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Не найден {p} — сначала прогони "
                    f"scripts/pack_scene_to_blob.py для сцены '{scene_name}'"
                )

        meta = _read_meta(meta_path)
        h, w, c = (int(x) for x in meta['frame_shape'].split(','))
        self.frame_shape  = (h, w, c)
        self.frame_nbytes = int(meta['frame_nbytes'])
        self.total_frames = int(meta['total_frames'])
        if meta['dtype'] != 'uint8':
            raise ValueError(
                f"Ожидался dtype=uint8 в {meta_path}, получено {meta['dtype']}"
            )

        # index — structured array (см. INDEX_DTYPE в pack_scene_to_blob.py):
        # rel_path (S256), level (S32), frame_idx (int64), offset (int64, в кадрах).
        # Грузим целиком в обычную (не memmap) память — это маленький массив
        # метаданных (десятки тысяч записей максимум на сцену), не сами пиксели.
        self.index = np.load(index_path)
        if len(self.index) != self.total_frames:
            raise ValueError(
                f"Несоответствие: index содержит {len(self.index)} записей, "
                f"а meta.total_frames={self.total_frames} — index.npy и blob "
                f"рассинхронизированы, пересобери сцену"
            )

        # БЕЗ .copy(): np.memmap открывает blob read-only и отдаёт view на
        # страницы файла через OS page cache, без полного чтения в RAM при
        # открытии и без копии всего файла в process memory — см. согласованное
        # решение в mempalace (room=decisions): "копия совсем без копий"
        # невозможна (приведение типа .float() всё равно копирует), но копия
        # НА ЭТАПЕ ЧТЕНИЯ С ДИСКА исключается этим способом.
        self._blob_mm = np.memmap(
            self.blob_path, dtype=np.uint8, mode='r',
        ).reshape(self.total_frames, h, w, c)

        # group_by level, в исходном порядке (low_light_10, low_light_20,
        # normal_light_10, normal_light_20) — соответствует тому, как
        # pack_scene_to_blob.py укладывал кадры в blob (ALL_LEVELS).
        self._frames_by_level = {}
        for level in LIGHT_LEVELS + GT_LEVELS:
            mask = self.index['level'] == level.encode('utf-8')
            idxs = np.nonzero(mask)[0]
            if len(idxs) > 0:
                # idxs уже идут по возрастанию frame_idx, потому что
                # pack_scene_to_blob.py писал кадры level'а подряд, без
                # перемешивания — но сортируем явно по frame_idx, чтобы не
                # зависеть от этого негласного порядка записи.
                order = np.argsort(self.index['frame_idx'][idxs])
                self._frames_by_level[level] = idxs[order]

        self.scene_name = scene_name

        # Строим сэмплы скользящего окна — та же логика, что в
        # BVIRLVDataset.__init__ (data/datasets.py), но здесь работаем не с
        # путями к файлам, а с позициями (int) внутри self._blob_mm.
        self.samples = []   # (lq_blob_idxs: list[int], gt_blob_idx: int, level: str, frame_name: str)
        for ll, nl in zip(LIGHT_LEVELS, GT_LEVELS):
            if ll not in self._frames_by_level or nl not in self._frames_by_level:
                continue
            low_idxs = self._frames_by_level[ll]
            gt_idxs  = self._frames_by_level[nl]
            n = min(len(low_idxs), len(gt_idxs))
            if n < window_size:
                continue
            for i in range(window_size - 1, n):
                window_lq = [
                    int(low_idxs[i - window_size + 1 + j]) for j in range(window_size)
                ]
                target_gt = int(gt_idxs[i])
                # frame_name берём из rel_path последнего кадра окна — то же
                # происхождение строки имени, что в BVIRLVDataset
                # (lq_paths[-1].stem), просто из индекса, а не из Path.
                last_rel = self.index['rel_path'][window_lq[-1]].decode('utf-8')
                frame_name = Path(last_rel).stem
                self.samples.append((window_lq, target_gt, ll, frame_name))

        if not self.samples:
            raise RuntimeError(
                f"BlobSceneDataset: не удалось построить ни одного сэмпла для "
                f"сцены '{scene_name}' (window_size={window_size}) — проверь, "
                f"что в index есть оба levels с достаточным числом кадров"
            )

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, blob_idx: int) -> torch.Tensor:
        """
        Кадр blob_idx -> тензор [3, H, W] float32 в [0, 1].

        self._blob_mm[blob_idx] — view на memmap, БЕЗ копии при индексации
        (numpy basic indexing по первой оси возвращает view, не copy — в
        отличие от fancy indexing списком/массивом индексов). torch.from_numpy
        также не копирует. Копия неизбежно происходит только на .float()
        (создание нового тензора другого dtype) — то же место, где она была
        бы в load_image_npy() из data/datasets.py.
        """
        arr = self._blob_mm[blob_idx]                         # (H, W, 3) uint8, view
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t.float().div(255.0)

    def _sync_crop(self, frames, target):
        """Буквальная копия BVIRLVDataset._sync_crop (data/datasets.py)."""
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
        """Буквальная копия BVIRLVDataset._sync_augment (data/datasets.py)."""
        if random.random() > 0.5:
            frames = [TF.hflip(f) for f in frames]
            target = TF.hflip(target)
        if random.random() > 0.5:
            frames = [TF.vflip(f) for f in frames]
            target = TF.vflip(target)
        return frames, target

    def __getitem__(self, idx):
        lq_blob_idxs, gt_blob_idx, level, frame_name = self.samples[idx]

        frames = [self._load_frame(i) for i in lq_blob_idxs]
        target = self._load_frame(gt_blob_idx)

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

        frames_t  = torch.stack(frames, dim=0)
        timespans = torch.full((self.window_size,), self.dt)
        return {
            'frames':    frames_t,
            'target':    target,
            'timespans': timespans,
            'name':      f"{self.scene_name}/{level}/{frame_name}",
        }
