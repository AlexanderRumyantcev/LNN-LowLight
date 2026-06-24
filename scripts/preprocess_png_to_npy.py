"""
Препроцессинг PNG -> .npy для BVI-RLV (вариант A: ускорение DataLoader без
изменения содержимого пикселей — см. data/datasets.py: load_image_npy,
BVIRLVDataset.npy_cache_root).

Идея: PNG-decode (Image.open().convert('RGB') + TF.to_tensor) выполняется заново
на каждый __getitem__, даже для кадров, которые уже декодировались в соседних
сэмплах скользящего окна. Этот скрипт декодирует каждый PNG РОВНО ОДИН РАЗ и
сохраняет результат как uint8 (H, W, 3) в [0, 255] — тот же массив, который
возвращает PIL.Image, без каста к float32 (чтобы кэш был в 4р легче на диске) —
нормализация /255.0 переносится в load_image_npy(), вызываемое при загрузке
каждого сэмпла (там же, где это делает TF.to_tensor() внутри load_image()).
Само чтение .npy становится быстрым memcpy (np.load), без zlib-инфляции и без
lossy-сжатия, которые делают PNG-decode дорогим.

ВАЖНО: это НЕ ресайз и НЕ сжатие — побитовое (с точностью uint8/255.0 == исходный
пиксель PNG) содержание кадра не меняется. Это отдельный вопрос (вариант B),
который нужно решать осознанно на уровне архитектуры, а не как побочный эффект ускорения I/O.

Быстрая проверка гипотезы на одной сцене (рекомендуется начинать именно с этого):

    python scripts/preprocess_png_to_npy.py \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-20-scene-subset \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-part-2 \
        --out-root /kaggle/working/bvi_rlv_npy \
        --scene S02 \
        --workers 4

Полный датасет (когда гипотеза подтвердится и потребуется место на все 20 сцен) —
просто убрать --scene:

    python scripts/preprocess_png_to_npy.py \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-20-scene-subset \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-part-2 \
        --out-root /kaggle/working/bvi_rlv_npy \
        --workers 4

После завершения — в конфиге добавить:
    dataset:
      npy_cache_root: /kaggle/working/bvi_rlv_npy

И пересоздать train/val датасеты (build_dataset подхватит npy_cache_root
автоматически через cfg.get('npy_cache_root')).
"""

import argparse
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image


LIGHT_LEVELS = ['low_light_10', 'low_light_20']
GT_LEVELS    = ['normal_light_10', 'normal_light_20']
EXTS         = {'.png', '.jpg', '.jpeg'}


def resolve_scene_content_dir(scene_dir: Path) -> Path:
    """
    Копия BVIRLVDataset._resolve_scene_content_dir — должна оставаться идентичной
    логике в data/datasets.py, иначе пути PNG и .npy разойдутся.
    """
    if (scene_dir / 'low_light_10').exists() or (scene_dir / 'low_light_20').exists():
        return scene_dir

    nested = scene_dir / scene_dir.name
    if nested.is_dir() and (
        (nested / 'low_light_10').exists() or (nested / 'low_light_20').exists()
    ):
        return nested

    subdirs = [d for d in scene_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and (
        (subdirs[0] / 'low_light_10').exists() or (subdirs[0] / 'low_light_20').exists()
    ):
        return subdirs[0]

    return scene_dir


def collect_png_paths(data_roots: list[Path], scene_filter: str | None = None) -> list[Path]:
    """
    Собирает все PNG-пути (low_light_* и normal_light_* во всех сценах всех
    data_roots) — без дублирования логики train/test split из BVIRLVDataset,
    потому что препроцессинг должен покрыть ВСЕ кадры (train и val вместе),
    чтобы npy_cache_root был полным независимо от того, как сделан сплит.

    scene_filter: если указан — оставляет только сцену(ы), имя которых совпадает
    (или совпадают по подстроке) с scene_dir.name — удобно для быстрого теста
    гипотезы на одной-двух сценах без конвертации всего датасета.
    """
    all_paths = []
    for root in data_roots:
        if not root.exists():
            print(f"[WARN] data_root не найден, пропускаю: {root}", file=sys.stderr)
            continue
        for scene_dir in sorted(root.iterdir()):
            if not scene_dir.is_dir():
                continue
            if scene_filter is not None and scene_filter not in scene_dir.name:
                continue
            content_dir = resolve_scene_content_dir(scene_dir)
            for level in LIGHT_LEVELS + GT_LEVELS:
                level_dir = content_dir / level
                if not level_dir.exists():
                    continue
                for p in sorted(level_dir.iterdir()):
                    if p.suffix.lower() in EXTS:
                        all_paths.append((p, root))
    return all_paths


def convert_one(args) -> tuple[bool, str]:
    """
    Декодирует один PNG и сохраняет .npy по зеркальному пути.
    Сохраняет ровно тот же uint8 [0, 255] массив, который возвращает PIL — без
    каста к float32, чтобы кэш был в 4р легче (см. load_image_npy в data/datasets.py,
    где делается /255.0 при загрузке каждого сэмпла).
    Возвращает (success, message) — для агрегации ошибок в основном процессе.
    """
    png_path, root, out_root = args
    try:
        rel = png_path.relative_to(root)
        npy_path = (out_root / rel).with_suffix('.npy')

        if npy_path.exists():
            return True, f"skip (exists): {rel}"

        npy_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(png_path).convert('RGB')
        arr = np.asarray(img, dtype=np.uint8)          # (H, W, 3), [0, 255] — без /255.0

        # Атомарность: пишем во временный файл и переименовываем — защита от
        # повреждённого .npy при прерывании сессии Kaggle посреди записи.
        # Имя временного файла нарочно заканчивается на .tmp.npy (не .npy.tmp) —
        # np.save() автоматически добавляет '.npy' к пути, если он ещё так не
        # заканчивается — с '.npy.tmp' он ошибочно создаёт 'X.npy.tmp.npy' вместо
        # ожидаемого 'X.npy.tmp', и следующий rename() падает с FileNotFoundError.
        tmp_path = npy_path.with_suffix('.tmp.npy')
        np.save(tmp_path, arr)
        tmp_path.rename(npy_path)

        return True, f"ok: {rel}"
    except Exception as e:
        return False, f"FAIL {png_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root', action='append', required=True,
        help='Путь к BVI-RLV data_root. Можно указать несколько раз для нескольких Kaggle-датасетов.',
    )
    parser.add_argument(
        '--out-root', required=True,
        help='Куда сохранять .npy-кэш (например /kaggle/working/bvi_rlv_npy).',
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Число процессов для параллельного decode (по умолчанию 4 — столько же, '
             'сколько num_workers в DataLoader после фикса).',
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Ограничить число файлов (для быстрого smoke-теста перед полным прогоном).',
    )
    parser.add_argument(
        '--scene', default=None,
        help='Конвертировать только сцену(ы), имя которых содержит эту подстроку '
             '(например --scene S02 возьмёт только S02_animals1). '
             'Для быстрой проверки гипотезы на одной сцене без конвертации всего датасета.',
    )
    args = parser.parse_args()

    data_roots = [Path(r) for r in args.data_root]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Собираю список PNG из {len(data_roots)} data_root(s)" + (f" (сцена фильтр: '{args.scene}')" if args.scene else "") + "...")
    t0 = time.time()
    pairs = collect_png_paths(data_roots, scene_filter=args.scene)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    print(f"      Найдено {len(pairs)} файлов за {time.time() - t0:.1f}с")

    if not pairs:
        print("[ERROR] Не найдено ни одного PNG — проверь --data-root", file=sys.stderr)
        sys.exit(1)

    tasks = [(p, root, out_root) for p, root in pairs]

    print(f"[2/3] Конвертирую PNG -> .npy ({args.workers} процессов)...")
    t0 = time.time()
    n_ok, n_fail, n_skip = 0, 0, 0
    failures = []

    with Pool(args.workers) as pool:
        for i, (success, msg) in enumerate(pool.imap_unordered(convert_one, tasks, chunksize=16), 1):
            if success:
                if msg.startswith("skip"):
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                failures.append(msg)

            if i % 500 == 0 or i == len(tasks):
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                print(
                    f"      {i}/{len(tasks)} | {rate:.1f} файлов/с | "
                    f"ok={n_ok} skip={n_skip} fail={n_fail}",
                    flush=True,
                )

    dt_total = time.time() - t0
    print(f"[3/3] Готово за {dt_total/60:.1f} мин. ok={n_ok} skip={n_skip} fail={n_fail}")

    if failures:
        print(f"\n[WARN] {len(failures)} файлов не удалось сконвертировать:", file=sys.stderr)
        for f in failures[:20]:
            print(f"  {f}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... и ещё {len(failures) - 20}", file=sys.stderr)

    # Грубая оценка объёма на диске — uint8 .npy без сжатия всё равно тяжелее PNG
    # (PNG исходно сжат deflate'ом), это ожидаемая плата за скорость чтения —
    # стоит явно показать цифру, чтобы оценить при расширении с одной сцены на весь датасет.
    total_npy_bytes = sum(f.stat().st_size for f in out_root.rglob('*.npy'))
    print(f"\nИтоговый размер .npy-кэша: {total_npy_bytes / 1024**3:.1f} GB")


if __name__ == '__main__':
    main()
