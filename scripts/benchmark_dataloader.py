"""
Бенчмарк трёх режимов DataLoader на одной сцене BVI-RLV — финальный шаг (3 из 3)
в проверке гипотезы "объединение кадров в один blob-файл + memmap ускоряет
загрузку данных против N отдельных файлов". Контекст и предыдущие шаги: см.
scripts/pack_scene_to_blob.py (упаковка) и data/blob_scene_dataset.py (Dataset).

РЕЖИМЫ:
  png   — BVIRLVDataset(npy_cache_root=None): чтение оригинальных PNG через PIL
          (Image.open + TF.to_tensor), baseline до любых оптимизаций.
  npy   — BVIRLVDataset(npy_cache_root=...): чтение из существующего .npy-кэша,
          один np.load() на кадр (текущее состояние прод-пайплайна).
  blob  — BlobSceneDataset: чтение из единого blob-файла сцены через np.memmap,
          без N отдельных файловых открытий (гипотеза этого эксперимента).

МЕТОДОЛОГИЯ — сопоставима с прошлым тестом PNG vs npy (см. mempalace
wing=LNN_LowLight room=decisions, "ПРЕДЛОЖЕНИЯ ПО УСКОРЕНИЮ DATALOADER"):
одна и та же сцена, одинаковый DataLoader(batch_size, num_workers), одинаковое
число прогреваемых и замеряемых батчей. Чтобы исключить влияние порядка
прогона (диск может попасть в page cache ОС после первого режима, искажая
последующие), каждый режим запускается в ОТДЕЛЬНОМ процессе (через --mode),
а внешний раннер (--all) вызывает их по очереди как subprocess — это даёт
каждому режиму чистый старт без переноса page cache между ними внутри
одного Python-процесса (хотя сам файл ОС всё равно может быть в кэше с
прошлого прогона — это намеренно: на Kaggle файлы датасета грузятся с
сетевой/виртуализированной FS впервые за сессию, так что "холодный" кэш
ближе к реальному паттерну использования, чем искусственный drop_caches,
который недоступен без root).

ВАЖНО: png и npy режимы используют РЕАЛЬНЫЙ BVIRLVDataset (data/datasets.py)
без каких-либо изменений — не переписанную копию — чтобы сравнение было
честным: один и тот же класс, тот же crop/augment, разница только в
npy_cache_root=None vs npy_cache_root=<путь>. Режим blob использует
BlobSceneDataset (data/blob_scene_dataset.py) с буквально скопированным
crop/augment контрактом для той же причины.

Использование:

    # Подготовка (если ещё не сделано для этой сцены):
    python scripts/preprocess_png_to_npy.py --data-root ... --out ... --scene S02
    python scripts/pack_scene_to_blob.py --npy-cache-root ... --data-root ... \
        --scene S02 --out /kaggle/working/bvi_rlv_blob --verify

    # Один режим:
    python scripts/benchmark_dataloader.py --mode png  --data-root ... --scene S02 ...
    python scripts/benchmark_dataloader.py --mode npy  --data-root ... --scene S02 \
        --npy-cache-root ... ...
    python scripts/benchmark_dataloader.py --mode blob --scene S02 \
        --blob-dir /kaggle/working/bvi_rlv_blob ...

    # Все три сразу, с итоговой сравнительной таблицей:
    python scripts/benchmark_dataloader.py --all \
        --data-root ... --npy-cache-root ... --blob-dir ... --scene S02

ПРОВЕРЕНО: на синтетической сцене (4 levels x 10 кадров, 300x320x3, реальные
декодируемые PNG) — все три режима по отдельности (--mode png/npy/blob) и в
составе --all (subprocess-раннер + сводная таблица + JSON), с num_workers=0
и num_workers=2. НАЙДЕН И ИСПРАВЛЕН БАГ: исходный train_ratio=0.85 в png/npy
режимах на одной тестовой сцене с двумя levels иногда отправлял один из
levels в test-сплит (BVIRLVDataset.train_ratio работает на уровне пар
low/gt, не сцен целиком), из-за чего png/npy получали 6 сэмплов, а blob
(всегда берёт оба level) — 12, что делало сравнение нечестным. Исправлено
форсированием train_ratio=1.0 в build_dataset() для png/npy режимов — после
фикса все три режима дают одинаковое число сэмплов на одной сцене. Также
проверена обработка отсутствующих путей в --all (--blob-dir без --data-root
корректно пропускает png/npy с понятным [SKIP] сообщением, таблица строится
даже с одним доступным режимом, без деления на None при отсутствии baseline).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# data/ должен быть импортируемым как пакет относительно корня проекта —
# скрипт предполагается запущенным из корня репозитория (python scripts/...),
# как и существующие scripts/preprocess_png_to_npy.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _filter_scene(dataset, scene_substr: str):
    """
    BVIRLVDataset строит сэмплы из ВСЕХ сцен train/test сплита — для бенчмарка
    нужна ровно одна сцена (как в прошлом тесте), поэтому после построения
    датасета отфильтровываем dataset.samples на месте, оставляя только сэмплы,
    чьё scene_name содержит scene_substr. Это проще и надёжнее, чем пытаться
    заставить train_ratio/seed оставить ровно одну нужную сцену в сплите.
    """
    before = len(dataset.samples)
    dataset.samples = [
        s for s in dataset.samples if scene_substr in s[2]   # s = (lq_paths, gt_path, scene_name, root)
    ]
    if not dataset.samples:
        raise RuntimeError(
            f"После фильтрации по сцене '{scene_substr}' не осталось сэмплов "
            f"(было {before}) — проверь, что сцена есть в train-сплите "
            f"(или попробуй другой --seed/--train-ratio)"
        )
    return dataset


def build_dataset(mode: str, args):
    if mode == 'png':
        from data.datasets import BVIRLVDataset
        ds = BVIRLVDataset(
            data_root=args.data_root,
            split='train',
            window_size=args.window_size,
            patch_size=args.patch_size,
            augment=True,
            # train_ratio=1.0 (не args.train_ratio) — ВАЖНО для честности
            # сравнения с blob-режимом. BVIRLVDataset.train_ratio работает на
            # уровне отдельных (low_level, gt_level) пар, не сцен целиком —
            # при единственной тестовой сцене с двумя levels (low_light_10,
            # low_light_20) обычный train_ratio=0.85 может случайно отправить
            # один из двух levels в test-сплит, оставив для бенчмарка только
            # один level вместо двух. BlobSceneDataset (blob-режим) всегда
            # строит сэмплы из ОБОИХ levels одной сцены без train/test сплита
            # — поэтому здесь форсируем train_ratio=1.0, чтобы png/npy тоже
            # включали оба level и число сэмплов совпадало с blob-режимом
            # (обнаружено при тестировании: с train_ratio=0.85, seed=42 на
            # одной сцене получалось 6 сэмплов вместо ожидаемых 12 — один
            # level отфильтровывался сплитом).
            train_ratio=1.0,
            seed=args.seed,
            npy_cache_root=None,
        )
        return _filter_scene(ds, args.scene)

    elif mode == 'npy':
        from data.datasets import BVIRLVDataset
        ds = BVIRLVDataset(
            data_root=args.data_root,
            split='train',
            window_size=args.window_size,
            patch_size=args.patch_size,
            augment=True,
            train_ratio=1.0,   # см. подробное обоснование в ветке 'png' выше
            seed=args.seed,
            npy_cache_root=args.npy_cache_root,
        )
        return _filter_scene(ds, args.scene)

    elif mode == 'blob':
        from data.blob_scene_dataset import BlobSceneDataset
        # BlobSceneDataset работает с одной конкретной сценой и ожидает полное
        # имя сцены (как у файла <scene_name>.blob), не подстроку — находим
        # полное имя по файлам в blob_dir, чтобы --scene S02 работал так же,
        # как и для png/npy режимов (где scene — подстрока).
        blob_dir = Path(args.blob_dir)
        matches = sorted(blob_dir.glob(f"*{args.scene}*.blob"))
        if not matches:
            raise FileNotFoundError(
                f"Не найден *{args.scene}*.blob в {blob_dir} — сначала прогони "
                f"scripts/pack_scene_to_blob.py для этой сцены"
            )
        if len(matches) > 1:
            raise RuntimeError(f"Найдено несколько blob для '{args.scene}': {matches}")
        scene_name = matches[0].stem
        return BlobSceneDataset(
            blob_dir=args.blob_dir,
            scene_name=scene_name,
            window_size=args.window_size,
            patch_size=args.patch_size,
            augment=True,
            fps=args.fps,
        )

    else:
        raise ValueError(f"Неизвестный режим: {mode}")


def run_single_mode(mode: str, args) -> dict:
    print(f"\n{'=' * 60}\nРЕЖИМ: {mode}\n{'=' * 60}")

    t_build_start = time.time()
    dataset = build_dataset(mode, args)
    t_build = time.time() - t_build_start
    print(f"Dataset построен за {t_build:.2f}с, сэмплов: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,   # сознательно False, как и в прошлой
                                     # диагностике COW-инцидента — не скрывать
                                     # потенциальный рост RSS персистентными
                                     # воркерами в этом замере
        pin_memory=False,           # выключено намеренно: цель — измерить
                                     # именно CPU-side loading, не смешивать
                                     # с H2D-копированием (это отдельный пункт
                                     # 3 из списка предложений, не сюда)
    )

    n_batches_total = len(loader)
    n_warmup = min(args.warmup_batches, max(0, n_batches_total - 1))
    n_measure = args.measure_batches
    if n_warmup + n_measure > n_batches_total:
        print(
            f"[WARN] warmup({n_warmup}) + measure({n_measure}) = "
            f"{n_warmup + n_measure} > доступных батчей ({n_batches_total}) — "
            f"уменьшаю measure_batches до {max(1, n_batches_total - n_warmup)}"
        )
        n_measure = max(1, n_batches_total - n_warmup)

    it = iter(loader)

    # Прогрев — первые батчи почти всегда медленнее (старт воркеров,
    # первое открытие файлов/первая загрузка страниц в page cache) и не
    # отражают устойчивую скорость — та же логика, что в прошлом тесте.
    for _ in range(n_warmup):
        next(it)

    batch_times = []
    t_prev = time.time()
    for i in range(n_measure):
        batch = next(it)
        t_now = time.time()
        batch_times.append(t_now - t_prev)
        t_prev = t_now
        # Лёгкая проверка формы — не часть замера времени (вне t_prev/t_now),
        # но даёт уверенность, что мы не замеряем скорость отдачи "пустых" батчей.
        assert batch['frames'].shape[0] <= args.batch_size

    avg = sum(batch_times) / len(batch_times)
    p50 = sorted(batch_times)[len(batch_times) // 2]
    result = {
        'mode': mode,
        'n_samples': len(dataset),
        'dataset_build_seconds': round(t_build, 3),
        'batches_measured': len(batch_times),
        'avg_seconds_per_batch': round(avg, 4),
        'median_seconds_per_batch': round(p50, 4),
        'min_seconds_per_batch': round(min(batch_times), 4),
        'max_seconds_per_batch': round(max(batch_times), 4),
        'samples_per_second': round(args.batch_size / avg, 2),
    }
    print(
        f"avg={result['avg_seconds_per_batch']}с/батч  "
        f"median={result['median_seconds_per_batch']}с/батч  "
        f"({result['samples_per_second']} сэмплов/с)"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['png', 'npy', 'blob'], help='Один конкретный режим.')
    parser.add_argument('--all', action='store_true', help='Прогнать все три режима последовательно (каждый в отдельном subprocess).')

    parser.add_argument('--scene', required=True, help="Подстрока имени сцены, например 'S02'.")
    parser.add_argument('--data-root', help='data_root для png/npy режимов (см. BVIRLVDataset).')
    parser.add_argument('--npy-cache-root', help='Корень .npy-кэша для npy режима.')
    parser.add_argument('--blob-dir', help='Директория с <scene>.blob/_index.npy/_meta.txt для blob режима.')

    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--warmup-batches', type=int, default=3)
    parser.add_argument('--measure-batches', type=int, default=29)

    parser.add_argument('--json-out', help='Опционально: куда сохранить результаты в JSON.')
    args = parser.parse_args()

    if not args.mode and not args.all:
        parser.error("Укажи --mode {png,npy,blob} или --all")

    if args.all:
        # Каждый режим — отдельный subprocess, чтобы page cache / память одного
        # режима не влияла на следующий внутри одного процесса (см. docstring
        # модуля). Передаём те же аргументы, кроме --all/--mode.
        results = []
        base_argv = [
            sys.executable, str(Path(__file__).resolve()),
            '--scene', args.scene,
            '--window-size', str(args.window_size),
            '--patch-size', str(args.patch_size),
            '--fps', str(args.fps),
            '--seed', str(args.seed),
            '--batch-size', str(args.batch_size),
            '--num-workers', str(args.num_workers),
            '--warmup-batches', str(args.warmup_batches),
            '--measure-batches', str(args.measure_batches),
        ]
        if args.data_root:
            base_argv += ['--data-root', args.data_root]
        if args.npy_cache_root:
            base_argv += ['--npy-cache-root', args.npy_cache_root]
        if args.blob_dir:
            base_argv += ['--blob-dir', args.blob_dir]

        for mode in ('png', 'npy', 'blob'):
            if mode == 'png' and not args.data_root:
                print(f"[SKIP] png: нужен --data-root")
                continue
            if mode == 'npy' and not (args.data_root and args.npy_cache_root):
                print(f"[SKIP] npy: нужны --data-root и --npy-cache-root")
                continue
            if mode == 'blob' and not args.blob_dir:
                print(f"[SKIP] blob: нужен --blob-dir")
                continue

            proc_argv = base_argv + ['--mode', mode, '--json-out', f'/tmp/_bench_{mode}.json']
            print(f"\n>>> Запускаю subprocess для режима '{mode}'...")
            proc = subprocess.run(proc_argv, capture_output=False)
            if proc.returncode != 0:
                print(f"[ERROR] режим '{mode}' завершился с кодом {proc.returncode}, пропускаю в сводке")
                continue
            with open(f'/tmp/_bench_{mode}.json') as f:
                results.append(json.load(f))

        print(f"\n{'=' * 70}\nСВОДНАЯ ТАБЛИЦА\n{'=' * 70}")
        header = f"{'режим':<6} {'avg с/батч':<12} {'median с/батч':<14} {'сэмплов/с':<10}"
        print(header)
        print('-' * len(header))
        baseline_avg = None
        for r in results:
            if r['mode'] == 'png':
                baseline_avg = r['avg_seconds_per_batch']
        for r in results:
            speedup = f" ({baseline_avg / r['avg_seconds_per_batch']:.2f}x vs png)" if baseline_avg else ""
            print(
                f"{r['mode']:<6} {r['avg_seconds_per_batch']:<12} "
                f"{r['median_seconds_per_batch']:<14} {r['samples_per_second']:<10}{speedup}"
            )

        if args.json_out:
            with open(args.json_out, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nРезультаты сохранены в {args.json_out}")
        return

    # Одиночный режим
    result = run_single_mode(args.mode, args)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
