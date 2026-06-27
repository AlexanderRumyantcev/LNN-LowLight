"""
Упаковка .npy-кэша одной сцены BVI-RLV в единый blob-файл + индекс (вариант A
из обсуждения: один файл на сцену + отдельный индекс с offset'ами, максимально
близкий к будущей прод-интеграции в BVIRLVDataset).

КОНТЕКСТ: после перехода PNG -> .npy (scripts/preprocess_png_to_npy.py) каждый
кадр читается через отдельный np.load() — то есть отдельный open()/read()/close()
syscall на каждый из тысяч мелких файлов на эпоху. Гипотеза: если собрать все
кадры сцены в один blob-файл и читать его через np.memmap (без открытия файла
заново на каждый __getitem__), число syscall'ов резко падает, что особенно важно
на сетевой/виртуализированной FS Kaggle.

ЭТОТ СКРИПТ ТОЛЬКО УПАКОВЫВАЕТ — не трогает существующий .npy-кэш и не меняет
BVIRLVDataset. Берёт уже сконвертированные .npy для одной сцены и складывает их в:

  <out>/<scene_name>.blob       — один файл, конкатенация всех кадров подряд,
                                   каждый кадр (H, W, 3) uint8, без зазоров
  <out>/<scene_name>_index.npy  — структурированный массив с метаданными на
                                   каждый кадр: исходный относительный путь
                                   (для отладки/проверки), offset (в кадрах),
                                   level (low_light_10/20, normal_light_10/20)

Формат index — structured numpy array (не JSON), чтобы не тащить лишнюю
зависимость и не плодить форматы: то же соглашение, что и у самого .npy-кэша.
Поля специально избыточны (rel_path хранится явно, не только level+frame_idx) —
это даёт возможность на этапе теста сверить, что blob[i] действительно
соответствует ожидаемому исходному файлу (см. --verify), не полагаясь только
на порядок сборки.

ВАЖНО: размер кадра (H, W) фиксируется по первому найденному файлу и проверяется
для всех остальных — если в сцене окажутся кадры разного размера, скрипт упадёт
с понятной ошибкой, а не молча запишет битый blob. Для BVI-RLV это не ожидается
(все кадры одной сцены сняты одной камерой), но проверка должна быть явной, а не
предположением.

Использование (на одной сцене, как и предыдущий тест PNG vs .npy):

    python scripts/pack_scene_to_blob.py \
        --npy-cache-root /kaggle/working/bvi_rlv_npy \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-20-scene-subset \
        --scene S02 \
        --out /kaggle/working/bvi_rlv_blob \
        --verify

--data-root нужен только чтобы найти исходную папку сцены и зеркалить ту же
структуру (low_light_10/20, normal_light_10/20), которую строит
BVIRLVDataset._resolve_scene_content_dir — без дублирования этой логики тут
было бы легко разойтись с реальной раскладкой на диске.

ПРОВЕРЕНО: на синтетической структуре, имитирующей реальную раскладку BVI-RLV
(двойная вложенность SceneName/SceneName/, 4 уровня по 6 кадров) — успешный путь
с --verify (0 расхождений) и аварийный путь (кадр с другим shape корректно
прерывает сборку с понятной ошибкой и подчищает временный .blob.tmp файл,
не оставляя битых артефактов).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


LIGHT_LEVELS = ['low_light_10', 'low_light_20']
GT_LEVELS    = ['normal_light_10', 'normal_light_20']
ALL_LEVELS   = LIGHT_LEVELS + GT_LEVELS
EXTS         = {'.png', '.jpg', '.jpeg'}

# Индекс хранит rel_path как фиксированную ширину байтовой строки — numpy
# structured array не умеет переменную длину строк без object dtype (а object
# dtype в .npy сериализуется через pickle, чего хотелось бы избежать для
# простого служебного индекса). 256 байт хватает с большим запасом на
# "low_light_10/00001.png" (меньше 30 символов).
REL_PATH_MAXLEN = 256
LEVEL_MAXLEN    = 32

INDEX_DTYPE = np.dtype([
    ('rel_path', f'S{REL_PATH_MAXLEN}'),
    ('level',    f'S{LEVEL_MAXLEN}'),
    ('frame_idx', np.int64),    # порядковый номер кадра внутри своего level (0-based)
    ('offset',    np.int64),    # offset В КАДРАХ (не байтах) от начала blob — см. ниже
])


def resolve_scene_content_dir(scene_dir: Path) -> Path:
    """
    Точная копия BVIRLVDataset._resolve_scene_content_dir (data/datasets.py) —
    держать идентичной логике там, иначе пути в индексе разойдутся с тем, что
    реально читает Dataset. Скопировано, а не импортировано, чтобы этот скрипт
    оставался самодостаточным служебным инструментом для эксперимента, не
    тянущим зависимость на основной пакет проекта.
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


def find_scene_dir(data_root: Path, scene_filter: str) -> Path:
    """Находит ровно одну папку сцены в data_root, чьё имя содержит scene_filter."""
    matches = [
        d for d in sorted(data_root.iterdir())
        if d.is_dir() and scene_filter in d.name
    ]
    if not matches:
        raise FileNotFoundError(
            f"Не найдено ни одной сцены с '{scene_filter}' в имени внутри {data_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Найдено несколько сцен с '{scene_filter}' в имени: "
            f"{[m.name for m in matches]} — уточни --scene"
        )
    return matches[0]


def collect_frame_paths(content_dir: Path, npy_cache_root: Path, data_root: Path):
    """
    Для каждого level (low_light_10/20, normal_light_10/20) собирает
    отсортированный список .npy путей (внутри npy_cache_root, зеркалящих
    структуру content_dir) — тот же relative_to(), которым пользуется
    BVIRLVDataset._npy_path_for, но здесь напрямую от data_root, не от
    динамического self.npy_cache_root.

    Возвращает dict: level -> list[(rel_path_str, npy_path)], упорядоченный
    так же, как level_dir.iterdir() в сортированном виде (== порядок, в
    котором BVIRLVDataset строит low_frames/gt_frames).
    """
    result = {}
    for level in ALL_LEVELS:
        level_dir = content_dir / level
        if not level_dir.exists():
            continue
        png_paths = sorted([p for p in level_dir.iterdir() if p.suffix.lower() in EXTS])
        pairs = []
        for png_path in png_paths:
            rel = png_path.relative_to(data_root)
            npy_path = (npy_cache_root / rel).with_suffix('.npy')
            pairs.append((str(rel), npy_path))
        if pairs:
            result[level] = pairs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--npy-cache-root', required=True,
        help='Корень уже существующего .npy-кэша (см. scripts/preprocess_png_to_npy.py).',
    )
    parser.add_argument(
        '--data-root', required=True,
        help='Исходный data_root (используется только для разрешения структуры '
             'папок сцены — относительно него строились пути в --npy-cache-root).',
    )
    parser.add_argument(
        '--scene', required=True,
        help="Подстрока имени сцены, например 'S02' для S02_animals1.",
    )
    parser.add_argument(
        '--out', required=True,
        help='Куда сохранить <scene>.blob и <scene>_index.npy.',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='После упаковки перечитать blob через memmap и сверить с исходными '
             '.npy файлами через np.array_equal — медленнее, но безопаснее для '
             'первого прогона на новой сцене.',
    )
    args = parser.parse_args()

    npy_cache_root = Path(args.npy_cache_root)
    data_root      = Path(args.data_root)
    out_dir        = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        print(f"[ERROR] data_root не найден: {data_root}", file=sys.stderr)
        sys.exit(1)
    if not npy_cache_root.exists():
        print(f"[ERROR] npy_cache_root не найден: {npy_cache_root}", file=sys.stderr)
        sys.exit(1)

    scene_dir = find_scene_dir(data_root, args.scene)
    content_dir = resolve_scene_content_dir(scene_dir)
    scene_name = scene_dir.name
    print(f"[1/4] Сцена: {scene_name} (содержимое в {content_dir})")

    frames_by_level = collect_frame_paths(content_dir, npy_cache_root, data_root)
    if not frames_by_level:
        print(
            f"[ERROR] Не найдено ни одного level (low_light_*/normal_light_*) "
            f"в {content_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_frames = sum(len(v) for v in frames_by_level.values())
    print(
        f"      Найдено уровней: {list(frames_by_level.keys())}, "
        f"всего кадров: {total_frames}"
    )

    # Порядок level'ов фиксируем явно (ALL_LEVELS), а не порядок dict — чтобы
    # сборка blob'а была детерминирована независимо от порядка iterdir().
    ordered_levels = [lv for lv in ALL_LEVELS if lv in frames_by_level]

    print("[2/4] Проверяю размер кадра по первому файлу и собираю список в нужном порядке...")
    flat_entries = []   # (rel_path, npy_path, level, frame_idx)
    for level in ordered_levels:
        for frame_idx, (rel_path, npy_path) in enumerate(frames_by_level[level]):
            if not npy_path.exists():
                print(
                    f"[ERROR] Ожидаемый .npy не найден: {npy_path} "
                    f"(rel={rel_path}) — сначала прогони preprocess_png_to_npy.py "
                    f"на эту сцену",
                    file=sys.stderr,
                )
                sys.exit(1)
            flat_entries.append((rel_path, npy_path, level, frame_idx))

    first_arr = np.load(flat_entries[0][1])
    if first_arr.dtype != np.uint8:
        print(
            f"[ERROR] Ожидался dtype uint8, получено {first_arr.dtype} — "
            f"проверь, что .npy-кэш создан текущей версией preprocess_png_to_npy.py",
            file=sys.stderr,
        )
        sys.exit(1)
    frame_shape = first_arr.shape   # (H, W, 3)
    frame_nbytes = first_arr.nbytes
    print(f"      Размер кадра: {frame_shape}, {frame_nbytes} байт/кадр")

    blob_path  = out_dir / f"{scene_name}.blob"
    index_path = out_dir / f"{scene_name}_index.npy"

    print(f"[3/4] Пишу blob ({total_frames} кадров x {frame_nbytes} байт)...")
    t0 = time.time()
    index_records = np.zeros(total_frames, dtype=INDEX_DTYPE)

    # Атомарность как и в preprocess_png_to_npy.py: пишем во временный файл,
    # переименовываем в конце — частично записанный blob после прерывания
    # сессии не должен выглядеть как валидный готовый файл.
    tmp_blob_path = blob_path.with_suffix('.blob.tmp')
    with open(tmp_blob_path, 'wb') as f:
        for i, (rel_path, npy_path, level, frame_idx) in enumerate(flat_entries):
            arr = np.load(npy_path)
            if arr.shape != frame_shape:
                print(
                    f"[ERROR] Кадр {rel_path} имеет shape {arr.shape}, "
                    f"ожидался {frame_shape} (по первому кадру сцены) — "
                    f"сборка единого blob'а требует одинакового размера всех "
                    f"кадров, прерываю",
                    file=sys.stderr,
                )
                tmp_blob_path.unlink(missing_ok=True)
                sys.exit(1)
            if arr.dtype != np.uint8:
                print(
                    f"[ERROR] Кадр {rel_path} имеет dtype {arr.dtype}, ожидался uint8",
                    file=sys.stderr,
                )
                tmp_blob_path.unlink(missing_ok=True)
                sys.exit(1)

            f.write(np.ascontiguousarray(arr).tobytes())

            rec = index_records[i]
            rec['rel_path']  = rel_path.encode('utf-8')[:REL_PATH_MAXLEN]
            rec['level']     = level.encode('utf-8')[:LEVEL_MAXLEN]
            rec['frame_idx'] = frame_idx
            rec['offset']    = i   # offset в КАДРАХ — байтовый offset = i * frame_nbytes,
                                    # вычисляется на стороне читателя зная frame_shape,
                                    # чтобы не дублировать frame_nbytes в каждой записи

            if (i + 1) % 500 == 0 or (i + 1) == total_frames:
                dt = time.time() - t0
                rate = (i + 1) / dt if dt > 0 else 0
                print(f"      {i + 1}/{total_frames} | {rate:.1f} кадров/с", flush=True)

    tmp_blob_path.rename(blob_path)

    # Индекс сохраняем отдельно через временный файл по той же причине.
    # Суффикс именно '.tmp.npy' (не '.npy.tmp') — np.save() сам добавляет '.npy'
    # к пути, если он ещё так не заканчивается; с '.npy.tmp' это привело бы к
    # файлу 'X.npy.tmp.npy' и последующему rename() с FileNotFoundError — та же
    # грабля, что уже была учтена в scripts/preprocess_png_to_npy.py.
    tmp_index_path = index_path.with_suffix('.tmp.npy')
    np.save(tmp_index_path, index_records)
    tmp_index_path.rename(index_path)

    dt_total = time.time() - t0
    blob_size_gb = blob_path.stat().st_size / 1024**3
    print(
        f"[4/4] Готово за {dt_total:.1f}с. "
        f"blob={blob_path} ({blob_size_gb:.2f} GB), index={index_path}"
    )

    # Метаданные о shape кадра сознательно не пишем внутрь index.npy (structured
    # dtype не несёт произвольных атрибутов) — сохраняем рядом отдельным,
    # человекочитаемым файлом, чтобы тестовый Dataset/бенчмарк (следующий шаг)
    # не пересчитывал и не угадывал H, W заново.
    meta_path = out_dir / f"{scene_name}_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"frame_shape={frame_shape[0]},{frame_shape[1]},{frame_shape[2]}\n")
        f.write(f"frame_nbytes={frame_nbytes}\n")
        f.write(f"total_frames={total_frames}\n")
        f.write(f"dtype=uint8\n")
    print(f"      meta={meta_path}")

    if args.verify:
        print("\n[VERIFY] Перечитываю blob через memmap и сравниваю с исходными .npy...")
        blob_mm = np.memmap(blob_path, dtype=np.uint8, mode='r')
        blob_mm = blob_mm.reshape(total_frames, *frame_shape)
        n_checked, n_mismatch = 0, 0
        t_verify = time.time()
        for i, (rel_path, npy_path, level, frame_idx) in enumerate(flat_entries):
            original = np.load(npy_path)
            if not np.array_equal(blob_mm[i], original):
                n_mismatch += 1
                print(f"  [MISMATCH] кадр {i} ({rel_path}) не совпадает с blob", file=sys.stderr)
            n_checked += 1
        del blob_mm
        print(
            f"[VERIFY] Проверено {n_checked} кадров за {time.time() - t_verify:.1f}с, "
            f"расхождений: {n_mismatch}"
        )
        if n_mismatch > 0:
            print("[VERIFY] FAILED — blob не идентичен исходным .npy, не используй его как есть", file=sys.stderr)
            sys.exit(1)
        else:
            print("[VERIFY] OK — blob побитово идентичен исходным .npy")


if __name__ == '__main__':
    main()
