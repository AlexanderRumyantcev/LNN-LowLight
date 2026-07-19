"""
Этап 2 ТЗ motion alignment: офлайн-предвычисление optical flow (i -> i+1)
для всего датасета BVI-RLV.

КОНТЕКСТ (см. TZ_motion_alignment_LNN_LowLight.md и mempalace wing=LNN_LowLight):
Текущий temporal-pipeline (окно 5 кадров) берёт кадры по фиксированным пиксельным
координатам без компенсации движения. Цель всей работы — align-then-fuse: сначала
выровнять кадры по движению (этот скрипт считает flow), потом (Этап 3) построить
композицию потоков для произвольного окна, потом (Этап 4) применить warping
онлайн в dataloader. Здесь — только вычисление и кэширование flow, ничего больше.

ПРИНЯТЫЕ РЕШЕНИЯ (Этапы 1/1.5, Блокеры 1-2 — не пересматривать без явной причины):
  - Разрешение flow: half-res (scale=0.5 от исходного 1080p, т.е. ~960x540) —
    точность выравнивания вдвое выше quarter-res, overhead по I/O незначителен
    (+18.5% на батч по замеру Этапа 1.5).
  - Метод: RAFT-small. Кэшируются flow-поля (fp16), не готовые warped-кадры —
    экономия места, warping будет выполняться онлайн в Этапе 4.
  - Бюджет: ~32.7GB на весь датасет (16172 flow-полей, half-res, fp16) —
    подтверждено диагностикой Блокера 1 (2026-07-18), считать бюджет на
    целевом диске ПЕРЕД полным прогоном (известная проблема Kaggle:
    переполнение /kaggle/working при одновременном хранении PNG+NPY+blob+flow).
  - Flow считается на normal_light_XX той же сцены/уровня (НЕ на low_light!) —
    см. ВАЖНО в докстринге collect_clips ниже: RAFT-small систематически даёт
    мусорный flow на low_light кадрах (эмпирически подтверждено 2026-07-19),
    низкоосвещённые и нормально-освещённые уровни синхронизированы по кадрам,
    поэтому flow с normal_light переиспользуется для выравнивания low_light.
    Результат сохраняется под путём low_light_XX (это цель выравнивания),
    источник данных (normal_light_XX) виден только внутри process_clip.
    Между разными levels/сценами flow не считается.
  - Загрузка RAFT-small: torch.hub недоступен на Kaggle (несовместим с версией
    torch) — модель через core.raft.RAFT (репозиторий princeton-vl/RAFT,
    склонированный отдельно), обёрнута в nn.DataParallel (веса сохранены с этим
    префиксом ключей), затем разворачивается через .module для инференса на
    одном GPU. Эта процедура провалидирована при закрытии Блокера 2
    (2026-07-17/18): load_state_dict(strict=False) дал Missing=[]/Unexpected=[],
    и диагностика подтвердила побитовую идентичность вызова через обёртку
    DataParallel и через .module напрямую — .module здесь выбран для простоты
    кода, не ради скорости.
  - КОНВЕНЦИЯ ВХОДА RAFT: raw pixel values [0, 255] float, БЕЗ /255.0 —
    RAFT нормализует вход сам внутри forward (2*(x/255)-1). Это ДРУГАЯ
    конвенция, чем load_image()/load_image_npy() в data/datasets.py
    (там [0,1] для RetinexFormer/CfC) — не путать между собой.
  - Известная проблема при запуске (Этап 1.5): RAFT требует H,W кратные 8 —
    паддинг реализован ниже (pad_to_multiple_of_8 / unpad), снимается сразу
    после инференса, чтобы в кэше не было паддинг-артефактов.
  - PNG с альфа-каналом — обязательный .convert('RGB') при загрузке.

ФОРМАТ ХРАНЕНИЯ (обоснование): один .pt файл на пару кадров (не один файл на
клип целиком) — по двум причинам. (1) Та же гранулярность, что уже
провалидирована в бенчмарке Этапа 1.5 (BENCH_CACHE/{scale}/{i:05d}.pt) и в
существующем .npy-кэше проекта (один файл на кадр) — не вводим новый паттерн
хранения без необходимости. (2) Resume по отдельным файлам работает "из
коробки" через простой exists()-чек на каждую пару, без необходимости хранить
отдельный прогресс-манифест или частично дописывать один большой файл клипа
(что было бы уязвимо к обрыву сессии Kaggle посреди записи). Компромисс —
много мелких файлов (как и в .npy-кэше), но для Этапа 3 (композиция потоков)
это даже удобнее: доступ к flow[i] не требует загрузки всего клипа в память.

Путь к flow: <out_root>/<SceneName>/<low_light_XX>/<i:05d>.pt
  — flow[i] это смещение от кадра i к кадру i+1 внутри этого level.
  Зеркалит структуру .npy-кэша (scripts/preprocess_png_to_npy.py), чтобы
  Этап 3/4 могли строить путь той же логикой (scene_name/level/frame_idx).

УСТОЙЧИВОСТЬ К ОБРЫВУ СЕССИИ:
  - Resume на уровне отдельной пары: dst.exists() -> skip, без пересчёта.
  - Resume на уровне level: маркер _complete.marker пишется только после
    того, как ВСЕ пары level'а посчитаны — позволяет быстро пропустить уже
    полностью готовый level без повторного перечисления/exists()-проверки
    каждого файла (важно при большом числе клипов на повторных запусках).
  - Атомарная запись: каждый .pt пишется во временный файл и переименовывается
    — тот же паттерн, что в preprocess_png_to_npy.py и pack_scene_to_blob.py,
    защита от битого файла при обрыве сессии посреди torch.save().

ИСПОЛЬЗОВАНИЕ:

  Быстрая проверка на одной сцене (рекомендуется перед полным прогоном):

    python scripts/precompute_flow.py \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-20-scene-subset \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-part-2 \
        --raft-repo /kaggle/working/RAFT \
        --raft-weights /kaggle/input/raft-small-dataset/raft-small.pth \
        --out-root /kaggle/working/bvi_rlv_flow \
        --scene S02

  Полный датасет (после проверки на S02):

    python scripts/precompute_flow.py \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-20-scene-subset \
        --data-root /kaggle/input/bvi-rlv-low-light-video-dataset-part-2 \
        --raft-repo /kaggle/working/RAFT \
        --raft-weights /kaggle/input/raft-small-dataset/raft-small.pth \
        --out-root /kaggle/working/bvi_rlv_flow

  Повторный запуск (после обрыва Kaggle-сессии) — та же команда, скрипт сам
  пропустит уже готовые level'ы (по _complete.marker) и уже готовые отдельные
  пары внутри незавершённых level'ов (по exists()).

КАК ПРОВЕРИТЬ КОРРЕКТНОСТЬ (acceptance criteria Этапа 2 — визуальная проверка
на 1-2 тестовых клипах; сам рендеринг вынесен за рамки этого скрипта):
  1. Загрузить сохранённый flow: `flow = torch.load(path).float()`  # [2, Hr, Wr]
  2. На статичном участке кадра (неподвижный фон) flow должен быть near-zero
     по обеим компонентам (dx, dy).
  3. На движущемся объекте — связная, плавно меняющаяся область ненулевых
     значений (не шум, не отдельные "выбросы" пикселей).
  4. Ориентир по величине (эмпирически подтверждён 2026-07-19 картированием
     ПО ВСЕМУ клипу S02/normal_light_10, а не по 1-2 точкам): mean|flow| по
     клипу должен быть в районе ~1-2px, доля пар с mean|flow|>10px — около 0%.
     (На low_light эти же метрики были mean~6.4px и 12.8% пар >10px — заведомо
     сломанный результат, именно поэтому flow теперь считается по normal_light,
     см. докстринг collect_clips. Если увидишь на новом прогоне похожие на
     low_light-паттерн скачки — источник кадров в process_clip где-то съехал
     обратно на low_light, проверь код.)
  5. Цветовое кодирование (опционально, для быстрой визуальной оценки):
     `hue = atan2(dy, dx)`, `sat = clip(sqrt(dx**2+dy**2) / max_flow, 0, 1)` ->
     HSV -> RGB — стандартная схема визуализации flow (Middlebury/Sintel).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


LIGHT_LEVELS = ['low_light_10', 'low_light_20']   # результат сохраняется под этими путями;
                                                    # источник кадров для RAFT — normal_light_XX (см. collect_clips)
EXTS = {'.png', '.jpg', '.jpeg'}


# ----------------------------------------------------------------------------
# Структура датасета на диске — копия BVIRLVDataset._resolve_scene_content_dir
# (data/datasets.py). Держать идентичной логике там, иначе пути в flow-кэше
# разойдутся с тем, что реально читает Dataset. Скопировано (не импортировано),
# чтобы этот скрипт оставался самодостаточным служебным инструментом — тот же
# принцип, что уже применён в pack_scene_to_blob.py и preprocess_png_to_npy.py.
# ----------------------------------------------------------------------------
def resolve_scene_content_dir(scene_dir: Path) -> Path:
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


def collect_clips(data_roots: list[Path], scene_filter: str = None):
    """
    Возвращает список клипов: (scene_name, level, low_frame_paths[sorted], normal_frame_paths[sorted]).
    Один клип = одна сцена x один low_light уровень.

    ВАЖНО (найдено эмпирически 2026-07-19, см. mempalace wing=LNN_LowLight):
    RAFT-small СИСТЕМАТИЧЕСКИ даёт мусорный flow на low_light кадрах —
    картирование по всему клипу S02/low_light_10 показало 12.8% пар с
    mean|flow|>10px (пики до 160px) на визуально статичной сцене, тогда как
    normal_light_10 той же сцены даёт стабильный mean~1.4px по всему клипу
    (0% пар >10px). Гамма-подсветка low_light перед RAFT не помогла (мусор
    той же формы и величины) — проблема не в видимости яркости для человека,
    а в том, что RAFT (обучен на Sintel/KITTI/FlyingChairs — хорошо
    освещённые кадры) не умеет отличать шум сенсора/H.264-компрессии на
    почти-чёрных пикселях от реального смещения.

    РЕШЕНИЕ: low_light_XX и normal_light_XX — синхронизированные по кадрам
    съёмки одной и той же физической сцены (проверено эмпирически: offset=0
    даёт минимальную MSE в 8/9 контрольных точек по всему клипу, 9-я точка —
    фактическая ничья между offset=0 и offset=-1). Поэтому flow считается на
    normal_light_XX (где RAFT работает надёжно), а сохраняется под путём
    low_light_XX — именно low_light кадры будут выравниваться этим flow на
    Этапе 4, а не normal_light (normal_light тут используется только как
    "чистый источник" для оценки реального движения камеры/сцены).
    """
    clips = []
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
            for level in LIGHT_LEVELS:
                level_dir = content_dir / level
                normal_level = level.replace('low_light_', 'normal_light_')
                normal_dir = content_dir / normal_level
                if not level_dir.exists():
                    continue
                if not normal_dir.exists():
                    print(
                        f"[WARN] {scene_dir.name}/{level}: нет соответствующей "
                        f"{normal_level} — пропускаю клип (flow считается по "
                        f"normal_light, без неё клип обработать нельзя)",
                        file=sys.stderr,
                    )
                    continue

                low_paths = sorted([p for p in level_dir.iterdir() if p.suffix.lower() in EXTS])
                normal_paths = sorted([p for p in normal_dir.iterdir() if p.suffix.lower() in EXTS])
                if len(low_paths) < 2:
                    continue
                if len(low_paths) != len(normal_paths):
                    print(
                        f"[WARN] {scene_dir.name}/{level}: расхождение числа кадров "
                        f"low={len(low_paths)} vs normal={len(normal_paths)} — "
                        f"пропускаю клип (нарушена синхронизация по индексу)",
                        file=sys.stderr,
                    )
                    continue
                clips.append((scene_dir.name, level, low_paths, normal_paths))
    return clips


# ----------------------------------------------------------------------------
# RAFT-small: загрузка модели с реальными весами (см. докстринг модуля —
# процедура провалидирована при закрытии Блокера 2).
# ----------------------------------------------------------------------------
def load_raft_small(raft_repo: Path, weights_path: Path, device: str):
    """
    ВАЖНО про импорт: модули внутри core/ (raft.py, update.py, extractor.py,
    corr.py, utils/) в оригинальном репозитории princeton-vl/RAFT используют
    ПЛОСКИЕ импорты друг друга (raft.py делает `from update import ...`,
    `from extractor import ...`), а не пакетные `core.update`/`core.extractor`.
    Официальный demo.py репозитория поэтому добавляет в sys.path именно папку
    core/, а не корень репо, и импортирует `from raft import RAFT` (НЕ
    `from core.raft import RAFT`). Если добавить в sys.path корень репозитория
    и импортировать через `core.raft`, сам класс RAFT импортируется, но при
    первой же попытке ОН импортировать `update`/`extractor` падает с
    ModuleNotFoundError — это воспроизведённая на практике ошибка (не
    гипотетическая), поэтому здесь намеренно добавляется raft_repo/'core'.
    """
    raft_repo = Path(raft_repo)
    core_dir = raft_repo / 'core'
    if not core_dir.is_dir():
        raise FileNotFoundError(
            f"Ожидалась папка core/ внутри {raft_repo} — проверь, что это "
            f"корень склонированного репозитория princeton-vl/RAFT, а не сам core/."
        )
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    try:
        from raft import RAFT  # noqa: E402  (импорт после sys.path.insert — намеренно)
    except ImportError as e:
        raise ImportError(
            f"Не удалось импортировать RAFT из {core_dir}. "
            f"Проверь, что репозиторий princeton-vl/RAFT склонирован полностью "
            f"(core/raft.py, core/update.py, core/extractor.py, core/corr.py, "
            f"core/utils/ должны присутствовать). Оригинальная ошибка: {e}"
        )

    args = argparse.Namespace(small=True, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    model = torch.nn.DataParallel(model)

    state_dict = torch.load(weights_path, map_location=device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing or unexpected:
        print(
            f"[WARN] load_state_dict: missing_keys={missing}, "
            f"unexpected_keys={unexpected} — проверь путь к весам",
            file=sys.stderr,
        )
    else:
        print("[OK] RAFT-small weights loaded: полное структурное совпадение ключей")

    model = model.module.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Паддинг до кратности 8 (известная проблема Этапа 1.5) и загрузка кадра в
# конвенции RAFT (raw [0,255], без /255.0 — см. докстринг модуля).
# ----------------------------------------------------------------------------
def pad_to_multiple_of_8(img: torch.Tensor):
    """img: [1, 3, H, W]. Возвращает (padded_img, (pad_w, pad_h))."""
    _, _, H, W = img.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    # replicate — стандартный паддинг RAFT (InputPadder в оригинальном
    # репозитории), не constant/zero, чтобы не вносить артефакт "чёрного края"
    # в область, где RAFT ищет correlation.
    img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
    return img, (pad_w, pad_h)


def unpad(flow: torch.Tensor, pad):
    pad_w, pad_h = pad
    H, W = flow.shape[-2], flow.shape[-1]
    if pad_h > 0:
        flow = flow[..., :H - pad_h, :]
    if pad_w > 0:
        flow = flow[..., :, :W - pad_w]
    return flow


def load_frame_raft(path: Path, scale: float, device: str) -> torch.Tensor:
    """PNG -> [1, 3, Hr, Wr] float, raw [0,255], уже в целевом разрешении."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device)

    H, W = t.shape[-2:]
    Hr, Wr = max(1, round(H * scale)), max(1, round(W * scale))
    t = F.interpolate(t, size=(Hr, Wr), mode='bilinear', align_corners=False)
    return t


@torch.no_grad()
def compute_flow(model, img1: torch.Tensor, img2: torch.Tensor, iters: int) -> torch.Tensor:
    """img1, img2: [1, 3, Hr, Wr] (raw RAFT convention). Возвращает [2, Hr, Wr] fp16 на CPU."""
    img1p, pad = pad_to_multiple_of_8(img1)
    img2p, _ = pad_to_multiple_of_8(img2)
    _, flow_up = model(img1p, img2p, iters=iters, test_mode=True)
    flow = unpad(flow_up, pad)
    return flow[0].half().cpu()


# ----------------------------------------------------------------------------
# Основной цикл: по клипам -> по парам (i, i+1), resume на обоих уровнях.
# ----------------------------------------------------------------------------
def process_clip(model, scene_name, level, low_paths, normal_paths, out_root: Path, scale: float, device: str, iters: int):
    """
    Считает flow ПО normal_paths (стабильный, надёжный RAFT-инференс — см.
    docstring collect_clips), но сохраняет результат под путём scene/level,
    где level = 'low_light_XX' — именно эти кадры будут выравниваться этим
    flow на Этапе 4. low_paths используется только чтобы знать n_pairs
    (должно совпадать с normal_paths — уже провалидировано в collect_clips).
    """
    level_dir = out_root / scene_name / level
    level_dir.mkdir(parents=True, exist_ok=True)
    marker = level_dir / '_complete.marker'

    n_pairs = len(low_paths) - 1
    if marker.exists():
        return n_pairs, n_pairs  # (обработано, всего) — уже готово, ничего не считаем

    n_done = 0
    for i in range(n_pairs):
        dst = level_dir / f"{i:05d}.pt"
        if dst.exists():
            n_done += 1
            continue

        img1 = load_frame_raft(normal_paths[i], scale, device)
        img2 = load_frame_raft(normal_paths[i + 1], scale, device)
        flow = compute_flow(model, img1, img2, iters)

        tmp_dst = dst.with_suffix('.tmp.pt')
        torch.save(flow, tmp_dst)
        tmp_dst.rename(dst)  # атомарно — как в preprocess_png_to_npy.py / pack_scene_to_blob.py
        n_done += 1

    if n_done == n_pairs:
        marker.touch()

    return n_done, n_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', action='append', required=True,
                         help='Путь к BVI-RLV data_root. Можно указать несколько раз.')
    parser.add_argument('--raft-repo', required=True,
                         help='Путь к склонированному репозиторию princeton-vl/RAFT (содержит core/).')
    parser.add_argument('--raft-weights', required=True,
                         help='Путь к чекпоинту raft-small.pth с реальными предобученными весами.')
    parser.add_argument('--out-root', required=True,
                         help='Куда сохранять flow-кэш (например /kaggle/working/bvi_rlv_flow).')
    parser.add_argument('--scale', type=float, default=0.5,
                         help='Масштаб относительно исходного разрешения (0.5 = half-res, решение Этапа 1.5).')
    parser.add_argument('--iters', type=int, default=20,
                         help='Число итераций уточнения RAFT (test_mode). 20 — стандартное значение для eval.')
    parser.add_argument('--scene', default=None,
                         help="Обработать только сцену(ы), имя которых содержит эту подстроку "
                              "(например --scene S02) — для проверки перед полным прогоном.")
    parser.add_argument('--device', default='cuda',
                         help='cuda или cpu (cuda обязателен для приемлемой скорости на полном датасете).')
    args = parser.parse_args()

    if args.scale <= 0 or args.scale > 1:
        print(f"[ERROR] --scale должен быть в (0, 1], получено {args.scale}", file=sys.stderr)
        sys.exit(1)

    data_roots = [Path(r) for r in args.data_root]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Загружаю RAFT-small (device={args.device})...")
    model = load_raft_small(Path(args.raft_repo), Path(args.raft_weights), args.device)

    print(f"[2/3] Собираю список клипов" + (f" (фильтр сцены: '{args.scene}')" if args.scene else "") + "...")
    clips = collect_clips(data_roots, scene_filter=args.scene)
    if not clips:
        print("[ERROR] Не найдено ни одного клипа — проверь --data-root/--scene", file=sys.stderr)
        sys.exit(1)

    total_pairs_expected = sum(len(low_p) - 1 for _, _, low_p, _ in clips)
    print(f"      Найдено клипов: {len(clips)}, всего пар кадров (ожидается): {total_pairs_expected}")

    print(f"[3/3] Считаю flow (scale={args.scale}, iters={args.iters}, источник=normal_light)...")
    t0 = time.time()
    total_done = 0
    for clip_idx, (scene_name, level, low_paths, normal_paths) in enumerate(clips, 1):
        clip_t0 = time.time()
        n_done, n_pairs = process_clip(
            model, scene_name, level, low_paths, normal_paths, out_root, args.scale, args.device, args.iters
        )
        total_done += n_done
        dt = time.time() - clip_t0
        elapsed = time.time() - t0
        avg_per_clip = elapsed / clip_idx
        remaining = avg_per_clip * (len(clips) - clip_idx)
        print(
            f"      [{clip_idx}/{len(clips)}] {scene_name}/{level}: "
            f"{n_done}/{n_pairs} пар за {dt:.1f}с | "
            f"итого {total_done}/{total_pairs_expected} | "
            f"осталось ~{remaining/60:.1f} мин",
            flush=True,
        )

    dt_total = time.time() - t0
    print(f"\nГотово за {dt_total/60:.1f} мин. Пар посчитано (за этот запуск + уже готовые): {total_done}")

    total_flow_bytes = sum(f.stat().st_size for f in out_root.rglob('*.pt'))
    print(f"Итоговый размер flow-кэша: {total_flow_bytes / 1024**3:.2f} GB (ориентир по Этапу 1: ~32.7 GB на весь датасет)")


if __name__ == '__main__':
    main()
