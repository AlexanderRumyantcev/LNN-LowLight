"""
Метрики и валидация (ТЗ §6) + численная реализация критерия §5.2 (порог
    перехода на per-pixel dense).

§6.1: основная метрика — error-vs-time-since-event (кривая по offset от скачка), НЕ
    единая агрегированная per-segment-type MSE (та маскирует структуру, см. §6.1/урок
    run_v3 spike-теста). Строится только для STEP-сегментов — для static/drift понятие
    "offset от события" неприменимо (§6.1).
§6.2: early-zone (offset 0..EARLY_ZONE_THRESHOLD после скачка) и floor (established regime:
    static/drift целиком + step-сегменты вдали от скачка) репортируются РАЗДЕЛЬНО.
§6.3: метрики — отдельно по типу сегмента (static/step/drift), не только по всему датасету.
§6.4: n_seeds >= 8 (не 3 — на n=3 оценки значимости были нестабильны в spike-тесте), paired
    bootstrap по последовательностям (не по отдельным шагам — шаги внутри последовательности
    коррелированы), 5000 resamples, 95% CI.
§5.2: критерий перехода на per-pixel dense — если CfC-B не показывает статистически
    значимого преимущества над ЧЕСТНОЙ версией NRD-style/NRC-style (§4) на реальных данных
    (по функции paired_bootstrap_significance ниже, применённой к per-sequence error-vs-offset/floor
    метрикам) — сигнал переключиться на per-pixel dense.

Segment-labeling здесь завязан на формат light_schedule из data/synthetic_probe_scene.py
(synthetic stand-in, §3 не Blender) — при переходе на реальные Blender-данные потребуется
аналогичная разметка сегментов из реального пайплайна генерации (§3.3), сама логика
метрик (per_segment_type_mse, early_zone_floor_split, paired_bootstrap_significance) от
источника данных не зависит.
"""

from dataclasses import dataclass

import numpy as np

from data.synthetic_probe_scene import SEG_STATIC, SEG_STEP, SEG_DRIFT

SEGMENT_NAMES = {SEG_STATIC: "static", SEG_STEP: "step", SEG_DRIFT: "drift"}

EARLY_ZONE_THRESHOLD = 4.0   # реальное время, см. §2.4/§6.2 (early-zone ~4-5 "кадров")
FLOOR_THRESHOLD = 8.0        # offset >= этого — step-сегмент считается "устаканившимся"
MIN_N_SEEDS = 8               # §6.4 — n=3 признан недостаточным в spike-тесте


def label_samples(t: np.ndarray, light_schedule) -> tuple[np.ndarray, np.ndarray]:
    """
    Для каждого t определяет (segment_type, offset_since_jump).
    offset_since_jump = NaN везде, КРОМЕ точек внутри STEP-сегмента (§6.1 — offset
    осмыслен только для step; для static/drift эталонного "момента скачка" нет).
    Внутри STEP-сегмента offset может быть отрицательным (точка ДО скачка, в пределах
    того же сегмента) — такие точки не попадают ни в early-zone, ни явно в floor этим
    вызовом; см. early_zone_floor_split, где floor берёт offset >= FLOOR_THRESHOLD ИЛИ
    сегмент static/drift, то есть pre-jump хвост STEP-сегмента ни туда, ни туда не
    относится (сознательно — это переходная зона к следующему событию, не установившийся
    режим).
    """
    segment_type = np.full(t.shape, -1, dtype=np.int64)
    offset = np.full(t.shape, np.nan, dtype=np.float64)
    for t_start, t_end, seg_type, p in light_schedule:
        mask = (t >= t_start) & (t < t_end)
        if not mask.any():
            continue
        segment_type[mask] = seg_type
        if seg_type == SEG_STEP:
            offset[mask] = t[mask] - p["jump_t"]
    # хвост за последним сегментом (округление) — считаем как последний сегмент
    tail = segment_type == -1
    if tail.any() and len(light_schedule) > 0:
        last_type = light_schedule[-1][2]
        segment_type[tail] = last_type
        if last_type == SEG_STEP:
            offset[tail] = t[tail] - light_schedule[-1][3]["jump_t"]
    return segment_type, offset


def per_segment_type_mse(
    pred: np.ndarray, true: np.ndarray, segment_type: np.ndarray
) -> dict:
    """§6.3 — MSE отдельно по типу сегмента, не одно усреднённое число."""
    sq_err = (pred - true) ** 2
    out = {}
    for seg_id, name in SEGMENT_NAMES.items():
        mask = segment_type == seg_id
        out[name] = float(sq_err[mask].mean()) if mask.any() else float("nan")
    return out


def error_vs_offset_curve(
    pred: np.ndarray,
    true: np.ndarray,
    segment_type: np.ndarray,
    offset: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """§6.1 — основная метрика: MSE по бинам offset (только STEP-сегмент, offset >= 0,
    т.е. ПОСЛЕ скачка). bin_edges по умолчанию соответствует масштабу dt_range проекта
    (~0.5-2.0 за обновление): [0,1,2,3,4,6,8,12,20,inf)."""
    if bin_edges is None:
        bin_edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])

    mask = (segment_type == SEG_STEP) & (offset >= 0) & ~np.isnan(offset)
    sq_err = (pred[mask] - true[mask]) ** 2
    off = offset[mask]

    bin_idx = np.digitize(off, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(sq_err[b_mask].mean()) if b_mask.any() else float("nan")
    return curve


def early_zone_floor_split(
    pred: np.ndarray,
    true: np.ndarray,
    segment_type: np.ndarray,
    offset: np.ndarray,
    early_threshold: float = EARLY_ZONE_THRESHOLD,
    floor_threshold: float = FLOOR_THRESHOLD,
) -> tuple[float, float]:
    """§6.2 — early-zone и floor раздельно, НЕ единое усреднённое число."""
    sq_err = (pred - true) ** 2

    early_mask = (segment_type == SEG_STEP) & (offset >= 0) & (offset < early_threshold)

    floor_mask = np.isin(segment_type, [SEG_STATIC, SEG_DRIFT]) | (
        (segment_type == SEG_STEP) & (offset >= floor_threshold)
    )

    early = float(sq_err[early_mask].mean()) if early_mask.any() else float("nan")
    floor = float(sq_err[floor_mask].mean()) if floor_mask.any() else float("nan")
    return early, floor


def paired_bootstrap_significance(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    §6.4 — paired bootstrap ПО ПОСЛЕДОВАТЕЛЬНОСТЯМ (errors_a/errors_b — один скаляр на
    последовательность/probe, не на отдельный шаг: шаги внутри последовательности
    коррелированы, ресэмплинг по ним нарушил бы предположение о независимости).

    errors_a, errors_b: [N_seeds] — например per-segment-type MSE или early-zone MSE,
        посчитанные отдельно на каждой из N_seeds независимых последовательностей/проб.

    Returns: dict(mean_diff, ci_low, ci_high, n_seeds, significant)
        significant = True, если 95% CI разницы (a - b) не содержит 0.
    """
    errors_a = np.asarray(errors_a, dtype=np.float64)
    errors_b = np.asarray(errors_b, dtype=np.float64)
    if errors_a.shape != errors_b.shape:
        raise ValueError("errors_a и errors_b должны быть одной формы (paired)")
    n_seeds = len(errors_a)
    if n_seeds < MIN_N_SEEDS:
        raise ValueError(
            f"n_seeds={n_seeds} < {MIN_N_SEEDS} — §6.4: на n=3 оценки значимости "
            f"были нестабильны в spike-тесте, минимум {MIN_N_SEEDS}."
        )

    rng = np.random.default_rng(seed)
    diff = errors_a - errors_b
    mean_diff = float(diff.mean())

    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n_seeds, size=n_seeds)  # paired resampling — тот же idx на a и b
        resampled_means[i] = diff[idx].mean()

    alpha = 1.0 - ci
    lo, hi = np.quantile(resampled_means, [alpha / 2, 1.0 - alpha / 2])
    significant = not (lo <= 0.0 <= hi)

    return dict(
        mean_diff=mean_diff, ci_low=float(lo), ci_high=float(hi),
        n_seeds=n_seeds, significant=significant,
    )


def seed_robustness_report(
    errors_a: np.ndarray, errors_b: np.ndarray,
) -> dict:
    """
    Диагностика устойчивости вывода к выбору набора сидов — ДОПОЛНЕНИЕ к
    paired_bootstrap_significance, не замена. Мотивация (сессия 2026-08-23):
    CI paired-bootstrap говорит "задевает ли разница ноль", но не говорит,
    насколько эффект велик относительно межсидового шума — а это отдельный
    вопрос, который на n_seeds=MIN_N_SEEDS=8 может быть некритично мал.
    Рекомендация читать std между сидами против размера эффекта — практика
    из статистического приложения DeepONet (Lu et al. 2021) и из общего анализа
    мощности в Colas & Sigaud ("How Many Random Seeds?", 2018): статистически
    значимый на конкретном n эффект может быть "хрупким",
    если |Cohen's d| мал — тогда для уверенности нужен стресс-тест на
    большем n_seeds, а не просто больше decimal places в CI.

    errors_a, errors_b: [N_seeds] — тот же формат paired-массивов, что и
        в paired_bootstrap_significance (один скаляр на сид/probe).

    Cohen's d здесь — mean(diff)/std(diff, ddof=1) для paired-разницы (не
    независимая version формулы) — грубый ориентир (Cohen 1988:
    0.2/0.5/0.8 = малый/средний/большой), НЕ формальный тест мощности.
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("errors_a и errors_b должны быть одной формы (paired)")
    diff = a - b
    n = len(diff)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("nan")

    if np.isnan(cohens_d):
        recommendation = "std_diff=0 — вырожденный случай, проверить входные данные"
    elif abs(cohens_d) >= 0.8:
        recommendation = "большой эффект (|d|>=0.8) — вывод устойчив, доп. сиды маловероятно не поменяют знак"
    elif abs(cohens_d) >= 0.5:
        recommendation = f"средний эффект (0.5<=|d|<0.8) при n_seeds={n} — вероятно устойчив, но стоит перепроверить на большем n (Colas & Sigaud)"
    else:
        recommendation = f"МАЛЫЙ эффект (|d|<0.5) при n_seeds={n} — значимость может быть хрупкой, перепроверить на большем n (Colas & Sigaud) ОБЯЗАТЕЛЬНО перед архитектурными выводами"

    return dict(
        mean_diff=mean_diff, std_diff=std_diff, cohens_d=cohens_d,
        n_seeds=n, recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# §6 delta (TZ_stage1b_per_pixel_dense_fallback.md) — новый срез по disocclusion,
# ОРТОГОНАЛЬНЫЙ static/step/drift разбивке выше (та завязана на световой
# сигнал light_schedule, эта — на геометрический сигнал disocclusion_flag §2.5).
#
# per_segment_type_mse/error_vs_offset_curve/early_zone_floor_split НЕ ТРОГАЮТСЯ
# (§6 ТЗ: "без изменений") — именно они остаются источником static/step/drift
# срезов и для dense-пути (та же логика применяется к light_schedule dense-
# генератора, см. blender/generate_dataset_dense.py/light_schedule.py).
# ---------------------------------------------------------------------------


def disocclusion_vs_stable_mse(
    pred: np.ndarray, true: np.ndarray, disocclusion_flag: np.ndarray
) -> dict:
    """§6 delta — MSE раздельно для disocclusion vs non-disocclusion пикселей
    (флаг §2.5), ОРТОГОНАЛЬНО к static/step/drift (per_segment_type_mse) —
    именно этот срез определяет исход §5 stage1b (три возможных сценария),
    не агрегат по всему кадру.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    flag = np.asarray(disocclusion_flag).astype(bool)
    if pred.shape != flag.shape:
        raise ValueError(
            f"pred/disocclusion_flag shape mismatch: {pred.shape} vs {flag.shape}"
        )
    sq_err = (pred - true) ** 2
    stable = ~flag
    return dict(
        disocclusion=float(sq_err[flag].mean()) if flag.any() else float("nan"),
        stable=float(sq_err[stable].mean()) if stable.any() else float("nan"),
    )


def dark_vs_bright_mse(
    pred: np.ndarray, true: np.ndarray, dark_threshold: float
) -> dict:
    """TZ_stage8 доп. диагностика (13.09.2026, чат) — MSE раздельно для
    тёмных/светлых кадров, срез по true (luminance) относительно
    dark_threshold, вычисленного один раз на eval-множество сида (см.
    run_capacity_sweep_zeroday_kan.py). ОРТОГОНАЛЬНО disocclusion_vs_
    stable_mse (та делит по геометрическому событию, эта — по
    радиометрической яркости) и НЕ входит в пре-регистрированный §4
decision rule TZ_stage8 — отдельная диагностика поверх него, не
    замена (изменение состава метрик задним числом после пре-регистрации
    иначе было бы p-hacking, см. TZ_stage8 §4).

    Мотивация: relative MC-шум растёт при низком радиансе (тот же
    физический эффект, из-за которого NRCRelativeL2Loss нормирует по
    предсказанию, а не по target, models/losses.py) — если KAN
    чувствительнее к шуму (arXiv 2407.14882), эффект должен быть
    сильнее именно в тёмных кадрах; срез по disocclusion/stable этого
    отдельно не покажет.

    pred, true: одной формы — одна последовательность/канал (см. вызов
    в _slice_mse_by_brightness).
    dark_threshold: скаляр — порог luminance; "тёмный" = true < threshold.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {pred.shape} vs {true.shape}")
    sq_err = (pred - true) ** 2
    dark = true < dark_threshold
    bright = ~dark
    return dict(
        dark=float(sq_err[dark].mean()) if dark.any() else float("nan"),
        bright=float(sq_err[bright].mean()) if bright.any() else float("nan"),
    )


def compute_age_since_disocclusion(disocclusion_flag: np.ndarray) -> np.ndarray:
    """Возраст (в шагах последовательности, НЕ реальном времени — регулярный
    Δt дозволяет это, §2.3) с последнего disocclusion-события: 0 в самом
    событии (disocclusion_flag=1), растёт на 1 с каждым следующим стабильным
    шагом. NaN — до первого события в этой последовательности (если весь
    префикс стабилен без единого события, что не должно случаться на
    практике: idx=0 всегда disocclusion_flag=1 по конвенции cold-start, см.
    blender/dataset_adapter_dense.py).

    Принимает 1D [T] (одна последовательность) ИЛИ 2D [B, T] (батч
    независимых последовательностей — возраст считается ОТДЕЛЬНО по каждой
    строке, события одного пикселя не влияют на другой).
    """
    flag = np.asarray(disocclusion_flag).astype(bool)
    if flag.ndim == 1:
        age = np.full(flag.shape, np.nan, dtype=np.float64)
        last_event = -1
        for i, f in enumerate(flag):
            if f:
                last_event = i
                age[i] = 0.0
            elif last_event >= 0:
                age[i] = float(i - last_event)
        return age
    if flag.ndim == 2:
        return np.stack([compute_age_since_disocclusion(row) for row in flag], axis=0)
    raise ValueError(f"disocclusion_flag должен быть 1D или 2D, получено ndim={flag.ndim}")


def error_vs_warp_age_curve(
    pred: np.ndarray,
    true: np.ndarray,
    disocclusion_flag: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """§6 delta — аналог error_vs_offset_curve (§6.1), но событие-триггер —
    disocclusion (§2.5), а не световой скачок light_schedule: "error как
    функция возраста истории после warp (сколько кадров прошло с
    последней валидной диссокклюзии в этом пикселе)".

    pred/true/disocclusion_flag — одной формы, 1D [T] или 2D [B, T] (батч
    пикселей, см. compute_age_since_disocclusion). Точки БЕЗ определённого
    возраста (NaN — до первого события) исключаются из кривой, тем же
    принципом, что error_vs_offset_curve исключает offset<0/NaN.
    """
    if bin_edges is None:
        bin_edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    age = compute_age_since_disocclusion(disocclusion_flag).reshape(-1)
    if pred.shape != age.shape:
        raise ValueError(f"pred/disocclusion_flag shape mismatch after flatten: {pred.shape} vs {age.shape}")

    mask = ~np.isnan(age)
    sq_err = (pred[mask] - true[mask]) ** 2
    a = age[mask]

    bin_idx = np.digitize(a, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(sq_err[b_mask].mean()) if b_mask.any() else float("nan")
    return curve


def error_vs_distance_to_train_curve(
    pred: np.ndarray,
    true: np.ndarray,
    eval_pixel_coords: np.ndarray,
    train_pixel_coords: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """07.09.2026, чат — механистический аналог error_vs_warp_age_curve, но
    ось не время, а ПРОСТРАНСТВО: ошибка как функция евклидова расстояния (в
    пикселях экрана) от eval-точки до ближайшей train-точки.

    Мотивация: агрегатный disocc/stable MSE не чувствителен к дырам в
    пространственном покрытии train-точек (см. mempalace/EXPERIMENT_LOG.md
    07.09.2026, продакшн-прогон tile_stratified_sample на Zero Day — MSE
    почти не сдвинулся). Эта кривая — прямая проверка гипотезы "модель хуже
    экстраполирует туда, где обучающих точек рядом не было", а также способ
    отличить "sample density зависит от координаты в пространстве" (реальный
    эффект) от "просто нужно больше точек вообще" (эффект объёма данных,
    ортогональный расположению).

    pred, true: [N_eval, ...] -- одна запись на eval-точку, произвольное число
        хвостовых осей (T, C — время, каналы, как в остальном проекте,
        см. _slice_mse в run_hybrid_v0.py). Квадратичная ошибка усредняется по
        ВСЕМ хвостовым осям сразу (не по T потом по C раздельно, как
        _slice_mse) -- здесь не нужна отдельная точность по каналам, только
        один скаляр на eval-точку для биннинга по расстоянию.
    eval_pixel_coords: [N_eval, 2] -- (x, y) для каждой eval-точки, тот же
        порядок, что и pred/true по оси 0.
    train_pixel_coords: [N_train, 2] -- (x, y) реально использованных
        train-точек этого сида (после tile_stratified_sample/perm[:n]).
    bin_edges: границы бинов расстояния в пикселях экрана. По умолчанию
        [0,5,10,20,30,50,80,120,inf) -- подобрано под масштаб Zero Day
        (320x180, ~90 точек -> характерный шаг тайла ~sqrt(320*180/90)~25px,
        так что дефолтные бины покрывают "внутри своего тайла" через
        "далеко от любой train-точки". Для другого H,W/бюджета бины стоит
        передавать явно.

    Возвращает dict той же формы, что error_vs_warp_age_curve/
    error_vs_offset_curve (label -> mean squared error, NaN для пустых
    бинов) -- пустой бин ЗНАЧИТ, что при данном бюджете точек ни одна
    eval-точка не оказалась на этом расстоянии от ближайшей train-точки
    (сам по себе диагностический факт про покрытие, не только про ошибку).

    Использует scipy.spatial.cKDTree для поиска ближайшего сосуда -- N_eval
    может быть большим (57600 на Zero Day), брутфорс O(N_eval*N_train) для
    большого train-бюджета был бы дороже без явной необходимости.
    """
    from scipy.spatial import cKDTree

    if bin_edges is None:
        bin_edges = np.array([0, 5, 10, 20, 30, 50, 80, 120, np.inf])

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {pred.shape} vs {true.shape}")
    n_eval = pred.shape[0]
    sq_err = ((pred - true) ** 2).reshape(n_eval, -1).mean(axis=1)

    eval_pixel_coords = np.asarray(eval_pixel_coords, dtype=np.float64)
    train_pixel_coords = np.asarray(train_pixel_coords, dtype=np.float64)
    if eval_pixel_coords.shape != (n_eval, 2):
        raise ValueError(
            f"eval_pixel_coords должен быть [{n_eval}, 2], получено {eval_pixel_coords.shape}"
        )

    tree = cKDTree(train_pixel_coords)
    dist, _ = tree.query(eval_pixel_coords, k=1)

    bin_idx = np.digitize(dist, bin_edges[1:-1])
    curve = {}
    for b in range(len(bin_edges) - 1):
        b_mask = bin_idx == b
        label = f"[{bin_edges[b]:g},{bin_edges[b+1]:g})"
        curve[label] = float(sq_err[b_mask].mean()) if b_mask.any() else float("nan")
    return curve
