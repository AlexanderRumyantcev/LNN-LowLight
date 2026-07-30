"""
Метрики и валидация (ТЗ §6) + числовая реализация критерия §5.2 (порог перехода на
per-pixel dense).

§6.1: основная метрика — error-vs-time-since-event (кривая по offset от скачка), НЕ единая
    агрегированная per-segment-type MSE (та маскирует структуру, см. §6.1/урок run_v3
    spike-теста). Строится только для STEP-сегментов — для static/drift понятие "offset от
    события" неприменимо (§6.1).
§6.2: early-zone (offset 0..EARLY_ZONE_THRESHOLD после скачка) и floor (established regime:
    static/drift целиком + step-сегменты вдали от скачка) репортируются РАЗДЕЛЬНО.
§6.3: метрики — отдельно по типу сегмента (static/step/drift), не только по всему датасету.
§6.4: n_seeds >= 8 (не 3 — на n=3 оценки значимости были нестабильны в spike-тесте), paired
    bootstrap по последовательностям (не по отдельным шагам — шаги внутри последовательности
    коррелированы), 5000 resamples, 95% CI.
§5.2: критерий перехода на per-pixel dense — если CfC-B не показывает статистически значимого
    преимущества над ЧЕСТНОЙ версией NRD-style/NRC-style (§4) на реальных данных (по функции
    paired_bootstrap_significance ниже, применённой к per-sequence error-vs-offset/floor
    метрикам) — сигнал переключиться на per-pixel dense.

Segment-labeling здесь завязан на формат light_schedule из data/synthetic_probe_scene.py
(synthetic stand-in, §3 не Blender) — при переходе на реальные Blender-данные потребуется
аналогичная разметка сегментов из реального пайплайна генерации (§3.3), сама логика метрик
(per_segment_type_mse, early_zone_floor_split, paired_bootstrap_significance) от источника
данных не зависит.
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
    сегмент static/drift, то есть pre-jump хвост STEP-сегмента ни туда, ни туда не относится
    (сознательно — это переходная зона к следующему событию, не установившийся режим).
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
