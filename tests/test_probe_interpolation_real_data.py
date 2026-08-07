"""
Тест №1 из плана "проверить, что уже можно, до написания адаптера" (2026-08-03).

Цель: пропустить РЕАЛЬНЫЕ данные из blender/generate_dataset_result.json через
УЖЕ СУЩЕСТВУЮЩИЙ models/probe_interpolation.py (formula-постпроцессинг §2.3),
который до сих пор проверялся только на synthetic_probe_scene.py. Формула не
имеет обучаемых параметров, поэтому её можно тестировать напрямую, без CfC и
без run_experiment.py — и без адаптера под формат build_batch.

Метод: leave-one-out per кадр. Для каждой пробы i её ground-truth irradiance —
это реально отрендеренное значение в этой точке (не синтетическое допущение).
Кандидаты для интерполяции — ВСЕ ОСТАЛЬНЫЕ пробы того же кадра (K = N-1),
с их реальными position/normal/irradiance и реальной visibility[i, k] из
precomputed raycast-матрицы. Формула не обязана быть точной (i не обязана
быть "близко" к соседям на сетке 8x8) — но обязана быть ЧИСЛЕННО КОРРЕКТНОЙ
(без NaN, веса нормализованы) и не деградировать в тривиальный fallback
на большей части проб.

НЕ проверяется здесь (осознанно):
- CfC-B / staleness / dt-последовательность
- согласование с run_experiment.py / build_batch (отдельный шаг после этого)
- многокадровая temporal-согласованность (кадры независимы)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from models.probe_interpolation import compute_interpolation_weights  # noqa: E402

RESULT_PATH = PROJECT_ROOT / "blender" / "generate_dataset_result.json"


@pytest.fixture(scope="module")
def real_dataset():
    assert RESULT_PATH.exists(), (
        f"{RESULT_PATH} not found — run blender/generate_dataset.py headless first."
    )
    return json.loads(RESULT_PATH.read_text())


def _leave_one_out_frame(frame, visibility_matrix):
    """Для каждой пробы кадра: интерполировать irradiance по ВСЕМ остальным
    пробам того же кадра (реальные position/normal/visibility/irradiance),
    сравнить с реально отрендеренным значением в этой точке.

    Возвращает per-probe relative L2 ошибку, флаг NaN/Inf, флаг degenerate
    (все кандидаты occluded/behind-normal — сработал distance-only fallback).
    """
    probes = frame["probes"]
    n = len(probes)
    positions = torch.tensor([p["position"] for p in probes], dtype=torch.float32)
    normals = torch.tensor([p["normal"] for p in probes], dtype=torch.float32)
    irradiance = torch.tensor([p["irradiance"] for p in probes], dtype=torch.float32)
    vis_full = torch.tensor(np.array(visibility_matrix), dtype=torch.float32)

    rel_errors, nan_flags, degenerate_flags = [], [], []
    for i in range(n):
        others = [k for k in range(n) if k != i]
        query_pos = positions[i:i + 1]
        query_normal = normals[i:i + 1]
        probe_pos = positions[others].unsqueeze(0)
        vis_i = vis_full[i, others].unsqueeze(0)

        weights = compute_interpolation_weights(query_pos, query_normal, probe_pos, vis_i)
        pred = torch.einsum("bk,bkc->bc", weights, irradiance[others].unsqueeze(0))[0]
        true = irradiance[i]

        nan_flags.append(bool(torch.isnan(pred).any() or torch.isinf(pred).any()))
        rel_err = (torch.linalg.norm(pred - true) / (torch.linalg.norm(true) + 1e-6)).item()
        rel_errors.append(rel_err)

        # воспроизводим pre-fallback сумму весов, чтобы засечь долю degenerate
        direction = probe_pos - query_pos.unsqueeze(1)
        distance = torch.linalg.norm(direction, dim=-1)
        direction_n = direction / (distance.unsqueeze(-1) + 1e-6)
        normal_term = torch.clamp(torch.einsum("bkc,bc->bk", direction_n, query_normal), min=0.0)
        distance_term = 1.0 / (distance.pow(2.0) + 1e-6)
        raw_sum = (vis_i * normal_term * distance_term).sum(dim=-1).item()
        degenerate_flags.append(raw_sum < 1e-6)

    return rel_errors, nan_flags, degenerate_flags


def test_probe_interpolation_leave_one_out_on_real_data(real_dataset):
    visibility_matrix = real_dataset["visibility_matrix"]
    frames = real_dataset["frames"]
    assert len(frames) > 0, "dataset has no frames"

    all_rel_errors = []
    all_nan_flags = []
    all_degenerate_flags = []

    for frame_idx, frame in enumerate(frames):
        rel_errors, nan_flags, degenerate_flags = _leave_one_out_frame(frame, visibility_matrix)
        all_rel_errors.extend(rel_errors)
        all_nan_flags.extend(nan_flags)
        all_degenerate_flags.extend(degenerate_flags)

        print(
            f"frame {frame_idx} (t={frame['t']:.3f}): "
            f"median rel_err={np.median(rel_errors):.4f}, "
            f"max rel_err={np.max(rel_errors):.4f}, "
            f"degenerate={sum(degenerate_flags)}/{len(degenerate_flags)}"
        )

    # (1) Численная корректность — жёсткое требование, без исключений.
    assert not any(all_nan_flags), "NaN/Inf в предсказании — формула ломается на реальной геометрии"

    # (2) Fallback не должен доминировать. Если бы происходил почти всегда —
    # это был бы прямой сигнал несостыковки (например geometry/visibility
    # рассинхронизированы, или normal-конвенция другая, чем ожидает формула)
    # и повод пересмотреть fallback, а не считать его "редким инженерным случаем".
    degenerate_fraction = sum(all_degenerate_flags) / len(all_degenerate_flags)
    print(f"OVERALL degenerate fraction: {degenerate_fraction:.3f}")
    assert degenerate_fraction < 0.5, (
        f"fallback сработал на {degenerate_fraction:.0%} проб — формула §2.3 "
        "не подходит для этой геометрии как есть, нужен пересмотр перед адаптером"
    )

    # (3) Не проверяем точность как таковую (leave-one-out на разреженной сетке
    # 8x8 — не то же самое, что реальный inference) — но фиксируем распределение
    # ошибок как БАЗОВУЮ ЛИНИЮ для сравнения после любых будущих изменений формулы.
    print(f"OVERALL median rel_err: {np.median(all_rel_errors):.4f}")
    print(f"OVERALL mean rel_err: {np.mean(all_rel_errors):.4f}")
