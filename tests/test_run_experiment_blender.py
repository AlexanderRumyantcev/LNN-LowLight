"""
Smoke-тест на run_experiment_blender.py — проверяет, что весь пайплайн
(адаптер -> build_batch -> обучение CfC-B/NRC-baseline'ов с obs_dim=3 ->
предсказание -> реконструированный light_schedule -> метрики) реально
проходит end-to-end на РЕАЛЬНЫХ Blender-данных без исключений/NaN.

Намеренно МАЛЫЕ epochs/hidden_dim (не про качество результата — это
отдельный вопрос, здесь только "пайплайн не падает и не даёт NaN").
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment_blender import run, DEFAULT_DATASET  # noqa: E402


def test_run_end_to_end_smoke():
    assert DEFAULT_DATASET.exists(), f"{DEFAULT_DATASET} not found"
    results = run(n_train_probes=8, epochs=3, hidden_dim=8, subsample_seed=0)

    model_kinds = {"cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
                   "gru_faithful", "gru_honest"}
    assert model_kinds <= set(results.keys())
    for kind in model_kinds:
        r = results[kind]
        for key in ("early", "floor"):
            val = r[key]
            assert not np.isnan(val), f"{kind}.{key} is NaN"
            assert np.isfinite(val), f"{kind}.{key} is not finite"
        for name, val in r["per_segment_type"].items():
            assert np.isfinite(val), f"{kind}.per_segment_type[{name}] is not finite"

    interp = results["_interpolation"]
    assert interp["n"] > 0, "интерполяция не дала ни одного сравнения — cold-start окно/query_times не согласованы"
    assert np.isfinite(interp["mean_rel_err"])
    assert np.isfinite(interp["median_rel_err"])
