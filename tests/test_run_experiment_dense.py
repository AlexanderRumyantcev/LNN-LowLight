"""
Smoke-тест на run_experiment_dense.py (§2.3+§4+§6 delta, per-pixel dense путь)
— проверяет, что весь пайплайн (dataset_adapter_dense -> build_batch_dense ->
обучение CfC-B/baseline'ов со staleness_dim=4 -> предсказание -> метрики,
включая §6 delta disocclusion/stable) реально проходит end-to-end на РЕАЛЬНЫХ
Blender-данных без исключений/NaN.

Тот же принцип, что test_run_experiment_blender.py: намеренно МАЛЫЕ
epochs/hidden_dim/n_train_pixels — не про качество результата, только
"пайплайн не падает и не даёт NaN".
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment_dense import run, DEFAULT_DATASET  # noqa: E402

MODEL_KINDS = {"cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
               "gru_faithful", "gru_honest"}


def test_run_end_to_end_smoke():
    assert DEFAULT_DATASET.exists(), f"{DEFAULT_DATASET} not found"
    results = run(n_train_pixels=15, epochs=3, hidden_dim=8, seed=0)

    assert MODEL_KINDS <= set(results.keys())
    for kind in MODEL_KINDS:
        r = results[kind]
        for key in ("early", "floor", "disocclusion", "stable"):
            val = r[key]
            assert not np.isnan(val), f"{kind}.{key} is NaN"
            assert np.isfinite(val), f"{kind}.{key} is not finite"
        for name, val in r["per_segment_type"].items():
            assert np.isfinite(val), f"{kind}.per_segment_type[{name}] is not finite"
        for label, val in r["warp_age_curve"].items():
            # часть возрастных бинов может быть пустой на маленьком n_train_pixels/
            # коротком датасете — NaN там допустим, но не должно быть inf/сломанных чисел
            assert not np.isinf(val), f"{kind}.warp_age_curve[{label}] is inf"
