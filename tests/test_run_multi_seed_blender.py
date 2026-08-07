"""
Smoke-тест на run_experiment_blender.run_multi_seed() — n_seeds-стратегия на реальных
Blender-данных (§6.4-аналог). Проверяет, что весь цикл (n_seeds прогонов _run_one_seed +
paired bootstrap значимость) реально проходит end-to-end без исключений/NaN на игрушечных
параметрах. Намеренно МАЛЫЕ epochs/hidden_dim/n_train_probes — не про качество результата.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment_blender import run_multi_seed, DEFAULT_DATASET  # noqa: E402


def test_run_multi_seed_smoke():
    assert DEFAULT_DATASET.exists(), f"{DEFAULT_DATASET} not found"
    results = run_multi_seed(
        n_seeds=8, n_train_probes=8, epochs=3, hidden_dim=8, verbose_per_seed=False,
    )

    model_kinds = {"cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
                   "gru_faithful", "gru_honest"}
    assert set(results["early_by_kind"].keys()) == model_kinds
    for kind in model_kinds:
        early_vals = results["early_by_kind"][kind]
        floor_vals = results["floor_by_kind"][kind]
        assert len(early_vals) == 8
        assert len(floor_vals) == 8

    assert len(results["interp_by_seed"]) == 8
    for d in results["interp_by_seed"]:
        assert d["n"] > 0, "интерполяция не дала сравнений хотя бы на одном сиде"
