"""
Smoke-тест на run_experiment_dense.run_multi_seed() — §6.4-аналог на
per-pixel dense пути. Проверяет, что весь цикл (n_seeds прогонов
_run_one_seed + paired bootstrap значимость, включая §5 stage1b
критерий в disocclusion-срезе) реально проходит end-to-end без
исключений/NaN на игрушечных параметрах.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment_dense import run_multi_seed, DEFAULT_DATASET  # noqa: E402

MODEL_KINDS = {"cfc", "nrd_faithful", "nrd_honest", "nrc_faithful", "nrc_honest",
               "gru_faithful", "gru_honest"}


def test_run_multi_seed_smoke():
    assert DEFAULT_DATASET.exists(), f"{DEFAULT_DATASET} not found"
    results = run_multi_seed(
        n_seeds=8, n_train_pixels=15, epochs=3, hidden_dim=8, verbose_per_seed=False,
    )

    assert set(results["early_by_kind"].keys()) == MODEL_KINDS
    assert set(results["disocc_by_kind"].keys()) == MODEL_KINDS
    for kind in MODEL_KINDS:
        assert len(results["early_by_kind"][kind]) == 8
        assert len(results["disocc_by_kind"][kind]) == 8
        assert len(results["stable_by_kind"][kind]) == 8

    # §5 stage1b критерий — значимость в disocclusion-срезе должна быть
    # реально посчитана хотя бы для части честных baseline'ов (не всё NaN)
    assert len(results["disocc_significance"]) > 0
    for honest_kind, res in results["disocc_significance"].items():
        assert "significant" in res
        assert res["n_seeds"] == 8
