"""
Тест на blender/dataset_adapter.py — переупаковка generate_dataset_result.json
в seqs[i] для build_batch (run_experiment.py), obs_dim=3 (РЕШЕНИЕ по дыре №3,
RGB не схлопывается). Продолжение test_probe_interpolation_real_data.py
("согласование с build_batch — отдельный шаг после этого").
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "blender"))

from probe_subsampling import ProbeSubsampleConfig  # noqa: E402
from dataset_adapter import build_all_probe_sequences  # noqa: E402
from run_experiment import build_batch  # noqa: E402

RESULT_PATH = PROJECT_ROOT / "blender" / "generate_dataset_result.json"


@pytest.fixture(scope="module")
def real_dataset():
    assert RESULT_PATH.exists(), (
        f"{RESULT_PATH} not found — run blender/generate_dataset.py headless first."
    )
    return json.loads(RESULT_PATH.read_text())


def test_seqs_shapes_and_count(real_dataset):
    cfg = ProbeSubsampleConfig(seed=0)
    seqs = build_all_probe_sequences(real_dataset, cfg, base_seed=100)

    assert len(seqs) == real_dataset["num_probes"]
    for seq in seqs:
        L = len(seq["t"])
        assert seq["obs"].shape == (L, 3)
        assert seq["true_irradiance"].shape == (L, 3)
        assert seq["spp"].shape == (L,)
        assert seq["dt"].shape == (L,)
        assert seq["cold_start"].shape == (L,)
        assert seq["t"][0] == pytest.approx(real_dataset["frames"][0]["t"])


def test_different_probes_get_independent_async_streams(real_dataset):
    """base_seed+p должен давать РАЗНЫЕ Δt-потоки на разных пробах — иначе
    вся идея per-probe асинхронности (c) теряется на реальных данных."""
    cfg = ProbeSubsampleConfig(seed=0)
    seqs = build_all_probe_sequences(real_dataset, cfg, base_seed=0)
    assert list(seqs[0]["pool_index"]) != list(seqs[1]["pool_index"])


def test_values_traceable_to_source_frames(real_dataset):
    """obs[k] пробы p должен совпадать с irradiance ИЗ РЕАЛЬНОГО кадра
    frames[pool_index[k]]['probes'][p] — не просто правильная форма, а
    честная переадресация значений, без перетасовки между пробами."""
    cfg = ProbeSubsampleConfig(seed=0)
    seqs = build_all_probe_sequences(real_dataset, cfg, base_seed=42)
    frames = real_dataset["frames"]

    for p in (0, 1, len(seqs) - 1):
        seq = seqs[p]
        for k, pool_idx in enumerate(seq["pool_index"]):
            expected = frames[pool_idx]["probes"][p]["irradiance"]
            assert np.allclose(seq["obs"][k], expected, atol=1e-6)


def test_build_batch_rgb_obs_dim(real_dataset):
    """Выход адаптера должен проходить через build_batch (run_experiment.py)
    с итоговым feature-измерением 3 (RGB), а не 1 — это и есть цель адаптера."""
    cfg = ProbeSubsampleConfig(seed=0)
    seqs = build_all_probe_sequences(real_dataset, cfg, base_seed=7)
    indices = list(range(min(5, len(seqs))))

    batch = build_batch(seqs, indices)

    n = len(indices)
    T = batch["obs"].shape[1]
    assert batch["obs"].shape == (n, T, 3)
    assert batch["true"].shape == (n, T, 3)
    assert batch["dt"].shape == (n, T)
    assert batch["cold"].shape == (n, T)
    assert batch["conf"].shape == (n, T)
    assert not batch["obs"].isnan().any()
    assert not batch["true"].isnan().any()


def test_build_batch_still_handles_scalar_obs():
    """Обратная совместимость: старый synthetic-путь (obs_dim=1, форма (T,))
    не должен был сломаться обобщением build_batch под RGB."""
    n_pool = 50
    seqs = []
    for i in range(3):
        seqs.append(dict(
            t=np.arange(n_pool, dtype=np.float32),
            dt=np.ones(n_pool, dtype=np.float32),
            cold_start=np.zeros(n_pool, dtype=np.float32),
            obs=np.random.rand(n_pool).astype(np.float32),
            spp=np.full(n_pool, 16.0, dtype=np.float32),
            true_irradiance=np.random.rand(n_pool).astype(np.float32),
        ))
    batch = build_batch(seqs, [0, 1, 2])
    assert batch["obs"].shape == (3, n_pool, 1)
    assert batch["true"].shape == (3, n_pool, 1)
