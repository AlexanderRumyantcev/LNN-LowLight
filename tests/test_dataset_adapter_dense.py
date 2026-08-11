"""
Тесты адаптера §2.3(продолжение)/§2.6: blender/dataset_adapter_dense.py.

Использует короткий прогон generate_dataset_dense.py (тот же приём,
что tests/test_generate_dataset_dense.py) — механика адаптера не
зависит от длительности; полный TOTAL_DURATION=400 датасет уже
сгенерирован вручную отдельно (см. mempalace).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "blender"))
from dataset_adapter_dense import build_all_pixel_sequences, CONFIDENCE_LOG_MAX
from conftest import run_blender_script

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.temporal.cfc_probe_module import CfCProbeModule

SMOKE_DURATION = 24  # -> 4 кадра при DT=8


def _npz_path():
    meta = run_blender_script(
        "generate_dataset_dense.py",
        args=[SMOKE_DURATION, "smoke"],
        result_filename="generate_dataset_dense_result_meta_smoke.json",
        timeout=180,
    )
    return meta["npz_path"], meta


def test_sequences_cover_grid_and_have_expected_length():
    npz_path, meta = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    assert len(seqs) > 0
    assert len(seqs) <= meta["grid_res"] ** 2
    for seq in seqs:
        assert seq["obs"].shape[0] == meta["n_frames"]
        assert seq["obs"].shape[1] == 3  # RGB, дыра №3, не схлопнуто в scalar


def test_cold_start_and_dt_convention_matches_probe_subsampling():
    """Тот же принцип, что probe_subsampling.build_probe_sequence:
    dt[0]=0.0, cold_start[0]=1.0, cold_start[1:]=0."""
    npz_path, _ = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    for seq in seqs:
        assert seq["dt"][0] == 0.0
        assert seq["cold_start"][0] == 1.0
        assert np.all(seq["cold_start"][1:] == 0.0)


def test_dt_regular_not_async():
    """§2.3: регулярная сетка — dt[1:] константен (в отличие от
    probe-пути, где dt варьируется per draw)."""
    npz_path, meta = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    for seq in seqs:
        assert np.allclose(seq["dt"][1:], meta["dt"])


def test_first_frame_treated_as_trivial_disocclusion():
    """§2.5: 'нет истории' на первом кадре — тривиальная дисокклюзия,
    тем же принципом, что out-of-bounds."""
    npz_path, _ = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    for seq in seqs:
        assert seq["disocclusion_flag"][0] == 1.0
        assert seq["geometric_mismatch_score"][0] == 1.0


def test_disocclusion_flag_and_score_are_well_formed():
    npz_path, _ = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    for seq in seqs:
        flags = seq["disocclusion_flag"]
        scores = seq["geometric_mismatch_score"]
        assert set(np.unique(flags)).issubset({0.0, 1.0})
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
        # score>threshold(0.5 по умолчанию) должен подразумевать flag=1
        # (см. disocclusion.is_disoccluded) — не обязательно наоборот
        # (out-of-bounds тоже даёт flag=1 при score, зависящем от ветки)
        high_score_mask = scores > 0.5
        assert np.all(flags[high_score_mask] == 1.0)


def test_confidence_matches_fixed_global_normalization():
    """Та же фиксированная log1p(64.0)-нормализация, что в
    run_experiment.py/run_experiment_blender.py — не пересчитана по
    датасету."""
    npz_path, _ = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    seq = seqs[0]
    expected = np.log1p(seq["spp"]) / CONFIDENCE_LOG_MAX
    assert np.allclose(seq["confidence"], expected)


def test_valid_flag_propagated_per_frame_not_dropped():
    """Пиксель может быть невалиден на ЧАСТИ кадров (камера ушла) —
    valid должен присутствовать per-frame, не константой на всю
    последовательность (в отличие от probe-пути)."""
    npz_path, meta = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    for seq in seqs:
        assert seq["valid"].shape == (meta["n_frames"],)
        assert seq["valid"].dtype == bool


def test_excludes_pixels_never_valid():
    """Пиксели, ни разу не попавшие в геометрию (постоянно небо/фон),
    не должны попадать в выдачу вообще."""
    npz_path, meta = _npz_path()
    data = np.load(npz_path)
    valid = data["valid"]
    never_valid_pixels = int((~valid.any(axis=0)).sum())
    seqs = build_all_pixel_sequences(npz_path)
    assert len(seqs) == meta["grid_res"] ** 2 - never_valid_pixels


def test_wires_directly_into_cfc_probe_module_staleness_dim_four():
    """Сквозная проверка: выход адаптера подаётся в
    CfCProbeModule.build_input(staleness_dim=4) без переделки формата —
    именно ради этого и собирался §2.6."""
    npz_path, meta = _npz_path()
    seqs = build_all_pixel_sequences(npz_path)
    batch = seqs[:4]  # маленький батч из первых валидных пикселей

    obs = torch.tensor(np.stack([s["obs"] for s in batch]))              # [B,T,3]
    dt = torch.tensor(np.stack([s["dt"] for s in batch]))                # [B,T]
    cold = torch.tensor(np.stack([s["cold_start"] for s in batch]))      # [B,T]
    conf = torch.tensor(np.stack([s["confidence"] for s in batch]))      # [B,T]
    disocc = torch.tensor(np.stack([s["disocclusion_flag"] for s in batch]))
    mismatch = torch.tensor(np.stack([s["geometric_mismatch_score"] for s in batch]))

    model = CfCProbeModule(obs_dim=3, hidden_dim=8, use_staleness=True, staleness_dim=4)
    u = CfCProbeModule.build_input(
        obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
    )
    assert u.shape == (len(batch), meta["n_frames"], 3 + 4)
    pred, h = model(u, dt)
    assert pred.shape == obs.shape
    assert h.shape == (len(batch), 8)

    pred.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
