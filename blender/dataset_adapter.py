"""
Адаптер: переупаковка результата generate_dataset.py (одна запись на
кадр, с "probes": [...] внутри) в формат seqs[i] для build_batch
(run_experiment.py) — по одной асинхронной per-probe последовательности
на пробу, той же формы, что уже используется для synthetic-данных
(data/synthetic_probe_scene.py:sample_probe_sequence).

Это ЧИСТАЯ переупаковка поверх уже готовых кусков:
- пул моментов (b) уже есть как frames[i]['t'] (порядок по возрастанию,
  как и требует build_probe_sequence);
- per-probe асинхронная подвыборка (c) уже реализована в
  probe_subsampling.py:build_probe_sequence — этот модуль лишь достаёт
  параллельные по кадрам массивы ('obs'=irradiance, 'spp', 'true_irradiance')
  для КАЖДОЙ пробы и прогоняет их через build_probe_sequence по отдельности,
  с независимым RNG на пробу (иначе все пробы получили бы идентичный
  Δt-поток — тот же принцип независимости, что уже проверен в
  test_different_probes_get_different_async_streams);
- RGB остаётся как есть (дыра №3, РЕШЕНО 2026-08-03, см. mempalace) —
  obs/true_irradiance здесь [n_pool, 3], build_probe_sequence уже
  общий по форме массива (test_build_probe_sequence_handles_rgb_vector_obs).

ЧЕГО ЗДЕСЬ НЕТ (сознательно, отдельные шаги): интерполяция между
пробами (models/probe_interpolation.py — spatial, не temporal),
wiring в CfCProbeModule/build_input (obs_dim=3 передаётся явно
вызывающим кодом при инстанцировании, сам модуль уже общий по obs_dim).
"""
from pathlib import Path
import json

import numpy as np

from probe_subsampling import ProbeSubsampleConfig, build_probe_sequence


def load_dataset(path) -> dict:
    return json.loads(Path(path).read_text())


def build_all_probe_sequences(dataset: dict, cfg: ProbeSubsampleConfig, base_seed: int = 0):
    """Одна асинхронная seqs[i]-последовательность на КАЖДУЮ пробу датасета.

    Порядок проб внутри frames[i]['probes'] один и тот же для всех кадров
    (генератор гарантирует это assert'ом на equal probe count при статичной
    геометрии/valid-маске, см. generate_dataset.py) — поэтому индекс p
    внутри probes-списка кадра однозначно адресует одну и ту же пробу
    во времени, без явного id.

    base_seed + p даёт независимый (но детерминированный) RNG на пробу —
    без этого все пробы получили бы ОДИН И ТОТ ЖЕ Δt-поток (тот же
    объект ProbeSubsampleConfig.seed для всех), что противоречит самой
    цели асинхронной подвыборки (c).
    """
    frames = dataset["frames"]
    assert len(frames) > 1, "нужно хотя бы 2 кадра, чтобы был пул моментов"
    pool_t = np.array([f["t"] for f in frames], dtype=np.float64)
    assert np.all(np.diff(pool_t) > 0), "frames должны быть отсортированы по t (как из generate_dataset.py)"

    num_probes = dataset["num_probes"]
    for f in frames:
        assert len(f["probes"]) == num_probes, "probe count разошёлся между кадрами"

    seqs = []
    for p in range(num_probes):
        pixel0 = frames[0]["probes"][p]["pixel"]
        obs = np.array([f["probes"][p]["irradiance"] for f in frames], dtype=np.float32)
        spp = np.array([f["probes"][p]["spp"] for f in frames], dtype=np.float32)
        true_irradiance = np.array(
            [f["probes"][p]["true_irradiance"] for f in frames], dtype=np.float32
        )
        # sanity: индекс p действительно адресует одну и ту же пробу во всех кадрах
        for f in frames:
            assert f["probes"][p]["pixel"] == pixel0, (
                f"probe index {p} указывает на разные пиксели в разных кадрах — "
                "порядок проб внутри кадра нестабилен"
            )

        probe_series = dict(obs=obs, spp=spp, true_irradiance=true_irradiance)
        rng = np.random.default_rng(base_seed + p)
        seq = build_probe_sequence(pool_t, probe_series, cfg, rng)
        seqs.append(seq)
    return seqs
