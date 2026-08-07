"""
Integration-тест на generate_dataset.py — структурная проверка
объединённого пайплайна (не пересчитывает фотометрику заново, это
работа юнит-тестов test_position/irradiance/visibility/spp/albedo/
normal.py — здесь проверяется только то, что не может быть проверено
на изолированных прототипах: сцепка всех буферов в единый рендер,
консистентность проб между кадрами, форма и симметрия visibility-
матрицы).

Написан ПОСЛЕ generate_dataset.py и по факту его реального вывода —
осознанно, чтобы не подгонять генератор под тест.
"""
import numpy as np

from conftest import run_blender_script

REQUIRED_PROBE_KEYS = {"pixel", "position", "normal", "albedo", "irradiance", "spp", "true_irradiance"}


def test_generate_dataset_structure_and_consistency():
    result = run_blender_script("generate_dataset.py", timeout=180)

    grid_res = result["grid_res"]
    num_probes = result["num_probes"]
    frames = result["frames"]

    assert grid_res > 0
    assert num_probes > 0
    assert len(frames) >= 2, "need at least 2 frames to check dt/consistency"

    # Каждый пробный словарь содержит все 6 буферов, и число проб на
    # кадр совпадает с заявленным (та же сцена/камера -> та же маска
    # valid-пикселей на каждый кадр).
    for i, frame in enumerate(frames):
        probes = frame["probes"]
        assert len(probes) == num_probes, (
            f"frame {i} (t={frame['t']}) has {len(probes)} probes, "
            f"expected {num_probes} to match declared num_probes"
        )
        for p in probes[:5] + probes[-5:]:  # выборочно, не все 64+ на каждый кадр
            assert REQUIRED_PROBE_KEYS.issubset(p.keys()), (
                f"probe missing keys: {REQUIRED_PROBE_KEYS - p.keys()}"
            )
            assert len(p["position"]) == 3
            assert len(p["normal"]) == 3
            assert len(p["albedo"]) == 3
            assert len(p["irradiance"]) == 3
            assert isinstance(p["spp"], (int, float))
            assert len(p["true_irradiance"]) == 3
            # (d): high-spp эталон и low-spp obs оценивают ОДИН и тот же
            # интеграл на том же t/свете -> должны быть одного порядка
            # величины, не оторванными друг от друга значениями (rtol
            # щедрый — obs при низком spp может отклоняться заметно,
            # это ровно тот шум, ради которого true_irradiance и нужен).
            obs = np.array(p["irradiance"])
            true = np.array(p["true_irradiance"])
            assert np.all(true >= 0.0), f"true_irradiance has negative component: {true}"
            assert np.allclose(obs, true, rtol=0.5, atol=0.05), (
                f"obs {obs} and true_irradiance {true} diverge beyond plausible "
                f"noise — check render_true_irradiance is using the same t/light"
            )

    # Позиции проб фиксированы геометрией сцены (§3.1 п.1) — должны
    # совпадать по всем кадрам пиксель-в-пиксель, независимо от того,
    # что irradiance/spp меняются вместе со светом.
    first_positions = [p["position"] for p in frames[0]["probes"]]
    for i, frame in enumerate(frames[1:], start=1):
        positions = [p["position"] for p in frame["probes"]]
        assert positions == first_positions, (
            f"probe positions changed between frame 0 and frame {i} — "
            f"they should be invariant to light movement"
        )

    # dt — вещественный, не завязан на целочисленный шаг Blender-кадра
    # (§3.3): у первого кадра dt=None, у остальных — не None и не 0,
    # и хотя бы один dt не является целым числом.
    assert frames[0]["dt_from_previous"] is None
    dts = [f["dt_from_previous"] for f in frames[1:]]
    assert all(dt is not None and dt > 0 for dt in dts)
    assert any(abs(dt - round(dt)) > 1e-6 for dt in dts), (
        f"all dt values are integers ({dts}) — real-valued sampling not demonstrated"
    )

    # Visibility-матрица: правильная форма, диагональ (проба видит
    # саму себя), симметрия (если A видит B, B видит A — при
    # одинаковой геометрии и одинаковом эпсилоне это должно держаться
    # точно, не приближённо).
    vis = np.array(result["visibility_matrix"], dtype=bool)
    assert vis.shape == (num_probes, num_probes)
    assert vis.diagonal().all(), "a probe must always be visible to itself"
    assert np.array_equal(vis, vis.T), "visibility matrix is not symmetric"

    # Санити: матрица не должна быть тривиальной (все True или все
    # False) — иначе occluder в сцене ни на что не влияет, и тест
    # ничего не проверяет по существу.
    density = vis.mean()
    assert 0.0 < density < 1.0, (
        f"visibility matrix is trivial (density={density}) — occluder has "
        f"no effect, fixture is not meaningful"
    )

    assert len(result["probe_positions"]) == num_probes
