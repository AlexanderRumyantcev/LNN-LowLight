"""
Проверка: не является ли немонотонность std(spp) у DIRECT-компоненты
(tests/analyze_noise_model_components.py) просто шумом нашей ЖЕ оценки
std по всего 32 повторам, а не реальным свойством рендерера — учитывая,
что распределение на низком spp тяжёлохвостое (excess kurtosis ~+9.6 на
spp=1, см. drawer от анализа гауссовости).

Метод: bootstrap. Пересэмплируем 32 повтора с возвращением 2000 раз,
считаем std каждый раз — получаем эмпирическое распределение самой
оценки std на каждом spp-уровне, сравниваем перекрытие доверительных
интервалов между соседними spp-уровнями. Данные уже есть, новый
рендер не нужен.
"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/Users/alexandrrumyantsev/Desktop/LNN_LowLight_Project")
RESULT_PATH = PROJECT_ROOT / "blender" / "verify_noise_model_components_result.json"

data = json.loads(RESULT_PATH.read_text())
spp_levels = data["spp_levels"]
stable_valid = np.array(data["stable_valid"], dtype=bool)
coords = list(zip(*np.where(stable_valid)))

rng = np.random.default_rng(0)
N_BOOT = 2000

print(f"{'spp':>4} | {'std (all probes, point)':>24} | {'bootstrap 95% CI (median probe)':>34}")
for spp in spp_levels:
    arr = np.array(data["repeats_by_spp_direct"][str(spp)])  # (32, H, W, 3)
    lum = arr.mean(axis=-1)  # (32, H, W)
    series = np.stack([lum[:, r, c] for r, c in coords], axis=1)  # (32, n_probes)

    point_std = np.median(series.std(axis=0, ddof=1))

    # bootstrap для одной "типичной" пробы (той, чей std ближе всего к медиане)
    per_probe_std = series.std(axis=0, ddof=1)
    median_probe_idx = np.argsort(per_probe_std)[len(per_probe_std) // 2]
    probe_series = series[:, median_probe_idx]

    boot_stds = []
    for _ in range(N_BOOT):
        resample = rng.choice(probe_series, size=len(probe_series), replace=True)
        boot_stds.append(resample.std(ddof=1))
    lo, hi = np.percentile(boot_stds, [2.5, 97.5])

    print(f"{spp:>4} | {point_std:>24.4f} | [{lo:.4f}, {hi:.4f}]")
