"""
Генератор синтетических данных для spike-теста (ТЗ раздел 3).

Формат одной последовательности:
    true_values: (T,)  — истинный сигнал (скрыт от модели, нужен только для лосса/метрик)
    obs:         (T,)  — зашумлённое наблюдение (вход модели)
    dt:          (T,)  — нерегулярный интервал с предыдущего наблюдения
    spp:         (T,)  — сколько сэмплов "использовано" для наблюдения в момент t
                          (влияет на дисперсию шума; также нужен для precision-weighted baseline)
    seg_type:    (T,)  — тип сегмента в этой точке: 0=static, 1=step, 2=drift
                          НЕ подаётся в модель, только для диагностики/метрик по типу сегмента
    event_mask:  (T,)  — 1 в кадре, где произошёл скачок (для error-vs-time-since-event и oracle)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GenConfig:
    # длина одной последовательности (число наблюдений)
    seq_len: int = 200
    # число сегментов на последовательность (диапазон) — открытый параметр ТЗ, дефолт разумный
    n_segments_range: tuple = (3, 5)
    # длина одного сегмента в шагах (диапазон). Минимум намеренно > горизонта метрики
    # error_vs_time_since_event (25 шагов), чтобы хвост кривой после скачка не захватывал
    # следующий сегмент/событие (баг, найденный в первом прогоне, см. ТЗ Приложение А.1)
    segment_len_range: tuple = (35, 60)
    # диапазон истинных значений сигнала (произвольные единицы, аналог radiance)
    value_range: tuple = (0.1, 10.0)
    # величина скачка для step-сегментов, как множитель текущего значения (диапазон)
    step_magnitude_range: tuple = (0.3, 3.0)
    # знак скачка случайный (может расти или падать)
    # диапазон spp (samples per pixel), определяет дисперсию шума наблюдения
    spp_range: tuple = (1, 64)
    # константа масштаба шума: var(obs) ≈ noise_c * true_value / spp (Monte Carlo-подобное поведение:
    # дисперсия убывает с ростом spp и растёт с яркостью сигнала — грубая, но качественно верная модель)
    noise_c: float = 0.5
    # диапазон Δt между наблюдениями (нерегулярная временная сетка) — БАЗОВАЯ мода,
    # без выбросов. Раньше это был единственный параметр (равномерный на всём диапазоне);
    # калибровка 2026-07-27 по измеренным данным реального frame pacing показала, что
    # это неверная форма распределения — см. dt_spike_prob/dt_spike_multiplier_range ниже.
    dt_range: tuple = (0.5, 2.0)
    # ВЕРОЯТНОСТЬ редкого тяжёлого выброса Δt на шаг (имитация GPU thermal throttling /
    # шейдер-компиляции на лету / LOD-переключений). Источник калибровки: измеренные данные
    # реального frame pacing показывают узкую базовую моду + редкие выбросы на порядок
    # величины (напр. троттлинг GPU при упоре в лимит мощности даёт скачок 50-100мс поверх
    # базовых ~8-16мс кадров — это ~5-12x; G-buffer generation time варьируется между
    # определёнными кадрами одной сцены в пределах четырёх порядков величины). Дефолт —
    # рабочая точка, не откалиброван на реальных логах ЭТОГО проекта (Kaggle не даёт живой
    # GPU-нагрузки, см. барьер №5) — предположение по индустриальным данным, требует
    # уточнения после результатов этого прогона.
    dt_spike_prob: float = 0.07
    # множитель на БАЗОВЫЙ dt при выбросе (диапазон, сэмплируется равномерно внутри)
    dt_spike_multiplier_range: tuple = (5.0, 15.0)
    # разрешить ли drift-сегментам менять скорость/направление внутри сегмента
    drift_rate_range: tuple = (-0.05, 0.05)  # относительное изменение истины за единицу dt

    seed: int = 0


SEG_STATIC, SEG_STEP, SEG_DRIFT = 0, 1, 2


def _gen_segment_plan(rng: np.random.Generator, cfg: GenConfig):
    """Разбивает последовательность на сегменты со случайными типами и длинами."""
    n_segments = rng.integers(cfg.n_segments_range[0], cfg.n_segments_range[1] + 1)
    lengths = rng.integers(cfg.segment_len_range[0], cfg.segment_len_range[1] + 1, size=n_segments)
    total = lengths.sum()
    # растягиваем/обрезаем под seq_len
    if total < cfg.seq_len:
        lengths[-1] += cfg.seq_len - total
    types = rng.integers(0, 3, size=n_segments)  # 0=static,1=step,2=drift
    return types, lengths


def _gen_true_signal(rng: np.random.Generator, cfg: GenConfig, types, lengths):
    T = int(lengths.sum())
    true_values = np.zeros(T)
    seg_type_arr = np.zeros(T, dtype=np.int64)
    event_mask = np.zeros(T, dtype=np.int64)

    cur_value = rng.uniform(*cfg.value_range)
    idx = 0
    for seg_type, seg_len in zip(types, lengths):
        seg_type = int(seg_type)
        seg_len = int(seg_len)
        if seg_type == SEG_STATIC:
            true_values[idx:idx + seg_len] = cur_value
            seg_type_arr[idx:idx + seg_len] = SEG_STATIC

        elif seg_type == SEG_STEP:
            # скачок в СЛУЧАЙНЫЙ момент внутри сегмента, не обязательно в начале
            jump_at = rng.integers(1, max(seg_len - 1, 2))
            mag = rng.uniform(*cfg.step_magnitude_range)
            sign = rng.choice([-1.0, 1.0])
            new_value = np.clip(cur_value * (1.0 + sign * mag), *cfg.value_range)
            true_values[idx:idx + jump_at] = cur_value
            true_values[idx + jump_at:idx + seg_len] = new_value
            seg_type_arr[idx:idx + seg_len] = SEG_STEP
            event_mask[idx + jump_at] = 1
            cur_value = new_value

        elif seg_type == SEG_DRIFT:
            rate = rng.uniform(*cfg.drift_rate_range)
            local = cur_value * np.exp(rate * np.arange(seg_len))
            local = np.clip(local, *cfg.value_range)
            true_values[idx:idx + seg_len] = local
            seg_type_arr[idx:idx + seg_len] = SEG_DRIFT
            cur_value = local[-1]

        idx += seg_len

    return true_values[:cfg.seq_len], seg_type_arr[:cfg.seq_len], event_mask[:cfg.seq_len]


def generate_sequence(rng: np.random.Generator, cfg: GenConfig):
    types, lengths = _gen_segment_plan(rng, cfg)
    true_values, seg_type_arr, event_mask = _gen_true_signal(rng, cfg, types, lengths)

    T = cfg.seq_len
    dt = rng.uniform(*cfg.dt_range, size=T)
    # Тяжёлохвостые выбросы Δt (калибровка 2026-07-27, см. GenConfig.dt_spike_prob) —
    # независимо от базовой моды, каждый шаг с малой вероятностью получает мультипликативный
    # выброс поверх базового dt, имитируя редкие GPU-стоппы (троттлинг/компиляция шейдеров/
    # LOD-переключения) вместо равномерного разброса на всём диапазоне.
    dt_spike_mask = rng.uniform(0.0, 1.0, size=T) < cfg.dt_spike_prob
    # ВСЕГДА расходуем одно и то же число случайных чисел из потока, независимо от
    # dt_spike_prob (в т.ч. при prob=0.0, когда маска пустая) — иначе поток RNG сдвигается
    # по-разному между prob=0.0 и prob>0.0, и сравнение перестаёт быть чистой абляцией
    # (см. обсуждение 2026-07-27: "конечность выборки" и чистое A/B сравнение).
    spike_mult = rng.uniform(*cfg.dt_spike_multiplier_range, size=T)
    dt = np.where(dt_spike_mask, dt * spike_mult, dt)
    dt[0] = 0.0  # первый шаг — нет предыдущего наблюдения
    dt_spike_mask[0] = False

    spp = rng.integers(cfg.spp_range[0], cfg.spp_range[1] + 1, size=T)
    noise_std = np.sqrt(cfg.noise_c * np.maximum(true_values, 1e-3) / spp)
    obs = true_values + rng.normal(0.0, 1.0, size=T) * noise_std

    return dict(
        true_values=true_values.astype(np.float32),
        obs=obs.astype(np.float32),
        dt=dt.astype(np.float32),
        spp=spp.astype(np.float32),
        seg_type=seg_type_arr,
        event_mask=event_mask,
        noise_std=noise_std.astype(np.float32),
        dt_spike_mask=dt_spike_mask,  # диагностика: НЕ подаётся в модель, только для анализа
    )


def generate_dataset(n_sequences: int, cfg: GenConfig):
    rng = np.random.default_rng(cfg.seed)
    return [generate_sequence(rng, cfg) for _ in range(n_sequences)]


def make_train_test_configs(base_seed: int = 0):
    """
    Раздельные train/test конфиги с НЕ совпадающими диапазонами величины скачка и seed,
    чтобы тестовые скачки не совпадали один в один с обучающими (ТЗ п.3.3).
    """
    train_cfg = GenConfig(
        seed=base_seed,
        step_magnitude_range=(0.3, 2.0),
        drift_rate_range=(-0.04, 0.04),
    )
    test_cfg = GenConfig(
        seed=base_seed + 10_000,  # непересекающийся поток случайности
        step_magnitude_range=(0.5, 3.0),  # смещённый диапазон — не идентичный train
        drift_rate_range=(-0.06, 0.06),
    )
    return train_cfg, test_cfg


if __name__ == "__main__":
    train_cfg, test_cfg = make_train_test_configs(seed_base := 0)
    seq = generate_sequence(np.random.default_rng(0), train_cfg)
    print("seq keys:", list(seq.keys()))
    print("T =", len(seq["true_values"]))
    print("seg types present:", np.unique(seq["seg_type"]))
    print("n events:", seq["event_mask"].sum())
    print("value range:", seq["true_values"].min(), seq["true_values"].max())
    print("obs noise std range:", seq["noise_std"].min(), seq["noise_std"].max())

    # Быстрая проверка калибровки Δt-выбросов (2026-07-27) на большей выборке для устойчивой статистики
    dts, spikes = [], []
    rng_check = np.random.default_rng(123)
    for _ in range(200):
        s = generate_sequence(rng_check, train_cfg)
        dts.append(s["dt"])
        spikes.append(s["dt_spike_mask"])
    dts = np.concatenate(dts)
    spikes = np.concatenate(spikes)
    print(f"\ndt stats over {len(dts)} steps:")
    print(f"  spike fraction: {spikes.mean():.4f} (target ~{train_cfg.dt_spike_prob})")
    print(f"  dt[normal] mean/std/max: {dts[~spikes].mean():.3f}/{dts[~spikes].std():.3f}/{dts[~spikes].max():.3f}")
    if spikes.any():
        print(f"  dt[spike]  mean/std/max: {dts[spikes].mean():.3f}/{dts[spikes].std():.3f}/{dts[spikes].max():.3f}")
