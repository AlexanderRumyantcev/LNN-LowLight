"""
Адаптер §2.3(продолжение)/§2.6: переупаковка generate_dataset_dense.py
(.npz растровые буферы) в per-pixel временные последовательности,
готовые для CfCProbeModule.build_input(staleness_dim=4).

Аналог blender/dataset_adapter.py для probe-пути (build_all_probe_sequences),
но с тремя структурными отличиями, вытекающими из §1.3/§2.3 ТЗ:
- одна последовательность на ЭКРАННЫЙ ПИКСЕЛЬ (x, y), а не на
  world-space пробу — пиксель не привязан к одной и той же геометрии
  между кадрами (движется камера), в отличие от probe-пути, где проба =
  фиксированная world-space точка на неподвижной ortho-камере;
- dt РЕГУЛЯРНЫЙ (= шаг датасета) на каждом шаге, кроме dt[0]=0
  (cold start) — переиспользован ТОТ ЖЕ принцип, что уже в
  probe_subsampling.build_probe_sequence (dt[0]=0.0, cold_start[0]=1.0),
  просто без асинхронной подвыборки (§2.3: "все GRID_RES² пикселей
  рендерятся синхронно");
- добавляет disocclusion_flag/geometric_mismatch_score (§2.5) через
  blender/disocclusion.reproject_and_check на паре соседних кадров
  (idx-1, idx) — то, чего нет и не может быть в probe-пути (там нет
  движения камеры, значит нет дисокклюзии).

ПЕРЕИСПОЛЬЗУЕТСЯ БЕЗ ИЗМЕНЕНИЙ: disocclusion.reproject_and_check (§2.4/
§2.5, уже провалидирован на реальных данных в 08_disocclusion_prototype.py),
cold_start/dt[0]=0.0 конвенция из probe_subsampling.py, confidence-
нормализация log1p(spp)/log1p(64.0) — тот же фиксированный глобальный
масштаб, что в run_experiment.py/run_experiment_blender.py (не
пересчитывается по датасету, чтобы probe- и dense-путь оставались
сопоставимы по этой оси).

МОЯ инженерная реализация (не задано ТЗ явно): при движущейся камере
экранный пиксель (x, y) не гарантированно валиден (§3.1: alpha>0.5) на
КАЖДОМ кадре последовательности — floor может уйти из кадра и вернуться
между двумя моментами t. В отличие от probe-пути (где valid — константа
на всю последовательность пробы), здесь valid отдаётся per-frame как
часть последовательности (ключ 'valid'), а не используется, чтобы
отбросить пиксель целиком — маскирование loss на невалидных шагах
оставлено вызывающему коду (models/losses.py), не декларируется здесь.
Пиксель исключается из выдачи только если НИ РАЗУ не был валиден за
весь прогон (полностью вне геометрии сцены на всех кадрах).

Что здесь НЕ делается (сознательно, отдельные шаги): wiring в
CfCProbeModule (obs_dim/staleness_dim передаются явно вызывающим кодом
при инстанцировании, сам модуль уже общий), интерполяция между
пикселями (не нужна — pixel-grid уже плотная, в отличие от разреженных
проб probe-пути).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from disocclusion import reproject_and_check

# Та же фиксированная нормализация confidence, что run_experiment.py/
# run_experiment_blender.py (log1p(64.0), не пересчитывается по
# конкретному датасету) — см. докстринг модуля.
CONFIDENCE_LOG_MAX = float(np.log1p(64.0))


def load_dense_dataset(npz_path):
    return np.load(npz_path)


def build_all_pixel_sequences(
    npz_path,
    distance_scale: float = 0.2,
    normal_threshold_deg: float = 45.0,
    mismatch_threshold: float = 0.5,
):
    """Список dict — одна запись на каждый (валидный хотя бы раз за
    прогон) пиксель сетки GRID_RES×GRID_RES, тот же общий стиль вывода,
    что build_all_probe_sequences (probe-путь), с дополнительными
    ключами disocclusion_flag/geometric_mismatch_score.

    distance_scale/normal_threshold_deg/mismatch_threshold — прокинуты
    напрямую в disocclusion.reproject_and_check (см. её докстринг про
    некалиброванность порогов, §2.7/§7 ТЗ) — не переоткалиброваны
    отдельно здесь.
    """
    data = np.load(npz_path)
    position = data["position"]                # [T, H, W, 3]
    normal = data["normal"]                     # [T, H, W, 3]
    irradiance = data["irradiance"]              # [T, H, W, 3]
    true_irradiance = data["true_irradiance"]    # [T, H, W, 3]
    spp = data["spp"]                            # [T, H, W]
    valid = data["valid"]                        # [T, H, W] bool
    motion_vector = data["motion_vector"]        # [T, H, W, 2] -- (Z,W) каналы
    camera_positions = data["camera_positions"]  # [T, 3]
    t_values = data["t_values"]                  # [T]

    T, H, W = position.shape[0], position.shape[1], position.shape[2]
    assert T > 1, "нужно хотя бы 2 кадра для дисокклюзии"
    dt_diffs = np.diff(t_values)
    assert np.allclose(dt_diffs, dt_diffs[0]), (
        "§2.3: регулярная сетка t_i=i*DT — ожидается константный шаг, "
        f"получено {dt_diffs}"
    )
    dt_const = float(dt_diffs[0])

    sequences = []
    for y in range(H):
        for x in range(W):
            if not valid[:, y, x].any():
                continue  # пиксель ни разу не попал в геометрию сцены

            dt = np.full(T, dt_const, dtype=np.float32)
            dt[0] = 0.0
            cold_start = np.zeros(T, dtype=np.float32)
            cold_start[0] = 1.0

            # idx=0: нет предыдущего кадра для warp -> трактуем как
            # тривиальную дисокклюзию (§2.5 out-of-bounds-семантика:
            # "history невалидна тривиально"), тем же принципом, что
            # cold_start[0]=1.0 уже кодирует "нет истории вообще".
            disocclusion_flag = np.zeros(T, dtype=np.float32)
            disocclusion_flag[0] = 1.0
            geometric_mismatch_score = np.zeros(T, dtype=np.float32)
            geometric_mismatch_score[0] = 1.0

            for idx in range(1, T):
                mv = motion_vector[idx, y, x]
                out = reproject_and_check(
                    x=float(x), y=float(y), motion_vector_zw=mv,
                    pos_curr=position[idx, y, x], normal_curr=normal[idx, y, x],
                    prev_position_buffer=position[idx - 1],
                    prev_normal_buffer=normal[idx - 1],
                    camera_pos=camera_positions[idx],
                    distance_scale=distance_scale,
                    normal_threshold_deg=normal_threshold_deg,
                    mismatch_threshold=mismatch_threshold,
                )
                disocclusion_flag[idx] = 1.0 if out["disocclusion_flag"] else 0.0
                geometric_mismatch_score[idx] = out["geometric_mismatch_score"]

            spp_seq = spp[:, y, x].astype(np.float32)
            confidence = (np.log1p(spp_seq) / CONFIDENCE_LOG_MAX).astype(np.float32)

            sequences.append(dict(
                pixel=(x, y),
                t=t_values.astype(np.float32),
                dt=dt,
                cold_start=cold_start,
                obs=irradiance[:, y, x, :].astype(np.float32),
                true_irradiance=true_irradiance[:, y, x, :].astype(np.float32),
                spp=spp_seq,
                confidence=confidence,
                valid=valid[:, y, x].astype(bool),
                disocclusion_flag=disocclusion_flag,
                geometric_mismatch_score=geometric_mismatch_score,
            ))
    return sequences
