"""
Единый скрипт прогона pipeline dense-пути (per-pixel,
TZ_stage1b_per_pixel_dense_fallback.md) на РЕАЛЬНЫХ Blender-данных
(blender/generate_dataset_dense_result_full.npz) — аналог
run_experiment_blender.py (probe-путь), максимально переиспользует
существующий код:

БЕЗ ИЗМЕНЕНИЙ переиспользуется: build_batch (обрезка по мин. длине +
вывод obs_dim из формы данных), _select_device, _summarize (per-
segment-type/early/floor метрики §6.1-6.3, канально усредняемые),
_reconstruct_light_schedule — всё из run_experiment.py/
run_experiment_blender.py; NRCRelativeL2Loss (models/losses.py),
calibrate_time_gate_init (models/temporal/cfc_probe_module.py) — без
изменений; CfCProbeModule/NRCStyleBaseline/GRUProbeBaseline/
NRDStyleBaseline — их staleness_dim=4/use_disocclusion_reset/
use_mismatch_honest расширения (§2.6/§4.2), реализованные ранее в
этой же сессии.

ЧТО ДЕЙСТВИТЕЛЬНО ОТЛИЧАЕТСЯ ОТ PROBE-ПУТИ (не может быть
переиспользовано как есть):
- Источник последовательностей: build_all_pixel_sequences
  (blender/dataset_adapter_dense.py) — одна последовательность на
  ЭКРАННЫЙ ПИКСЕЛЬ, не world-space пробу (§2.3).
- train_model_dense/predict_dense: honest-версии CfC/NRC/GRU получают
  staleness_dim=4/extra_staleness_dim=2 с extra_staleness=
  [disocclusion_flag, geometric_mismatch_score] (§2.6/§4.2) вместо
  старого 2-/3-элементного вектора. NRD-style использует НОВЫЕ
  режимы (use_disocclusion_reset для faithful, use_mismatch_honest для
  honest, §4.2) вместо старого use_honest_dt — Δt-awareness для
  NRD-style в dense-пути неактуальна тем же образом, что в probe-пути.
- Метрики: ДОБАВЛЯЕТСЯ §6 delta (disocclusion_vs_stable_mse,
  error_vs_warp_age_curve, evaluation/metrics.py) поверх старых
  per_segment_type/early_zone_floor/error_vs_offset (те остаются без
  изменений — §6 ТЗ явно требует этого для static/step/drift среза;
  dense-датасет тоже использует light_schedule.py, тот же сигнал).
- Интерполяция между пробами (models/probe_interpolation.py) НЕ
  ПЕРЕНОСИТСЯ — per-pixel dense не имеет разреженных проб, которые
  нужно интерполировать между собой (§2.3: сама warp/reprojection —
  это и есть механизм переноса между КАДРАМИ, не между пикселями).
- "Seeds" здесь ЕЩЁ СЛАБЕЕ, чем в run_experiment_blender.py (см. её
  докстринг): там вариация — per-probe subsampling + train/eval split
  ПРОБ поверх одного рендера. Здесь Δt РЕГУЛЯРНЫЙ (§2.3) — subsampling
  не при чём вообще; единственная вариация между "сидами" — train/eval
  РАЗБИЕНИЕ ПИКСЕЛЕЙ поверх одного и того же рендера (одна геометрия,
  одна камера-траектория, одно световое расписание). Значимость
  измеряет устойчивость ТОЛЬКО к разбиению пикселей — печатается явно.

МОЁ ИНЖЕНЕРНОЕ РЕШЕНИЕ (не задано ТЗ явно): для этой первой версии
скрипта используются только пиксели, валидные НА ВСЕХ кадрах
(seq["valid"].all()) — не "валиден хотя бы раз", как в
dataset_adapter_dense.py. NRCRelativeL2Loss (models/losses.py) не
поддерживает per-timestep маскирование (только continuous
confidence_weight, не boolean mask) — обучение/оценка на пикселях с
невалидными (sky) кадрами внутри последовательности без маскирования
означало бы тренировку на бессмысленных значениях. Полное маскирование
loss — отдельная доработка на будущее, не в этой сессии; текущий
фильтр — простой и честный способ не завязывать sanity-check на
нерешённой проблеме. На полном датасете (generate_dataset_dense_result_
full.npz) это оставляет 29 из 64 пикселей (см. mempalace) — достаточно
для sanity-прогона, недостаточно для финальных выводов о качестве.

ГЛАВНЫЙ КРИТЕРИЙ §5 stage1b — три возможных исхода, определяемые
ИМЕННО disocclusion-срезом (§5/§6 delta), не агрегатом по кадру —
печатается отдельным блоком в конце run_multi_seed().
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "blender"))
from dataset_adapter_dense import build_all_pixel_sequences  # noqa: E402

from run_experiment import build_batch as _build_batch_base
from run_experiment_blender import _select_device, _reconstruct_light_schedule, _summarize
from models.temporal.cfc_probe_module import CfCProbeModule, calibrate_time_gate_init, full_gate_diagnostics
from models.baselines import NRDStyleBaseline, NRCStyleBaseline, GRUProbeBaseline, ODELSTMProbeBaseline
from models.losses import NRCRelativeL2Loss
from models.spatial_features import build_spatial_conditioning, spatial_dim as _spatial_dim_fn
from evaluation.metrics import (
    label_samples, paired_bootstrap_significance, MIN_N_SEEDS, SEGMENT_NAMES,
    disocclusion_vs_stable_mse, error_vs_warp_age_curve,
)
from determinism import set_determinism, capture_env_metadata  # 2026-08-28, methodology plan

BIN_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, np.inf])
BIN_LABELS = [f"[{BIN_EDGES[i]:g},{BIN_EDGES[i+1]:g})" for i in range(len(BIN_EDGES) - 1)]
SEGMENT_TYPE_NAMES = list(SEGMENT_NAMES.values())

DEFAULT_DATASET = Path(__file__).parent / "blender" / "generate_dataset_dense_result_full.npz"

# Позиционное кондиционирование (2026-08-20, чат/mempalace) — фиксированная
# размерность spatial-входа для ВСЕХ "_spatial" конфигураций (единая точка, не
# пересчитывается по месту вызова, чтобы __init__ модели и build_input использовали заведомо одно
# и то же число). roughness не передаётся (include_roughness=False) — в текущей Blender-сцене
# это константа (Roughness=1.0, одна общая материал), не рендерится отдельным буфером (см. models/
# spatial_features.py докстринг).
SPATIAL_DIM = _spatial_dim_fn(n_freqs_position=4, n_freqs_direction=2, include_roughness=False)

# Масштаб сцены для нормализации position перед positional encoding (2026-08-20 багфикс,
# см. models/spatial_features.py::build_spatial_conditioning докстринг position_scale).
# blender/generate_dataset_dense.py::build_static_geometry — bpy.ops.mesh.primitive_plane_add(
# size=8) -> пол простирается от -4 до +4 по x/y -> половина протяжённости = 4.0.
FLOOR_HALF_EXTENT = 4.0

MODEL_KINDS = ["cfc", "cfc_hardjump", "cfc_hardjump_skip", "cfc_mmrnn", "nrd_faithful",
               "nrd_honest", "nrc_faithful", "nrc_honest", "gru_faithful", "gru_honest",
               "ode_lstm_honest", "cfc_spatial", "nrc_honest_spatial", "gru_honest_spatial",
               "ode_lstm_spatial"]
# SKIP_KINDS (31.08.2026, чат) — гипотеза триумвирата с ODE-LSTM опровергнута
# (paired bootstrap по warp_age, n=16, dt1.npz — ode_lstm_honest нигде не превосходит
# nrc_honest значимо, значимо хуже сразу после disocclusion; см. mempalace). ode_lstm_*
# на порядок дороже остальных моделей (RK4) и больше не нужен в продакшн-прогонах —
# --skip_kinds позволяет пропустить его обучение (и обучение любых других MODEL_KINDS)
# не трогая остальной пайплайн. Пропущенные kind'ы просто отсутствуют в preds (тот же
# паттерн, что уже используется для cfc_mmrnn — см. guard в _summarize/_summarize_
# disocclusion), их метрики в summary выводятся как nan. По умолчанию пусто (обратная
# совместимость).
SKIP_KINDS = set()

# *_spatial (2026-08-20, чат/mempalace) — те же 4 честные конфигурации с общими весами
# (cfc/nrc_honest/gru_honest/ode_lstm_honest), плюс позиционное кондиционирование (models/
# spatial_features.py::build_spatial_conditioning — позиция+направление+normal+albedo,
# positional encoding, §5.2 Müller et al. 2021). Применено ОДИНАКОВО ко всем четырём (не
# только к nrc/cfc) — решение из чата: иначе разница cfc vs honest была бы смесью
# "closed-form преимущество" и "доступ к позиции", а не изолированным тестом ни одного из
# двух вопросов (а)/(б). nrd_* и *_faithful сюда не входят (NRD — 0 обучаемых параметров,
# кондиционировать нечего; *_faithful по определению не видит ничего сверх obs; добавление spatial
# сделало бы его наполовину honest, смешивая вопросы (а)/(б), см. models/baselines.py доктрины).
# cfc_hardjump_skip (2026-08-19, models/temporal/cfc_probe_module.py::use_jump_output_skip):
# та же cfc_hardjump (h/jump_map/рекуррентная динамика НЕ меняются), но на шаге disocclusion=1
# предсказание — out_t = obs_t + zero-init residual(h_jump), а не readout(h_jump) через две
Tanh-прослойки с нуля. Проверяет найденный разрыв с NRDStyleBaseline.use_disocclusion_reset
# (out_t=obs_t, ноль параметров) — cfc_hardjump статистически значимо ХУЖЕ nrd_honest на
# disocclusion (см. mempalace), гипотеза: разрыв инженерный (нет прямого пути к obs_t), не
# архитектурный. Если cfc_hardjump_skip закрывает разрыв с nrd_honest — гипотеза подтверждена.
# cfc_hardjump (2026-08-16, ablation по Krach et al. 2022 / Herrera et al. 2021, см. mempalace):
# та же CfCProbeModule, но с use_hard_jump_reset=True — на шагах disocclusion=1 состояние
# ПОЛНОСТЬЮ заменяется отдельной MLP от u_t (NJ-ODE-style жёсткий jump, вес 1), а не смешивается
# через sigma_tau. Проверяет гипотезу "у CfC-B нет отдельного hard-reset примитива" напрямую —
# если disocc_mse(cfc_hardjump) << disocc_mse(cfc) и приближается к nrc_honest, гипотеза
# подтверждена; если нет — узкое место не в отсутствии jump-канала как такового.
#
# cfc_mmrnn (2026-08-22, TZ_stage3_cfc_mmrnn.md, models/temporal/cfc_probe_module.py::
# use_mixed_memory): та же cfc_hardjump_skip (disocclusion уже закрыт, не трогаем) ПЛЮС
mixed-memory RNN буфер (Lechner & Hasani, OpenReview rOGm97YR22N/NeurIPS 2022) — LSTM
# ПЕРЕД CfC-шагом каждый раз, отдельный c_state, никогда не проходящий через continuous-
# time часть. Цель — STABLE-срез (не disocclusion): cfc_hardjump_skip сравнялся с nrc_honest
# на disocclusion, но остался значимо хуже nrd_honest/gru_honest на stable (см. mempalace,
# 2026-08-19) — проверяем гипотезу "это vanishing/exploding gradient на длинных сегментах,
# то, что чинит именно mmRNN" против альтернативы "узкое место не в этом". На disocclusion
# (TZ_stage3 §3.2, Вариант A): c_state тоже жёстко сбрасывается через jump_map, тем же
# принципом, что и h — см. докстринг forward() в cfc_probe_module.py.
#
# ode_lstm_honest (2026-08-18, TZ_stage2_efficiency_and_baseline_validation.md §2.2):
# ODELSTMProbeBaseline (models/baselines.py) — настоящий numerical-integration ODE-RNN
# baseline (ODE-LSTM, Lechner & Hasani 2020), не closed-form как CfC-B. Только honest-версия
# (Δt архитектурно обязателен для этой модели — см. докстринг класса, faithful-вариант без
# Δt не имеет смысла в отличие от NRD/NRC/GRU). Отвечает на вопрос §2: теряет ли CfC-B
# точность именно из-за closed-form приближения относительно настоящего интегрирования.


def load_fully_valid_pixel_sequences(npz_path):
    """build_all_pixel_sequences (blender/dataset_adapter_dense.py) без
    изменений + фильтр на "валиден на ВСЕХ кадрах" (см. докстринг
    модуля про причину)."""
    seqs = build_all_pixel_sequences(npz_path)
    filtered = [s for s in seqs if s["valid"].all()]
    print(f"пикселей всего с последовательностью: {len(seqs)}, "
          f"валидных на всех кадрах: {len(filtered)}")
    return filtered


def build_batch_dense(seqs, indices):
    """build_batch (run_experiment.py) БЕЗ ИЗМЕНЕНИЙ для obs/dt/cold/
    conf/true/t (ключи совпадают — dataset_adapter_dense.py намеренно
    использует ту же номенклатуру, что probe_subsampling.py/
    dataset_adapter.py) + доп. dense-специфичные каналы
    (disocclusion_flag/geometric_mismatch_score/valid), обрезанные до
    той же длины T, что вычислил build_batch."""
    batch = _build_batch_base(seqs, indices)
    T = batch["dt"].shape[1]
    batch["disocclusion_flag"] = torch.tensor(
        np.stack([seqs[i]["disocclusion_flag"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["geometric_mismatch_score"] = torch.tensor(
        np.stack([seqs[i]["geometric_mismatch_score"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["valid"] = torch.tensor(
        np.stack([seqs[i]["valid"][:T] for i in indices]), dtype=torch.bool,
    )
    # Позиционное кондиционирование (2026-08-20, чат) — сырые буферы, ЕЩЁ БЕЗ positional
    # encoding (encoding применяется позже, в _get_spatial(), одинаково при train и predict,
    # models/spatial_features.py::build_spatial_conditioning).
    batch["position"] = torch.tensor(
        np.stack([seqs[i]["position"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["normal"] = torch.tensor(
        np.stack([seqs[i]["normal"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["albedo"] = torch.tensor(
        np.stack([seqs[i]["albedo"][:T] for i in indices]), dtype=torch.float32,
    )
    batch["direction"] = torch.tensor(
        np.stack([seqs[i]["direction"][:T] for i in indices]), dtype=torch.float32,
    )
    return batch


def _get_spatial(batch, device):
    """Единая точка сборки spatial-входа (2026-08-20, чат) — вызывается ОДИНАКОВО
    из train_model_dense/predict_dense для всех "_spatial" конфигураций, чтобы positional
    encoding (n_freqs_position/n_freqs_direction/position_scale) не разъехался между
    train и eval. position_scale=FLOOR_HALF_EXTENT (2026-08-20 багфикс, см. models/
    spatial_features.py докстринг position_scale — без нормализации positional encoding
    алиасировался на сырых мировых координатах пола, 4.5x больше данных эту
    деградацию не лечило, что и подтвердило: дело в масштабе encoding'а, не в объёме
    выборки)."""
    return build_spatial_conditioning(
        batch["position"].to(device), batch["direction"].to(device),
        batch["normal"].to(device), batch["albedo"].to(device),
        position_scale=FLOOR_HALF_EXTENT,
    )



# Модели без развитой recurrence (per-frame независимые MLP/GRU-cell на маленьких
# матрицах) — 2026-08-24 (стресс-тест n_seeds=16, чат): MPS-бэкенд документированно не
# гарантирует детерминизм даже при фиксированном torch.manual_seed (github.com/pytorch/
# pytorch#97236, module:determinism+module:mps; PyTorch Reproducibility notes — atomic-
# операции в reduction-ядрах не имеют фиксированного порядка суммирования). Эмпирически
# обнаружено: nrc_honest даёt РАЗНЫЕ per-seed числа между двумя прогонами на одинаковых
# сидах (nrc_honest stable seed0: 0.0169 vs 0.0221), тогда как cfc_mmrnn/cfc_hardjump_skip
# воспроизводятся побитово. Все модели в этом множестве дёшевы (<1с/сид на nrc/nrd, ~40с на
# gru) — форсируем CPU для них конкретно, что убирает MPS-недетерминизм почти бесплатно по
# времени, вместо общего torch.use_deterministic_algorithms(True) (неполная поддержка MPS,
# может падать на неподдерживаемых операциях).
_FORCE_CPU_KINDS = {
    "nrc_honest", "nrc_faithful", "nrc_honest_spatial",
    "gru_honest", "gru_faithful", "gru_honest_spatial",
}


def train_model_dense(kind: str, batch, device, epochs: int, lr: float, hidden_dim: int,
                       gate_lr: float | None = None):
    """Аналог train_model (run_experiment.py) — honest-версии получают
    staleness_dim=4/extra_staleness_dim=2 с extra_staleness=
    [disocclusion_flag, geometric_mismatch_score] (§2.6/§4.2) вместо
    старого 2-/3-элементного вектора. nrd_faithful/nrd_honest сюда не
    попадают — у NRDStyleBaseline нет обучаемых параметров (как и в
    train_model для probe-пути).

    gate_lr (только для kind='cfc', МОЙ инженерный эксперимент, не из
    ТЗ, 2026-08-09): отдельный learning rate для model.cell.W_a/W_b
    (time-gate readout) поверх общего lr для всей остальной сети.
    Мотивация — эмпирическая, не догадка: gate-диагностика (см.
    mempalace) показала, что backbone z РЕАЛЬНО различает disocclusion/
    stable контексты (cos_sim=0.886, не ~1.0), но W_a/W_b не
    конвертируют это различие в разный sigma_tau — а их градиент на
    порядок меньше остального backbone (0.0038/0.0018 vs 0.063/0.023).
    None (по умолчанию) = старое поведение, единый lr на все параметры,
    один Adam без param groups.
    """
    if kind in _FORCE_CPU_KINDS:
        device = torch.device("cpu")
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf, true = batch["cold"].to(device), batch["conf"].to(device), batch["true"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    obs_dim = obs.shape[-1]

    if kind == "cfc":
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt)[0]
    elif kind == "cfc_hardjump":
        # ablation (2026-08-16, см. MODEL_KINDS докстринг выше) — та же модель/калибровка/lr,
        # что и "cfc", единственное отличие: use_hard_jump_reset=True + disocc_seq в forward.
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
    elif kind == "cfc_hardjump_skip":
        # ablation (2026-08-19, см. MODEL_KINDS докстринг выше) — та же cfc_hardjump, плюс
        # use_jump_output_skip=True (h/jump_map не меняются, меняется только сам оутпут на
        # шаге disocclusion=1, см. cfc_probe_module.py).
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True,
                                use_jump_output_skip=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
    elif kind == "cfc_mmrnn":
        # ablation (2026-08-22, см. MODEL_KINDS докстринг выше / TZ_stage3_cfc_mmrnn.md) —
        # та же cfc_hardjump_skip, плюс use_mixed_memory=True (LSTM-буфер перед CfC-шагом).
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, use_hard_jump_reset=True,
                                use_jump_output_skip=True, use_mixed_memory=True).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch])
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt, disocc_seq=disocc)[0]
    elif kind == "nrc_honest":
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                  staleness_dim=4).to(device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch])
        forward = lambda: model(u)
    elif kind == "nrc_faithful":
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=False).to(device)
        u = NRCStyleBaseline.build_input(obs, use_staleness=False)
        forward = lambda: model(u)
    elif kind == "gru_honest":
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=True,
                                  extra_staleness_dim=2).to(device)
        u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                          extra_staleness=[disocc, mismatch])
        forward = lambda: model(u)[0]
    elif kind == "gru_faithful":
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=False).to(device)
        u = GRUProbeBaseline.build_input(obs, use_dt_staleness=False)
        forward = lambda: model(u)[0]
    elif kind == "ode_lstm_honest":
        # §2.2 TZ_stage2 — только honest (Δt архитектурно обязателен, faithful не существует
        # для этой архитектуры), staleness_dim=4 симметрично cfc/cfc_hardjump/nrc_honest/gru_honest.
        model = ODELSTMProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                      staleness_dim=4).to(device)
        u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch])
        forward = lambda: model(u, dt)[0]
    elif kind == "cfc_spatial":
        # 2026-08-20 (чат) — та же cfc, плюс позиционное кондиционирование (models/
        # spatial_features.py). calibrate_time_gate_init остаётся без изменений — она
        # калибрует ТОЛЬКО W_a (time-gate), который спереди видит тот же расширенный z из
        # backbone, но сам принцип калибровки (масштаб t_a*log1p(dt)) не завязан на состав
        # входа backbone'а.
        spatial = _get_spatial(batch, device)
        model = CfCProbeModule(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = model.build_input(obs, cold, conf, use_staleness=True, extra_staleness=[disocc, mismatch],
                               spatial=spatial)
        calibrate_time_gate_init(model, u, dt, target_pre_sigmoid_std=2.0)
        forward = lambda: model(u, dt)[0]
    elif kind == "nrc_honest_spatial":
        spatial = _get_spatial(batch, device)
        model = NRCStyleBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                  staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u)
    elif kind == "gru_honest_spatial":
        spatial = _get_spatial(batch, device)
        model = GRUProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_dt_staleness=True,
                                  extra_staleness_dim=2, spatial_dim=SPATIAL_DIM).to(device)
        u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                          extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u)[0]
    elif kind == "ode_lstm_spatial":
        spatial = _get_spatial(batch, device)
        model = ODELSTMProbeBaseline(obs_dim=obs_dim, hidden_dim=hidden_dim, use_staleness=True,
                                      staleness_dim=4, spatial_dim=SPATIAL_DIM).to(device)
        u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
        forward = lambda: model(u, dt)[0]
    else:
        raise ValueError(kind)

    loss_fn = NRCRelativeL2Loss(use_confidence_weight=False)
    if kind == "cfc" and gate_lr is not None:
        gate_params = list(model.cell.W_a.parameters()) + list(model.cell.W_b.parameters())
        gate_param_ids = {id(p) for p in gate_params}
        rest_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
        opt = torch.optim.Adam([
            {"params": rest_params, "lr": lr},
            {"params": gate_params, "lr": gate_lr},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = forward()
        loss = loss_fn(pred, true)
        loss.backward()
        opt.step()
    return model


def predict_dense(kind: str, model_or_none, batch, device, tau_nrd: float = 3.0, alpha_nrd: float = 0.3):
    if kind in _FORCE_CPU_KINDS:
        device = torch.device("cpu")
    obs, dt = batch["obs"].to(device), batch["dt"].to(device)
    cold, conf = batch["cold"].to(device), batch["conf"].to(device)
    disocc = batch["disocclusion_flag"].to(device)
    mismatch = batch["geometric_mismatch_score"].to(device)
    with torch.no_grad():
        if kind == "cfc":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "cfc_hardjump":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
        if kind == "cfc_hardjump_skip":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
        if kind == "cfc_mmrnn":
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt, disocc_seq=disocc)[0].cpu().numpy()
        if kind == "nrc_honest":
            u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch])
            return model_or_none(u).cpu().numpy()
        if kind == "nrc_faithful":
            u = NRCStyleBaseline.build_input(obs, use_staleness=False)
            return model_or_none(u).cpu().numpy()
        if kind == "gru_honest":
            u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                              extra_staleness=[disocc, mismatch])
            return model_or_none(u)[0].cpu().numpy()
        if kind == "gru_faithful":
            u = GRUProbeBaseline.build_input(obs, use_dt_staleness=False)
            return model_or_none(u)[0].cpu().numpy()
        if kind == "ode_lstm_honest":
            u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                                  extra_staleness=[disocc, mismatch])
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "cfc_spatial":
            spatial = _get_spatial(batch, device)
            u = model_or_none.build_input(obs, cold, conf, use_staleness=True,
                                           extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrc_honest_spatial":
            spatial = _get_spatial(batch, device)
            u = NRCStyleBaseline.build_input(obs, cold, conf, use_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u).cpu().numpy()
        if kind == "gru_honest_spatial":
            spatial = _get_spatial(batch, device)
            u = GRUProbeBaseline.build_input(obs, dt, cold, conf, use_dt_staleness=True,
                                              extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u)[0].cpu().numpy()
        if kind == "ode_lstm_spatial":
            spatial = _get_spatial(batch, device)
            u = ODELSTMProbeBaseline.build_input(obs, cold, conf, use_staleness=True,
                                                  extra_staleness=[disocc, mismatch], spatial=spatial)
            return model_or_none(u, dt)[0].cpu().numpy()
        if kind == "nrd_faithful":
            m = NRDStyleBaseline(alpha=alpha_nrd, use_disocclusion_reset=True).to(device)
            return m(obs, dt, disocclusion_flag_seq=disocc).cpu().numpy()
        if kind == "nrd_honest":
            m = NRDStyleBaseline(tau=tau_nrd, use_mismatch_honest=True).to(device)
            return m(obs, dt, geometric_mismatch_seq=mismatch).cpu().numpy()
    raise ValueError(kind)


def _summarize_disocclusion(model_kinds, preds, true, disocc_flags, n_eval, verbose=True):
    """§6 delta — MSE disocclusion vs stable + error-vs-warp-age, тем же
    способом канального усреднения (obs_dim каналов -> одно число на
    пиксель -> одно число на весь eval), что _summarize (run_experiment_
    blender.py) делает для static/step/drift."""
    results = {}
    for kind in model_kinds:
        if kind not in preds:
            # ВРЕМЕННЫЙ ФИКС (2026-08-28, methodology plan) — тот же случай, что в аналогичном
            # гарде в _summarize() (run_experiment_blender.py): `cfc_mmrnn` есть в MODEL_KINDS, но
            # НЕ вызывается в preds выше в _run_one_seed() (train_model_dense/predict_dense ЕГО
            # уже умеют, но вызов для него там не добавлен — cfc_mmrnn обучается/оценивается
            # только через отдельный run_cfc_mmrnn.py). Уже было зафиксировано как
            # известное, не исправленное "побочное" в EXPERIMENT_LOG.md до этой сессии.
            # TODO: либо прописать cfc_mmrnn в _run_one_seed()/preds, либо убрать его из
            # MODEL_KINDS для этого пути — решить отдельно, не в рамках methodology plan.
            results[kind] = dict(
                disocclusion=float("nan"), stable=float("nan"),
                warp_age_curve={label: float("nan") for label in BIN_LABELS},
            )
            continue
        disocc_vals, stable_vals = [], []
        curve_accum = {label: [] for label in BIN_LABELS}
        for p in range(n_eval):
            pred_p = preds[kind][p].astype(np.float64)  # [T, C]
            true_p = true[p].astype(np.float64)
            flag_p = disocc_flags[p].astype(bool)

            d_c, s_c = [], []
            curve_c = {label: [] for label in BIN_LABELS}
            for c in range(pred_p.shape[-1]):
                split = disocclusion_vs_stable_mse(pred_p[:, c], true_p[:, c], flag_p)
                if not np.isnan(split["disocclusion"]):
                    d_c.append(split["disocclusion"])
                if not np.isnan(split["stable"]):
                    s_c.append(split["stable"])
                curve = error_vs_warp_age_curve(pred_p[:, c], true_p[:, c], flag_p, bin_edges=BIN_EDGES)
                for label, val in curve.items():
                    if not np.isnan(val):
                        curve_c[label].append(val)
            if d_c:
                disocc_vals.append(float(np.mean(d_c)))
            if s_c:
                stable_vals.append(float(np.mean(s_c)))
            for label in BIN_LABELS:
                if curve_c[label]:
                    curve_accum[label].append(float(np.mean(curve_c[label])))

        results[kind] = dict(
            disocclusion=float(np.mean(disocc_vals)) if disocc_vals else float("nan"),
            stable=float(np.mean(stable_vals)) if stable_vals else float("nan"),
            warp_age_curve={label: (float(np.mean(vals)) if vals else float("nan"))
                             for label, vals in curve_accum.items()},
        )

    if verbose:
        print("\n=== §6 delta: disocclusion vs stable MSE ===")
        for kind in model_kinds:
            r = results[kind]
            print(f"{kind:14s}  disocclusion={r['disocclusion']:.4f}  stable={r['stable']:.4f}")

    return results


def _aggregate_gate_diag(diag_list: list[dict]) -> dict:
    """Усредняет gate-диагностику (full_gate_diagnostics) ПО СИДАМ в одну строку —
    2026-08-11, TZ_stage1b_gate_theory_sweep.md §2/§6.2: sweep по gate_lr должен давать
    ОДНУ сравнимую строку на конфиг, а не список из n_seeds отдельных словарей. Простое
    среднее по сидам (не взвешенное по n) — n_train_pixels фиксирован по ТЗ §6.1, поэтому
    n по группам varies минимально между сидами (только из-за случайного train/eval split).
    """
    def _mean_field(get_group_dict, group_key, field):
        vals = [get_group_dict(d)[group_key][field] for d in diag_list
                 if group_key in get_group_dict(d)]
        return float(np.mean(vals)) if vals else float("nan")

    agg = dict(n_seeds=len(diag_list))

    # top-level (агрегат по ВСЕМ шагам, без разбивки на группы)
    agg["sigma_tau_mean"] = float(np.mean([d["sigma_tau"]["mean"] for d in diag_list]))
    agg["sat_low_frac"] = float(np.mean([d["sigma_tau"]["sat_low(<0.02)"] for d in diag_list]))
    agg["sat_high_frac"] = float(np.mean([d["sigma_tau"]["sat_high(>0.98)"] for d in diag_list]))
    agg["t_a_abs_mean"] = float(np.mean([d["t_a_dt_scale"]["t_a_abs_mean"] for d in diag_list]))
    agg["t_b_abs_mean"] = float(np.mean([abs(d["t_b=W_b(z)"]["mean"]) for d in diag_list]))

    # градиентная норма W_a/W_b относительно backbone (§6.2 требование ТЗ)
    for layer in ("backbone", "W_a", "W_b"):
        w = [d["grad"][layer]["weight_grad_norm"] for d in diag_list if d.get("grad")]
        b = [d["grad"][layer]["bias_grad_norm"] for d in diag_list if d.get("grad")]
        agg[f"grad_{layer}_weight_norm"] = float(np.mean(w)) if w else float("nan")
        agg[f"grad_{layer}_bias_norm"] = float(np.mean(b)) if b else float("nan")

    # разбивка (seg_type × disocclusion_flag) — ГЛАВНОЕ для H1/H2 (§6.3 ТЗ)
    all_joint_keys = set()
    for d in diag_list:
        all_joint_keys.update(d.get("by_seg_type_x_disocclusion", {}).keys())
    by_joint = {}
    for key in sorted(all_joint_keys):
        by_joint[key] = dict(
            sigma_tau_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sigma_tau_mean"),
            sat_low_frac=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sat_low_frac"),
            sat_high_frac=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "sat_high_frac"),
            t_a_abs_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "t_a_abs_mean"),
            t_b_abs_mean=_mean_field(lambda d: d.get("by_seg_type_x_disocclusion", {}), key, "t_b_abs_mean"),
        )
    agg["by_seg_type_x_disocclusion"] = by_joint

    # маргинал по disocclusion (§1 by_disocclusion) — для H1/H2 на более грубом уровне
    all_disocc_keys = set()
    for d in diag_list:
        all_disocc_keys.update(d.get("by_disocclusion", {}).keys())
    by_disocc = {}
    for key in sorted(all_disocc_keys):
        by_disocc[key] = dict(
            sigma_tau_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sigma_tau_mean"),
            sat_low_frac=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sat_low_frac"),
            sat_high_frac=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "sat_high_frac"),
            t_a_abs_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "t_a_abs_mean"),
            t_b_abs_mean=_mean_field(lambda d: d.get("by_disocclusion", {}), key, "t_b_abs_mean"),
        )
    agg["by_disocclusion"] = by_disocc

    return agg


def _train_with_progress(kind, seed, n_seeds, train_batch, device, epochs, lr, hidden_dim,
                          gate_lr=None, verbose=True):
    """Обёртка над train_model_dense с печатью прогресса по этапам (не по эпохам —
    train_model_dense не даёт хуков внутрь цикла эпох) — ode_lstm_honest на порядок
    дороже остальных (RK4, rk4_substeps forward-проходов f_node на шаг вместо одного),
    поэтому без этого печать вида выглядит как зависание на несколько минут.

    31.08.2026 (чат) — если kind в SKIP_KINDS, обучение полностью пропускается,
    возвращается None (см. докстринг SKIP_KINDS)."""
    if kind in SKIP_KINDS:
        if verbose:
            print(f"  [seed {seed+1}/{n_seeds}] {kind} — пропущено (SKIP_KINDS)", flush=True)
        return None
    if verbose:
        print(f"  [seed {seed+1}/{n_seeds}] обучение {kind} ({epochs} эпох)...",
              end="", flush=True)
    t0 = time.perf_counter()
    model = train_model_dense(kind, train_batch, device, epochs, lr, hidden_dim, gate_lr=gate_lr)
    if verbose:
        print(f" готово за {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def _run_one_seed(seqs, light_schedule, seed, n_train_pixels, epochs, lr, hidden_dim, device,
                   verbose=True, gate_lr=None, n_seeds=1):
    """Один 'сид' = train/eval разбиение ПИКСЕЛЕЙ поверх ОДНИМ и тем
    же рендером (см. предупреждение в докстринге модуля — Δt регулярный,
    значит здесь нет второй оси вариации, которая была в probe-пути
    через per-probe subsampling): — обрезано, остновок дальше идет путём сильного—

... (обрезано, лимит вывода инструмента)