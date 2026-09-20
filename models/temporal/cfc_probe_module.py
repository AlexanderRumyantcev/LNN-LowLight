"""
CfC-B: closed-form continuous-time temporal-модель для world-space проб (ТЗ §2.2).

Отличие от models/temporal/cfc_module.py (старый видео-модуль, до пивота 24.07):
    там вход — spatial illumination feature map [B, C, H, W], здесь — скалярная/векторная
    временная последовательность НАБЛЮДЕНИЙ ОДНОЙ ПРОБЫ (world-space irradiance probe),
    без какой-либо spatial pooling/broadcast — проба уже точка в пространстве, а не карта.

Архитектура (ТЗ §2.2): staleness-вектор (cold-start флаг + confidence из spp, §2.2/§3.1)
    вплетён в тот же backbone z, из которого считаются и кандидаты (g, h_cand), и параметры
    time-gate (sigma_tau) — т.е. влияет на closed-form решение НЕ как последующая коррекция
    выхода, а через ту же представление z, что и обычный вход. Именно так это провалидировано
    в изолированном spike-тесте (TZ_spike_test_CfC_step_detection.md, mode='B') —
    см. spike_test/models/models.py: CfCCell/CfCSequenceModel, mode='B'. Код здесь —
    production-версия того же механизма для формата данных полного пайплайна (§3.1: obs,
    dt, cold_start, confidence), без синтетических oracle-режимов spike-теста.

dirty-флаг от движения сцены сюда НЕ входит (сознательно, §2.2/§7.3) — появится на этапе-2.

Интерполяция между пробами (visibility+normal weighting, §2.3) — забота ДРУГОГО модуля
    (деталь постпроцессинга ДО подачи в CfC-B): этот модуль получает на вход уже готовое
    per-probe наблюдение irradiance, не сырые буферы visibility/normal.
"""

import torch
import torch.nn as nn


class CfCProbeCell(nn.Module):
    """
    Один шаг closed-form update (Hasani et al. 2022), staleness вплетена в backbone z:
        z          = backbone([u_t, h_prev])
        g          = tanh(W_g z)                      -- "новая" кандидат-оценка
        h_cand     = tanh(W_h z)                       -- steady-state кандидат
        sigma_tau  = sigmoid(W_a z * dt_t + W_b z)     -- time-gate (dt = elapsed time)
        h'         = h_cand * sigma_tau + g * (1 - sigma_tau)

    u_t включает staleness-вектор (см. CfCProbeModule.build_input) — поэтому z, а через
    него и sigma_tau/g/h_cand, зависят от staleness не только от dt.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, hidden_dim)  # множитель на dt
        self.W_b = nn.Linear(hidden_dim, hidden_dim)  # смещение
        self.last_sigma_tau: torch.Tensor | None = None  # диагностика saturation гейта
        self.last_gate_diag: dict | None = None  # z, g, h_cand, t_a, t_b, pre_sigmoid, sigma_tau — см. full_gate_diagnostics()

    def forward(
        self,
        u_t: torch.Tensor,
        h_prev: torch.Tensor,
        dt_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        u_t:    [B, input_dim]  — obs (+ staleness-вектор), см. build_input
        h_prev: [B, hidden_dim]
        dt_t:   [B, 1]          — Δt с прошлого обновления этой пробы
        """
        z = self.backbone(torch.cat([u_t, h_prev], dim=-1))
        g = torch.tanh(self.W_g(z))
        h_cand = torch.tanh(self.W_h(z))
        t_a = self.W_a(z)
        t_b = self.W_b(z)
        # log1p(dt) вместо сырого dt (2026-08-01, см. full_gate_diagnostics-диагностику масштаба
        # t_a): душит тяжёлый хвост Δt-выбросов, сохраняя dt=0 -> log1p(dt)=0 (т.е. по-прежнему
        # sigma_tau при dt->0 зависит только от t_b — корректное continuous-time поведение).
        # Масштаб самого t_a калибруется ОТДЕЛЬНО и один раз, см. calibrate_time_gate_init() —
        # НЕ хардкодится здесь под конкретный датасет.
        dt_log = torch.log1p(dt_t)
        pre_sigmoid = t_a * dt_log + t_b
        sigma_tau = torch.sigmoid(pre_sigmoid)
        self.last_sigma_tau = sigma_tau.detach()
        # без detach — full_gate_diagnostics делает backward через ВСЕ эти величины, чтобы
        # проверить, доходит ли до backbone/W_g/W_h/W_a/W_b полезный градиент
        self.last_gate_diag = dict(z=z, g=g, h_cand=h_cand, t_a=t_a, t_b=t_b,
                                    pre_sigmoid=pre_sigmoid, sigma_tau=sigma_tau)
        return h_cand * sigma_tau + g * (1.0 - sigma_tau)


class CfCProbeModule(nn.Module):
    """
    Оборачивает CfCProbeCell по последовательности обновлений ОДНОЙ или БАТЧА проб.

    Батч-размерность — пробы (или последовательности проб), не пиксели/кадры видео:
    веса общие для всех проб (та же логика, что у per-scene MLP в NRC-style baseline, §4),
    сама модель ничего не знает о конкретной геометрии сцены — вся geometry-специфика
    (какая проба где) закодирована в наблюдениях (obs), не в весах.

    Args:
        obs_dim:      размерность самого наблюдения irradiance (обычно 1 — скаляр
                      яркости на канал; при RGB можно вызывать по одному на канал
                      или расширить obs_dim=3, зависит от финального решения по каналам)
        hidden_dim:   размер скрытого состояния CfC
        use_staleness: включать ли staleness-вектор во вход (§2.2 — основная
                      конфигурация CfC-B; False даёт "голый" CfC, нужен как
                      ablation-точка сравнения, см. spike-test mode='A')
        staleness_dim: размер staleness-вектора при use_staleness=True. По
                      умолчанию 2 — [cold_start, confidence] (§2.2 основного ТЗ,
                      probe-путь, поведение без изменений). Per-pixel dense путь
                      (§2.6 TZ_stage1b_per_pixel_dense_fallback.md) передаёт
                      staleness_dim=4 — [cold_start, confidence, disocclusion_flag,
                      geometric_mismatch_score] — см. build_input(extra_staleness=...)
                      ниже. Архитектура ячейки не меняется вообще (§2.6: "CfC-B в
                      исходной конфигурации переносится без переделки, меняется
                      только состав входного staleness-вектора") — расширяется
                      только input_dim backbone'а.
    """

    def __init__(
        self,
        obs_dim: int = 1,
        hidden_dim: int = 32,
        use_staleness: bool = True,
        staleness_dim: int = 2,
        use_hard_jump_reset: bool = False,
        use_jump_output_skip: bool = False,
        spatial_dim: int = 0,
        use_mixed_memory: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        self.use_hard_jump_reset = use_hard_jump_reset
        self.use_jump_output_skip = use_jump_output_skip
        # mixed-memory RNN буфер (2026-08-22, TZ_stage3_cfc_mmrnn.md §2/§3) — Lechner & Hasani,
        # "Mixed-Memory RNNs for Learning Long-term Dependencies in Irregularly Sampled Time
        # Series" (OpenReview rOGm97YR22N, NeurIPS 2022). Доказанный в статье факт: continuous-
        # time RNN страдает vanishing/exploding gradient на длинных зависимостях НЕЗАВИСИМО от
        # выбора ODE-солвера (в т.ч. closed-form CfC); решение — отдельный, дискретный memory
        # compartment (LSTM-подобный, c_state), разделённый с continuous-time состоянием h.
        # Реализация — тот же рецепт, что у официальной библиотеки ncps
        # (ncps.torch.cfc.CfC(mixed_memory=True), см. .venv/.../ncps/torch/cfc.py): на каждом
        # шаге LSTM ПЕРЕД CfC-ячейкой, его h-выход идёт КАК h_prev В CfC-ячейку (не отдельный
        # readout-путь). c_state никогда не проходит через continuous-time часть.
        self.use_mixed_memory = use_mixed_memory
        # spatial_dim (2026-08-20, models/spatial_features.py): позиционное кондиционирование
        # (позиция+направление+normal+albedo, positional encoding) — закрывает найденный разрыв
        # с настоящим NRC (Müller et al. 2021, чат/mempalace 2026-08-19/20). Применяется
        # ОДИНАКОВО к CfC-B и honest-baseline'ам (nrc_honest/gru_honest/ode_lstm_honest) — иначе
        # архитектурное сравнение теряет интерпретируемость (см. чат 2026-08-20). 0 по умолчанию
        # = старое поведение без изменений (обратная совместимость).
        self.spatial_dim = spatial_dim

        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        self.cell = CfCProbeCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

        if use_mixed_memory:
            # §4 TZ_stage3: LSTMCell(input_dim, hidden_dim) добавляет 4*hidden_dim*
            # (input_dim+hidden_dim+1) параметров — существенно относительно cell'а выше.
            # Параметр-каунт ОБЯЗАТЕЛЬНО пересчитывать перед выводами (см. run скрипт).
            self.memory_lstm = nn.LSTMCell(input_dim, hidden_dim)

        # NJ-ODE-style жёсткий jump-канал (2026-08-16, ablation по мотивам Krach et al. 2022 /
        # Herrera et al. 2021 — см. mempalace): dH_t = f(...)dt + (rho(X_t) - H_{t-})*du_t, вес
        # jump-слагаемого В МОМЕНТ НАБЛЮДЕНИЯ равен 1 (полная замена H_t := rho(X_t)), НЕ мягкий
        # blend, как sigma_tau в CfCProbeCell. rho здесь — ОТДЕЛЬНАЯ маленькая MLP от u_t (НЕ от
        # h_prev вообще, как в базовой не-path-dependent версии NJ-ODE, Eq. 7-8) — сознательно
        # простейший вариант гипотезы "у CfC-B нет отдельного hard-reset примитива для
        # disocclusion", а не любое усложнение self.cell. Bounded output (Tanh) — как
        # Definition 3.12 (bounded output neural networks) у Krach et al. Используется ТОЛЬКО
        # если use_hard_jump_reset=True И вызывающий код передаёт disocc_seq в forward()
        # (иначе полностью обратно совместимо со старым поведением).
        if use_hard_jump_reset:
            self.jump_map = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

        # Output-level residual skip к obs_t (2026-08-19, ablation по найденному разрыву с
        # NRDStyleBaseline.use_disocclusion_reset): jump_map выше заменяет СКРЫТОЕ состояние
        # (bounded Tanh, hidden_dim) — чтобы попасть в out_t ~ obs_t, этому состоянию нужно
        # пройти ЕЩЁ через общий self.readout, т.е. jump_map ОБЯЗАН выучить с нуля
        # отображение, аппроксимирующее identity через две Tanh-прослойки, на ~6% кадров
        # (disocclusion=1). У NRD's disocclusion-reset такого разрыва нет вообще: out_t = obs_t, ноль параметров.
        # Здесь — НЕ трогаем h/jump_map/рекуррентную динамику (что переносится в следующий шаг
        # как h_prev, остаётся как было, полностью совместимо со старым cfc_hardjump-поведением
        # по состоянию) — правим ТОЛЬКО что репортируется как предсказание НА САМОМ шаге
        # disocclusion=1: out_t := obs_t + jump_output_residual(h_jump), где
        # jump_output_residual — ДОПОЛНИТЕЛЬНАЯ Linear(hidden_dim, obs_dim), инициализированная
        # В НОЛЬ (вес и bias). При инициализации out_t == obs_t ТОЧНО (NRD-эквивалент), градиент
        # учит МАЛУЮ ПОПРАВКУ поверх уже готового NRD-подобного ответа, а не identity с нуля —
        # принципиально более лёгкая оптимизационная задача (zero-init residual, не
        # random-init через 2 нелинейности). Отдельный флаг (НЕ переопределяет
        # use_hard_jump_reset) — старое поведение cfc_hardjump остаётся доступным без изменений
        # для сравнения (см. MODEL_KINDS: cfc_hardjump_skip — новый пункт, cfc_hardjump — как
        # был). Требует use_hard_jump_reset=True (иначе нет jump_map/h_jump, к которым
        # привязываться).
        if use_jump_output_skip:
            if not use_hard_jump_reset:
                raise ValueError("use_jump_output_skip=True требует use_hard_jump_reset=True")
            self.jump_output_residual = nn.Linear(hidden_dim, obs_dim)
            nn.init.zeros_(self.jump_output_residual.weight)
            nn.init.zeros_(self.jump_output_residual.bias)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = True,
        extra_staleness: list[torch.Tensor] | None = None,
        spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Собирает u_t из сырых буферов пробы (§3.1) в формат входа модели.

        obs:        [B, T, obs_dim]  — уже интерполированное (§2.3) наблюдение irradiance
        cold_start: [B, T]           — флаг "проба ни разу не обновлялась" (§2.2)
        confidence: [B, T]           — log1p(spp) сигнал, нормализованный по известному
                                        глобальному диапазону spp (см. spike-test
                                        models.py::build_features — фиксированная,
                                        не по-батчевая нормализация, важно для
                                        воспроизводимости между запусками)
        extra_staleness: список дополнительных [B, T] каналов, добавляемых ПОСЛЕ
                                        cold_start/confidence — probe-путь его не
                                        передаёт (None, поведение идентично прежнему
                                        2-элементному вектору). Per-pixel dense путь
                                        (§2.6) передаёт [disocclusion_flag,
                                        geometric_mismatch_score] (§2.5), полученные
                                        из blender/disocclusion.reproject_and_check
                                        на паре соседних кадров — НЕ вычисляется этим
                                        модулем, подаётся уже готовым. Порядок каналов
                                        должен совпадать с staleness_dim, который модель
                                        получила в __init__.
        spatial: [B, T, spatial_dim] позиционное кондиционирование (models/spatial_
                                        features.py::build_spatial_conditioning,
                                        2026-08-20) — None по умолчанию = старое
                                        поведение без пространственного входа. Идёт
                                        ПОСЛЕ staleness-вектора; должен совпадать со
                                        spatial_dim, переданным модели в __init__.
        """
        parts = [obs]
        if use_staleness:
            if cold_start is None or confidence is None:
                raise ValueError(
                    "use_staleness=True требует cold_start и confidence "
                    "(получить из spp-метаданных пробы, см. ТЗ §3.1/§3.2)"
                )
            components = [cold_start, confidence]
            if extra_staleness:
                components.extend(extra_staleness)
            parts.append(torch.stack(components, dim=-1))  # [B, T, staleness_dim]
        if spatial is not None:
            parts.append(spatial)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        record_gates: bool = False,
        disocc_seq: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_seq:  [B, T, input_dim] — вход на каждый шаг (см. build_input)
            dt_seq: [B, T] или [B, T, 1] — Δt между последовательными обновлениями пробы
            h0:     [B, hidden_dim] или None (тогда нулевое состояние — cold-start)
            record_gates: сохранить ли sigma_tau по шагам (диагностика насыщения гейта)
            disocc_seq: [B, T] 0/1, опционально — используется, ТОЛЬКО если
                        use_hard_jump_reset=True (иначе игнорируется). На шагах, где
                        disocc_seq==1, h полностью заменяется на self.jump_map(u_t)
                        (NJ-ODE-style hard reset, вес 1, БЕЗ участия h_prev/обычного
                        CfC-обновления на этом шаге) — см. докстринг __init__.
            c0:     [B, hidden_dim] или None — начальное состояние LSTM-буфера
                        mixed-memory (см. TZ_stage3 §3.2, Вариант A). Игнорируется,
                        если use_mixed_memory=False. None -> нулевое (cold-start).

        Returns:
            (pred_seq, h_final):
                pred_seq: [B, T, obs_dim] — предсказанная irradiance на каждом шаге
                          (причинно: шаг t не видит входов t+1..T-1)
                h_final:  [B, hidden_dim] — финальное скрытое состояние (для continuation
                          между вызовами — проба живёт дольше одного обучающего окна)

        Если use_mixed_memory=True, финальное состояние LSTM-буфера c_state сохраняется
        в self.last_c_final ([B, hidden_dim]) — НЕ добавлено в кортеж возврата, чтобы не
        ломать существующие вызовы (run_experiment_dense.py и т.д. распаковывают ровно
        (pred, h_final)); вызывающий код, которому нужна continuation по c_state между
        вызовами, читает атрибут явно и передаёт его следующим вызовом как c0.
        """
        if dt_seq.dim() == 2:
            dt_seq = dt_seq.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        B, T, _ = u_seq.shape
        h = (
            h0
            if h0 is not None
            else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        )
        c = None
        if self.use_mixed_memory:
            c = (
                c0
                if c0 is not None
                else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
            )

        use_jump = self.use_hard_jump_reset and disocc_seq is not None
        if use_jump and disocc_seq.dim() == 1:
            disocc_seq = disocc_seq.unsqueeze(0)  # на случай [T] без батч-оси

        outputs = []
        gate_log = [] if record_gates else None
        diag_logs = {k: [] for k in ("z", "g", "h_cand", "t_a", "t_b", "pre_sigmoid", "sigma_tau")} if record_gates else None
        for t in range(T):
            if self.use_mixed_memory:
                # mmRNN-рецепт (ncps, mixed_memory=True): LSTM ПЕРЕД CfC-шагом каждый раз,
                # его h-выход идёт КАК h_prev в CfC-ячейку. c обновляется здесь же, но НИКОГДА
                # не проходит через continuous-time часть (self.cell) — см. §2 TZ_stage3.
                h_for_cell, c = self.memory_lstm(u_seq[:, t], (h, c))
            else:
                h_for_cell = h
            h_cfc = self.cell(u_seq[:, t], h_for_cell, dt_seq[:, t])
            if use_jump:
                # NJ-ODE-style: H_t := rho(u_t) ПОЛНОСТЬЮ (вес 1, без h_prev/h_cfc на этом шаге)
                # на шагах disocc==1; иначе обычное CfC-обновление без изменений. Эта часть НЕ
                # меняется use_jump_output_skip'ом — рекуррентная динамика (что несётся в
                # следующий шаг как h_prev) идентична исходному cfc_hardjump.
                h_jump = self.jump_map(u_seq[:, t])
                mask = disocc_seq[:, t].to(h_cfc.dtype).unsqueeze(-1)  # [B,1]
                h = mask * h_jump + (1.0 - mask) * h_cfc
                if self.use_mixed_memory:
                    # TZ_stage3 §3.2, Вариант A: на disocc==1 сбрасывается И c_state тоже —
                    # накопленная LSTM-память с прошлой геометрии не должна пережить разрыв
                    # причинности. Переиспользуем h_jump (тот же bounded Tanh-выход) как reset-
                    # цель для c — простейшая инстанциация Варианта A (не вводит вторую сеть
                    # rho_c, см. обсуждение вариантов в TZ), не отдельная сеть.
                    c = mask * h_jump + (1.0 - mask) * c
                if self.use_jump_output_skip:
                    # См. докстринг __init__: меняется ТОЛЬКО репортируемое предсказание на
                    # шаге disocc==1, не h. obs_component — сырые каналы наблюдения (всегда
                    # первые obs_dim каналов u_t, см. build_input: cat([obs, staleness])).
                    obs_component = u_seq[:, t, : self.obs_dim]
                    out_jump = obs_component + self.jump_output_residual(h_jump)
                    out = mask * out_jump + (1.0 - mask) * self.readout(h_cfc)
                else:
                    out = self.readout(h)
            else:
                h = h_cfc
                out = self.readout(h)
            if record_gates:
                gate_log.append(self.cell.last_sigma_tau)
                for k in diag_logs:
                    diag_logs[k].append(self.cell.last_gate_diag[k])
            outputs.append(out)

        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)  # [B, T, hidden_dim]
            # каждый ключ — [B, T, hidden_dim]; держит граф, если вызывающий код не под no_grad()
            # (нужно full_gate_diagnostics для backward через t_a/t_b)
            self.last_diag_log = {k: torch.stack(v, dim=1) for k, v in diag_logs.items()}

        if self.use_mixed_memory:
            self.last_c_final = c

        return torch.stack(outputs, dim=1), h


# ---------------------------------------------------------------------------
# Data-driven калибровка масштаба time-gate (2026-08-01) — НЕ хардкодит числа под конкретный
# датасет (см. риск в mempalace: dt_range/dt_spike_* в synthetic_probe_scene.py — временная
# синтетика, копия калибровки из spike-теста, не связана с реальными Blender-данными). Вместо
# этого один раз ПЕРЕД обучением измеряет фактический масштаб t_a*log1p(dt) НА ТЕХ ДАННЫХ, что
# переданы (синтетика сейчас, Blender позже — не важно), и домножает W_a на коэффициент так,
# чтобы typical pre-sigmoid попадал в чувствительный диапазон сигмоиды. При смене датасета
# (другой масштаб dt) калибровка автоматически пересчитывается заново — ничего не захардкожено.
# ---------------------------------------------------------------------------
def calibrate_time_gate_init(model: "CfCProbeModule", u_seq: torch.Tensor, dt_seq: torch.Tensor,
                              target_pre_sigmoid_std: float = 2.0, verbose: bool = True) -> float:
    """
    Измеряет std(t_a * log1p(dt)) на переданном батче (при текущей, ещё не откалиброванной
    инициализации W_a) и домножает W_a.weight/bias на scale_factor = target_std / current_std,
    так что std этой величины после калибровки ~= target_pre_sigmoid_std.

    Почему масштабируем ИМЕННО W_a (не W_b, не backbone): W_b — это offset (значение гейта при
    dt->0), его масштаб не связан с чувствительностью к dt и трогать не нужно. t_a должен
    определять НАСКОЛЬКО СИЛЬНО сигмоида реагирует на log1p(dt) — это единственный параметр,
    отвечающий за эту чувствительность (см. полный разбор роли t_a в чате/mempalace 2026-08-01).

    target_pre_sigmoid_std=2.0 — эвристика: sigmoid(±2) ~= {0.12, 0.88}, т.е. typical dt должен
    быть способен сдвинуть гейт к границам разумного рабочего диапазона, не только к крайним
    выбросам dt. Не привязано к конкретным числам синтетики — это просто "насколько чувствительна
    сигмоида должна быть по построению", безразмерная величина.

    Возвращает применённый scale_factor (для логирования/воспроизводимости).
    """
    with torch.no_grad():
        model(u_seq, dt_seq, record_gates=True)
        diag = model.last_diag_log
        t_a = diag["t_a"]
        dt_seq_3d = dt_seq.unsqueeze(-1) if dt_seq.dim() == 2 else dt_seq
        dt_log = torch.log1p(dt_seq_3d)
        current_component = t_a * dt_log  # тот же член, что входит в pre_sigmoid = t_a*log1p(dt)+t_b
        current_std = float(current_component.std())
        scale_factor = target_pre_sigmoid_std / max(current_std, 1e-6)

        model.cell.W_a.weight.mul_(scale_factor)
        model.cell.W_a.bias.mul_(scale_factor)

    if verbose:
        print(f"    [calibrate_time_gate_init] current_std={current_std:.4f} "
              f"target_std={target_pre_sigmoid_std:.4f} scale_factor={scale_factor:.3f}")

    return scale_factor


# ---------------------------------------------------------------------------
# Комплексная gate-диагностика (2026-08-01) — ВСЕ метрики за один проход, вместо проверки по
# одной. Портирована из spike_test/models/models.py::full_gate_diagnostics на формат этого
# production-модуля (u_seq/dt_seq вместо features/tau). Отвечает на вопрос: если σ_τ зажат в
# узком диапазоне, это (а) W_a/W_b не получают полезный градиент, или (б) z (выход backbone) сам
# не варьируется между staleness/dt-паттернами настолько, чтобы "пробить" W_a/W_b.
# ---------------------------------------------------------------------------
def full_gate_diagnostics(model: "CfCProbeModule", u_seq: torch.Tensor, dt_seq: torch.Tensor,
                           true: torch.Tensor = None, seg_type: "np.ndarray" = None,
                           segment_names: dict | None = None,
                           disocclusion_flag: "np.ndarray | torch.Tensor" = None,
                           label: str = "sigma_tau",
                           verbose: bool = True) -> dict:
    """
    Один проход даёт срез сразу по ВСЕМ параметрам CfC-ячейки (2026-08-01, расширено по запросу
    пользователя — раньше не хватало g/h_cand, градиентов backbone/W_g/W_h, распределения dt и
    прямой (не через offset-since-jump) связи гейта со своим фактическим входом):
      - saturation sigma_tau (как в predict_cfc_with_gates)
      - pre_sigmoid = t_a*log1p(dt) + t_b — ДО сжатия сигмоидой (2026-08-01: log1p(dt), не сырой
        dt, см. calibrate_time_gate_init() и CfCProbeCell.forward)
      - t_a = W_a(z) и t_b = W_b(z) — отдельно друг от друга и от dt
      - g = tanh(W_g z) и h_cand = tanh(W_h z) — ДВА кандидата состояния (не только гейт;
        если они сами насыщены в ±1, узкий диапазон h' объясняется НЕ гейтом, а кандидатами)
      - z (выход backbone) — общая статистика + разбивка по seg_type (static/step/drift), если
        передан. Если z почти не отличается между сегментами — backbone не различает паттерны,
        через которые ДОЛЖЕН идти staleness-сигнал.
      - если передан `true` — градиентная норма ВСЕХ обучаемых слоёв ячейки (backbone[0],
        W_g, W_h, W_a, W_b) после ОДНОГО backward (не эпохи обучения) — прямая проверка,
        какой из пяти слоёв (если хоть один) не получает полезный градиент.
      - распределение самого dt (перцентили) рядом с типичным масштабом |t_a|*dt — отвечает на
        вопрос "гейт физически МОЖЕТ уйти далеко от 0.5 на типичном dt, или t_a слишком мал
        относительно реального диапазона dt, и это ограничение по конструкции, а не по обучению".
      - sigma_tau/pre_sigmoid, забинченные НАПРЯМУЮ по величине dt (не по offset-since-jump,
        который является другой переменной и может не коррелировать с dt пробы) — прямая
        проверка "реагирует ли гейт хотя бы на собственный прямой вход".
      - если передан `disocclusion_flag` (2026-08-11, TZ_stage1b_gate_theory_sweep.md §1):
        разбивка sat_low/sat_high/|t_a|/|t_b|/sigma_tau по disocclusion=0/1 отдельно
        (`by_disocclusion`), И, если ОДНОВРЕМЕННО передан seg_type, совместная разбивка
        по (seg_type × disocclusion_flag) (`by_seg_type_x_disocclusion`) — раньше это
        требовало одноразового патча в run_experiment_dense.py поверх model.last_diag_log
        (см. mempalace, сессия 2026-08-09), теперь встроено сюда как основной путь, чтобы
        разные прогоны (sweep по gate_lr/lr_Wa/lr_Wb) давали сравнимые таблицы без
        ad hoc-кода на каждый прогон.

    u_seq: [B, T, input_dim], dt_seq: [B, T] или [B, T, 1], true: [B, T, obs_dim] опционально.
    seg_type: [B, T] int-коды сегментов (см. evaluation.metrics.SEGMENT_NAMES), опционально.
    disocclusion_flag: [B, T] 0/1 (bool-совместимый), опционально. Форма ДОЛЖНА совпадать
        с первыми двумя измерениями z/sigma_tau/t_a/t_b (та же конвенция, что у seg_type).
    """
    import numpy as np  # локальный импорт — модуль не тянет numpy на верхнем уровне

    model.zero_grad(set_to_none=True)
    grad_info = {}
    if true is not None:
        pred, _ = model(u_seq, dt_seq, record_gates=True)
        diag = model.last_diag_log
        loss = ((pred - true) ** 2).mean()
        loss.backward()
        layers = {"backbone": model.cell.backbone[0], "W_g": model.cell.W_g,
                  "W_h": model.cell.W_h, "W_a": model.cell.W_a, "W_b": model.cell.W_b}
        for name, layer in layers.items():
            w_grad, b_grad = layer.weight.grad, layer.bias.grad
            grad_info[name] = dict(
                weight_grad_norm=float(w_grad.norm().item()) if w_grad is not None else None,
                bias_grad_norm=float(b_grad.norm().item()) if b_grad is not None else None,
            )
        model.zero_grad(set_to_none=True)
        diag = {k: v.detach() for k, v in diag.items()}
    else:
        with torch.no_grad():
            model(u_seq, dt_seq, record_gates=True)
            diag = model.last_diag_log

    def stats(t):
        t = t.detach()
        return dict(mean=float(t.mean()), std=float(t.std()), min=float(t.min()), max=float(t.max()))

    z, g, h_cand = diag["z"], diag["g"], diag["h_cand"]
    t_a, t_b, pre_sig, sig_tau = diag["t_a"], diag["t_b"], diag["pre_sigmoid"], diag["sigma_tau"]

    def sat_frac(t, lo, hi):
        return float(((t > lo) & (t < hi)).float().mean())

    result = {
        "sigma_tau": {**stats(sig_tau),
                      "sat_low(<0.02)": float((sig_tau < 0.02).float().mean()),
                      "sat_high(>0.98)": float((sig_tau > 0.98).float().mean())},
        "pre_sigmoid(t_a*log1p(dt)+t_b)": stats(pre_sig),
        "t_a=W_a(z)": stats(t_a),
        "t_b=W_b(z)": stats(t_b),
        "g=tanh(W_g*z)": {**stats(g), "sat(|g|>0.98)": 1.0 - sat_frac(g.abs(), -1.0, 0.98)},
        "h_cand=tanh(W_h*z)": {**stats(h_cand), "sat(|h_cand|>0.98)": 1.0 - sat_frac(h_cand.abs(), -1.0, 0.98)},
        "z(backbone_out)": stats(z),
        "z_per_dim_std_mean": float(z.std(dim=(0, 1)).mean()),
        "grad": grad_info,
    }

    # Единая функция сбора статистики по произвольной boolean-маске [B, T] — используется
    # и для seg_type, и для disocclusion_flag, и для их пересечения (2026-08-11, TZ
    # stage1b_gate_theory_sweep.md §1), чтобы разные разрезы были ПОЛНОСТЬЮ сравнимы между
    # собой (одни и те же поля), а не разного состава, как было раньше (by_seg_type содержал
    # только z_norm/t_a_mean/t_b_mean, без sat_low/sat_high/sigma_tau — их приходилось
    # доставать отдельным одноразовым кодом, см. старый "[disocclusion gate reaction check]"
    # патч в run_experiment_dense.py).
    def _masked_gate_stats(mask: torch.Tensor) -> dict | None:
        if not mask.any():
            return None
        st = sig_tau[mask]
        return dict(
            n=int(mask.sum()),
            sigma_tau_mean=float(st.mean()), sigma_tau_std=float(st.std()),
            sat_low_frac=float((st < 0.02).float().mean()),
            sat_high_frac=float((st > 0.98).float().mean()),
            t_a_mean=float(t_a[mask].mean()), t_a_abs_mean=float(t_a[mask].abs().mean()),
            t_b_mean=float(t_b[mask].mean()), t_b_abs_mean=float(t_b[mask].abs().mean()),
            z_norm_mean=float(z[mask].norm(dim=-1).mean()),
        )

    # Обе маски принудительно приводим к устройству z (не полагаемся на то, что вызывающий
    # код передал seg_type/disocclusion_flag уже на нужном device — оба тензора могли прийти
    # с разных источников, как выяснилось на MPS: seg_type собирается через np.stack на CPU,
    # а disocclusion_flag вызывающий код обычно уже переносит на device модели).
    seg_t = torch.as_tensor(seg_type, device=z.device) if seg_type is not None else None
    seg_names = segment_names or {0: "static", 1: "step", 2: "drift"}
    if seg_t is not None:
        by_seg = {}
        for code, name in seg_names.items():
            stats_ = _masked_gate_stats(seg_t == code)
            if stats_ is not None:
                by_seg[name] = stats_
        result["by_seg_type"] = by_seg

    disocc_t = None
    if disocclusion_flag is not None:
        disocc_t = torch.as_tensor(disocclusion_flag, device=z.device).bool()
        # приводим к [B, T], если пришло с лишней последней размерностью (как dt_seq выше)
        if disocc_t.dim() > 2:
            disocc_t = disocc_t.reshape(disocc_t.shape[0], disocc_t.shape[1])
        by_disocc = {}
        for name, mask in (("disocclusion=1", disocc_t), ("disocclusion=0", ~disocc_t)):
            stats_ = _masked_gate_stats(mask)
            if stats_ is not None:
                by_disocc[name] = stats_
        result["by_disocclusion"] = by_disocc

        if seg_t is not None:
            by_joint = {}
            for code, seg_name in seg_names.items():
                for disocc_val, disocc_name in ((True, "disocc=1"), (False, "disocc=0")):
                    mask = (seg_t == code) & (disocc_t == disocc_val)
                    stats_ = _masked_gate_stats(mask)
                    if stats_ is not None:
                        by_joint[f"{seg_name}/{disocc_name}"] = stats_
            result["by_seg_type_x_disocclusion"] = by_joint

    # --- dt-распределение vs масштаб t_a (2026-08-01, обновлено под log1p(dt)-фикс) -----------
    # dt_seq на входе может быть [B,T] или [B,T,1]; приводим к [B,T] для перцентилей.
    dt_flat = dt_seq.detach().reshape(dt_seq.shape[0], dt_seq.shape[1]).reshape(-1).float()
    dt_log_flat = torch.log1p(dt_flat)  # реальный вход гейта после фикса (не сырой dt)
    q_levels = torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99], device=dt_flat.device)
    dt_q = torch.quantile(dt_flat, q_levels)
    dt_log_q = torch.quantile(dt_log_flat, q_levels)
    t_a_abs_mean = float(t_a.abs().mean())
    result["dt_distribution"] = {
        "p01": float(dt_q[0]), "p10": float(dt_q[1]), "p25": float(dt_q[2]),
        "p50": float(dt_q[3]), "p75": float(dt_q[4]), "p90": float(dt_q[5]), "p99": float(dt_q[6]),
        "min": float(dt_flat.min()), "max": float(dt_flat.max()),
    }
    result["t_a_dt_scale"] = {
        "t_a_abs_mean": t_a_abs_mean,
        # |t_a|*log1p(dt) — РЕАЛЬНЫЙ член pre_sigmoid (после log1p-фикса), не |t_a|*сырой dt
        "|t_a|*log1p(dt)_p50": t_a_abs_mean * float(dt_log_q[3]),
        "|t_a|*log1p(dt)_p90": t_a_abs_mean * float(dt_log_q[5]),
        "|t_a|*log1p(dt)_p99": t_a_abs_mean * float(dt_log_q[6]),
    }

    # --- sigma_tau/pre_sigmoid забинченные напрямую по dt (не по offset-since-jump) --------
    sig_tau_bt = sig_tau.mean(dim=-1).reshape(-1)     # (B,T) -> (B*T,), усреднено по hidden
    pre_sig_bt = pre_sig.mean(dim=-1).reshape(-1)
    dt_bin_edges = torch.quantile(dt_flat, torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=dt_flat.device))
    dt_bin_edges[-1] = dt_bin_edges[-1] + 1e-6  # включить максимум в последний бин
    by_dt_bin = {}
    for i in range(5):
        lo, hi = float(dt_bin_edges[i]), float(dt_bin_edges[i + 1])
        mask = (dt_flat >= lo) & (dt_flat < hi)
        if mask.any():
            by_dt_bin[f"dt[{lo:.2f},{hi:.2f})"] = dict(
                n=int(mask.sum()),
                sigma_tau_mean=float(sig_tau_bt[mask].mean()),
                pre_sigmoid_mean=float(pre_sig_bt[mask].mean()),
            )
    result["by_dt_bin"] = by_dt_bin

    if verbose:
        print(f"    [full-gate-diag] {label}")
        st = result["sigma_tau"]
        print(f"      sigma_tau:      mean={st['mean']:.3f} std={st['std']:.3f} "
              f"sat_low={st['sat_low(<0.02)']:.1%} sat_high={st['sat_high(>0.98)']:.1%}")
        ps = result["pre_sigmoid(t_a*log1p(dt)+t_b)"]
        print(f"      pre-sigmoid:    mean={ps['mean']:.3f} std={ps['std']:.3f} range=[{ps['min']:.3f}, {ps['max']:.3f}]")
        ta, tb = result["t_a=W_a(z)"], result["t_b=W_b(z)"]
        print(f"      t_a=W_a(z):     mean={ta['mean']:.3f} std={ta['std']:.3f} range=[{ta['min']:.3f}, {ta['max']:.3f}]")
        print(f"      t_b=W_b(z):     mean={tb['mean']:.3f} std={tb['std']:.3f} range=[{tb['min']:.3f}, {tb['max']:.3f}]")
        gg, hc = result["g=tanh(W_g*z)"], result["h_cand=tanh(W_h*z)"]
        print(f"      g=tanh(W_g z):  mean={gg['mean']:.3f} std={gg['std']:.3f} sat(|g|>0.98)={gg['sat(|g|>0.98)']:.1%}")
        print(f"      h_cand:         mean={hc['mean']:.3f} std={hc['std']:.3f} sat(|h_cand|>0.98)={hc['sat(|h_cand|>0.98)']:.1%}")
        zz = result["z(backbone_out)"]
        print(f"      z(backbone):    mean={zz['mean']:.3f} std={zz['std']:.3f} per_dim_std_mean={result['z_per_dim_std_mean']:.4f}")
        if grad_info:
            for name in ("backbone", "W_g", "W_h", "W_a", "W_b"):
                gi = grad_info[name]
                wn, bn = gi['weight_grad_norm'], gi['bias_grad_norm']
                print(f"      grad[{name:8s}]: weight_norm={wn:.5f} bias_norm={bn:.5f}" if wn is not None
                      else f"      grad[{name:8s}]: None (нет градиента)")
        def _print_group_row(prefix_width: int, name: str, s: dict) -> None:
            print(f"      {name:<{prefix_width}s} n={s['n']:6d} "
                  f"sigma_tau={s['sigma_tau_mean']:.3f}±{s['sigma_tau_std']:.3f} "
                  f"sat_low={s['sat_low_frac']:.1%} sat_high={s['sat_high_frac']:.1%} "
                  f"|t_a|={s['t_a_abs_mean']:.4f} |t_b|={s['t_b_abs_mean']:.4f} "
                  f"z_norm={s['z_norm_mean']:.3f}")

        if "by_seg_type" in result:
            for seg, s in result["by_seg_type"].items():
                _print_group_row(9, f"seg={seg}", s)
        if "by_disocclusion" in result:
            for name, s in result["by_disocclusion"].items():
                _print_group_row(16, name, s)
        if "by_seg_type_x_disocclusion" in result:
            for name, s in result["by_seg_type_x_disocclusion"].items():
                _print_group_row(18, name, s)
        dd = result["dt_distribution"]
        print(f"      dt dist:        p01={dd['p01']:.3f} p10={dd['p10']:.3f} p50={dd['p50']:.3f} "
              f"p90={dd['p90']:.3f} p99={dd['p99']:.3f} max={dd['max']:.3f}")
        ts = result["t_a_dt_scale"]
        print(f"      |t_a|*log1p(dt): |t_a|_mean={ts['t_a_abs_mean']:.4f} "
              f"at_p50={ts['|t_a|*log1p(dt)_p50']:.4f} at_p90={ts['|t_a|*log1p(dt)_p90']:.4f} "
              f"at_p99={ts['|t_a|*log1p(dt)_p99']:.4f}")
        for bin_label, s in by_dt_bin.items():
            print(f"      {bin_label:18s} n={s['n']:5d} sigma_tau={s['sigma_tau_mean']:.4f} "
                  f"pre_sigmoid={s['pre_sigmoid_mean']:.4f}")

    return result
