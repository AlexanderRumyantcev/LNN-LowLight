"""
Baseline'ы для сравнения с CfC-B (ТЗ §4) — 4 конфигурации, по 2 версии на 2 семейства.

Faithful vs честная версия разводит два разных вопроса (см. ТЗ §4): (а) помогает ли вообще
доступ к Δt/staleness-информации, (б) даёт ли именно CfC-формулировка преимущество СВЕРХ
простого доступа к той же информации. Faithful воспроизводит то, как реально устроен
NRD/NRC в проде (без знания о Δt/staleness); честная версия — тот же механизм + равный
CfC-B доступ к информации.

Сознательно НЕ повторяется инженерная сложность продакшн-версий (variance-based clamping
в NRD, полный online-distillation цикл NRC на GPU) — не меняет алгоритмический вопрос,
который тестирует sanity-check этапа-1 (§1.3/§7.7).
"""

import torch
import torch.nn as nn


class NRDStyleBaseline(nn.Module):
    """
    Hand-crafted exponential accumulation (аналог NVIDIA NRD/ReBLUR/RELAX) — БЕЗ нейросети
    во temporal-части, поэтому нет обучаемых параметров: alpha/tau — ФИКСИРОВАННЫЕ
    гиперпараметры (как и в реальном NRD, где decay задаётся вручную/эвристикой, не обучением).

    faithful (use_honest_dt=False): out_t = out_{t-1}*(1-alpha) + obs_t*alpha, alpha ФИКСИРОВАН,
        Δt игнорируется вообще (как в реальном NRD, рассчитанном на стабильный fps).
    честная версия (use_honest_dt=True): alpha_t = 1 - exp(-dt_t/tau) — continuous-time
        аналог EMA, даёт baseline'у ту же информацию о времени, что видит CfC.

    Dense-путь (§4.2 TZ_stage1b_per_pixel_dense_fallback.md) добавляет ДВА новых
    независимых режима, оба ОПЦИОНАЛЬНЫ и по умолчанию выключены (обратная
    совместимость с probe-путём, где reprojection не имел смысла — §1.2, статичные
    пробы):

    use_disocclusion_reset (faithful dense): жёсткий сброс out_t = obs_t на
        disocclusion_flag_t=1, БЕЗ какого-либо обучаемого параметра — именно так
        устроен реальный NRD (invalidate history on disocclusion + reseed), это
        механизм, не признак. §4.2 ТЗ: "faithful-версия использует warp/reprojection
        как встроенный механизм... но не получает geometric_mismatch_score как
        отдельный обучаемый признак сверху" — здесь ровно это: булев флаг как
        триггер, не continuous score как фича.

    use_mismatch_honest (честная dense): МОЯ инженерная реализация формулы для
        honest-версии (ТЗ задаёт ТРЕБОВАНИЕ симметрии staleness-доступа, но не
        точную формулу для baseline'а без обучаемых параметров) —
        alpha_t = max(alpha_t_from_dt_or_fixed, geometric_mismatch_score_t):
        continuous-score МЯГКО подмешивается в тот же alpha, не через жёсткий
        reset (в отличие от faithful) — тот же исходный сигнал geometric_mismatch,
        что видит CfC-B, но использованный простой формулой без gate-архитектуры
        (тот же принцип разделения вопросов (а)/(б), что уже применён к
        use_honest_dt).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        tau: float = 3.0,
        use_honest_dt: bool = False,
        use_disocclusion_reset: bool = False,
        use_mismatch_honest: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.use_honest_dt = use_honest_dt
        self.use_disocclusion_reset = use_disocclusion_reset
        self.use_mismatch_honest = use_mismatch_honest

    def forward(
        self,
        obs_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        disocclusion_flag_seq: torch.Tensor | None = None,
        geometric_mismatch_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        obs_seq: [B, T, obs_dim]
        dt_seq:  [B, T] или [B, T, 1] — используется только если use_honest_dt=True
        disocclusion_flag_seq:  [B, T] или [B, T, 1] — нужен при use_disocclusion_reset=True
        geometric_mismatch_seq: [B, T] или [B, T, 1] — нужен при use_mismatch_honest=True
        Returns: pred_seq [B, T, obs_dim]
        """
        if dt_seq.dim() == 2:
            dt_seq = dt_seq.unsqueeze(-1)  # [B, T, 1]
        if self.use_disocclusion_reset:
            if disocclusion_flag_seq is None:
                raise ValueError("use_disocclusion_reset=True требует disocclusion_flag_seq")
            if disocclusion_flag_seq.dim() == 2:
                disocclusion_flag_seq = disocclusion_flag_seq.unsqueeze(-1)
        if self.use_mismatch_honest:
            if geometric_mismatch_seq is None:
                raise ValueError("use_mismatch_honest=True требует geometric_mismatch_seq")
            if geometric_mismatch_seq.dim() == 2:
                geometric_mismatch_seq = geometric_mismatch_seq.unsqueeze(-1)

        B, T, _ = obs_seq.shape
        out = obs_seq[:, 0]  # cold-start: первое наблюдение как есть (нет истории)
        outputs = [out]
        for t in range(1, T):
            if self.use_honest_dt:
                a = 1.0 - torch.exp(-dt_seq[:, t] / self.tau)  # [B, 1]
            else:
                a = self.alpha
            if self.use_mismatch_honest:
                a = torch.maximum(
                    a if torch.is_tensor(a) else torch.full_like(geometric_mismatch_seq[:, t], a),
                    geometric_mismatch_seq[:, t],
                )
            out = out * (1.0 - a) + obs_seq[:, t] * a
            if self.use_disocclusion_reset:
                reset_mask = disocclusion_flag_seq[:, t] > 0.5
                out = torch.where(reset_mask, obs_seq[:, t], out)
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class NRCStyleBaseline(nn.Module):
    """
    Online per-scene per-frame MLP (аналог NVIDIA NRC / AMD FSR Radiance Cache) — БЕЗ
    рекуррентности: каждый шаг обрабатывается независимо тем же MLP (как в
    оригинальном NRC, где каждый query — независимый forward-pass, а обучение online
    происходит по накопленной статистике сцены, а не через carried-over hidden state между
    кадрами).

    faithful (use_staleness=False): вход = только obs, как в оригинальном NRC.
    честная версия (use_staleness=True): вход = obs + [cold_start, confidence], как у CfC-B
        (§2.2) — тот же staleness-вектор, тот же источник (log1p(spp) нормализованный).

    Dense-путь (§4.2 TZ_stage1b): staleness_dim расширяется до 4
        ([cold_start, confidence, disocclusion_flag, geometric_mismatch_score]) —
        тот же принцип, что CfCProbeModule.staleness_dim, для симметрии сравнения.
        По умолчанию staleness_dim=2 (probe-путь, поведение без изменений).
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_staleness: bool = False,
                 staleness_dim: int = 2, spatial_dim: int = 0, n_hidden_layers: int = 2,
                 bias: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        # bias (09.09.2026, TZ_stage6_nrc_patent_capacity_check.md §3.2): True по умолчанию =
        # старое поведение (обратная совместимость со всеми существующими 14 конфигурациями
        # MODEL_KINDS). False — patent-faithful режим (US11610360, MLP без bias во всех слоях),
        # используется ТОЛЬКО новым kind'ом nrc_honest_boosted (run_nrc_boosted_zeroday.py).
        self.bias = bias
        # spatial_dim (2026-08-20, models/spatial_features.py): позиционное кондиционирование
        # (позиция+направление+normal+albedo, positional encoding) — закрывает найденный разрыв
        # с настоящим NRC (Müller et al. 2021), см. чат/mempalace 2026-08-19/20. 0 по умолчанию
        # = старое поведение без изменений (обратная совместимость).
        self.spatial_dim = spatial_dim
        # n_hidden_layers (2026-08-20, чат — литературная проверка после эксперимента с
        # деградацией nrc_honest_spatial): 2 по умолчанию = старое поведение (обратная
        # совместимость). Референс из литературы (Neural Radiance Cache Implementation on
        # Mobile GPU, SIGGRAPH Asia 2025, сравнение с оригинальным Müller et al. 2021;
        # tiny-cuda-nn DOCUMENTATION.md конфиг, воспроизводящий encoding NRC) — настоящий NRC
        # использует 5 скрытых слоёв по 64 нейрона, у нас было 2×32. Neural Visibility Cache
        # for Real-Time Light Sampling (arXiv 2506.05930) прямо пишет: NRC требует БОЛЬШЕЙ MLP
        # и БОЛЬШЕГО числа шагов обучения для приемлемых результатов, чем более простые
        # альтернативы — литературное подтверждение гипотезы недообучения после расширения
        # входа spatial-кондиционированием (48 доп. измерений).
        self.n_hidden_layers = n_hidden_layers
        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, obs_dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = False,
        extra_staleness: list[torch.Tensor] | None = None,
        spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Тот же формат входа, что CfCProbeModule.build_input (§3.1/§3.2) — для честного
        сравнения оба baseline'а и CfC-B видят идентично собранный staleness-вектор.
        extra_staleness — доп. каналы (§4.2 dense-путь: [disocclusion_flag,
        geometric_mismatch_score]), None по умолчанию = старое 2-элементное поведение.
        spatial — [B, T, spatial_dim] позиционное кондиционирование (models/spatial_
        features.py::build_spatial_conditioning), None по умолчанию = старое поведение без
        пространственного входа. Порядок каналов: obs, затем staleness (если есть), затем
        spatial (если есть) — должен совпадать с spatial_dim, переданным модели в __init__."""
        parts = [obs]
        if use_staleness:
            if cold_start is None or confidence is None:
                raise ValueError("use_staleness=True требует cold_start и confidence")
            components = [cold_start, confidence]
            if extra_staleness:
                components.extend(extra_staleness)
            parts.append(torch.stack(components, dim=-1))
        if spatial is not None:
            parts.append(spatial)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        u_seq: [B, T, input_dim] (см. build_input)
        Returns: pred_seq [B, T, obs_dim] — каждый шаг обработан НЕЗАВИСИМО, без
                 рекуррентности (архитектурное отличие от CfC-B/GRU-baseline'ов).
        """
        return self.mlp(u_seq)


class GRUProbeBaseline(nn.Module):
    """
    Дискретный GRU-Δt baseline — 4-я конфигурация (доп. к NRD-style/NRC-style), закрывает
    вопрос "нужен честный baseline: RNN с Δt как доп. фичей, а не голая дискретная версия"
    (методология Vid-ODE Table 5; подтверждено в проекте ранее для видео-варианта до пивота).

    В отличие от NRD-style (без нейросети) и NRC-style (per-frame MLP, без рекуррентности),
    здесь ЕСТЬ и обучаемая нейросеть, И рекуррентность (carried-over hidden state между
    шагами) — но ДИСКРЕТНАЯ (nn.GRUCell), без closed-form continuous-time gate CfC-B.
    Разница honest-GRU vs CfC-B при РАВНОМ доступе к Δt/staleness изолирует именно
    архитектурное преимущество closed-form gate'а, а не сам факт наличия информации о
    времени (то самое разделение вопросов (а)/(б), которое задаёт faithful/честная схема).

    faithful (use_dt_staleness=False): голый discrete GRU по наблюдениям — один шаг сети
        на одно наблюдение, БЕЗ доступа к Δt/staleness вообще. Типичная наивная дискретная
        RNN, не "знающая" физического времени между шагами (не отличает частое обновление
        от редкого) — именно для такого случая литература документирует деградацию на
        нерегулярной последовательности.
    честная версия (use_dt_staleness=True): тот же GRUCell, но log1p(dt) и staleness
        [cold_start, confidence] подаются КАК ДОПОЛНИТЕЛЬНАЯ ФИЧА на входе (конкатенация
        с obs, не через архитектурный механизм) — та же информация, что видит CfC-B через
        свой gate, но использованная безо всякой continuous-time структуры.

    Dense-путь (§4.2/§4.3 TZ_stage1b): extra_staleness_dim расширяет тройку
        [log1p(dt), cold_start, confidence] до [log1p(dt), cold_start, confidence,
        disocclusion_flag, geometric_mismatch_score] (extra_staleness_dim=2) — тот же
        принцип симметрии, что у CfC-B/NRC-style. §4.3 отдельно фиксирует: на
        регулярной сетке Δt (в отличие от probe-пути) GRU-Δt перестаёт быть архитектурно
        ущемлённым и может оказаться заметно более конкурентным baseline'ом — не баг интерпретации,
        если сближение с CfC-B появится на dense-данных.
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_dt_staleness: bool = False,
                 extra_staleness_dim: int = 0, spatial_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_dt_staleness = use_dt_staleness
        self.extra_staleness_dim = extra_staleness_dim
        # spatial_dim (2026-08-20, models/spatial_features.py) — см. NRCStyleBaseline.__init__
        # докстринг-комментарий, тот же принцип симметрии.
        self.spatial_dim = spatial_dim
        # +[log1p(dt), cold_start, confidence] (+ доп. каналы dense-пути, если есть) + spatial
        input_dim = obs_dim + ((3 + extra_staleness_dim) if use_dt_staleness else 0) + spatial_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        dt: torch.Tensor | None = None,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_dt_staleness: bool = False,
        extra_staleness: list[torch.Tensor] | None = None,
        spatial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        obs: [B, T, obs_dim]; dt/cold_start/confidence: [B, T] (нужны только при
        use_dt_staleness=True — та же тройка сигналов, что видит CfC-B, но здесь просто
        конкатенируется с obs как обычная фича, а не подаётся в архитектурный gate).
        extra_staleness — доп. каналы после cold_start/confidence (§4.2 dense-путь:
        [disocclusion_flag, geometric_mismatch_score]), None = старое поведение.
        spatial — [B, T, spatial_dim] позиционное кондиционирование (models/spatial_
        features.py), None по умолчанию = старое поведение без пространственного входа.
        """
        parts = [obs]
        if use_dt_staleness:
            if dt is None or cold_start is None or confidence is None:
                raise ValueError("use_dt_staleness=True требует dt, cold_start и confidence")
            dt_log = torch.log1p(dt)
            components = [dt_log, cold_start, confidence]
            if extra_staleness:
                components.extend(extra_staleness)
            parts.append(torch.stack(components, dim=-1))
        if spatial is not None:
            parts.append(spatial)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def forward(self, u_seq: torch.Tensor, h0: torch.Tensor | None = None):
        """
        u_seq: [B, T, input_dim] (см. build_input)
        h0:    [B, hidden_dim] или None (нулевое состояние — cold-start)
        Returns: (pred_seq, h_final) — тот же интерфейс, что CfCProbeModule.forward,
                 для единообразного вызова из train_model/predict.
        """
        B, T, _ = u_seq.shape
        h = (
            h0 if h0 is not None
            else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        )
        outputs = []
        for t in range(T):
            h = self.cell(u_seq[:, t], h)
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1), h


class ODELSTMProbeBaseline(nn.Module):
    """
    Настоящий (не closed-form) ODE-RNN baseline — 5-я конфигурация, TZ_stage2_efficiency_
    and_baseline_validation.md §2. Закрывает вопрос: теряет ли CfC-B точность именно из-за
    closed-form приближения (Hasani et al. 2022) относительно настоящего численного
    интегрирования, на данных ЭТОГО проекта (нерегулярный Δt + disocclusion-скачки).

    НЕ голый Neural ODE (`dh/dt = f_θ(h,t)`, ODE-RNN) — по §2.1 ТЗ статья самого CfC
    (Extended Data Fig. 5) и Lechner & Hasani 2020 (arXiv 2006.04418) показывают, что голый
    ODE-RNN проваливается на нерегулярной выборке из-за vanishing/exploding gradient
    НЕЗАВИСИМО от солвера — это была бы патология обучения, а не честный тест архитектуры.
    Здесь — ODE-LSTM (Lechner & Hasani 2020): архитектура, устойчивая к этой патологии,
    но всё ещё использующая настоящее численное интегрирование (не аналитическую формулу,
    как CfC). Портировано с `torch_node_cell.py:ODELSTMCell` (репозиторий mlech26l/ode-lstms,
    официальная реализация авторов) под конвенции этого проекта.

    Механизм (в отличие от CfC-B, где обе кандидатные ветви видят h_prev через общий
    backbone и гейт σ_τ только смешивает — TZ_stage1c §5):
    1. nn.LSTMCell делает ДИСКРЕТНОЕ обновление (new_h, new_c) = lstm(u_t, (h_prev, c_prev))
       при поступлении наблюдения — cell state `c` НЕ эволюционирует численно между
       наблюдениями (остаётся дискретным) — это и даёт устойчивый путь градиента (в отличие
       от голого ODE-RNN, где всё состояние эволюционирует через ОДУ).
    2. Только new_h эволюционирует ВПЕРЁД по времени на dt_t через fixed-step RK4
       (f_node: 2-слойный MLP, Tanh посередине), 3 под-шага (`dt/3`, как в референсе —
       "3 unfolds").
    3. Readout: Linear(hidden_dim, obs_dim) от эволюционировавшего h на каждом шаге.

    Δt здесь НЕ опциональная честная фича (как у NRD-style/NRC-style/GRU-Δt, где есть
    faithful-версия без неё) — continuous-evolution фаза архитектурно требует Δt всегда,
    поэтому faithful-варианта (без Δt) для этой архитектуры не существует в принципе.
    Единственная честная ось сравнения с CfC-B — доступ к staleness-вектору
    [cold_start, confidence, disocclusion_flag, geometric_mismatch_score] сверх obs, как и
    у остальных honest-baseline'ов (симметрия input_dim с CfCProbeModule.build_input).
    """

    def __init__(self, obs_dim: int = 1, hidden_dim: int = 32, use_staleness: bool = True,
                 staleness_dim: int = 2, rk4_substeps: int = 3, spatial_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        self.rk4_substeps = rk4_substeps
        # spatial_dim (2026-08-20, models/spatial_features.py) — см. NRCStyleBaseline.__init__
        # докстринг-комментарий (§2.2 ТЗ требует симметрии input_dim с CfC-B/NRC-honest/
        # GRU-honest, spatial_dim не исключение).
        self.spatial_dim = spatial_dim

        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        # f_node: 1 hidden layer NODE, точная структура ODELSTMCell.f_node из референса
        self.f_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.readout = nn.Linear(hidden_dim, obs_dim)

    # build_input — идентичен NRCStyleBaseline.build_input по сигнатуре (тот же формат
    # staleness-вектора, что видят CfC-B/NRC-honest/GRU-honest — для честного сравнения)
    build_input = staticmethod(NRCStyleBaseline.build_input)

    def _rk4_evolve(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """dt: [B, 1] — эволюционирует h вперёд на dt через fixed-step RK4, rk4_substeps
        под-шагов (как solve_fixed в референсе: 3 unfolds по dt/3 каждый)."""
        step = dt / self.rk4_substeps
        for _ in range(self.rk4_substeps):
            k1 = self.f_node(h)
            k2 = self.f_node(h + k1 * step * 0.5)
            k3 = self.f_node(h + k2 * step * 0.5)
            k4 = self.f_node(h + k3 * step)
            h = h + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return h

    def forward(
        self,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u_seq:  [B, T, input_dim] (см. build_input)
        dt_seq: [B, T] или [B, T, 1] — Δt, АРХИТЕКТУРНО обязателен (не опция, см. докстринг)
        h0/c0:  [B, hidden_dim] или None (нулевое состояние — cold-start)
        Returns: (pred_seq, h_final) — тот же интерфейс, что CfCProbeModule.forward/
                 GRUProbeBaseline.forward, для единообразного вызова из train_model_dense/
                 predict_dense.
        """
        if dt_seq.dim() == 3:
            dt_seq = dt_seq.squeeze(-1)  # [B, T]
        B, T, _ = u_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        c = c0 if c0 is not None else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)

        outputs = []
        for t in range(T):
            h, c = self.lstm(u_seq[:, t], (h, c))
            h = self._rk4_evolve(h, dt_seq[:, t].unsqueeze(-1))
            outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1), h
