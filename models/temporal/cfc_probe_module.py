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
        use_staleness: включать ли staleness-вектор [cold_start, confidence] во вход
                      (§2.2 — основная конфигурация CfC-B; False даёт "голый" CfC,
                      нужен как ablation-точка сравнения, см. spike-test mode='A')
    """

    def __init__(
        self,
        obs_dim: int = 1,
        hidden_dim: int = 32,
        use_staleness: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.use_staleness = use_staleness

        input_dim = obs_dim + (2 if use_staleness else 0)  # +[cold_start, confidence]
        self.cell = CfCProbeCell(input_dim=input_dim, hidden_dim=hidden_dim)
        self.readout = nn.Linear(hidden_dim, obs_dim)

    @staticmethod
    def build_input(
        obs: torch.Tensor,
        cold_start: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        use_staleness: bool = True,
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
        """
        if not use_staleness:
            return obs
        if cold_start is None or confidence is None:
            raise ValueError(
                "use_staleness=True требует cold_start и confidence "
                "(получить из spp-метаданных пробы, см. ТЗ §3.1/§3.2)"
            )
        staleness = torch.stack([cold_start, confidence], dim=-1)  # [B, T, 2]
        return torch.cat([obs, staleness], dim=-1)

    def forward(
        self,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
        record_gates: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            u_seq:  [B, T, input_dim] — вход на каждый шаг (см. build_input)
            dt_seq: [B, T] или [B, T, 1] — Δt между последовательными обновлениями пробы
            h0:     [B, hidden_dim] или None (тогда нулевое состояние — cold-start)
            record_gates: сохранить ли sigma_tau по шагам (диагностика насыщения гейта)

        Returns:
            (pred_seq, h_final):
                pred_seq: [B, T, obs_dim] — предсказанная irradiance на каждом шаге
                          (причинно: шаг t не видит входов t+1..T-1)
                h_final:  [B, hidden_dim] — финальное скрытое состояние (для continuation
                          между вызовами — проба живёт дольше одного обучающего окна)
        """
        if dt_seq.dim() == 2:
            dt_seq = dt_seq.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        B, T, _ = u_seq.shape
        h = (
            h0
            if h0 is not None
            else torch.zeros(B, self.hidden_dim, device=u_seq.device, dtype=u_seq.dtype)
        )

        outputs = []
        gate_log = [] if record_gates else None
        diag_logs = {k: [] for k in ("z", "g", "h_cand", "t_a", "t_b", "pre_sigmoid", "sigma_tau")} if record_gates else None
        for t in range(T):
            h = self.cell(u_seq[:, t], h, dt_seq[:, t])
            if record_gates:
                gate_log.append(self.cell.last_sigma_tau)
                for k in diag_logs:
                    diag_logs[k].append(self.cell.last_gate_diag[k])
            outputs.append(self.readout(h))

        if record_gates:
            self.last_gate_log = torch.stack(gate_log, dim=1)  # [B, T, hidden_dim]
            # каждый ключ — [B, T, hidden_dim]; держит граф, если вызывающий код не под no_grad()
            # (нужно full_gate_diagnostics для backward через t_a/t_b)
            self.last_diag_log = {k: torch.stack(v, dim=1) for k, v in diag_logs.items()}

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
    определять НАСКОЛЬКО сильно сигмоида реагирует на log1p(dt) — это единственный параметр,
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
                           segment_names: dict | None = None, label: str = "sigma_tau",
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

    u_seq: [B, T, input_dim], dt_seq: [B, T] или [B, T, 1], true: [B, T, obs_dim] опционально.
    seg_type: [B, T] int-коды сегментов (см. evaluation.metrics.SEGMENT_NAMES), опционально.
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

    if seg_type is not None:
        seg_t = torch.as_tensor(seg_type)
        names = segment_names or {0: "static", 1: "step", 2: "drift"}
        by_seg = {}
        for code, name in names.items():
            mask = seg_t == code
            if mask.any():
                by_seg[name] = dict(z_norm_mean=float(z[mask].norm(dim=-1).mean()),
                                     t_a_mean=float(t_a[mask].mean()), t_b_mean=float(t_b[mask].mean()))
        result["by_seg_type"] = by_seg

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
        if "by_seg_type" in result:
            for seg, s in result["by_seg_type"].items():
                print(f"      seg={seg:7s}     z_norm={s['z_norm_mean']:.3f} t_a={s['t_a_mean']:.3f} t_b={s['t_b_mean']:.3f}")
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
