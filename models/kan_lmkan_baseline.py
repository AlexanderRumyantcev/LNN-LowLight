"""
models/kan_lmkan_baseline.py — TZ_stage9 §3.2 (18.09.2026): NRC-style baseline
с fused-kernel KAN (lmKAN, schwallergroup/lmkan, spline lookup tables) вместо
MLP внутри NRCStyleBaseline — GPU-only (CUDA-кернелы lmkan), в отличие от
models/kan_baseline.py::NRCKANBaseline (efficient-kan, CPU, TZ_stage8).

Интерфейс — та же схема, что NRCKANBaseline: build_input переиспользован из
NRCStyleBaseline, forward-контракт per-frame без рекуррентности. Отличие —
lmKAN работает ТОЛЬКО на CUDA (fused-kernel), batch-last layout ([features,
batch], не [batch,...,features] как у efficient-kan/обычных nn.Module) и
требует input_dim/output_dim каждого слоя, кратные tile_size_forward=8 и
tile_size_backward=4 (см. README lmkan) — оба ограничения обрабатываются
здесь автоматически (паддинг), без изменения вызывающего кода.

ФАКТИЧЕСКИЕ РАЗМЕРНОСТИ ПРОЕКТА (проверено 18.09.2026 напрямую по
generate_dataset_zeroday_result_nostep_v1.npz + models/spatial_features.py,
см. чат/EXPERIMENT_LOG.md): obs_dim=3 (RGB irradiance), staleness_dim=4
(dense-путь), spatial_dim=48 → input_dim=55. Ни 55, ни obs_dim=3 не кратны 8
— решение по паддингу (см. докстринг __init__ ниже) явно зафиксировано,
не принято молча, как требовал TZ_stage9 §3.2.
"""
import torch
import torch.nn as nn
from lmKAN import LMKAN2DLayer

from models.baselines import NRCStyleBaseline


def _pad_to_multiple(n: int, multiple: int) -> int:
    """Наименьшее число >= n, кратное multiple."""
    return ((n + multiple - 1) // multiple) * multiple


class NRCLmKANBaseline(nn.Module):
    """lmKAN-версия NRCStyleBaseline (fused CUDA-кернелы, GPU-only). См.
    models/baselines.py::NRCStyleBaseline докстринг для контекста
    faithful/честной схемы staleness-входа — здесь специфика
    TZ_stage9 только в самом регрессоре (fused-kernel KAN вместо MLP/
    efficient-kan)."""

    def __init__(self, obs_dim: int = 3, hidden_dim: int = 16, use_staleness: bool = True,
                 staleness_dim: int = 4, spatial_dim: int = 48, n_hidden_layers: int = 5,
                 num_grids: int = 10, tile_size_forward: int = 8, tile_size_backward: int = 4,
                 block_size_forward: int = 1024, block_size_backward: int = 512):
        """
        Дефолты obs_dim/staleness_dim/spatial_dim — фактические значения проекта
        (см. докстринг модуля), не общие заглушки, как в NRCStyleBaseline/
        NRCKANBaseline (obs_dim=1 и т.п.) — lmKAN требует знать точные размеры
        ДО инстанцирования CUDA-кернелов, ошибка в дефолте здесь дороже.

        num_grids=10 — README lmkan: "start with 8-10, increase after getting
        a working example" — не тюнится в этом этапе (TZ_stage9 §3.2, тот же
        принцип, что grid_size/spline_order зафиксированы в TZ_stage8 §3.2 для
        efficient-kan).

        ПАДДИНГ (решение TZ_stage9 §3.2, явно, не молча): input_dim и obs_dim
        (=output_dim последнего слоя) паддятся нулями до ближайшего кратного
        tile_size_forward=8 (кратность 4 для tile_size_backward автоматически
        выполняется, т.к. 8 кратно 4). При obs_dim=3/staleness=4/spatial=48 →
        input_dim=55 -> паддинг до 56 (1 лишний нулевой признак на входе,
        нулевой сигнал, не участвует в реальном признаковом пространстве).
        obs_dim=3 -> паддинг до 8 на выходе (5 лишних каналов вычисляются
        кернелом, но отбрасываются в forward() и НЕ должны попадать в
        loss — вызывающий код получает уже обрезанный до obs_dim выход,
        как у NRCStyleBaseline/NRCKANBaseline, ничего дополнительно маскировать
        не нужно). Скрытые слои (hidden_dim) тоже паддятся на случай, если
        вызывающий код передаст hidden_dim, не кратный 8 (16/64 из TZ_stage9
        §3.2 уже кратны — паддинг для них не срабатывает, no-op).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.use_staleness = use_staleness
        self.staleness_dim = staleness_dim
        self.spatial_dim = spatial_dim
        self.n_hidden_layers = n_hidden_layers
        self.num_grids = num_grids

        input_dim = obs_dim + (staleness_dim if use_staleness else 0) + spatial_dim
        input_dim_padded = _pad_to_multiple(input_dim, tile_size_forward)
        hidden_dim_padded = _pad_to_multiple(hidden_dim, tile_size_forward)
        output_dim_padded = _pad_to_multiple(obs_dim, tile_size_forward)
        self._input_dim = input_dim
        self._input_dim_padded = input_dim_padded
        self._output_dim_padded = output_dim_padded

        dims = [input_dim_padded] + [hidden_dim_padded] * n_hidden_layers + [output_dim_padded]
        self.kan_layers = nn.ModuleList([
            LMKAN2DLayer(
                num_grids=num_grids, input_dim=dims[i], output_dim=dims[i + 1],
                tile_size_forward=tile_size_forward, tile_size_backward=tile_size_backward,
                block_size_forward=block_size_forward, block_size_backward=block_size_backward,
            )
            for i in range(len(dims) - 1)
        ])

    # build_input идентичен NRCStyleBaseline (та же схема staleness/spatial-
    # входа, для честного сравнения с nrc_strong на одном eval-батче)
    build_input = staticmethod(NRCStyleBaseline.build_input)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        """
        u_seq: [B, T, input_dim] (см. build_input, тот же контракт, что у
               NRCStyleBaseline/NRCKANBaseline).
        Returns: pred_seq [B, T, obs_dim] — каждый шаг обработан независимо
                 (без рекуррентности).

        lmKAN использует batch-last layout ([features, N]), в отличие от
        обычных nn.Module ([..., features]) — в отличие от NRCKANBaseline
        (efficient-kan принимает произвольную форму входа с последней осью
        == features и делает reshape сам), здесь транспонирование и
        схлопывание батчевых осей нужно делать явно, вручную, один раз на
        входе и один раз на выходе слоёв.
        """
        B, T, D = u_seq.shape
        assert D == self._input_dim, (
            f"Ожидался input_dim={self._input_dim} (obs_dim+staleness_dim+"
            f"spatial_dim), получено {D} — проверьте use_staleness/spatial "
            f"аргументы build_input."
        )

        x = u_seq.reshape(B * T, D)  # [N, D]
        pad = self._input_dim_padded - D
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))  # нулевой паддинг по фичам
        x = x.transpose(0, 1).contiguous()  # -> [D_padded, N], batch-last

        for layer in self.kan_layers:
            x = layer(x)  # [D_out_padded, N] -> следующий вход как есть

        x = x.transpose(0, 1)  # -> [N, output_dim_padded]
        x = x[:, : self.obs_dim]  # обрезка паддинга выхода до реального obs_dim
        return x.reshape(B, T, self.obs_dim)

    def hessian_regularization_loss(self) -> torch.Tensor:
        """Аналог models/kan_baseline.py::NRCKANBaseline.curvature_regularization_
        loss, но встроенная в самую библиотеку lmkan реализация (README:
        "Each lmKAN layer has the method get_hessian_regularization()... sum
        these values across all layers, add lambda * total to the loss") —
        вместо самостоятельного пересчёта второй разности control points, как
        в TZ_stage8. Авторы рекомендуют начинать с сильной регуляризации и
        постепенно её ослаблять по ходу обучения (decay schedule) — не
        реализовано здесь намеренно: TZ_stage9 измеряет только скорость
        (§3.5 ТЗ), не переоценивает точность/сходимость (вне scope, §5 ТЗ) —
        добавить при переходе к вопросу точности, не раньше."""
        total = torch.zeros((), device=next(self.parameters()).device)
        for layer in self.kan_layers:
            total = total + layer.get_hessian_regularization()
        return total
