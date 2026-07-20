"""
Этап 3 ТЗ motion alignment: композиция flow для sliding window + occlusion-маска.

КОНТЕКСТ (см. TZ_motion_alignment_LNN_LowLight.md, mempalace wing=LNN_LowLight):
Этап 2 (scripts/precompute_flow.py) кэшировал offline только flow между
СОСЕДНИМИ кадрами (i -> i+1), half-res, RAFT-small на normal_light, для всего
датасета. Этот модуль НЕ пишет ничего на диск и не запускается отдельным
скриптом — согласно принятому решению в ТЗ ("Flow между произвольными t-k и
t для конкретного окна получается композицией соседних потоков в момент
чтения батча"), функции здесь предназначены для вызова ОНЛАЙН из dataloader
(интеграция — Этап 4, из этого модуля пока не делается).

ДВЕ РАЗНЫЕ ОПЕРАЦИИ ПОД КАПОТОМ (ВАЖНО: направление ниже — исправлено
2026-07-20 после ручной перепроверки конвенции grid_sample; в первой версии
модуля было перепутано, см. mempalace wing=LNN_LowLight, decisions):

  Ключевое правило: grid_sample(input, grid) требует, чтобы grid был
  ОПРЕДЕЛЁН НА СЕТКЕ РЕЗУЛЬТАТА (anchor), а не на сетке input. Композиция
  neighbor-flow всегда получается на сетке ПЕРВОГО элемента цепочки — значит,
  нужна ли инверсия, зависит от того, с чьей сетки цепочка стартует.

  1. FORWARD-КОМПОЗИЦИЯ (compose_flow_chain), gather-семантика, БЕЗ инверсии:
     Применяется, когда исходный кадр ПОЗЖЕ опорного (frame_idx > anchor_idx,
     "будущий" кадр относительно anchor — в текущей каузальной схеме окна
     (anchor = последний кадр) такого не бывает, но функция общая). Цепочка
     flow[anchor_idx], ..., flow[frame_idx-1] стартует с flow[anchor_idx],
     который уже живёт на сетке anchor_idx — то есть результат композиции
     сразу на нужной (anchor) сетке, инверсия не нужна.

  2. SPLAT-ИНВЕРСИЯ (splat_invert_flow), scatter-семантика:
     Применяется, когда исходный кадр РАНЬШЕ опорного (frame_idx < anchor_idx
     — это ОСНОВНОЙ случай при текущей каузальной схеме окна: все 4
     не-опорных кадра окна прошлые). Цепочка flow[frame_idx], ...,
     flow[anchor_idx-1] стартует с flow[frame_idx], который живёт на сетке
     frame_idx — то есть композиция получается на сетке ИСТОЧНИКА, а не
     anchor, и напрямую как grid для grid_sample непригодна (тихая ошибка
     выравнивания, а не падение). Поэтому:
       a) forward-композицией строим flow frame_idx -> anchor_idx (на сетке
          frame_idx, обычная операция #1);
       b) полученный flow ИНВЕРТИРУЕМ через billinear splatting: каждый
          вектор -flow(x,y) "разбрызгивается" (в отличие от grid_sample,
          который "собирает") из своей исходной позиции (регулярная сетка
          frame_idx) в целевую, обычно дробную позицию в кадре anchor_idx,
          с весами по 4 ближайшим целым пикселям.
     После splatting неизбежно возникают:
       - ДЫРЫ (ни один вектор не долетел) = дезокклюзия: область, видимая в
         frame_idx, но не имевшая соответствия в anchor_idx (например, вышла
         из-за движущегося объекта). Backward flow там принципиально не
         определён.
       - КОЛЛИЗИИ (несколько source-пикселей попали в одну целевую точку) =
         окклюзия: например, передний план надвинулся на задний фон,
         двигавшийся иначе. Разрешается эвристикой softmax splatting
         (Niklaus & Liu, 2020): вес = billinear_weight * exp(scale*|flow|)
         — более быстро движущаяся точка считается "ближе к камере" и
         перекрывает более медленную.
     Обе ситуации попадают в occlusion_mask.

ГРАНИЦЫ ОКНА: для frame_idx вне диапазона клипа (t-k < 0 или t+k >= n_frames)
вызывающий код (Этап 4, dataloader) должен обрезать окно, а не звать эти
функции с несуществующим индексом — здесь эта проверка НЕ делается (модуль
работает с уже готовыми списками flow-тензоров, не знает о границах клипа).

ИНТЕРФЕЙС ДЛЯ ЭТАПА 4 (dataloader):
  get_flow_to_anchor(neighbor_flows, frame_idx, anchor_idx, importance_scale)
    -> (flow, occlusion_mask)
  где neighbor_flows — dict[int, Tensor[2,H,W]], neighbor_flows[t] = flow
  (t -> t+1), загруженный из кэша Этапа 2 (scripts/precompute_flow.py) для
  всех t в диапазоне [min(frame_idx,anchor_idx), max(frame_idx,anchor_idx)).
  flow: Tensor[2,H,W] — grid-смещение (в пикселях, на сетке anchor_idx),
  готовое для прямого использования как sampling grid в grid_sample при
  warping frame_idx на сетку anchor_idx.
  occlusion_mask: Tensor[1,H,W] uint8, 1 = не доверять flow в этом пикселе.
  ПРИ ТЕКУЩЕЙ КАУЗАЛЬНОЙ СХЕМЕ ОКНА (anchor = последний кадр, все прочие —
  прошлые, frame_idx < anchor_idx) маска НЕ нулевая — это основной, а не
  краевой случай: splat-инверсия и её дыры/коллизии участвуют в каждом
  вызове. Нулевой она гарантированно будет только при frame_idx > anchor_idx
  (будущий кадр, при текущей схеме окна не встречается).

Unit-тест на синтетическом сдвиге: см. test_motion_alignment.py в корне репо.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Общие геометрические помощники
# ---------------------------------------------------------------------------

def _make_base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Регулярная сетка координат в пикселях, shape [2, H, W] (x, y)."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)


def _sample_flow_at(flow_field: torch.Tensor, sample_xy_pix: torch.Tensor) -> torch.Tensor:
    """
    Билинейно читает flow_field (gather) в точках sample_xy_pix.

    flow_field:    [2, H, W] — flow t -> t+1, в пикселях, на сетке кадра t
    sample_xy_pix: [2, H, W] — координаты в пикселях (не обязательно целые)
    Возвращает:    [2, H, W] — flow_field, прочитанный в этих точках
    """
    _, h, w = flow_field.shape
    x_norm = 2.0 * sample_xy_pix[0] / max(w - 1, 1) - 1.0
    y_norm = 2.0 * sample_xy_pix[1] / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    sampled = F.grid_sample(
        flow_field.unsqueeze(0).float(), grid.float(),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return sampled[0]


# ---------------------------------------------------------------------------
# 1. Forward-композиция (t-k -> t, gather-семантика)
# ---------------------------------------------------------------------------

def compose_flow_chain(flows: list[torch.Tensor]) -> torch.Tensor:
    """
    Композирует цепочку соседних forward flow-полей в один flow start -> end.

    flows: список длины k, flows[j] = flow (start+j) -> (start+j+1), [2,H,W],
           в том порядке, в котором время идёт ВПЕРЁД (flows[0] — самый
           ранний шаг).
    Возвращает: flow start -> end, [2, H, W], на сетке кадра start.
    """
    assert len(flows) >= 1, "нужен хотя бы один flow (|frame_idx - anchor_idx| >= 1)"
    device = flows[0].device
    h, w = flows[0].shape[-2:]
    base_grid = _make_base_grid(h, w, device, torch.float32)

    acc = flows[0].float().clone()
    for j in range(1, len(flows)):
        next_flow = flows[j].float()          # flow (start+j) -> (start+j+1)
        current_pos = base_grid + acc          # где мы сейчас, в пикселях (start+j)
        acc = acc + _sample_flow_at(next_flow, current_pos)
    return acc


# ---------------------------------------------------------------------------
# 2. Splat-инверсия (scatter-семантика) + occlusion mask
# ---------------------------------------------------------------------------

def splat_invert_flow(flow_fwd: torch.Tensor, importance_scale: float = 1.0):
    """
    Инвертирует forward flow (anchor -> frame) в backward flow (frame ->
    anchor) методом билинейного softmax-splatting.

    flow_fwd: [2, H, W] — flow anchor -> frame, на сетке кадра anchor
    importance_scale: масштаб веса exp(scale*|flow|) при разрешении
                       коллизий (см. докстринг модуля)

    Возвращает:
      backward_flow:  [2, H, W] — flow frame -> anchor, на сетке кадра frame
      occlusion_mask: [1, H, W] uint8, 1 = дырка (дезокклюзия), backward
                       flow здесь не определён (заполнен нулями)
    """
    device = flow_fwd.device
    _, h, w = flow_fwd.shape
    flow_fwd = flow_fwd.float()

    base_grid = _make_base_grid(h, w, device, torch.float32)
    dst = base_grid + flow_fwd
    dst_x, dst_y = dst[0], dst[1]

    val_x, val_y = -flow_fwd[0], -flow_fwd[1]
    magnitude = torch.sqrt(flow_fwd[0] ** 2 + flow_fwd[1] ** 2)
    importance = torch.exp(importance_scale * magnitude)

    x0 = torch.floor(dst_x).long()
    y0 = torch.floor(dst_y).long()
    x1, y1 = x0 + 1, y0 + 1
    wx1 = (dst_x - x0.float()).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy1 = (dst_y - y0.float()).clamp(0, 1)
    wy0 = 1.0 - wy1

    flat_out_x = torch.zeros(h * w, device=device, dtype=torch.float32)
    flat_out_y = torch.zeros(h * w, device=device, dtype=torch.float32)
    flat_weight = torch.zeros(h * w, device=device, dtype=torch.float32)
    flat_val_x, flat_val_y = val_x.reshape(-1), val_y.reshape(-1)

    for xi, yi, bw in [(x0, y0, wx0 * wy0), (x1, y0, wx1 * wy0),
                        (x0, y1, wx0 * wy1), (x1, y1, wx1 * wy1)]:
        valid = ((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)).reshape(-1)
        if not torch.any(valid):
            continue
        idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).reshape(-1)[valid]
        w_v = (bw.reshape(-1)[valid] * importance.reshape(-1)[valid])
        flat_out_x.scatter_add_(0, idx, flat_val_x[valid] * w_v)
        flat_out_y.scatter_add_(0, idx, flat_val_y[valid] * w_v)
        flat_weight.scatter_add_(0, idx, w_v)

    weight = flat_weight.reshape(h, w)
    occlusion_mask = weight <= 1e-6
    safe_weight = weight.clamp_min(1e-6)

    backward_flow = torch.stack([
        flat_out_x.reshape(h, w) / safe_weight,
        flat_out_y.reshape(h, w) / safe_weight,
    ], dim=0)
    backward_flow[:, occlusion_mask] = 0.0

    return backward_flow, occlusion_mask.unsqueeze(0).to(torch.uint8)


# ---------------------------------------------------------------------------
# 3. Единая точка входа для Этапа 4 (dataloader)
# ---------------------------------------------------------------------------

def get_flow_to_anchor(
    neighbor_flows: dict[int, torch.Tensor],
    frame_idx: int,
    anchor_idx: int,
    importance_scale: float = 1.0,
):
    """
    Возвращает flow, которым нужно warp-нуть frame_idx, чтобы он совпал с
    сеткой anchor_idx, плюс occlusion-маску.

    neighbor_flows: dict[t] = flow (t -> t+1), [2,H,W]; должен содержать все
                     t в диапазоне [min(frame_idx,anchor_idx), max(...)) —
                     это соседние flow из кэша Этапа 2.
    frame_idx, anchor_idx: индексы кадров внутри клипа (anchor — опорный
                            кадр окна, к которому всё выравнивается).

    Возвращает: (flow [2,H,W], occlusion_mask [1,H,W] uint8)
    """
    if frame_idx == anchor_idx:
        h, w = next(iter(neighbor_flows.values())).shape[-2:]
        device = next(iter(neighbor_flows.values())).device
        zero_flow = torch.zeros(2, h, w, device=device)
        zero_mask = torch.zeros(1, h, w, dtype=torch.uint8, device=device)
        return zero_flow, zero_mask

    if frame_idx < anchor_idx:
        # ОСНОВНОЙ случай при текущей каузальной схеме окна: frame_idx —
        # прошлый кадр. Цепочка flow[frame_idx..anchor_idx-1] стартует с
        # flow[frame_idx] -> композиция получается НА СЕТКЕ frame_idx, а не
        # anchor_idx. Для grid_sample нужна сетка anchor_idx -> инвертируем
        # через splatting (см. докстринг модуля).
        chain = [neighbor_flows[t] for t in range(frame_idx, anchor_idx)]
        forward_composed = compose_flow_chain(chain)          # на сетке frame_idx
        return splat_invert_flow(forward_composed, importance_scale)  # на сетке anchor_idx

    # frame_idx > anchor_idx: будущий кадр (при текущей каузальной схеме окна
    # не встречается, но функция общая). Цепочка flow[anchor_idx..frame_idx-1]
    # стартует с flow[anchor_idx] -> композиция УЖЕ на сетке anchor_idx,
    # инверсия не нужна, дыр не даёт (gather-семантика).
    chain = [neighbor_flows[t] for t in range(anchor_idx, frame_idx)]
    flow = compose_flow_chain(chain)
    occlusion = torch.zeros(1, *flow.shape[-2:], dtype=torch.uint8, device=flow.device)
    return flow, occlusion
