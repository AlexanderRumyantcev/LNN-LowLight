# TZ_stage9 §3.0 — Kaggle T4x2 smoke-test для lmKAN
# Формат: ячейки Jupytext-style (# %%), вставить в Kaggle Notebook как есть,
# либо запустить целиком как .py в Kaggle-кернеле с GPU.
#
# ПЕРЕД ЗАПУСКОМ (настройки Kaggle Notebook):
#   Settings -> Accelerator -> GPU T4 x2 (P100 выведен из строя 15.09.2026)
#   Settings -> Internet -> On (нужен для git clone + pip install)
#
# Цель ТОЛЬКО: убедиться, что nvcc собирает CUDA-кернелы lmkan и toy
# forward+backward отрабатывают. НЕ полный протокол TZ_stage9 (это следующий
# шаг после успеха этого смоук-теста) — здесь нет ни nrc_strong, ни реальной
# конфигурации проекта, только пример 1:1 из README библиотеки.

# %% [markdown]
# ## Ячейка 1 — проверка железа

# %%
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("PyTorch version:", torch.__version__, "| CUDA build:", torch.version.cuda)

# %% [markdown]
# ## Ячейка 2 — установка lmkan (компиляция CUDA-кернелов через nvcc)
# Если эта ячейка падает с ошибкой сборки — это и есть ответ на вопрос
# §3.0 TZ_stage9 ("Шаг 0 провален") и повод сразу пробовать PolyKAN вместо
# продолжения адаптации под lmkan.

# %%
get_ipython().system('git clone https://github.com/schwallergroup/lmkan.git /kaggle/working/lmkan')
get_ipython().system('cd /kaggle/working/lmkan && pip install .')
# Если запускаете как чистый .py вне Jupyter (не через ipykernel) — замените
# две строки выше на обычные subprocess.run(["git", "clone", ...]) /
# subprocess.run(["pip", "install", "."], cwd=...).

# %% [markdown]
# ## Ячейка 3 — toy forward, пример 1:1 из README lmkan
# Формы (128x128, batch 1024) — НЕ конфигурация проекта, это чистая проверка
# "библиотека вообще работает". Адаптация под input_dim≈55/hidden_dim∈{16,64}
# — отдельный, следующий шаг (§3.2 TZ_stage9), не часть смоук-теста.

# %%
import torch
from lmKAN import LMKAN2DLayer

NUM_GRIDS = 10          # README рекомендует стартовать с 8-10, не сразу большую сетку
BATCH_SIZE = 1024
INPUT_DIM = 128         # кратно tile_size_forward=8 и tile_size_backward=4
OUTPUT_DIM = 128        # кратно tile_size_forward=8 и tile_size_backward=4

layer = LMKAN2DLayer(
    num_grids=NUM_GRIDS,
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    tile_size_forward=8,
    tile_size_backward=4,
    block_size_forward=1024,
    block_size_backward=512,
).cuda()

# lmKAN использует batch-last layout — ВАЖНО, не (batch, features), как в
# обычных nn.Linear/nn.Module проекта; при адаптации под NRCStyleBaseline
# понадобится транспонирование на входе/выходе слоя.
x = torch.randn(INPUT_DIM, BATCH_SIZE).cuda()

output = layer(x)
print("Forward OK. Output shape:", output.shape)  # ожидается [OUTPUT_DIM, BATCH_SIZE]
assert output.shape == (OUTPUT_DIM, BATCH_SIZE), "Форма выхода не совпадает с ожидаемой"

# %% [markdown]
# ## Ячейка 4 — toy backward (online-режим TZ_stage9 нуждается в forward+backward,
# не только в инференсе — проверяем именно это, не только форму выхода)

# %%
target = torch.randn(OUTPUT_DIM, BATCH_SIZE).cuda()
loss = torch.nn.functional.mse_loss(output, target)
loss.backward()

has_grad = any(p.grad is not None for p in layer.parameters())
print("Backward OK. Loss:", loss.item(), "| gradients populated:", has_grad)
assert has_grad, "Backward прошёл без ошибок, но градиенты не посчитаны — тоже провал Шага 0"

# %% [markdown]
# ## Ячейка 5 — грубая (не протокольная!) прикидка задержки, только для
# ориентира перед основным прогоном TZ_stage9 §3.3-3.4. НЕ заменяет
# нормированный по батчу/железу протокол из основного ТЗ — здесь нет
# warm-up, нет усреднения по многим запускам, нет sync-барьеров помимо
# базового synchronize().

# %%
import time

torch.cuda.synchronize()
n_steps = 50
t0 = time.perf_counter()
for _ in range(n_steps):
    out = layer(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"~{(t1 - t0) / n_steps * 1000:.3f} мс/шаг (forward+backward, грубая оценка, "
      f"input_dim={INPUT_DIM}, output_dim={OUTPUT_DIM}, num_grids={NUM_GRIDS})")

# %% [markdown]
# ## Итог Шага 0
# Если все ассерты выше прошли без исключений — сборка и базовый
# online-режим (forward+backward) на T4 работают. Можно переходить к §3.2
# TZ_stage9 (адаптация под input_dim≈55, hidden_dim∈{16,64}, 5 слоёв,
# с паддингом под tile_size_forward/backward — 55 и типичный output_dim
# проекта НЕ кратны 8/4 "из коробки", это первое, что нужно решить в
# следующем шаге, не здесь).
#
# Если что-то из ассертов упало — зафиксировать точный текст ошибки в
# EXPERIMENT_LOG.md (раздел TZ_stage9) и переходить к варианту (а)/(б)
# из §3.0: пробовать PolyKAN, либо закрывать online-GPU-направление как
# инфраструктурно недоступное (не путать с "KAN неэффективен" — см. §3.0).
