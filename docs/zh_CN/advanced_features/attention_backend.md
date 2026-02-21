# 注意力后端 (Attention Backend)

SGLang 支持多种注意力后端，每种后端各有优缺点。您可以根据自身需求进行测试选择。

```{important}
选择最优的注意力后端对于最大化性能至关重要。不同的后端在不同场景下表现各异，请根据您的模型、硬件和使用场景进行选择。并非所有后端都支持所有平台和模型架构。

如果您未指定 `--attention-backend`，SGLang 会尽力根据您的硬件和模型架构自动选择性能最佳的后端。
```

## 支持矩阵

支持矩阵分为两部分：MHA（标准注意力）和 MLA（多头潜在注意力）。有关 MHA 和 MLA 之间关键差异的说明，请参阅 [SGLang 关于 DeepSeek MLA 的文档](../basic_usage/deepseek_v3.md#multi-head-latent-attention-mla-throughput-optimizations) 和原始 [DeepSeek MLA 论文](https://arxiv.org/pdf/2405.04434)。

### MHA 后端

| **Backend**                     | **Page Size > 1 (native)** | **FP8 KV Cache** | **FP4 KV Cache** | **Spec topk=1** | **Spec topk>1** | **Sliding Window** | **MultiModal** |
|---------------------------------|-----------------------------|------------------|-----------------|-----------------|-----------------|--------------------|----------------|
| **FlashInfer**                  | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ❌             |
| **FA3 (FlashAttention 3)**      | ✅                          | ✅               | ❌              | ✅              | ✅              | ✅                 | ✅             |
| **FA4 (FlashAttention 4)**      | 128                         | ❌               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **Triton**                      | ❌                          | ❌               | ✅              | ✅              | ✅              | ✅                 | ✅             |
| **Torch Native (SDPA)**         | ❌                          | ✅               | ✅              | ❌              | ❌              | ❌                 | ✅             |
| **FlexAttention (PyTorch)**     | ❌                          | ❌               | ✅              | ❌              | ❌              | ❌                 | ❌             |
| **TRTLLM MHA**                  | 16, 32 or 64                | ✅               | ✅              | ✅              | ❌              | ✅                 | ❌             |
| **Dual Chunk FlashAttention**   | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **AITER (ROCm)**                | ✅                          | ✅               | ❌              | ✅              | ✅              | ❌                 | ✅             |
| **Wave (ROCm)**                 | ✅                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |
| **Ascend (NPU)**                | ✅                          | ❌               | ❌              | ✅              | ❌              | ❌                 | ✅             |
| **Intel XPU**                   | ✅                          | ❌               | ❌              | ❌              | ❌              | ✅                 | ❌             |
| **Intel AMX (CPU)**             | ❌                          | ❌               | ❌              | ❌              | ❌              | ❌                 | ❌             |

### MLA 后端

| **Backend**                | **Native Page Sizes**     | **FP8 KV Cache** | **FP4 KV Cache** | **Chunked Prefix Cache** | **Spec topk=1** | **Spec topk>1** |
|----------------------------|---------------------------|------------------|------------------|--------------------------|-----------------|-----------------|
| **FlashInfer MLA**         | 1                         | ❌               | ✅               | ✅                       | ✅              | ❌              |
| **FlashMLA**               | 64                        | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **Cutlass MLA**            | 128                       | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **TRTLLM MLA (Blackwell)** | 32 or 64                  | ✅               | ✅               | ✅                       | ✅              | ❌              |
| **FA3 (FlashAttention 3)** | n/a                       | ❌               | ❌               | ✅                       | ✅              | ⚠️ (page_size=1 only) |
| **Triton**                 | n/a                       | ❌               | ❌               | ❌                       | ✅              | ⚠️ (page_size=1 only) |
| **FA4**                    | 1                         | ❌               | ✅               | ❌                       | ❌              | ❌              |
| **Ascend MLA (NPU)**       | 128                       | ❌               | ❌               | ❌                       | ❌              | ❌              |

```{note}
多模态注意力通过 `--mm-attention-backend` 选择。"MultiModal" 列表示该后端系列是否存在对应的多模态实现。
```

```{note}
- FlashAttention 4 目前仅支持预填充（prefill）阶段。
- NSA 专为 [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/) 设计。
```

```{note}
对于 KV4 FA4 场景，FA4 需要使用不同的 --decode-attention-backend 来运行。除 trtllm_mha 与 FA4 不兼容外，其他所有解码后端的行为如表中所示。
```

```{tip}
推测解码 topk：`topk` 是每步从草稿模型中采样的草稿 token 数。`topk = 1` 遵循经典 EAGLE 方法；`topk > 1` 探索多个分支，需要草稿和验证路径中的后端支持。
```

```{tip}
页大小（Page size）控制将多少个 token 分组到一个 KV 缓存块中。要使前缀缓存生效，token 数量必须至少填满一个完整的页。例如，如果您的提示只有 32 个 token 而 `page_size = 64`，则无法填满一个完整的页，因此无法在前缀缓存中匹配（页不能被填充）。如果有 65 个 token 且 `page_size = 64`，则只有前 64 个 token 的第一页会被缓存和匹配；剩余的 1 个 token 将被丢弃。使用 `page_size = 1` 可实现最大程度的前缀复用（token 级匹配）。
```

许多不原生支持页操作的后端可以在封装层通过将页表展开为逐 token 索引来模拟 `page_size > 1`。"Page Size > 1 (native)" 列表示真正的内核级分页。某些后端需要固定的原生页大小，不能缩减或以其他方式模拟：TRTLLM MHA (16/32/64)、TRTLLM MLA (32/64)、FlashMLA (64)、Cutlass MLA (128)、Ascend (128)。

MLA 页大小约束：
- FlashInfer MLA：page_size = 1。
- FlashMLA：page_size = 64。
- Cutlass MLA：page_size = 128。
- TRTLLM MLA：page_size ∈ {32, 64}。

### 混合注意力（预填充与解码使用不同后端）（实验性功能）

```{warning}
混合注意力是一项实验性功能。
```

您可以为预填充和解码阶段混合搭配不同的注意力后端。当一个后端擅长预填充而另一个擅长解码时，这非常有用。有关实现细节，请参阅 `python/sglang/srt/layers/attention/hybrid_attn_backend.py`。

```bash
# 示例：使用 FA4 进行预填充，使用 TRTLLM MLA 进行解码（Blackwell 架构）
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --prefill-attention-backend fa4
```

#### 推测解码与混合注意力

混合注意力也支持推测解码。用于草稿解码和目标验证的后端取决于 `--speculative-attention-mode`：

- `--speculative-attention-mode decode`（推荐）：草稿/验证使用解码后端。
- `--speculative-attention-mode prefill`（默认）：草稿/验证使用预填充后端。

混合注意力与推测解码结合时的约束条件：

- 如果任一注意力后端是 `trtllm_mha`，推测解码仅支持 `--speculative-eagle-topk 1`。
- 对于 `--page-size > 1` 且 `--speculative-eagle-topk > 1` 的分页 MHA 后端，仅支持 `flashinfer`。
- CUDA Graph：解码后端始终会被捕获；预填充后端仅在 `--speculative-attention-mode prefill` 时被捕获。


```{tip}
如果您只设置了 `--prefill-attention-backend` 或 `--decode-attention-backend` 中的一个，未指定的阶段将继承 `--attention-backend` 的设置。
如果两者都被指定且不同，SGLang 会自动启用混合封装器，根据每个阶段分派到所选的后端。
```

## 注意力后端选择指南 (CUDA)

如果未指定 `--attention-backend` 参数，SGLang 会根据硬件 (CUDA) 和模型架构自动选择最佳后端。

### 自动选择逻辑

**1. MHA 模型（如 Llama、Qwen）**
- **Hopper 架构（如 H100、H200）**：如果使用 CUDA 12.3+ 且模型配置受支持，默认使用 `fa3`。
- **Blackwell 架构（如 B200）**：默认使用 `trtllm_mha`，除非使用 `topk > 1` 的推测解码。
- **其他架构（Ampere、Ada 等）**：如果可用则默认使用 `flashinfer`；否则回退到 `triton`。

**2. MLA 模型（如 DeepSeek V3）**
- **Hopper 架构**：默认使用 `fa3`（需要 CUDA 12.3+）。
- **Blackwell 架构**：默认使用 `trtllm_mla`。
- **其他架构**：默认使用 `triton`。


## 使用指南

### 不同注意力后端的启动命令

- FlashInfer（非 Hopper 机器的默认选项，如 A100、A40）
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flashinfer
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend flashinfer \
  --trust-remote-code
```

- FlashAttention 3（Hopper 机器的默认选项，如 H100、H200、H20）
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend fa3
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --trust-remote-code \
  --attention-backend fa3
```

- Triton
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend triton
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-V3 \
  --attention-backend triton \
  --trust-remote-code
```

- FlashMLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --trust-remote-code
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- TRTLLM MLA（针对 Blackwell 架构优化，如 B200）
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --trust-remote-code
```

- TRTLLM MLA 配合 FP8 KV Cache（更高并发，更低内存占用）
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code
```

- FlashAttention 4（MHA 和 MLA）
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --prefill-attention-backend fa4 \
  --trust-remote-code
```

- Cutlass MLA
```bash
python3 -m sglang.launch_server \
  --tp 8 \
  --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla \
  --trust-remote-code
```

- Ascend
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend
```

- Intel XPU
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend intel_xpu
```

- Wave
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend wave
```

- FlexAttention
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flex_attention
```

- Dual Chunk FlashAttention
```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn
```

- Torch Native
```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
```

## 添加新注意力后端的步骤
要添加新的注意力后端，您可以参考现有后端
（`python/sglang/srt/layers/attention/triton_backend.py`、`python/sglang/srt/layers/attention/flashattention_backend.py`）
并按照以下步骤操作。

1. 在不使用 CUDA Graph 的情况下运行。支持两个前向函数
    - forward_extend
        - 用于预填充、带 KV 缓存的预填充以及目标验证
        - 每层调用一次
    - forward_decode
        - 用于普通解码和草稿解码
        - 每层调用一次
    - init_forward_metadata
        - 初始化类和所有层共享的通用元数据
        - 调用 plan 函数进行优化，如 split_kv
        - 每次前向传播调用一次
2. 使用 CUDA Graph 运行。包含两个阶段（捕获和重放），您需要实现三个函数
    - init_cuda_graph_state
        - 在生命周期中仅调用一次
        - 创建所有通用的共享缓冲区
    - init_forward_metadata_capture_cuda_graph
        - 在捕获 CUDA Graph 之前调用
        - 类似于 init_forward_metadata，但将元数据写入一些预定义的缓冲区
    - init_forward_metadata_replay_cuda_graph
        - 在重放 CUDA Graph 之前调用
        - 此函数处于关键路径上，需要保证快速执行
