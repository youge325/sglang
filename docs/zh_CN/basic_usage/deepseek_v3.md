# DeepSeek V3/V3.1/R1 使用指南

SGLang 为 DeepSeek 模型提供了大量专属优化，使其成为官方 [DeepSeek 团队](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended) 从第一天起就推荐的推理引擎。

本文档概述了当前针对 DeepSeek 的优化。
有关已实现功能的概览，请参阅已完成的 [路线图](https://github.com/sgl-project/sglang/issues/2591)。

## 使用 SGLang 启动 DeepSeek V3.1/V3/R1

运行 DeepSeek V3.1/V3/R1 模型的推荐配置如下：

| 权重类型 | 配置 |
|---------|------|
| **全精度 [FP8](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**<br>*（推荐）* | 8 x H200 |
| | 8 x B200 |
| | 8 x MI300X |
| | 2 x 8 x H100/800/20 |
| | Xeon 6980P CPU |
| **全精度 ([BF16](https://huggingface.co/unsloth/DeepSeek-R1-0528-BF16))**（从原始 FP8 上转换） | 2 x 8 x H200 |
| | 2 x 8 x MI300X |
| | 4 x 8 x H100/800/20 |
| | 4 x 8 x A100/A800 |
| **量化权重 ([INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8))** | 16 x A100/800 |
| | 32 x L40S |
| | Xeon 6980P CPU |
| | 4 x Atlas 800I A3 |
| **量化权重 ([W4A8](https://huggingface.co/novita/Deepseek-R1-0528-W4AFP8))** | 8 x H20/100, 4 x H200 |
| **量化权重 ([AWQ](https://huggingface.co/QuixiAI/DeepSeek-R1-0528-AWQ))** | 8 x H100/800/20 |
| | 8 x A100/A800 |
| **量化权重 ([MXFP4](https://huggingface.co/amd/DeepSeek-R1-MXFP4-Preview))** | 8, 4 x MI355X/350X |
| **量化权重 ([NVFP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2))** | 8, 4 x B200 |

```{important}
官方 DeepSeek V3 已经是 FP8 格式，因此你不应该使用任何量化参数（如 `--quantization fp8`）来运行它。
```

详细命令参考：

- [8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#using-docker-recommended)
- [4 x B200, 8 x B200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-one-b200-node)
- [8 x MI300X](../platforms/amd_gpu.md#running-deepseek-v3)
- [2 x 8 x H200](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)
- [4 x 8 x A100](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)
- [8 x A100 (AWQ)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-8-a100a800-with-awq-quantization)
- [16 x A100 (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-16-a100a800-with-int8-quantization)
- [32 x L40S (INT8)](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-32-l40s-with-int8-quantization)
- [Xeon 6980P CPU](../platforms/cpu_server.md#example-running-deepseek-r1)
- [4 x Atlas 800I A3 (int8)](../platforms/ascend_npu_deepseek_example.md#running-deepseek-with-pd-disaggregation-on-4-x-atlas-800i-a3)

### 下载权重
如果启动服务器时遇到错误，请确保权重已下载完成。建议提前下载或多次重启直到所有权重下载完毕。请参阅 [DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base#61-inference-with-deepseek-infer-demo-example-only) 官方指南下载权重。

### 使用单节点 8 x H200 启动
请参阅[示例](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#installation--launch)。

### 多节点运行示例

- [在 GB200 NVL72 上使用 PD 和大规模 EP 部署 DeepSeek](https://lmsys.org/blog/2025-06-16-gb200-part-1/) ([第一部分](https://lmsys.org/blog/2025-06-16-gb200-part-1/), [第二部分](https://lmsys.org/blog/2025-09-25-gb200-part-2/)) — GB200 优化综合指南。

- [在 96 张 H100 GPU 上使用 PD 分离和大规模专家并行部署 DeepSeek](https://lmsys.org/blog/2025-05-05-deepseek-pd-ep/) — PD 分离和大规模 EP 指南。

- [使用两个 H20*8 节点提供服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h208-nodes)。

- [在 H20 上提供 DeepSeek-R1 的最佳实践](https://lmsys.org/blog/2025-09-26-sglang-ant-group/) — H20 优化、部署和性能综合指南。

- [使用两个 H200*8 节点和 Docker 提供服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker)。

- [使用四个 A100*8 节点提供服务](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-four-a1008-nodes)。

## 优化

### 多头潜在注意力 (MLA) 吞吐量优化

**描述**：[MLA](https://arxiv.org/pdf/2405.04434) 是 DeepSeek 团队引入的创新注意力机制，旨在提高推理效率。SGLang 为此实现了专门的优化，包括：

- **权重吸收**：通过应用矩阵乘法结合律重新排序计算步骤，该方法在解码阶段平衡了计算和内存访问并提高了效率。

- **MLA 注意力后端**：目前 SGLang 支持多种优化的 MLA 注意力后端，包括 [FlashAttention3](https://github.com/Dao-AILab/flash-attention)、[Flashinfer](https://docs.flashinfer.ai/api/attention.html#flashinfer-mla)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[CutlassMLA](https://github.com/sgl-project/sglang/pull/5390)、**TRTLLM MLA**（针对 Blackwell 架构优化）和 [Triton](https://github.com/triton-lang/triton) 后端。默认的 FA3 在各种工作负载中提供良好性能。

- **FP8 量化**：W8A8 FP8 和 KV Cache FP8 量化可实现高效的 FP8 推理。此外，我们还实现了批量矩阵乘法 (BMM) 算子，以促进带有权重吸收的 MLA 中的 FP8 推理。

- **CUDA Graph 和 Torch.compile**：MLA 和混合专家 (MoE) 均与 CUDA Graph 和 Torch.compile 兼容，这降低了延迟并加速了小批量大小的解码速度。

- **分块前缀缓存**：分块前缀缓存优化可以通过将前缀缓存切分为块、使用多头注意力处理并合并状态来提高吞吐量。在对长序列进行分块预填充时，其改进效果显著。目前此优化仅适用于 FlashAttention3 后端。

总体而言，通过这些优化，我们与之前版本相比实现了高达 **7 倍**的输出吞吐量加速。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_3/deepseek_mla.svg" alt="DeepSeek 系列模型的多头潜在注意力">
</p>

**用法**：MLA 优化默认启用。

**参考**：查看 [博客](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) 和 [幻灯片](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/lmsys_1st_meetup_deepseek_mla.pdf) 了解更多详情。

### 数据并行注意力

**描述**：此优化涉及 DeepSeek 系列模型 MLA 注意力机制的数据并行 (DP)，可以显著减少 KV 缓存大小，从而支持更大的批量大小。每个 DP worker 独立处理不同类型的批次（预填充、解码、空闲），然后在混合专家 (MoE) 层处理前后进行同步。如果不使用 DP 注意力，KV 缓存将在所有 TP rank 之间复制。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/dp_attention.svg" alt="DeepSeek 系列模型的数据并行注意力">
</p>

启用数据并行注意力后，我们与之前版本相比实现了高达 **1.9 倍**的解码吞吐量提升。

<p align="center">
  <img src="https://lmsys.org/images/blog/sglang_v0_4/deepseek_coder_v2.svg" alt="数据并行注意力性能对比">
</p>

**用法**：
- 使用 8 张 H200 GPU 时，在服务器参数中添加 `--enable-dp-attention --tp 8 --dp 8`。此优化在高批量大小场景中提高峰值吞吐量，此时服务器受限于 KV 缓存容量。
- DP 和 TP 注意力可以灵活组合。例如，要在 2 节点 × 8 张 H100 GPU 上部署 DeepSeek-V3/R1，可以指定 `--enable-dp-attention --tp 16 --dp 2`。此配置以 2 个 DP 组运行注意力，每组包含 8 个 TP GPU。

```{caution}
数据并行注意力不推荐用于低延迟、小批量的用例。它针对大批量大小的高吞吐量场景进行了优化。
```

**参考**：查看[博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models)。

### 多节点张量并行

**描述**：对于单节点内存有限的用户，SGLang 支持使用张量并行跨多节点提供 DeepSeek 系列模型（包括 DeepSeek V3）的服务。此方法将模型参数分布到多个 GPU 或节点上，以处理单节点内存无法容纳的模型。

**用法**：查看[此处](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208)的使用示例。

### 块级 FP8

**描述**：SGLang 实现了块级 FP8 量化，包含两个关键优化：

- **激活**：使用 E4M3 格式，每 token 每 128 通道子向量缩放，在线转换。

- **权重**：每 128x128 块量化，以获得更好的数值稳定性。

- **DeepGEMM**：[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 核函数库，针对 FP8 矩阵乘法优化。

**用法**：上述激活和权重优化在 DeepSeek V3 模型上默认开启。DeepGEMM 在 NVIDIA Hopper/Blackwell GPU 上默认启用，在其他设备上默认禁用。也可以通过设置环境变量 `SGLANG_ENABLE_JIT_DEEPGEMM=0` 手动关闭 DeepGEMM。

```{tip}
在提供 DeepSeek 模型服务之前，预编译 DeepGEMM 核函数以提高首次运行性能。预编译过程通常需要约 10 分钟完成。
```

```bash
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### 多 Token 预测
**描述**：SGLang 基于 [EAGLE 投机解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 实现了 DeepSeek V3 多 Token 预测 (MTP)。通过此优化，在 H200 TP8 配置下，批量大小为 1 时可提高 **1.8 倍**解码速度，批量大小为 32 时可提高 **1.5 倍**。

**用法**：
添加 `--speculative-algorithm EAGLE`。其他参数如 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 是可选的。例如：
```
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --speculative-algorithm EAGLE \
  --trust-remote-code \
  --tp 8
```
- DeepSeek 模型的默认配置为 `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。可使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 脚本为给定批量大小搜索最佳配置。最小配置为 `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`，可以在较大批量大小时实现加速。
- 大多数 MLA 注意力后端完全支持 MTP 使用。详见 [MLA 后端](../advanced_features/attention_backend.md#mla-backends)。

```{note}
要为大批量大小（>48）启用 DeepSeek MTP，需要调整一些参数（参考[此讨论](https://github.com/sgl-project/sglang/issues/4543#issuecomment-2737413756)）：
- 将 `--max-running-requests` 调整为更大的数值。MTP 的默认值为 `48`。对于更大的批量大小，应将此值增加到超过默认值。
- 设置 `--cuda-graph-bs`。这是 CUDA Graph 捕获的批量大小列表。[投机解码的默认捕获批量大小](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L888-L895)为 48。你可以包含更多批量大小来自定义此参数。
```

```{tip}
要启用 EAGLE 投机解码的实验性重叠调度器，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。这可以通过启用草稿和验证阶段之间的重叠调度来提高性能。
```

### DeepSeek R1 和 V3.1 的推理内容

参见[推理解析器](https://docs.sglang.io/advanced_features/separate_reasoning.html)和 [DeepSeek V3.1 的 Thinking 参数](https://docs.sglang.io/basic_usage/openai_api_completions.html#Example:-DeepSeek-V3-Models)。

### DeepSeek 模型的函数调用

添加参数 `--tool-call-parser deepseekv3` 和 `--chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja`（推荐）来启用此功能。例如（在 1 个 H20 节点上运行）：

```
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3-0324 \
  --tp 8 \
  --port 30000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.9 \
  --tool-call-parser deepseekv3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv3.jinja
```

请求示例：

```
curl "http://127.0.0.1:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{"temperature": 0, "max_tokens": 100, "model": "deepseek-ai/DeepSeek-V3-0324", "tools": [{"type": "function", "function": {"name": "query_weather", "description": "获取城市天气，用户需要先提供城市名", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名，例如北京"}}, "required": ["city"]}}}], "messages": [{"role": "user", "content": "今天青岛天气怎么样"}]}'
```

```{important}
1. 使用较低的 `"temperature"` 值以获得更好的结果。
2. 为获得更一致的工具调用结果，建议使用 `--chat-template examples/chat_template/tool_chat_template_deepseekv3.jinja`。它提供了改进的统一提示词。
```

### DeepSeek R1 的思考预算

在 SGLang 中，我们可以使用 `CustomLogitProcessor` 实现思考预算。

启动服务器时启用 `--enable-custom-logit-processor` 标志。

```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.9 --disable-cuda-graph --reasoning-parser deepseek-r1 --enable-custom-logit-processor
```

请求示例：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "user",
            "content": "问题：巴黎是法国的首都吗？",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```

## 常见问题

**问：模型加载时间过长，遇到 NCCL 超时怎么办？**

答：如果遇到模型加载时间过长和 NCCL 超时，可以尝试增加超时时间。启动模型时添加参数 `--dist-timeout 3600`。这将超时设置为一小时，通常可以解决此问题。
