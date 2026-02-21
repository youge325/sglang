# 推测解码

SGLang 提供多种推测解码选项，包括 EAGLE-2/EAGLE-3、MTP、经典草稿模型解码和基于 NGRAM 的变体。我们的实现旨在最大化速度和效率，被认为是开源 LLM 引擎中最快的之一。

## 总览

### 跳转到各节

- [EAGLE 解码](#eagle-decoding)
  - [EAGLE-2 解码](#eagle-2-decoding)
  - [EAGLE-2 使用 torch.compile 解码](#eagle-2-decoding-with-torchcompile)
  - [EAGLE-2 频率排序推测采样解码](#eagle-2-decoding-via-frequency-ranked-speculative-sampling)
  - [EAGLE-3 解码](#eagle-3-decoding)
- [多 Token 预测](#multi-token-prediction)
- [独立推测解码（小型草稿模型）](#standalone-speculative-decoding-small-draft-model)
- [推测解码 V2（重叠调度器）](#speculative-decoding-v2-overlap-scheduler)
- [Ngram 推测解码](#ngram-speculative-decoding)
- [完整参数参考](#full-parameter-reference)
- [OOM 故障排除](#oom-troubleshooting)
- [参考文献](#references)

### 快速指南

- **最佳速度/质量（推荐）**：使用 **EAGLE-3** 配合 `--speculative-algorithm EAGLE3`。
- **强默认 / 广泛兼容**：使用 **EAGLE-2** 配合 `--speculative-algorithm EAGLE`。
- **降低 EAGLE-2 的 `lm_head` 开销**：启用 **FR-Spec** 配合 `--speculative-token-map`。
- **模型支持 MTP**：使用**通过推测解码的 MTP**（通常使用较小的 `speculative_num_steps/topk/num_draft_tokens`，请参阅示例部分）。
- **有较小的草稿 LLM**：使用 **STANDALONE**（`--speculative-algorithm STANDALONE`）。
- **无额外可用模型**：使用 **NGRAM**（`--speculative-algorithm NGRAM`，仅 CUDA）。
- **希望使用重叠调度器（实验性）**：启用 **SpecV2** 配合 `SGLANG_ENABLE_SPEC_V2=True`（需要 `--speculative-eagle-topk 1`）。

### 方法比较（简表）

| 方法 | 草稿来源 | 需要单独的草稿模型？ | 启用方式 | 注意事项/约束 |
|---|---|---:|---|---|
| EAGLE-2 | EAGLE 草稿模型（特征草拟 + 树） | 通常是 | `--speculative-algorithm EAGLE` + `--speculative-draft-model-path ...` | 调优 `--speculative-num-steps`、`--speculative-eagle-topk`、`--speculative-num-draft-tokens` |
| EAGLE-2 + `torch.compile` | 同 EAGLE-2 | 通常是 | 添加 `--enable-torch-compile`（可选 `--torch-compile-max-bs`） | 进一步的内核级优化 |
| EAGLE-2 + FR-Spec | 同 EAGLE-2 + Token 子集 | 通常是 | 添加 `--speculative-token-map ...` | 通过高频 Token 词表降低 `lm_head` 开销 |
| EAGLE-3 | EAGLE3 草稿模型 | 是 | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path ...` | 上述基准测试中最佳吞吐量 |
| MTP | 内置多 Token 预测头（模型特定） | 通常不需要 | 参见**多 Token 预测**部分 | 使用推测工作流；某些模型的草稿路径可能自动处理 |
| STANDALONE | 较小的草稿 LLM（Token 级） | 是 | `--speculative-algorithm STANDALONE` + `--speculative-draft-model-path ...` | **不支持** `--enable-dp-attention` |
| SpecV2（实验性） | V2 worker + 重叠调度器 | 不适用 | `SGLANG_ENABLE_SPEC_V2=True` | 仅支持 `--speculative-eagle-topk 1`；适用于 `EAGLE`、`EAGLE3`、`STANDALONE` |
| NGRAM | 从先前 Token 构建的 Ngram 缓存 | 否 | `--speculative-algorithm NGRAM` | 仅 CUDA；不支持 `--enable-dp-attention`；禁用重叠调度器和混合分块预填充 |

### 性能亮点

请参阅下方通过 EAGLE3 解码在 MT bench 上测试 LLaMA-Instruct 3.1 8B 所实现的巨大吞吐量提升。
更多详情请参阅 [EAGLE3 论文](https://arxiv.org/pdf/2503.01840)。

| 方法 | 吞吐量 (tokens/s) |
|--------|----------------|
| SGLang（无推测，1x H100） | 158.34 tokens/s |
| SGLang + EAGLE-2（1x H100） | 244.10 tokens/s |
| SGLang + EAGLE-3（1x H100） | 373.25 tokens/s |

---

## EAGLE 解码

要启用 EAGLE 推测解码，以下参数是相关的：

| 参数 | 描述 | 默认值 |
|---|---|---|
| `--speculative-draft-model-path` | 草稿模型路径/权重。EAGLE/EAGLE3 和 STANDALONE **通常需要**此参数。对于某些支持 MTP 的模型，可以省略。 | `None` |
| `--speculative-num-steps` | 自回归草拟深度。增加推测范围但有拒绝级联的风险。 | 自动（Llama/Grok 为 `5`；其他多数模型为 `3`） |
| `--speculative-eagle-topk` | 每步分支因子。提高候选多样性和接受率，但增加内存/计算消耗。 | 自动（Llama/Grok 为 `4`；其他多数模型为 `1`） |
| `--speculative-num-draft-tokens` | 最大并行验证容量。允许更深的树评估但增加 GPU 内存使用。 | 自动（Llama/Grok 为 `8`；其他多数模型为 `4`）。如果 `topk=1`，则调整为 `num_steps + 1`。 |
| `--speculative-accept-threshold-single` | 单 Token 验证的接受阈值。较低的值接受更积极。 | `1.0` |
| `--speculative-accept-threshold-acc` | 跨步累积接受阈值。 | `1.0` |
| `--speculative-attention-mode` | 推测操作的注意力模式（`prefill` 或 `decode`），影响目标验证和草稿扩展。 | `"prefill"` |
| `--speculative-draft-attention-backend` | 覆盖草稿模型的注意力后端。 | `None`（与目标相同） |
| `--speculative-draft-model-quantization` | 草稿模型的量化方法。使用 `"unquant"` 可以在目标模型量化时强制草稿模型不量化。 | 与目标模型相同 |
| `--speculative-draft-model-revision` | 要加载的草稿模型的特定版本/提交。 | `None`（当设置了 `--speculative-draft-model-path` 且省略版本时自动设为 `"main"`） |
| `--speculative-draft-load-format` | 草稿模型权重的加载格式。 | `None` |

这些参数对于 EAGLE-2 和 EAGLE-3 基本相同。EAGLE-3 模型会忽略 `--speculative-token-map`。
对于 `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens`：全部不设置以使用自动调优，或者在调优时全部显式设置。

您可以使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 找到这些参数的最佳组合。


### EAGLE-2 解码

您可以通过设置 `--speculative-algorithm EAGLE` 并选择适当的模型来启用 EAGLE-2 解码。

**启动服务器：**

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 使用 `torch.compile` 解码

您还可以启用 `torch.compile` 以进行进一步优化，并可选地设置 `--torch-compile-max-bs`：

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --enable-torch-compile \
    --torch-compile-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 频率排序推测采样解码

通过在草稿模型中使用截断的高频 Token 词表，EAGLE 推测解码减少了 `lm_head` 的计算开销，同时在不降低质量的情况下加速了流水线。更多详情请查阅[论文](https://arxiv.org/pdf/2502.14856)。

在我们的实现中，设置 `--speculative-token-map` 以启用该优化。您可以从[此模型](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec)获取 FR-Spec 中的高频 Token。或者您可以直接从[此仓库](https://github.com/thunlp/FR-Spec/tree/main?tab=readme-ov-file#prepare-fr-spec-vocabulary-subset)下载这些 Token。

感谢 [Weilin Zhao](https://github.com/Achazwl) 和 [Zhousx](https://github.com/Zhou-sx) 的贡献。

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-3 解码

您可以通过设置 `--speculative-algorithm EAGLE3` 并选择适当的模型来启用 EAGLE-3 解码。

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 多 Token 预测

我们通过推测解码在 SGLang 中支持 [MTP（多 Token 预测）](https://arxiv.org/pdf/2404.19737)。这里以 `XiaomiMiMo/MiMo-7B-RL` 为例（关于 DeepSeek MTP 的用法，请参阅 [deepseek_v32 文档](../basic_usage/deepseek_v32.md#multi-token-prediction)）。

```bash
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --host 0.0.0.0 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "XiaomiMiMo/MiMo-7B-RL",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 独立推测解码（小型草稿模型）

除了 EAGLE/MTP，SGLang 还支持使用较小的**草稿模型**进行 **Token 级推测解码**。通过 `--speculative-algorithm STANDALONE` 启用，并通过 `--speculative-draft-model-path` 提供草稿模型。

相关参数：

| 参数 | 描述 | 默认值 |
|---|---|---|
| `--speculative-draft-model-path` | 草稿模型权重（小于目标模型）。 | `None` |
| `--speculative-num-steps` | 草拟深度（草稿模型自回归运行的步数）。 | `3`（STANDALONE 的自动默认值） |
| `--speculative-eagle-topk` | 分支因子（每步的 Token 候选数）。 | `1`（STANDALONE 的自动默认值） |
| `--speculative-num-draft-tokens` | 验证容量。 | `4`（STANDALONE 的自动默认值） |
| `--speculative-draft-model-quantization` | 草稿模型的量化。使用 `"unquant"` 可以在目标模型量化时禁用草稿模型的量化。 | 与目标相同 |

> **注意：** 独立推测解码目前**不支持** `--enable-dp-attention`。

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 推测解码 V2（重叠调度器）

SGLang 提供了**实验性的推测解码 V2** 实现，启用重叠调度器并使用 V2 推测 worker（例如 `StandaloneWorkerV2`、`EAGLEWorkerV2`）。

要启用它，设置环境变量：
- `SGLANG_ENABLE_SPEC_V2=True`

注意事项：
- SpecV2 目前仅支持 `--speculative-eagle-topk 1`。启用 SpecV2 时，请**显式设置 `--speculative-eagle-topk 1`**。
- 如果您显式设置 `--speculative-eagle-topk > 1`，服务器将报错。
- 如果您省略 `--speculative-eagle-topk`，自动调优可能会为某些模型（例如 Llama）选择 `topk > 1`。这与 SpecV2 不兼容，可能不会始终触发即时配置错误，因此请显式设置 `--speculative-eagle-topk 1`。
- 此功能适用于 `EAGLE`、`EAGLE3` 和 `STANDALONE`。

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Ngram 推测解码

SGLang 还支持**基于 Ngram 的推测解码**（无需单独的草稿模型）。它从先前生成的 Token 构建的 Ngram 缓存中检索草稿 Token，然后使用目标模型进行验证。

启用方式：
- `--speculative-algorithm NGRAM`

### Ngram 特定参数

| 参数 | 描述 | 默认值 |
|---|---|---|
| `--speculative-num-draft-tokens` | 每步验证的草稿 Token 数量。如果省略，默认为 `--speculative-ngram-max-match-window-size`。 | `12`（使用默认 Ngram 设置） |
| `--speculative-ngram-min-match-window-size` | 最小匹配窗口大小。 | `1` |
| `--speculative-ngram-max-match-window-size` | 最大匹配窗口大小。 | `12` |
| `--speculative-ngram-min-bfs-breadth` | 最小 BFS 宽度。 | `1` |
| `--speculative-ngram-max-bfs-breadth` | 最大 BFS 宽度。 | `10` |
| `--speculative-ngram-match-type` | 匹配类型：`"BFS"` 或 `"PROB"`。 | `"BFS"` |
| `--speculative-ngram-branch-length` | 插入缓存的最近 Token 数量。 | `18` |
| `--speculative-ngram-capacity` | 缓存容量（条目数）。 | `10,000,000` |

注意事项：
- Ngram 推测解码**仅支持 CUDA**。
- 目前**不支持** `--enable-dp-attention`。
- 它会禁用重叠调度器和混合分块预填充。
- 如果 `--speculative-ngram-max-bfs-breadth > 1`（因此 `speculative_eagle_topk > 1`）且 `page_size > 1`，请使用 `--attention-backend flashinfer`；否则服务器将报错。
- 可选：设置 `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` 以强制贪婪验证。

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-match-window-size 12 \
    --speculative-ngram-max-bfs-breadth 10 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**发送请求：**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 完整参数参考

以下是 SGLang 中所有推测解码参数的完整列表：

### 核心参数

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--speculative-algorithm` | `str` | `None` | 使用的算法：`EAGLE`、`EAGLE3`、`STANDALONE`、`NGRAM`、`NEXTN`（`EAGLE` 的别名） |
| `--speculative-draft-model-path` | `str` | `None` | 草稿模型权重的路径 |
| `--speculative-draft-model-revision` | `str` | `None` | 草稿模型的特定版本/提交（当设置了草稿路径且省略版本时自动使用 `"main"`） |
| `--speculative-draft-load-format` | `str` | `None` | 草稿模型权重的加载格式 |
| `--speculative-num-steps` | `int` | `None`（省略时自动选择） | 自回归草拟深度 |
| `--speculative-eagle-topk` | `int` | `None`（省略时自动选择） | 每个草拟步的分支因子 |
| `--speculative-num-draft-tokens` | `int` | `None`（省略时自动选择） | 用于验证的最大草稿 Token 数量 |
| `--speculative-accept-threshold-single` | `float` | `1.0` | 单 Token 接受阈值 |
| `--speculative-accept-threshold-acc` | `float` | `1.0` | 累积接受阈值 |
| `--speculative-token-map` | `str` | `None` | FR-Spec 高频 Token 映射路径 |
| `--speculative-attention-mode` | `str` | `"prefill"` | 推测操作的注意力模式（`"prefill"` 或 `"decode"`） |
| `--speculative-draft-attention-backend` | `str` | `None` | 覆盖草稿模型的注意力后端 |
| `--speculative-moe-runner-backend` | `str` | `None` | 草稿模型的 MoE runner 后端 |
| `--speculative-moe-a2a-backend` | `str` | `None` | 草稿模型的 MoE all-to-all 后端 |
| `--speculative-draft-model-quantization` | `str` | 与目标相同 | 草稿模型的量化（`"unquant"` 禁用） |

### Ngram 特定参数

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `--speculative-ngram-min-match-window-size` | `int` | `1` | 最小 Ngram 匹配窗口 |
| `--speculative-ngram-max-match-window-size` | `int` | `12` | 最大 Ngram 匹配窗口 |
| `--speculative-ngram-min-bfs-breadth` | `int` | `1` | 最小 BFS 宽度 |
| `--speculative-ngram-max-bfs-breadth` | `int` | `10` | 最大 BFS 宽度 |
| `--speculative-ngram-match-type` | `str` | `"BFS"` | 匹配类型：`"BFS"` 或 `"PROB"` |
| `--speculative-ngram-branch-length` | `int` | `18` | 插入缓存的最近 Token 数 |
| `--speculative-ngram-capacity` | `int` | `10,000,000` | 缓存容量 |

### 环境变量

| 变量 | 默认值 | 描述 |
|---|---|---|
| `SGLANG_ENABLE_SPEC_V2` | `False` | 启用推测解码 V2（重叠调度器） |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `False` | 对 Ngram 解码强制贪婪验证 |

### 其他相关标志

| 参数 | 描述 |
|---|---|
| `--enable-multi-layer-eagle` | 启用多层 EAGLE（MiMoV2 和 Step3p5 模型自动启用） |
| `--enable-torch-compile` | 启用 `torch.compile` 进行内核级优化 |
| `--torch-compile-max-bs` | `torch.compile` 的最大批大小 |

---

## OOM 故障排除

> [!WARNING]
> **内存不足（OOM）？** 推测解码可能会增加 GPU 内存使用，因为草稿树、CUDA 图和验证相关缓冲区会消耗额外的显存。如果遇到 OOM 错误，请尝试以下调整。

### 第 1 步：减小草稿树大小（最有效）

这三个参数直接控制草稿树消耗的内存量：

```bash
# 调整前（激进，高内存）
--speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64

# 调整后（保守，低内存）
--speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16
```

- **`--speculative-num-draft-tokens`**：这是影响最大的单一参数。从 64 减少到 16 可以将草稿相关内存减少约 75%。从这里开始。
- **`--speculative-eagle-topk`**：从 8 减少到 4 甚至 2 可以将分支因子减半。
- **`--speculative-num-steps`**：从 5 减少到 3 可以缩短草拟深度。

### 第 2 步：降低静态内存比例

```bash
# 为动态分配（CUDA 图、草稿模型等）留出更多空间
--mem-fraction-static 0.5   # 省略时此值会自动计算
```

### 第 3 步：减小 CUDA 图批大小

```bash
# 更少的 CUDA 图捕获 = 更少的预留内存
--cuda-graph-max-bs 4   # 在内存紧张的情况下甚至可以设为 2
```

### 第 4 步：限制并发请求数

```bash
# 更少的并发请求降低运行中的负载，可以减少 OOM 风险
--max-running-requests 4
```

### 第 5 步：使用量化

```bash
# 量化目标模型（如果您的检查点/硬件支持）
--quantization fp8

# 或仅量化草稿模型（保持目标为全精度）
--speculative-draft-model-quantization fp8
```

### 第 6 步：使用较小的数据类型

```bash
--dtype float16   # 代替 bfloat16/float32（在支持的情况下）
```

### 第 7 步：使用 FR-Spec 减少 lm_head 内存（EAGLE-2 / STANDALONE）

```bash
--speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt
```
> 注意：对于 EAGLE-3，`--speculative-token-map` 会被忽略，因为 EAGLE-3 模型已内置热门 Token 处理。

### 快速 OOM 恢复方案

如果您遇到 OOM 且只想快速恢复正常运行，请从以下最小配置开始并逐步扩大：

```bash
python3 -m sglang.launch_server \
    --model <your-model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <your-draft-model> \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 2 \
    --mem-fraction-static 0.5 \
    --max-running-requests 4 \
    --dtype float16 \
    --log-level warning
```

然后逐步增加 `--speculative-num-draft-tokens`、`--speculative-eagle-topk` 和 `--cuda-graph-max-bs`，直到找到适合您 GPU 的最佳配置。

> [!TIP]
> **内存预算经验法则**：在自动估算 `--mem-fraction-static` 时，STANDALONE 预留约 6 GB，EAGLE/EAGLE3 预留约 2 GB 作为额外余量。请相应地规划您的 `--mem-fraction-static`。

---

## 参考文献

EAGLE 流程如下：

- 在 EAGLE 中，草稿模型使用特征序列 $(f_1, ..., f_k)$ 和 Token 序列 $(t_2, ..., t_{k+1})$ 来预测下一个特征向量，即原始 LLM 的最后一个隐藏状态。
- 然后从 $p_{k+2}=\text{LMHead}(f_{k+1})$ 中采样下一个 Token。之后，两个序列以树状方式扩展——分支出多个可能的延续，每步的分支因子由 `speculative_eagle_topk` 参数控制——以确保上下文的连贯性，并再次作为输入。
- 在 SGLang 的 EAGLE-2 实现中，草稿树在配置的步数内扩展，然后重新排序以选择前 `speculative_num_draft_tokens` 个最终节点作为草稿 Token。
- EAGLE-3 移除了特征预测目标，融入了低层和中层特征，并以在线策略方式进行训练。

通过在特征而非 Token 上操作以获得更规则的输入，以及额外传递下一个时间步的 Token 以减少采样随机性，这提高了草拟精度。更多详情请参阅 [EAGLE-2](https://arxiv.org/abs/2406.16858) 和 [EAGLE-3](https://arxiv.org/abs/2503.01840) 论文。

关于如何训练自己的 EAGLE 模型的指南，请参阅 [EAGLE 仓库](https://github.com/SafeAILab/EAGLE/tree/main?tab=readme-ov-file#train)。
