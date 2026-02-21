# 确定性推理 (Deterministic Inference)

## 为什么确定性推理很重要

确定性推理确保 LLM 在多次运行中产生一致的输出，这对以下场景至关重要：
- **强化学习**：确保多次运行中 logprobs 的一致性，减少随机噪声，使强化学习训练更加稳定、可复现和可调试。
- **测试与调试**：实现可复现的验证
- **生产环境**：提高可靠性和用户体验

即使设置 `temperature=0`，由于动态批处理和 GPU 内核中不同的规约顺序，标准 LLM 推理仍可能产生不同的输出。

## 非确定性的根本原因

主要来源是**不同的批量大小**。不同的批量大小会导致 GPU 内核以不同方式拆分规约操作，从而产生不同的加法顺序。由于浮点数的非结合性（`(a + b) + c ≠ a + (b + c)`），即使对于相同的输入也会产生不同的结果。


## SGLang 的解决方案

基于 [Thinking Machines Lab 的批量不变算子](https://github.com/thinking-machines-lab/batch_invariant_ops)，SGLang 实现了完全确定性的推理，同时保持与分块预填充（chunked prefill）、CUDA Graph、基数缓存（radix cache）和非贪心采样的兼容性。确定性推理功能的开发路线图可在此 [issue](https://github.com/sgl-project/sglang/issues/10278) 中找到。

### 支持的后端

确定性推理仅支持以下三种注意力后端：**FlashInfer**、**FlashAttention 3 (FA3)** 和 **Triton**。

下表展示了不同注意力后端上确定性推理的功能兼容性：

| Attention Backend | CUDA Graph | Chunked Prefill | Radix Cache | Non-greedy Sampling (Temp > 0) |
|-------------------|------------|-----------------|-------------|---------------------|
| **FlashInfer** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **FlashAttention 3 (FA3)** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Triton** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

## 使用方法

### 基本用法

通过添加 `--enable-deterministic-inference` 标志启用确定性推理：

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### 服务器参数

| Argument | Type/Default | Description |
|----------|--------------|-------------|
| `--enable-deterministic-inference` | flag; default: disabled | 启用使用批量不变操作的确定性推理 |
| `--attention-backend` | string; default: fa3 | 选择注意力后端（flashinfer、fa3 或 triton） |

### 示例配置

#### Qwen3-8B
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

#### Llama 模型
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

#### Qwen3-30B-A3B（MoE 模型）
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### 非贪心采样（Temperature > 0）下的确定性推理

SGLang 通过使用采样种子（sampling seeds）支持非贪心采样下的确定性推理。这对于强化学习场景（如 GRPO，即组相对策略优化）特别有用，在这些场景中您需要多个多样但可复现的响应。

#### 默认行为

默认情况下，SGLang 使用采样种子 `42` 以实现可复现的采样：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a joke",
        "sampling_params": {
            "temperature": 0.8,  # 非贪心采样
            "max_new_tokens": 128,
        },
    },
)
print(response.json())
# 这将在多次运行中始终产生相同的响应
```

#### 生成多个可复现的响应

要从同一提示采样不同的响应同时保持可复现性（例如用于 GRPO 训练），请在请求中提供不同的采样种子：

```python
import requests

# 为不同响应准备采样种子列表
sampling_seeds = [42, 43, 44, 45, 46]

responses = []
for seed in sampling_seeds:
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "Tell me a joke",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 128,
                "sampling_seed": seed,  # 指定采样种子
            },
        },
    )
    responses.append(response.json())

# 每个种子会产生不同但可复现的响应
# 使用相同的种子将始终产生相同的响应
```

这种方法确保了：
- 不同的种子产生多样化的响应
- 相同的种子在不同运行中始终产生相同的响应
- 结果可用于调试和评估的复现


## 验证

运行确定性测试以验证输出的一致性：

```bash
# 单一测试：相同提示，不同批量大小
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# 前缀测试：具有不同前缀长度的提示
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# 基数缓存一致性模式：测试基数缓存的确定性（缓存与未缓存的预填充）
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

预期结果：所有测试应显示 `Unique samples: 1`（完全确定性）。
