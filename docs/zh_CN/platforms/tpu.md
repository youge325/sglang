# TPU

SGLang 通过 SGLang-JAX 后端支持高性能 TPU 推理，该后端专门针对 Google Cloud TPU 进行了优化。基于 JAX 的实现为 TPU 硬件上的大语言模型（LLM）服务工作负载提供了卓越的吞吐量和低延迟。

如需报告 TPU 相关问题或功能请求，请访问 [sglang-jax GitHub issues 页面](https://github.com/sgl-project/sglang-jax/issues)。

**注意：** SGLang TPU 支持通过 SGLang-JAX 后端实现，这是一个专用的基于 JAX 的推理引擎，作为独立仓库维护在 [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax)。

## 系统要求

### 支持的 TPU 硬件

| TPU 类型 | HBM 内存 | 可用性 |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 每核 96 GB | Google Cloud |

### 软件要求

- **Python：** 3.12 或更高版本
- **JAX：** 支持 TPU 的最新版本
- **环境：** Google Cloud TPU VM 或兼容的 TPU 运行时
- **可选：** SkyPilot，用于简化云部署

## 功能支持矩阵

SGLang-JAX 为生产环境 LLM 服务提供全面的 TPU 优化功能：

| 功能 | 支持状态 | 描述 |
|---------|---------------|-------------|
| 高吞吐量连续批处理 | ✅ | 动态请求批处理，最大化 TPU 利用率 |
| Radix Tree KV 缓存 | ✅ | 请求间前缀共享的内存高效方案 |
| FlashAttention 后端 | ✅ | TPU 优化的注意力内核，支持长序列 |
| 张量并行 | ✅ | 将模型分布到多个 TPU 核心 |
| 分页注意力 | ✅ | 灵活的 KV 缓存分页管理 |
| 投机解码（EAGLE/EAGLE3） | ✅ | 兼容模型吞吐量提升 20-40% |
| 分块预填充 | ✅ | 混合预填充-解码批处理 |
| OpenAI 兼容 API | ✅ | OpenAI API 的直接替代 |
| 数据并行注意力 | 🚧 | 开发中 - 数据并行的注意力计算 |
| 量化 | 🚧 | 开发中 - 模型量化以减少内存使用 |
| 多 LoRA | 🚧 | 开发中 - 同时服务多个 LoRA 适配器 |

### 注意力后端比较

| 后端 | 分页注意力 | 投机解码 | MLA | 滑动窗口 |
|---------|----------------|---------------|-----|----------------|
| FlashAttention (fa) | ✅ | ✅ | ❌ | ✅ |
| Native | ❌ | ❌ | ❌ | ❌ |

**注意：** 建议在生产工作负载中使用 FlashAttention 后端，因为它具有更优的内存效率和性能。

## 已优化模型列表

以下模型已经过 TPU 部署的测试和优化：

| 模型系列 | 性能状态 |
|--------------|-------------------|
| [Qwen 3](https://huggingface.co/Qwen) | ⭐ 推荐用于生产环境 |
| [Qwen 3 MoE](https://huggingface.co/Qwen) | ⭐ 最佳性能 |
| [Qwen 2](https://huggingface.co/Qwen) | 待改进 |
| [Qwen 2 MoE](https://huggingface.co/Qwen) | 待改进 |
| [Qwen 1.5](https://huggingface.co/Qwen) | 待改进 |
| [Llama/LLaMA](https://huggingface.co/meta-llama) | 待改进 |
| [Grok-2](https://huggingface.co/xai-org) | 待改进 |
| [Gemma 2](https://huggingface.co/google) | 已在 TPU 上验证 |
| Bailing MoE | 待改进 |

## 安装

### 方法一：使用 PyPI（推荐）

```bash
pip install sglang-jax
```

### 方法二：从源码安装

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### 方法三：使用 Docker

**注意：** TPU 的 Docker 支持目前正在开发中。请使用 PyPI 或源码安装方式。

### 方法四：使用 SkyPilot 部署到 Cloud TPU

[SkyPilot](https://github.com/skypilot-org/skypilot) 提供了在 Google Cloud TPU 上的简化部署方案：

1. 安装 SkyPilot 并配置 GCP 访问权限（参见 [SkyPilot 文档](https://skypilot.readthedocs.io/)）

2. 创建 SkyPilot 配置文件：

<details>
<summary>SkyPilot YAML：<code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e

run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

3. 启动您的 TPU 集群：

```bash
# 标准部署
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp

# 使用竞价实例以节省成本
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp --use-spot
```

## 启动服务引擎

### 基础示例：Qwen-7B

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

**关键参数说明：**

1. `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` - 启用 JIT 编译缓存，加速后续服务器启动
2. `--tp-size=4` - 张量并行大小；将其与您的 TPU 核心数匹配（通常为 1、4 或 8）
3. `--device=tpu` - 指定 TPU 设备（sglang-jax 的默认设置）
4. `--dtype=bfloat16` - 使用 bfloat16 精度，TPU 特别针对此精度进行了优化
5. `--mem-fraction-static=0.8` - 将 80% 的 TPU HBM 分配给静态内存（可在 0.2 到 0.9 之间调整）
6. `--max-prefill-tokens=8192` - 预填充阶段处理的最大 token 数

### 高性能配置：Qwen3-8B

用于生产工作负载的最佳吞吐量配置：

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --tp-size=4 \
    --device=tpu \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --dtype=bfloat16 \
    --max-running-requests=256 \
    --page-size=128 \
    --attention-backend=fa
```

### 进阶：投机解码（EAGLE3）

投机解码可以为兼容模型提升 20-40% 的吞吐量：

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-32B \
    --trust-remote-code \
    --device=tpu \
    --tp-size=4 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=4096 \
    --attention-backend=fa \
    --dtype=bfloat16 \
    --port=30000 \
    --host=0.0.0.0 \
    --disable-overlap-schedule \
    --speculative-algorithm=EAGLE3 \
    --speculative-draft-model-path=AngelSlim/Qwen3-32B_eagle3 \
    --page-size=64 \
    --speculative-eagle-topk=1 \
    --speculative-num-steps=3 \
    --speculative-num-draft-tokens=4
```

**注意：** 投机解码目前支持 Qwen3 和 LLaMA 模型系列。详细配置指南请参阅[投机解码文档](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)。


### 多节点分布式服务

对于需要多个 TPU VM 的大型模型：

```bash
# 节点 0（协调器）
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=0 \
    --tp-size=8 \
    [其他参数...]

# 节点 1（工作节点）
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=1 \
    --tp-size=8 \
    [其他参数...]
```

## 请求基准测试

### 吞吐量测试

基础吞吐量基准测试：

```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts=100 \
    --random-input=512 \
    --random-output=128 \
    --max-concurrency=8 \
    --random-range-ratio=1 \
    --warmup-requests=0
```

### 延迟测试

测量单批次延迟：

```bash
python3 -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path Qwen/Qwen-7B-Chat \
    --batch-size=32 \
    --input-len=256 \
    --output-len=32
```

### 综合基准测试脚本

用于在不同配置下进行系统性能评估：

```bash
#!/bin/bash
set -e

backend=${1:-sgl-jax}
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
    for output_seq_len in "${output_seq_lens[@]}"; do
        echo "======================================="
        echo "Testing ISL/OSL: $input_seq_len/$output_seq_len"
        echo "======================================="
        for max_concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((num_prompts_per_concurrency * max_concurrency))
            python3 -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input ${input_seq_len} \
                --random-output ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

查看所有基准测试参数的详细帮助：

```bash
python3 -m sgl_jax.bench_serving --help
```

参阅[基准测试和性能分析指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)了解高级基准测试技术和 JAX Profiler 性能分析。

## 性能优化

### 内存优化

**减少内存使用：**
- 降低 `--mem-fraction-static`（从 0.8 → 0.5 → 0.3）
- 减少 `--max-prefill-tokens`（从 16384 → 8192 → 4096）
- 减少 `--max-running-requests`

**处理 OOM 错误：**
- 从保守的内存设置开始（`--mem-fraction-static=0.5`）
- 逐步增加直到找到最佳平衡点
- 增大 `--page-size` 以获得更好的内存局部性（1 → 16 → 64 → 128）

### 吞吐量优化

要最大化每秒 token 数：

- 使用 FlashAttention 后端：`--attention-backend=fa`
- 为 Qwen3 模型启用投机解码（EAGLE3）（提升 20-40%）
- 将 `--max-running-requests` 增加到 256+
- 将 `--mem-fraction-static` 设置为 0.8+（如果内存允许）
- 使用更大的页面大小（64-128）
- 启用分块预填充：`--chunked-prefill-size=2048`

### 延迟优化

要最小化首 token 生成时间（TTFT）和 token 间延迟：

- 将 `--page-size` 减小到 1-4
- 降低 `--max-running-requests`（16-32）以减小批次大小
- 减小 `--chunked-prefill-size`
- 使用保守的内存设置以避免 GC 暂停

### TPU 专项优化

1. **JIT 编译缓存：**
   ```bash
   export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
   ```
   始终设置此环境变量以缓存编译后的内核并加速服务器启动。

2. **数据类型优化：**
   使用 `--dtype=bfloat16` 以获得 TPU 原生优化。TPU 专门为 bfloat16 计算而设计。

3. **张量并行：**
   将 `--tp-size` 与您的 TPU 核心配置（1、4 或 8）匹配，以实现最优的模型分布。

4. **注意力后端：**
   生产工作负载请始终使用 `--attention-backend=fa`（FlashAttention）。

## 故障排除

### OOM（内存不足）错误

如果遇到内存不足错误：

1. 将 `--mem-fraction-static` 从 0.8 降低到 0.5 或更低
2. 将 `--max-prefill-tokens` 从 8192 降低到 4096 或 2048
3. 降低 `--max-running-requests` 以减少并发批次大小
4. 增大 `--page-size` 以获得更好的内存布局效率

### 编译时间过长

如果服务器启动时间过长：

1. 确保正确设置了 `JAX_COMPILATION_CACHE_DIR`
2. 首次运行需要 JIT 编译（这是正常的）
3. 使用缓存编译后，后续运行将显著加快
4. 考虑使用 `--skip-server-warmup` 将编译推迟到首次请求时

### 吞吐量低

如果未达到预期吞吐量：

1. 验证 `--tp-size` 是否与您的 TPU 核心配置匹配
2. 检查是否启用了 `--attention-backend=fa`
3. 增加 `--max-running-requests` 以支持更大的批次组建
4. 考虑为兼容模型启用投机解码
5. 确保内存设置允许足够的批次大小

### 连接问题

如果客户端无法连接到服务器：

1. 确保使用 `--host=0.0.0.0` 用于外部访问（而非仅 `127.0.0.1`）
2. 验证防火墙规则允许指定端口的流量（默认：30000）
3. 检查服务器进程是否正在运行：`curl http://localhost:30000/health`

## 高级功能

### 投机解码

SGLang-JAX 支持 EAGLE 和 EAGLE3 投机解码算法，适用于 Qwen3 和 LLaMA 模型系列。投机解码可以在不影响输出质量的情况下提升 20-40% 的吞吐量。

详细配置和支持的模型组合请参阅[投机解码文档](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)。

### 分块预填充

启用混合预填充-解码批处理以提高 TPU 利用率：

```bash
--chunked-prefill-size=2048 --enable-mixed-chunk
```

这允许调度器在同一批次中混合预填充和解码操作，从而提高整体吞吐量。

### 自定义注意力后端

SGLang-JAX 支持基于插件的注意力后端系统。您可以实现针对特定用例优化的自定义注意力内核。

实现详情请参阅[注意力后端文档](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/attention_backend.md)。

### 环境验证

在部署前验证您的 TPU 设置：

```bash
python -c "from sgl_jax import check_env; check_env.check_env()"
```

此命令将检查：
- 已安装的包版本
- TPU 设备可用性和规格
- 系统资源和配置
- 设置的兼容性

## 贡献

我们欢迎为改善 SGLang-JAX 的 TPU 支持做出贡献！

### 贡献方向

**查看[开发路线图](https://github.com/sgl-project/sglang-jax/issues/190)** 了解计划中的功能，寻找贡献新功能的机会。

当前贡献方向包括：

- 针对特定 TPU 代际的性能优化
- 支持更多模型架构
- 文档改进和示例
- 错误报告和修复
- 基准测试结果和性能分析

### 如何贡献

1. 访问 [sglang-jax 仓库](https://github.com/sgl-project/sglang-jax)
2. 阅读[贡献指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
3. 加入 [SGL-JAX Slack 社区](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X) 参与讨论
4. 在 [sglang-jax/issues](https://github.com/sgl-project/sglang-jax/issues) 报告问题

### 在 TPU 上测试

需要 TPU 访问权限进行测试的贡献者：

- 参阅 [TPU 资源指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/tpu_resources_guide.md) 了解如何获取 TPU 硬件访问
- 使用 SkyPilot 配合竞价实例进行低成本测试
- 遵循[基准测试和性能分析指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) 进行性能验证

## 参考资料

### 文档

- [SGLang-JAX 仓库](https://github.com/sgl-project/sglang-jax)
- [SGLang-JAX 安装指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Qwen 模型快速入门](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [基准测试和性能分析指南](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [投机解码](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)

### 外部资源

- [JAX 文档](https://jax.readthedocs.io/)
- [Google Cloud TPU 文档](https://cloud.google.com/tpu/docs)
- [SkyPilot 文档](https://skypilot.readthedocs.io/)
