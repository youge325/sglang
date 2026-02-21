# 多模态生成性能分析

本指南介绍了 SGLang 中多模态生成流水线的性能分析技术。

## PyTorch Profiler

PyTorch Profiler 提供详细的内核执行时间、调用栈和 GPU 利用率指标。

### 去噪阶段性能分析

对去噪阶段进行采样时间步的性能分析（默认：1 个预热步骤后分析 5 个步骤）：

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile
```

**参数：**
- `--profile`：启用去噪阶段的性能分析
- `--num-profiled-timesteps N`：预热后要分析的时间步数（默认：5）
  - 较小的值可减少 trace 文件大小
  - 示例：`--num-profiled-timesteps 10` 在 1 个预热步骤后分析 10 个步骤

### 完整流水线性能分析

分析所有流水线阶段（文本编码、去噪、VAE 解码等）：

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile \
  --profile-all-stages
```

**参数：**
- `--profile-all-stages`：与 `--profile` 配合使用，分析所有流水线阶段而非仅去噪阶段

### 输出位置

默认情况下，trace 文件保存在 ./logs/ 目录中。

确切的输出文件路径将在控制台输出中显示，例如：

```bash
[mm-dd hh:mm:ss] Saved profiler traces to: /sgl-workspace/sglang/logs/mocked_fake_id_for_offline_generate-5_steps-global-rank0.trace.json.gz
```

### 查看 Trace 文件

在以下位置加载和可视化 trace 文件：
- https://ui.perfetto.dev/ （推荐）
- chrome://tracing（仅限 Chrome）

对于大型 trace 文件，请减小 `--num-profiled-timesteps` 或避免使用 `--profile-all-stages`。


### `--perf-dump-path`（阶段/步骤计时导出）

除了 profiler trace 外，你还可以导出一个轻量级 JSON 报告，其中包含：
- 完整流水线的阶段级计时分解
- 去噪阶段的步骤级计时分解（每个扩散步骤）

这对于快速识别哪个阶段主导端到端延迟，以及去噪步骤是否具有均匀的运行时间（如果不均匀，哪个步骤存在异常峰值）非常有用。

导出的 JSON 包含一个 `denoise_steps_ms` 字段，格式为对象数组，每个对象包含一个 `step` 键（步骤索引）和一个 `duration_ms` 键。

示例：

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "<PROMPT>" \
  --perf-dump-path perf.json
```

## Nsight Systems

Nsight Systems 提供低级 CUDA 性能分析，包括内核详情、寄存器使用和内存访问模式。

### 安装

安装说明请参阅 [SGLang 性能分析指南](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md#profile-with-nsight)。

### 基本性能分析

分析整个流水线执行：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o QwenImage \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

### 定向阶段性能分析

使用 `--delay` 和 `--duration` 捕获特定阶段并减小文件大小：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o QwenImage_denoising \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

**参数：**
- `--delay N`：开始捕获前等待 N 秒（跳过初始化开销）
- `--duration N`：捕获 N 秒（聚焦特定阶段）
- `--force-overwrite`：覆盖已有的输出文件

## 注意事项

- **减小 trace 大小**：使用较小的 `--num-profiled-timesteps` 值或在 Nsight Systems 中使用 `--delay`/`--duration`
- **特定阶段分析**：单独使用 `--profile` 分析去噪阶段，添加 `--profile-all-stages` 分析完整流水线
- **多次运行**：使用不同的提示和分辨率进行性能分析，以识别不同工作负载下的瓶颈

## 常见问题

- 如果你使用 Nsight Systems 对 `sglang generate` 进行性能分析，发现生成的 profiler 文件未捕获任何 CUDA 内核，可以通过增加模型的推理步数来延长执行时间以解决此问题。
