# Cache-DiT 加速

SGLang 集成了 [Cache-DiT](https://github.com/vipshop/cache-dit)，一个用于 Diffusion Transformer (DiT) 的缓存加速引擎，可实现最高 **1.69 倍推理加速**且质量损失最小。

## 概述

**Cache-DiT** 使用智能缓存策略来跳过去噪循环中的冗余计算：

- **DBCache（双块缓存）**：基于残差差异动态决定何时缓存 transformer 块
- **TaylorSeer**：使用 Taylor 展开进行校准，优化缓存决策
- **SCM（步级计算掩码）**：步级缓存控制，提供额外加速

## 基本用法

通过导出环境变量并使用 `sglang generate` 或 `sglang serve` 启用 Cache-DiT：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

## Diffusers 后端配置

Cache-DiT 支持从自定义 YAML 文件加载加速配置。对于 diffusers 流水线，通过 `--cache-dit-config` 传递 YAML/JSON 路径。此流程需要 cache-dit >= 1.2.0（`cache_dit.load_configs`）。

### 单 GPU 推理

定义一个包含以下内容的 `config.yaml` 文件：

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
```

然后应用配置：

```bash
sglang generate --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config config.yaml \
  --prompt "A beautiful sunset over the mountains"
```

### 分布式推理

定义一个包含以下内容的 `parallel_config.yaml` 文件：

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```

`ulysses_size: auto` 表示 Cache-DiT 将自动检测 world_size。否则，设置为特定整数（例如 `4`）。

然后应用分布式配置：

```bash
sglang generate --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config parallel_config.yaml \
  --prompt "A futuristic cityscape at sunset"
```

## 高级配置

### DBCache 参数

DBCache 控制块级缓存行为：

| 参数 | 环境变量              | 默认值 | 描述                              |
|-----------|---------------------------|---------|------------------------------------------|
| Fn        | `SGLANG_CACHE_DIT_FN`     | 1       | 始终计算的前几个块的数量 |
| Bn        | `SGLANG_CACHE_DIT_BN`     | 0       | 始终计算的后几个块的数量  |
| W         | `SGLANG_CACHE_DIT_WARMUP` | 4       | 缓存开始前的预热步数       |
| R         | `SGLANG_CACHE_DIT_RDT`    | 0.24    | 残差差异阈值            |
| MC        | `SGLANG_CACHE_DIT_MC`     | 3       | 最大连续缓存步数          |

### TaylorSeer 配置

TaylorSeer 使用 Taylor 展开提高缓存准确性：

| 参数 | 环境变量                  | 默认值 | 描述                     |
|-----------|-------------------------------|---------|---------------------------------|
| 启用    | `SGLANG_CACHE_DIT_TAYLORSEER` | false   | 启用 TaylorSeer 校准器    |
| 阶数     | `SGLANG_CACHE_DIT_TS_ORDER`   | 1       | Taylor 展开阶数（1 或 2） |

### 组合配置示例

DBCache 和 TaylorSeer 是互补的策略，可以协同工作，你可以同时配置两组参数：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A curious raccoon in a forest"
```

### SCM（步级计算掩码）

SCM 提供步级缓存控制以获得额外加速。它决定哪些去噪步骤需要完整计算，哪些可以使用缓存结果。

**SCM 预设**

SCM 通过预设进行配置：

| 预设   | 计算比例 | 速度    | 质量    |
|----------|---------------|----------|------------|
| `none`   | 100%          | 基线 | 最佳       |
| `slow`   | ~75%          | ~1.3x    | 高       |
| `medium` | ~50%          | ~2x      | 良好       |
| `fast`   | ~35%          | ~3x      | 可接受 |
| `ultra`  | ~25%          | ~4x      | 较低      |

**用法**

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**自定义 SCM 分箱**

对于哪些步骤计算、哪些步骤缓存的精细控制：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_COMPUTE_BINS="8,3,3,2,2" \
SGLANG_CACHE_DIT_SCM_CACHE_BINS="1,2,2,2,3" \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**SCM 策略**

| 策略    | 环境变量                          | 描述                                 |
|-----------|---------------------------------------|---------------------------------------------|
| `dynamic` | `SGLANG_CACHE_DIT_SCM_POLICY=dynamic` | 基于内容的自适应缓存（默认） |
| `static`  | `SGLANG_CACHE_DIT_SCM_POLICY=static`  | 固定缓存模式                       |

## 环境变量

所有 Cache-DiT 参数都可以通过环境变量进行配置。
完整列表请参阅[环境变量](../../environment_variables.md)。

## 支持的模型

SGLang Diffusion × Cache-DiT 支持 SGLang Diffusion 原本支持的几乎所有模型：

| 模型系列 | 示例模型              |
|--------------|-----------------------------|
| Wan          | Wan2.1、Wan2.2              |
| Flux         | FLUX.1-dev、FLUX.2-dev      |
| Z-Image      | Z-Image-Turbo               |
| Qwen         | Qwen-Image、Qwen-Image-Edit |
| Hunyuan      | HunyuanVideo                |

## 性能提示

1. **从默认值开始**：默认参数对大多数模型效果良好
2. **使用 TaylorSeer**：它通常能同时提升速度和质量
3. **调整 R 阈值**：较低的值 = 更好的质量，较高的值 = 更快速度
4. **SCM 获取额外加速**：使用 `medium` 预设可获得良好的速度/质量平衡
5. **预热很重要**：更高的预热值 = 更稳定的缓存决策

## 限制

- **SGLang 原生流水线**：分布式支持（TP/SP）尚未验证；当 `world_size > 1` 时 Cache-DiT 将自动禁用。
- **SCM 最低步数**：SCM 需要 >= 8 个推理步骤才能有效
- **模型支持**：仅支持在 Cache-DiT 的 BlockAdapterRegister 中注册的模型

## 故障排除

### 分布式环境警告

```
WARNING: cache-dit is disabled in distributed environment (world_size=N)
```

这是预期行为。Cache-DiT 目前仅支持单 GPU 推理。

### 低步数时 SCM 被禁用

对于推理步数 < 8 的模型（例如 DMD 蒸馏模型），SCM 将自动禁用。DBCache 加速仍然有效。

## 参考链接

- [Cache-Dit](https://github.com/vipshop/cache-dit)
- [SGLang Diffusion](../index.md)
