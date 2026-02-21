# 环境变量

SGLang 支持多种环境变量来配置其运行时行为。本文档提供了完整的列表，并会持续更新。

*注意：SGLang 使用两种环境变量前缀：`SGL_` 和 `SGLANG_`。这可能是历史原因。虽然目前两者都支持不同的设置，但未来版本可能会统一它们。*

## 通用配置

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `SGLANG_USE_MODELSCOPE` | 启用使用 ModelScope 的模型 | `false` |
| `SGLANG_HOST_IP` | 服务器主机 IP 地址 | `0.0.0.0` |
| `SGLANG_PORT` | 服务器端口 | 自动检测 |
| `SGLANG_LOGGING_CONFIG_PATH` | 自定义日志配置路径 | 未设置 |
| `SGLANG_DISABLE_REQUEST_LOGGING` | 禁用请求日志 | `false` |
| `SGLANG_HEALTH_CHECK_TIMEOUT` | 健康检查超时时间（秒） | `20` |
| `SGLANG_FORWARD_UNKNOWN_TOOLS` | 转发未知工具调用给客户端而非丢弃 | `false` |
| `SGLANG_REQ_WAITING_TIMEOUT` | 请求在队列中等待调度的超时时间（秒） | `-1` |
| `SGLANG_REQ_RUNNING_TIMEOUT` | 请求在解码批次中运行的超时时间（秒） | `-1` |

## 性能调优

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | 控制是否使用 torch.inference_mode | `false` |
| `SGLANG_ENABLE_TORCH_COMPILE` | 启用 torch.compile | `true` |
| `SGLANG_SET_CPU_AFFINITY` | 启用 CPU 亲和性设置 | `0` |
| `SGLANG_IS_FLASHINFER_AVAILABLE` | 控制 FlashInfer 可用性检查 | `true` |
| `SGLANG_SKIP_P2P_CHECK` | 跳过 P2P（点对点）访问检查 | `false` |
| `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` | 设置启用分块前缀缓存的阈值 | `8192` |

## DeepGEMM 配置（高级优化）

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `SGLANG_ENABLE_JIT_DEEPGEMM` | 启用 DeepGEMM 内核的 JIT 编译 | `"true"` |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | 启用 DeepGEMM 内核预编译 | `"true"` |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | 并行 DeepGEMM 内核编译的工作线程数 | `4` |
| `SGLANG_DG_CACHE_DIR` | 编译后 DeepGEMM 内核的缓存目录 | `~/.cache/deep_gemm` |

## DeepEP 配置

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `SGLANG_DEEPEP_BF16_DISPATCH` | 使用 Bfloat16 进行分发 | `"false"` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 每个 GPU 上分发的最大 token 数 | `"128"` |

## 内存管理

| 环境变量 | 描述 | 默认值 |
|----------|------|--------|
| `SGLANG_DEBUG_MEMORY_POOL` | 启用内存池调试 | `false` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | 为内存规划裁剪最大新 token 估计 | `4096` |

```{note}
这是环境变量文档的精简中文版本。完整的环境变量列表请参阅 [英文文档](../../en/references/environment_variables.html)。
```
