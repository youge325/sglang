# 专家并行

SGLang 中的专家并行 (EP) 将混合专家 (MoE) 模型中的专家权重分布在多个设备上，解决内存瓶颈并实现高性能推理的高效扩展。它对于服务大规模 MoE 模型特别重要，在这些模型中 token 动态路由到跨 GPU 的专门专家。通过利用优化的 all-to-all 通信和分组矩阵乘法 (GEMM)，EP 降低了延迟、提高了吞吐量并最小化了 GPU 空闲时间。SGLang 的 EP 通过其模块化框架提供了强大的可扩展性，允许无缝集成自定义内核、后端和优化，无需重构核心逻辑。

## 支持的后端和选择指南

SGLang 的 EP 集成了多种高效后端，适用于不同场景，允许对性能权衡进行细粒度控制。用户通过命令行标志指定后端：
- `--moe-a2a-backend`：选择 all-to-all 通信后端。
- `--moe-runner-backend`：选择 MoE 计算后端。

### All-to-All 通信后端

| 后端 | 描述 | 使用场景 |
|------|------|----------|
| **`none`（默认）** | 禁用 EP 的 all-to-all。使用 All-Reduce 或 All-Gather 进行 token 分发。 | 混合 EP 和 TP 设置。 |
| `deepep` | DeepEP，用于 MoE 模型中高效 token 混洗的通信库。 | 大规模 EP 部署。 |
| `mooncake` | DeepEP 的扩展，利用 RDMA 进行弹性推理和高性能数据传输。 | 弹性 EP 服务。 |
| `mori` | MORI-EP，AMD 针对 ROCm 优化的原生 all-to-all 通信实现。 | AMD GPU 部署。 |
| `flashinfer` | Flashinfer 的 all-to-all 实现。 | 大规模 EP 部署。 |
| `ascend_fuseep` | 昇腾 NPU 原生融合 all-to-all 通信。 | 昇腾 NPU 部署。 |

DeepEP 和 Mooncake 后端支持两种 token 分发模式：`normal` 模式（针对预填充工作负载优化，高吞吐）和 `low_latency` 模式（针对解码工作负载优化，低延迟且兼容 CUDA Graph）。推荐设置 `--deepep-mode auto` 以在运行时自动切换分发模式。

目前，DeepEP、Mooncake、`ascend_fuseep` 和 MORI 仅支持 `ep_size = tp_size` 的情况。对于混合 EP 和 TP（即 `ep_size < tp_size`），目前仅支持 `none` 后端。

### MoE 计算后端

| 后端 | 描述 | 使用场景 |
|------|------|----------|
| **`auto`（默认）** | 根据模型架构、硬件、量化方案和运行时条件自动选择最优后端。 | 通用部署。 |
| `triton` | 基于 Triton 的分组 GEMM 实现。 | 自定义内核开发。 |
| `deep_gemm` | DeepGEMM 后端，针对 MoE 矩阵乘法优化。 | 大规模 FP8 块级量化 EP 部署。 |
| `cutlass` | 基于 CUTLASS 的高效 GEMM。 | 支持 CUTLASS 的 NVIDIA 架构。 |
| `flashinfer_trtllm` | FlashInfer 与 TensorRT-LLM 集成。 | Blackwell + TRT-LLM。 |
| `flashinfer_cutlass` | FlashInfer 与 CUTLASS 组合。 | Blackwell + FP4/FP8 模型。 |

### 启动示例

使用 DeepEP 和 DeepGEMM 启动 DeepSeek-V3：

```bash
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --moe-a2a-backend deepep --moe-runner-backend deep_gemm --tp 8 --ep 8
```

## 可扩展 EP 框架

SGLang 的 EP 框架提供了模块化抽象，便于集成自定义内核、后端和优化。它将 MoE 前向传递解耦为多个阶段（分发 → 前置排列 → 核心运行器 → 后置排列 → 合并），使得扩展无需重构核心逻辑。

### 框架概览

框架以 `FusedMoE` 作为统一入口点。关键组件包括：
- **Dispatcher**：管理 DeepEP 等后端的分发/合并（实现 `BaseDispatcher` 子类）。
- **MoeRunner**：通过 `MoeRunnerCore` 实现编排分组 GEMM 执行。
- **PermuteMethodPool**：自动注册布局转换。
- **TopK Router**：后端无关的专家选择。

```{note}
更多细节请参阅 [MoE 重构路线图](https://github.com/sgl-project/sglang/issues/8715) 和 [英文文档](../../en/advanced_features/expert_parallelism.html)。
```
