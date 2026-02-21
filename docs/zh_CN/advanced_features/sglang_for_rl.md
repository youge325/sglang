# SGLang 用于强化学习系统

本文档是基础设施团队将 SGLang 集成到 RL 和后训练系统的实践指南。它聚焦于循环中的运维痛点（rollout、评估、训练、权重同步），并将它们映射到具体的 SGLang API、标志和集成模式。重点是最大化 rollout 效率、准确性和稳定性。

## 为什么选择 SGLang 作为 RL 生命周期？

让我们遵循 DeepMind 早期 RL 工程的指导原则：

**做一个库，而不是一个框架。**

使用 SGLang 用于 RL 生命周期的五个理由：

* **细粒度引擎休眠与唤醒**：促进最大化 rollout 和训练能力
* **开放易用的 Refit 功能**：多种方法支持共置或分离部署
* **支持延迟生成**：启用部分 rollout 和专用 rollout 控制
* **确定性推理**：实现确定性推理以消除训练-推理不匹配
* **负载均衡路由**：缓存感知的负载均衡实现高吞吐量 rollout

## 细粒度引擎休眠与唤醒

Rollout 和训练都是内存密集型的，在同一 GPU 上共置它们通常导致内存压力和缓慢切换。SGLang 提供了内存感知的休眠/唤醒机制，释放 KV 缓存和权重同时保持服务器进程存活，然后在不完全重启的情况下恢复 rollout。

### 服务器标志

启动服务器时启用内存节省支持：

```
--enable-memory-saver
```

### 释放内存

**端点:** `POST /release_memory_occupation`

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `tags` | 要释放的内存区域。省略则全部释放。 | `None` | `kv_cache`, `weights` |

### 恢复内存

**端点:** `POST /resume_memory_occupation`

| 字段 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `tags` | 要恢复的内存区域。省略则全部恢复。 | `None` | `kv_cache`, `weights` |

## 开放易用的 Refit 功能

训练完成每步后，rollout 引擎必须使用新权重进行 refit。SGLang 支持三种 refit 策略：

- **从磁盘更新**：最简单，适合弹性 rollout 扩展和检查点。
- **从张量更新**：适合共置训练/rollout，可传递内存中的张量。
- **从分布式更新**：适合带有专用通信组（NCCL/IB）的分离训练/rollout。

### 从磁盘更新权重

**端点:** `POST /update_weights_from_disk`

| 字段 | 描述 | 默认值 |
| --- | --- | --- |
| `model_path` | 新权重的模型路径。 | 必填 |
| `load_format` | 加载权重的格式。 | `None` |
| `flush_cache` | 更新后刷新 KV 缓存。 | `True` |

### 从张量更新权重

**端点:** `POST /update_weights_from_tensor`

使用内存中的张量直接更新权重，无需磁盘 I/O。

### 从分布式更新权重

**端点:** `POST /update_weights_from_distributed`

使用 NCCL 通信在多个 GPU 实例间同步权重。

```{note}
更多详细的 API 参数和集成模式请参阅 [英文文档](../../en/advanced_features/sglang_for_rl.html)。
```
