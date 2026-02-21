# DP、DPA 和 SGLang DP 路由器

本指南介绍数据并行 (Data Parallelism, DP) 和数据并行注意力 (Data Parallelism Attention, DPA) 的区别、如何正确启用每种模式，以及如何使用 SGLang 模型网关 (SGLang Model Gateway, SMG) 进行生产级 DP 部署。

## 数据并行 (DP)

**数据并行 (Data Parallelism, DP)** 是最常见的并行策略，它将整个模型复制到多组 GPU 上，并行处理不同批次的请求。每组 GPU 处理独立的请求。结合专用的路由策略，正如我们稍后将介绍的 SGLang 模型网关中的合适路由算法，您的推理系统吞吐量可以几乎线性增长。

### 关键特性

- 每个副本拥有模型的完整副本
- 请求被分发/分散到各个副本
- 单个请求推理期间无副本间通信（对于简单 DP）

## 数据并行注意力 (DPA)

**数据并行注意力 (Data Parallelism Attention, DPA)**，也称为 DP Attention，是一种高级并行策略。虽然 DPA 对 **多头潜在注意力 (Multi-Head Latent Attention, MLA)** 模型（如 DeepSeek、MiniMax、Kimi-K2）提供最显著的收益，但它也支持 **标准注意力模型**（如 Qwen）。

### 张量并行对 MLA 模型的问题

最常见的推理并行策略是**张量并行 (Tensor Parallelism, TP)**。然而，TP 对于某些模型可能不是最高效的策略。例如，DeepSeek 模型使用 MLA 且只有**一个 KV 头**。如果我们在 8 个 GPU 上使用张量并行，将导致：

- 所有 GPU 上的 **KV 缓存重复**
- **不必要的内存使用**，限制了批量大小
- 由于内存限制导致**吞吐量降低**

### DPA 的工作原理

DPA 通过**专门对注意力组件应用数据并行**来解决这些限制。

<table>
<tr>
<td width="50%">
<img src="../_static/image/dpa.png" alt="DPA + EP Architecture" width="100%">
</td>
<td width="50%" valign="top">

**每个 DP 副本：**

- 独立处理不同批次（可以处于不同的前向模式：预填充、解码或空闲）
- 维护自己的 KV 缓存（无重复）
- 由于内存节省，可支持显著更大的批量大小

**DPA + EP 中的通信模式：**
-
- **All2All（Dispatch）**：根据门控决策将 token 路由到专家子组
- **All2All（Combine）**：将专家计算的结果收集回原始 token 位置

</td>
</tr>
</table>

### DPA 的关键优势

1. **显著减少 KV 缓存内存**：每个 DP 副本只存储其自身批次的 KV 缓存
2. **更大的批量大小**：内存节省使得更大的批量大小成为可能
3. **改善解码吞吐量**：对基于 MLA 的模型有显著的吞吐量提升
4. **独立的前向模式**：每个 DP 副本可以处于不同的前向模式（预填充、解码或空闲），并在注意力计算期间独立处理其分配的批次

### DPA 配合 MoE 的专家并行

对于像 DeepSeek 这样的 MoE 模型，DPA **通常**与专家并行 (Expert Parallelism, EP) 搭配使用以获得最佳吞吐量。然而，**DPA 不要求 EP**：如果您的部署不需要专家分片，可以在不启用 EP 的情况下启用 DPA。

- 将 256+ 个专家权重分布在各 GPU 上（无法放在单个 GPU 上）
- 通过 DeepEP 实现高效的 all-to-all token 路由
- 扩展到大型集群（相比普通 TP 最高可提升 5 倍吞吐量）

### DeepSeek 推荐配置

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --ep 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

> **注意**：使用 `--enable-dp-attention` 时必须显式设置 `--dp-size`。如果 `dp_size` 为 1（默认值），DPA 将被禁用。

有关详细的 EP 配置（DeepEP、双批次重叠、EPLB），请参阅 [专家并行](expert_parallelism.md)。

### 目标模型

DPA 支持以下模型架构：

- **MLA（多头潜在注意力）模型** — DPA 可提供最显著的收益：
  - DeepSeek 系列（DeepSeek-V2、DeepSeek-V3、DeepSeek-R1）
  - MiniMax 模型
  - Kimi-K2
  - 其他使用 MLA 架构的模型

- **标准注意力模型** — 同样支持：
  - Qwen 模型（参见 [PR #6121](https://github.com/sgl-project/sglang/pull/6121)）

对于使用标准 GQA 的 Llama 等模型，通常推荐使用标准 DP 或 TP。

要启用 DPA，请在服务器启动命令中添加 `--enable-dp-attention`。

### 激活逻辑

DPA 通过服务器参数（CLI 或配置）显式启用。您必须同时设置 `--dp-size` 和 `--enable-dp-attention`：

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp-size 8 \
    --enable-dp-attention
```

**重要说明**：`--dp-size` 必须大于 1 才能使 DPA 生效。当 `dp_size == 1`（默认值）时，`--enable-dp-attention` 会自动禁用。还必须满足 `tp_size % dp_size == 0` 的约束条件。

### MLA 模型的标准 DP

请注意，MLA 模型当然也支持 DP。假设您想为 MLA 模型启用标准 DP。首先，独立启动每个 MLA 模型的副本。您可以逐个启动这些启用了 DPA 的副本。启动全部 MLA 模型副本后，启动一个 SMG 并将所有副本连接到 SMG。以下是 SMG 的详细说明。

## 现代数据并行 SGLang 模型网关 (SMG)

### 原生 DP 模式

SGLang 中的原生 DP（内置数据并行）在单个 SGLang 实例中创建多个工作进程，由 `DataParallelController` 通过启动参数 `dp-size` 控制。


```bash
# 原生 DP 模式
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

**局限性：**

- 仅内置进程内负载均衡（如 `round_robin`、`total_requests`、`total_tokens`）
- 无缓存感知路由
- 有限的可观测性和指标
- 无容错或熔断器
- 不适合生产工作负载

⚠️ 原生 DP **目前强烈不推荐使用**。它仅用于一些旧版/过时的强化学习框架。您可以使用 SGLang 模型网关 (SMG) 来增强任何使用场景中的数据并行。

### 基于 SMG 的 DP（推荐）

从 2024 年 9 月开始，SGLang 模型网关（即 SMG，前身为 SGLang DP Router）专门作为基于 Rust 的生产级 DP 路由系统构建。它从 DP 路由开始，但后来我们进一步扩展了其范围以协调强化学习、PD 分离和其他场景。本文档仅讨论 SMG 在 DP 路由中的用法。其他用途请参阅 [SGLang 模型网关文档](sgl_model_gateway.md)。

> 为了实现最佳的生产级路由性能并将开销降至最低，我们使用 Rust 构建 SMG，而非 Python，因为 Python 永远不够快。

**我们强烈建议使用 SGLang 模型网关 (SMG) 进行生产级数据并行。** SMG 相比原生 DP 模式具有显著优势。

```bash
# 基于 SMG 的 DP 模式（推荐）
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4
```

⚠️ 请注意，**SMG 和原生 DP 共享相同的启动参数 `--dp-size`**。但原生 DP 的入口点是 `python -m sglang.launch_server`，而 SMG 的入口点是 `python -m sglang_router.launch_server`。

**基于 SMG 的 DP 优势：**

| Feature | Native DP | SMG-Based DP |
|---------|-----------|--------------|
| **Load Balancing** | 内置进程内方法 | 高级策略（缓存感知、二选一等） |
| **Cache Awareness** | ❌ 无 | ✅ 有 - 显著提高缓存命中率 |
| **Throughput** | 基线 | 显著提升 |
| **Multi-Node Support** | 有限 | ✅ 完全支持 |
| **Worker Health Monitoring** | 基础 | ✅ 熔断器、健康检查 |
| **Reliability** | 基础 | ✅ 重试、限流、排队 |
| **Observability** | 基础指标 | ✅ 40+ Prometheus 指标、OpenTelemetry |
| **Hot Worker Add/Remove** | ❌ 无 | ✅ 有 |

### SMG 的性能

SMG 中的缓存感知路由策略对具有共享前缀的工作负载显著提升了性能：

| Metric | Without Cache-Aware | With Cache-Aware SMG |
|--------|---------------------|----------------------|
| Throughput (token/s) | 82,665 | 158,596 (+92%) |
| Cache Hit Rate | 20% | 75% (+275%) |

*基准测试来自 [SGLang v0.4 博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)，使用多个长前缀组的工作负载，8x A100 80GB GPU，dp-size=8*

### 何时使用各种模式

**使用原生 DP 的场景：**

- ~永远不要使用原生 DP~
- DP 路由的学习材料

**使用基于 SMG 的 DP 的场景：**

- 在任何您认为需要 DP 的场景
- 生产部署
- 多节点分布式部署
- 具有共享前缀的工作负载（高缓存复用潜力）
- 需要高可用性和可靠性功能
- 需要详细的可观测性和指标
- 需要高效的强化学习 rollout 系统

请注意，对于强化学习 rollout 系统，**有四个关键原因使得基于 SMG 的 DP 远优于原生 DP 路由**。详情请参阅 [强化学习中的负载均衡路由器](./sglang_for_rl.md#load-balancing-router)。

### SMG 快速入门

**安装**

```bash
pip install sglang-router
# 或
pip install "sglang[all]"
```

**选项 A：同时启动 Worker 和 SMG（最简单）**

这是最简单的入门方式 - SMG 和 worker 一起启动：

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

**选项 B：分离启动（多节点）**

用于跨多台机器的分布式部署：

1. 在每个节点上启动 worker

```bash
# 节点 1
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000

# 节点 2
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000
```

2. 启动 SMG 并指向 worker

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:8000 http://node2:8000 \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000
```

**选项 C：动态 Worker 注册**

用于可动态添加/移除 worker 的弹性部署：

```bash
# 先启动 SMG
python -m sglang_router.launch_router \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 30000

# 动态注册 worker
curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker1:8000"}'

curl -X POST http://localhost:30000/workers \
    -H "Content-Type: application/json" \
    -d '{"url": "http://worker2:8000"}'
```

### 负载均衡策略

SMG 支持多种负载均衡策略：

| Policy | Description | Best For |
|--------|-------------|----------|
| `cache_aware` | 结合缓存局部性与负载均衡 | **推荐用于大多数工作负载** |
| `round_robin` | 按顺序循环分配 worker | 简单、可预测的分配 |
| `random` | 随机选择 worker | 基线测试 |
| `power_of_two` | 采样两个 worker，选择负载较轻的 | 低延迟需求 |

**缓存感知策略（默认，推荐）**

缓存感知策略为大多数工作负载提供最佳性能：

```bash
python -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 120 \
    --max-tree-size 67108864
```

**工作原理：**

1. 根据请求历史为每个 worker 维护一个近似基数树
2. 将请求路由到前缀匹配（缓存命中）最高的 worker
3. 当负载不均衡时回退到最短队列路由
4. 自动驱逐旧条目以防止内存溢出

### 最佳实践

1. **从 `cache_aware` 策略开始** - 对大多数工作负载而言，它在缓存局部性和负载分配之间提供了最佳平衡
2. **在生产环境使用 SMG** - 优先使用 `sglang_router.launch_server` 而非 `sglang.launch_server`，以获得更好的可靠性和可观测性
3. **启用健康检查** - 配置 `--router-health-check-interval-secs` 以自动检测和移除不健康的 worker

**应用最佳实践的推荐命令：**

```bash
python -m sglang_router.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dp-size 4 \
    --router-policy cache_aware \
    --router-health-check-interval-secs 30 \
    --router-prometheus-port 10001 \
    --host 0.0.0.0 \
    --port 30000
```

有关高级配置（熔断器、重试、Prometheus 指标、K8s 集成），请参阅 [SGLang 模型网关文档](sgl_model_gateway.md)。

### 验证流量分配

启动 SMG 后，验证流量是否正确分配：

**1. 检查 worker 状态：**

```bash
curl http://localhost:30000/workers
```

**2. 检查负载分配：**

```bash
curl http://localhost:30000/get_loads
```

**3. 监控指标（如果启用了 Prometheus）：**

```bash
# 需要关注的关键指标
smg_router_requests_total{model="..."}
smg_worker_requests_active{worker="..."}
sglang_cache_hit_rate{source="..."}
```

有关详细的指标和监控配置，请参阅 [SGLang 模型网关文档](sgl_model_gateway.md)。

## 参考

| Strategy | Use Case | Key Benefit |
|----------|----------|-------------|
| **Native DP** (`--dp-size`) | 从不使用 | 易于理解，非基于 Rust |
| **SMG-Based DP** | **生产环境（推荐）** | 缓存感知路由、高可用性 |
| **DPA** (`--dp-size N --enable-dp-attention`) | DeepSeek/MLA 模型 | 消除 KV 缓存重复、提升吞吐量 |
| **DPA + EP** | DeepSeek MoE 模型 | 相比普通 TP 显著提升吞吐量 |

**DeepSeek 推荐的生产配置：**
1. 为注意力层启用 **DPA**（`--dp-size 8 --enable-dp-attention`）
2. 为 MoE 层启用 **EP**（`--ep 8 --moe-a2a-backend deepep`）
3. 使用 **SMG** 配合 **cache_aware** 策略

**相关文档：**
- [专家并行](expert_parallelism.md) - DeepEP、双批次重叠、EPLB
- [SGLang 模型网关文档](sgl_model_gateway.md) - SMG 配置与故障排除
- [大规模 EP 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - 96 GPU 部署指南
