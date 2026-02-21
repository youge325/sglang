# SGLang Model Gateway

SGLang Model Gateway 是一个面向大规模 LLM 部署的高性能模型路由网关。它集中管理 Worker 生命周期，在异构协议（HTTP、gRPC、OpenAI 兼容）间平衡流量，并为历史存储、MCP 工具调用和隐私敏感工作流提供企业级控制能力。该网关针对 SGLang 服务运行时进行了深度优化，但也可以路由到任何 OpenAI 兼容的后端。

---

## 目录

1. [概述](#概述)
2. [架构](#架构)
   - [控制平面](#控制平面)
   - [数据平面](#数据平面)
   - [存储与隐私](#存储与隐私)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [部署模式](#部署模式)
   - [协同启动路由器和 Worker](#协同启动路由器和-worker)
   - [独立启动（HTTP）](#独立启动http)
   - [gRPC 启动](#grpc-启动)
   - [预填充-解码分离](#预填充-解码分离)
   - [OpenAI 后端代理](#openai-后端代理)
   - [多模型推理网关](#多模型推理网关)
6. [API 参考](#api-参考)
   - [推理端点](#推理端点)
   - [分词端点](#分词端点)
   - [解析器端点](#解析器端点)
   - [分类 API](#分类-api)
   - [对话和响应 API](#对话和响应-api)
   - [Worker 管理 API](#worker-管理-api)
   - [管理和健康检查端点](#管理和健康检查端点)
7. [负载均衡策略](#负载均衡策略)
8. [可靠性与流量控制](#可靠性与流量控制)
   - [重试](#重试)
   - [熔断器](#熔断器)
   - [限流与排队](#限流与排队)
   - [健康检查](#健康检查)
9. [推理解析器集成](#推理解析器集成)
10. [工具调用解析](#工具调用解析)
11. [分词器管理](#分词器管理)
12. [MCP 集成](#mcp-集成)
13. [服务发现（Kubernetes）](#服务发现kubernetes)
14. [历史记录与数据连接器](#历史记录与数据连接器)
15. [WASM 中间件](#wasm-中间件)
16. [语言绑定](#语言绑定)
17. [安全与认证](#安全与认证)
    - [网关服务器 TLS (HTTPS)](#网关服务器-tls-https)
    - [Worker 通信 mTLS](#worker-通信-mtls)
18. [可观测性](#可观测性)
    - [Prometheus 指标](#prometheus-指标)
    - [OpenTelemetry 追踪](#opentelemetry-追踪)
    - [日志](#日志)
19. [生产环境建议](#生产环境建议)
    - [安全最佳实践](#安全最佳实践)
    - [高可用性](#高可用性)
    - [性能](#性能)
    - [Kubernetes 部署](#kubernetes-部署)
    - [使用 PromQL 进行监控](#使用-promql-进行监控)
20. [配置参考](#配置参考)
21. [故障排除](#故障排除)

---

## 概述

- **统一控制平面**：用于在异构模型集群中注册、监控和编排常规、预填充和解码 Worker。
- **多协议数据平面**：跨 HTTP、PD（预填充/解码）、gRPC 和 OpenAI 兼容后端路由流量，共享可靠性原语。
- **业界首创的 gRPC 管道**：具备原生 Rust 分词、推理解析器和工具调用执行能力，用于高吞吐量、OpenAI 兼容的服务；支持单阶段和 PD 拓扑。
- **推理网关模式（`--enable-igw`）**：动态实例化多个路由器栈（HTTP 常规/PD、gRPC），并为多租户部署应用逐模型策略。
- **对话与响应连接器**：在路由器内部集中管理聊天历史，使同一上下文可在不同模型和 MCP 循环中复用，而无需向上游供应商泄露数据（memory、none、Oracle ATP、PostgreSQL）。
- **企业隐私**：代理式多轮 `/v1/responses`、原生 MCP 客户端（STDIO/HTTP/SSE/Streamable）以及历史存储均在路由器边界内运行。
- **可靠性核心**：带抖动的重试、Worker 级熔断器、令牌桶限流与排队、后台健康检查和缓存感知负载监控。
- **全面可观测性**：40+ Prometheus 指标、OpenTelemetry 分布式追踪、结构化日志和请求 ID 传播。

---

## 架构

### 控制平面

- **Worker Manager** 发现能力（`/get_server_info`、`/get_model_info`），跟踪负载，并在共享注册表中注册/移除 Worker。
- **Job Queue** 序列化添加/移除请求，并暴露状态（`/workers/{worker_id}`），以便客户端追踪注册进度。
- **Load Monitor** 向缓存感知和 power-of-two 策略提供实时 Worker 负载统计数据。
- **Health Checker** 持续探测 Worker 并更新就绪状态、熔断器状态和路由器指标。
- **Tokenizer Registry** 管理动态注册的分词器，支持从 HuggingFace 或本地路径异步加载。

### 数据平面

- **HTTP 路由器**（常规和 PD）实现 `/generate`、`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`、`/v1/rerank`、`/v1/classify`、`/v1/tokenize`、`/v1/detokenize` 及相关管理端点。
- **gRPC 路由器** 将分词后的请求直接流式传输到 SRT gRPC Worker，完全在 Rust 中运行——分词器、推理解析器和工具解析器都在进程内。支持单阶段和 PD 路由，包括嵌入和分类。
- **OpenAI 路由器** 将 OpenAI 兼容端点代理到外部供应商（OpenAI、xAI 等），同时在本地保留聊天历史和多轮编排。

### 存储与隐私

- 对话和响应历史存储在路由器层（memory、none、Oracle ATP 或 PostgreSQL）。相同的历史记录可为多个模型或 MCP 循环提供支持，而无需向上游供应商发送数据。
- `/v1/responses` 代理式流程、MCP 会话和对话 API 共享同一存储层，支持受监管工作负载的合规需求。

---

## 安装

### Docker

预构建的 Docker 镜像已在 Docker Hub 上提供，支持多架构（x86_64 和 ARM64）：

```bash
docker pull lmsysorg/sgl-model-gateway:latest
```

### 前提条件

- **Rust 和 Cargo**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustc --version
  cargo --version
  ```
- **Python**：需要 `pip` 和 virtualenv 工具。

### Rust 二进制文件

```bash
cd sgl-model-gateway
cargo build --release
```

### Python 包

```bash
pip install maturin

# 快速开发模式
cd sgl-model-gateway/bindings/python
maturin develop

# 生产构建
maturin build --release --out dist --features vendored-openssl
pip install --force-reinstall dist/*.whl
```

---

## 快速开始

### 常规 HTTP 路由

```bash
# Rust 二进制
./target/release/sgl-model-gateway \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware

# Python 启动器
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware
```

### gRPC 路由

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

---

## 部署模式

### 协同启动路由器和 Worker

在一个进程中启动路由器和一组 SGLang Worker：

```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

包含路由器参数（以 `--router-` 为前缀）的完整示例：

```bash
python -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-tool-call-parser llama \
  --router-model-path meta-llama/Llama-3.1-8B-Instruct \
  --router-policy round_robin \
  --router-log-level debug
```

### 独立启动（HTTP）

独立运行 Worker 并将路由器指向其 HTTP 端点：

```bash
# Worker 节点
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001

# 路由器节点
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

### gRPC 启动

使用 SRT gRPC Worker 以解锁最高吞吐量并访问原生推理/工具管道：

```bash
# Worker 暴露 gRPC 端点
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# 路由器
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080
```

gRPC 路由器同时支持常规 HTTP 等效服务和 PD（预填充/解码）服务。当连接模式解析为 gRPC 时，需提供 `--tokenizer-path` 或 `--model-path`（HuggingFace ID 或本地目录）。

### 预填充-解码分离

将预填充和解码 Worker 分离，以实现 PD 感知的缓存和负载均衡：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

预填充条目接受可选的 bootstrap 端口。PD 模式将预填充元数据与解码输出合并，并将结果流式返回给客户端。

### OpenAI 后端代理

代理 OpenAI 兼容端点，同时在本地保留历史记录和 MCP 会话：

```bash
python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend memory
```

OpenAI 后端模式要求每个路由器实例恰好有一个 `--worker-urls` 条目。

### 多模型推理网关

启用 IGW 模式，通过单个路由器路由多个模型：

```bash
./target/release/sgl-model-gateway \
  --enable-igw \
  --policy cache_aware \
  --max-concurrent-requests 512

# 动态注册 Worker
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{
        "url": "http://worker-a:8000",
        "model_id": "mistral",
        "priority": 10,
        "labels": {"tier": "gold"}
      }'
```

---

## API 参考

### 推理端点

| Method | Path | 描述 |
|--------|------|------|
| `POST` | `/generate` | SGLang generate API |
| `POST` | `/v1/chat/completions` | OpenAI 兼容的聊天补全（流式/工具调用） |
| `POST` | `/v1/completions` | OpenAI 兼容的文本补全 |
| `POST` | `/v1/embeddings` | 嵌入向量生成（HTTP 和 gRPC） |
| `POST` | `/v1/rerank`, `/rerank` | 重排序请求 |
| `POST` | `/v1/classify` | 文本分类 |

### 分词端点

网关提供用于文本分词的 HTTP 端点，支持批处理，设计上与 SGLang Python 分词 API 保持一致。

| Method | Path | 描述 |
|--------|------|------|
| `POST` | `/v1/tokenize` | 将文本分词为 token ID（单条或批量） |
| `POST` | `/v1/detokenize` | 将 token ID 转换回文本（单条或批量） |
| `POST` | `/v1/tokenizers` | 注册新分词器（异步，返回任务状态） |
| `GET` | `/v1/tokenizers` | 列出所有已注册的分词器 |
| `GET` | `/v1/tokenizers/{id}` | 按 UUID 获取分词器信息 |
| `GET` | `/v1/tokenizers/{id}/status` | 检查异步分词器加载状态 |
| `DELETE` | `/v1/tokenizers/{id}` | 从注册表中移除分词器 |

#### 分词请求

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

#### 批量分词请求

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": ["Hello", "World", "How are you?"]
}
```

#### 分词响应

```json
{
  "tokens": [15339, 11, 1917, 0],
  "count": 4,
  "char_count": 13
}
```

#### 反分词请求

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [15339, 11, 1917, 0],
  "skip_special_tokens": true
}
```

#### 反分词响应

```json
{
  "text": "Hello, world!"
}
```

#### 添加分词器（异步）

```bash
curl -X POST http://localhost:30000/v1/tokenizers \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3", "source": "meta-llama/Llama-3.1-8B-Instruct"}'
```

响应：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Tokenizer registration queued"
}
```

检查状态：
```bash
curl http://localhost:30000/v1/tokenizers/550e8400-e29b-41d4-a716-446655440000/status
```

### 解析器端点

网关提供管理端点，用于从 LLM 输出中解析推理内容和函数调用。

| Method | Path | 描述 |
|--------|------|------|
| `POST` | `/parse/reasoning` | 将推理内容（`<think>`）与普通文本分离 |
| `POST` | `/parse/function_call` | 从文本中解析函数/工具调用 |

#### 推理分离请求

```json
{
  "text": "<think>Let me analyze this step by step...</think>The answer is 42.",
  "parser": "deepseek-r1"
}
```

#### 响应

```json
{
  "normal_text": "The answer is 42.",
  "reasoning_text": "Let me analyze this step by step..."
}
```

#### 函数调用解析

```json
{
  "text": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}",
  "parser": "json"
}
```

### 分类 API

`/v1/classify` 端点使用序列分类模型（例如 `Qwen2ForSequenceClassification`、`BertForSequenceClassification`）提供文本分类功能。

#### 请求

```bash
curl http://localhost:30000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "I love this product!"
  }'
```

#### 响应

```json
{
  "id": "classify-a1b2c3d4-5678-90ab-cdef-1234567890ab",
  "object": "list",
  "created": 1767034308,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.12, 0.88],
      "num_classes": 2
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 0,
    "total_tokens": 6
  }
}
```

#### 响应字段

| 字段 | 描述 |
|------|------|
| `label` | 预测的类别标签（来自模型的 `id2label` 配置，若无则回退为 `LABEL_N`） |
| `probs` | 所有类别的概率分布（logits 的 softmax） |
| `num_classes` | 分类类别数量 |

#### 注意事项

- 分类复用嵌入后端——调度器返回 logits，通过 softmax 转换为概率
- 标签来自模型的 HuggingFace 配置（`id2label` 字段）；没有此映射的模型使用通用标签（`LABEL_0`、`LABEL_1` 等）
- HTTP 和 gRPC 路由器均支持分类

### 对话和响应 API

| Method | Path | 描述 |
|--------|------|------|
| `POST` | `/v1/responses` | 创建后台响应（代理式循环） |
| `GET` | `/v1/responses/{id}` | 检索已存储的响应 |
| `POST` | `/v1/responses/{id}/cancel` | 取消后台响应 |
| `DELETE` | `/v1/responses/{id}` | 删除响应 |
| `GET` | `/v1/responses/{id}/input_items` | 列出响应输入项 |
| `POST` | `/v1/conversations` | 创建对话 |
| `GET` | `/v1/conversations/{id}` | 获取对话 |
| `POST` | `/v1/conversations/{id}` | 更新对话 |
| `DELETE` | `/v1/conversations/{id}` | 删除对话 |
| `GET` | `/v1/conversations/{id}/items` | 列出对话项 |
| `POST` | `/v1/conversations/{id}/items` | 向对话添加项 |
| `GET` | `/v1/conversations/{id}/items/{item_id}` | 获取对话项 |
| `DELETE` | `/v1/conversations/{id}/items/{item_id}` | 删除对话项 |

### Worker 管理 API

| Method | Path | 描述 |
|--------|------|------|
| `POST` | `/workers` | 排队 Worker 注册（返回 202 Accepted） |
| `GET` | `/workers` | 列出 Worker 及其健康状态、负载和策略元数据 |
| `GET` | `/workers/{worker_id}` | 检查特定 Worker 或任务队列条目 |
| `PUT` | `/workers/{worker_id}` | 排队 Worker 更新 |
| `DELETE` | `/workers/{worker_id}` | 排队 Worker 移除 |

#### 添加 Worker

```bash
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"grpc://0.0.0.0:31000","worker_type":"regular"}'
```

#### 列出 Worker

```bash
curl http://localhost:30000/workers
```

响应：
```json
{
  "workers": [
    {
      "id": "2f3a0c3e-3a7b-4c3f-8c70-1b7d4c3a6e1f",
      "url": "http://0.0.0.0:31378",
      "model_id": "mistral",
      "priority": 50,
      "cost": 1.0,
      "worker_type": "regular",
      "is_healthy": true,
      "load": 0,
      "connection_mode": "Http"
    }
  ],
  "total": 1,
  "stats": {
    "prefill_count": 0,
    "decode_count": 0,
    "regular_count": 1
  }
}
```

### 管理和健康检查端点

| Method | Path | 描述 |
|--------|------|------|
| `GET` | `/liveness` | 存活检查（始终返回 OK） |
| `GET` | `/readiness` | 就绪检查（检查是否有可用的健康 Worker） |
| `GET` | `/health` | 存活检查的别名 |
| `GET` | `/health_generate` | 健康生成测试 |
| `GET` | `/engine_metrics` | 从 Worker 获取引擎级指标 |
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/get_model_info` | 获取模型信息 |
| `GET` | `/get_server_info` | 获取服务器信息 |
| `POST` | `/flush_cache` | 清除所有缓存 |
| `GET` | `/get_loads` | 获取所有 Worker 负载 |
| `POST` | `/wasm` | 上传 WASM 模块 |
| `GET` | `/wasm` | 列出 WASM 模块 |
| `DELETE` | `/wasm/{module_uuid}` | 移除 WASM 模块 |

---

## 负载均衡策略

| 策略 | 描述 | 用法 |
|------|------|------|
| `random` | 均匀随机选择 | `--policy random` |
| `round_robin` | 按顺序轮询 Worker | `--policy round_robin` |
| `power_of_two` | 随机采样两个 Worker 并选择负载较轻的 | `--policy power_of_two` |
| `cache_aware` | 结合缓存局部性与负载均衡（默认） | `--policy cache_aware` |
| `bucket` | 将 Worker 分为动态边界的负载桶 | `--policy bucket` |

### Cache-Aware 策略调优

```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval-secs 120 \
--max-tree-size 67108864
```

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--cache-threshold` | 0.3 | 缓存命中的最小前缀匹配率 |
| `--balance-abs-threshold` | 64 | 触发再平衡的绝对负载差异 |
| `--balance-rel-threshold` | 1.5 | 触发再平衡的相对负载比率 |
| `--eviction-interval-secs` | 120 | 缓存淘汰间隔（秒） |
| `--max-tree-size` | 67108864 | 缓存树的最大节点数 |

---

## 可靠性与流量控制

### 重试

配置指数退避重试：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--retry-max-retries` | 5 | 最大重试次数 |
| `--retry-initial-backoff-ms` | 50 | 初始退避时长（毫秒） |
| `--retry-max-backoff-ms` | 5000 | 最大退避时长（毫秒） |
| `--retry-backoff-multiplier` | 2.0 | 指数退避乘数 |
| `--retry-jitter-factor` | 0.1 | 随机抖动因子（0.0-1.0） |
| `--disable-retries` | false | 完全禁用重试 |

**可重试状态码：** 408、429、500、502、503、504

### 熔断器

每个 Worker 的熔断器可防止级联故障：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --cb-failure-threshold 5 \
  --cb-success-threshold 2 \
  --cb-timeout-duration-secs 30 \
  --cb-window-duration-secs 60
```

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--cb-failure-threshold` | 5 | 触发断开的连续失败次数 |
| `--cb-success-threshold` | 2 | 从半开状态恢复关闭所需的成功次数 |
| `--cb-timeout-duration-secs` | 30 | 尝试半开前的等待时间 |
| `--cb-window-duration-secs` | 60 | 失败计数窗口 |
| `--disable-circuit-breaker` | false | 禁用熔断器 |

**熔断器状态：**
- **Closed（关闭）**：正常运行，请求被允许
- **Open（断开）**：故障中，请求立即被拒绝
- **Half-Open（半开）**：测试恢复中，允许有限请求

### 限流与排队

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --max-concurrent-requests 256 \
  --rate-limit-tokens-per-second 512 \
  --queue-size 128 \
  --queue-timeout-secs 30
```

超出并发限制的请求将在 FIFO 队列中等待。返回：
- `429 Too Many Requests`：当队列已满时
- `408 Request Timeout`：当队列超时到期时

### 健康检查

```bash
--health-check-interval-secs 30 \
--health-check-timeout-secs 10 \
--health-success-threshold 2 \
--health-failure-threshold 3 \
--health-check-endpoint /health
```

---

## 推理解析器集成

网关内置了推理解析器，适用于使用 Chain-of-Thought (CoT) 推理并带有显式思考块的模型。

### 支持的解析器

| Parser ID | 模型系列 | 思考标记 |
|-----------|----------|----------|
| `deepseek-r1` | DeepSeek-R1 | `<think>...</think>`（初始推理） |
| `qwen3` | Qwen-3 | `<think>...</think>` |
| `qwen3-thinking` | Qwen-3 Thinking | `<think>...</think>`（初始推理） |
| `kimi` | Kimi K2 | Unicode 思考标记 |
| `glm45` | GLM-4.5/4.6/4.7 | `<think>...</think>` |
| `step3` | Step-3 | `<think>...</think>` |
| `minimax` | MiniMax | `<think>...</think>` |

### 用法

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek-r1
```

gRPC 路由器自动执行以下操作：
1. 检测流式输出中的推理块
2. 将推理内容与普通文本分离
3. 应用带缓冲区管理的增量流式解析
4. 处理部分 token 检测以确保正确的流式行为

---

## 工具调用解析

网关支持以多种格式从 LLM 输出中解析函数/工具调用。

### 支持的格式

| 解析器 | 格式 | 描述 |
|--------|------|------|
| `json` | JSON | 标准 JSON 工具调用 |
| `python` | Pythonic | Python 函数调用语法 |
| `xml` | XML | XML 格式的工具调用 |

### 用法

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tool-call-parser json
```

---

## 分词器管理

### 分词器来源

网关支持多种分词器后端：
- **HuggingFace**：通过模型 ID 从 HuggingFace Hub 加载
- **本地**：从本地 `tokenizer.json` 或目录加载
- **Tiktoken**：自动检测 OpenAI GPT 模型（gpt-4、davinci 等）

### 配置

```bash
# HuggingFace 模型
--model-path meta-llama/Llama-3.1-8B-Instruct

# 本地分词器
--tokenizer-path /path/to/tokenizer.json

# 覆盖聊天模板
--chat-template /path/to/template.jinja
```

### 分词器缓存

两级缓存以获得最佳性能：

| 缓存 | 类型 | 描述 |
|------|------|------|
| L0 | 精确匹配 | 对重复提示的全字符串缓存 |
| L1 | 前缀匹配 | 对增量提示的前缀边界匹配 |

```bash
--enable-l0-cache \
--l0-max-entries 10000 \
--enable-l1-cache \
--l1-max-memory 52428800  # 50MB
```

---

## MCP 集成

网关提供原生 Model Context Protocol (MCP) 客户端集成，用于工具执行。

### 支持的传输协议

| 传输协议 | 描述 |
|----------|------|
| STDIO | 本地进程执行 |
| SSE | Server-Sent Events (HTTP) |
| Streamable | 双向流式传输 |

### 配置

```bash
python -m sglang_router.launch_router \
  --mcp-config-path /path/to/mcp-config.yaml \
  --worker-urls http://worker1:8000
```

### MCP 配置文件

```yaml
servers:
  - name: "filesystem"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    protocol: "stdio"
    required: false

  - name: "github"
    url: "https://api.github.com/mcp"
    token: "ghp_xxxxx"
    protocol: "sse"
    required: false

  - name: "custom-tools"
    url: "https://tools.example.com/mcp"
    protocol: "streamable"
    required: true

pool:
  max_connections: 100
  idle_timeout: 300

proxy:
  http: "http://proxy.internal:8080"
  https: "https://proxy.internal:8443"
  no_proxy: "localhost,127.0.0.1,*.internal"

inventory:
  enable_refresh: true
  tool_ttl: 300
  refresh_interval: 300
```

---

## 服务发现（Kubernetes）

通过 Kubernetes Pod 选择器启用自动 Worker 发现：

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

### PD 模式发现

```bash
--pd-disaggregation \
--prefill-selector app=sglang component=prefill \
--decode-selector app=sglang component=decode \
--service-discovery
```

预填充 Pod 可通过 `sglang.ai/bootstrap-port` 注解暴露 bootstrap 端口。RBAC 必须允许对 Pod 执行 `get`、`list` 和 `watch` 操作。

---

## 历史记录与数据连接器

| 后端 | 描述 | 用法 |
|------|------|------|
| `memory` | 内存存储（默认） | `--history-backend memory` |
| `none` | 不持久化 | `--history-backend none` |
| `oracle` | Oracle 自治数据库 | `--history-backend oracle` |
| `postgres` | PostgreSQL 数据库 | `--history-backend postgres` |
| `redis` | Redis | `--history-backend redis` |

### Oracle 配置

```bash
# 连接描述符
export ATP_DSN="(description=(address=(protocol=tcps)(port=1522)(host=adb.region.oraclecloud.com))(connect_data=(service_name=service_name)))"

# 或 TNS 别名（需要钱包）
export ATP_TNS_ALIAS="sglroutertestatp_high"
export ATP_WALLET_PATH="/path/to/wallet"

# 凭据
export ATP_USER="admin"
export ATP_PASSWORD="secret"
export ATP_POOL_MIN=4
export ATP_POOL_MAX=32

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend oracle
```

### PostgreSQL 配置

```bash
export POSTGRES_DB_URL="postgres://user:password@host:5432/dbname"

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend postgres
```

### Redis 配置

```bash
export REDIS_URL="redis://localhost:6379"
export REDIS_POOL_MAX=16
export REDIS_RETENTION_DAYS=30

python -m sglang_router.launch_router \
  --backend openai \
  --worker-urls https://api.openai.com \
  --history-backend redis \
  --redis-retention-days 30
```

使用 `--redis-retention-days -1` 进行持久存储（默认为 30 天）。

---

## WASM 中间件

网关支持 WebAssembly (WASM) 中间件模块，用于自定义请求/响应处理。这使得组织可以实现认证、限流、计费、日志等特定逻辑，而无需修改或重新编译网关。

### 概述

WASM 中间件运行在沙盒环境中，具有内存隔离，无网络/文件系统访问权限，并可配置资源限制。

| 挂载点 | 执行时机 | 用例 |
|--------|----------|------|
| `OnRequest` | 转发到 Worker 之前 | 认证、限流、请求修改 |
| `OnResponse` | 收到 Worker 响应之后 | 日志记录、响应修改、错误处理 |

| 动作 | 描述 |
|------|------|
| `Continue` | 继续处理，不做修改 |
| `Reject(status)` | 以 HTTP 状态码拒绝请求 |
| `Modify(...)` | 修改请求头、正文或状态 |

### 示例

完整的工作示例位于 `examples/wasm/`：

| 示例 | 描述 |
|------|------|
| `auth/` | 受保护路由的 API 密钥认证 |
| `rate_limit/` | 每客户端限流（请求数/分钟） |
| `logging/` | 请求追踪头和响应修改 |

接口定义位于 `src/wasm/interface`。

### 构建模块

```bash
# 前提条件
rustup target add wasm32-wasip2
cargo install wasm-tools

# 构建
cargo build --target wasm32-wasip2 --release

# 转换为组件格式
wasm-tools component new \
  target/wasm32-wasip2/release/my_middleware.wasm \
  -o my_middleware.component.wasm
```

### 部署模块

```bash
# 启用 WASM 支持
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-wasm

# 上传模块
curl -X POST http://localhost:30000/wasm \
  -H "Content-Type: application/json" \
  -d '{
    "modules": [{
      "name": "auth-middleware",
      "file_path": "/absolute/path/to/auth.component.wasm",
      "module_type": "Middleware",
      "attach_points": [{"Middleware": "OnRequest"}]
    }]
  }'

# 列出模块
curl http://localhost:30000/wasm

# 移除模块
curl -X DELETE http://localhost:30000/wasm/{module_uuid}
```

### 运行时配置

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `max_memory_pages` | 1024 (64MB) | 最大 WASM 内存 |
| `max_execution_time_ms` | 1000 | 执行超时 |
| `max_stack_size` | 1MB | 栈大小限制 |
| `module_cache_size` | 10 | 每个 Worker 线程的缓存模块数 |

**注意：** 限流状态是每个 Worker 线程独立的，不会在网关副本之间共享。在生产环境中，建议在共享层（如 Redis）实现限流。

---

## 语言绑定

SGLang Model Gateway 提供 Python 和 Go 的官方语言绑定，支持与不同技术栈和组织需求的集成。

### Python 绑定

Python 绑定提供基于 PyO3 的 Rust 网关库封装。这是一个直接的绑定，从 Python 调用网关服务器启动。

#### 安装

```bash
# 从 PyPI 安装
pip install sglang-router

# 开发构建
cd sgl-model-gateway/bindings/python
pip install maturin && maturin develop --features vendored-openssl
```

#### 用法

Python 绑定在本文档中广泛使用。详细示例请参阅[快速开始](#快速开始)和[部署模式](#部署模式)部分。

关键组件：
- `RouterArgs` 数据类，包含 50+ 个配置选项
- `Router.from_args()` 用于编程式启动
- CLI 命令：`smg launch`、`smg server`、`python -m sglang_router.launch_router`

### Go 绑定

Go 绑定为拥有 Go 基础设施的组织提供高性能 gRPC 客户端库。适用于：

- 与内部 Go 服务和工具集成
- 高性能客户端应用
- 构建自定义 OpenAI 兼容代理服务器

#### 架构

```
┌─────────────────────────────────────────┐
│         高级 Go API                      │
│   (client.go - OpenAI 风格接口)          │
├─────────────────────────────────────────┤
│         gRPC 层                          │
├─────────────────────────────────────────┤
│         Rust FFI 层                      │
│   (分词、解析、转换)                      │
└─────────────────────────────────────────┘
```

**主要特性：**
- 通过 FFI 实现原生 Rust 分词（线程安全、无锁）
- 完整的流式支持，支持上下文取消
- 可配置的通道缓冲区大小，适用于高并发场景
- 内置工具调用解析和聊天模板应用

#### 安装

```bash
# 首先构建 FFI 库
cd sgl-model-gateway/bindings/golang
make build && make lib

# 然后在 Go 项目中使用
go get github.com/sgl-project/sgl-go-sdk
```

**要求：** Go 1.24+、Rust 工具链

#### 示例

完整的工作示例位于 `bindings/golang/examples/`：

| 示例 | 描述 |
|------|------|
| `simple/` | 非流式聊天补全 |
| `streaming/` | 带 SSE 的流式聊天补全 |
| `oai_server/` | 完整的 OpenAI 兼容 HTTP 服务器 |

```bash
# 运行示例
cd sgl-model-gateway/bindings/golang/examples/simple && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/streaming && ./run.sh
cd sgl-model-gateway/bindings/golang/examples/oai_server && ./run.sh
```

#### 测试

```bash
cd sgl-model-gateway/bindings/golang

# 单元测试
go test -v ./...

# 集成测试（需要运行中的 SGLang 服务器）
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
export SGL_TOKENIZER_PATH=/path/to/tokenizer
go test -tags=integration -v ./...
```

### 对比

| 特性 | Python | Go |
|------|--------|-----|
| **主要用途** | 网关服务器启动器 | gRPC 客户端库 |
| **CLI 支持** | 完整 CLI (smg, sglang-router) | 仅库 |
| **Kubernetes 发现** | 原生支持 | 不适用（客户端库） |
| **PD 模式** | 内置 | 不适用（客户端库） |

**何时使用 Python：** 启动和管理网关服务器、服务发现、PD 分离。

**何时使用 Go：** 构建自定义客户端应用、与 Go 微服务集成、OpenAI 兼容代理服务器。

---

## 安全与认证

### 路由器 API 密钥

```bash
python -m sglang_router.launch_router \
  --api-key "your-router-api-key" \
  --worker-urls http://worker1:8000
```

客户端必须为受保护的端点提供 `Authorization: Bearer <key>`。

### Worker API 密钥

```bash
# 添加带有显式密钥的 Worker
curl -H "Authorization: Bearer router-key" \
  -X POST http://localhost:8080/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://worker:8000","api_key":"worker-key"}'
```

### 安全配置

1. **无认证**（默认）：仅在可信环境中使用
2. **仅路由器认证**：客户端向路由器认证
3. **仅 Worker 认证**：路由器开放，Worker 需要密钥
4. **完全认证**：路由器和 Worker 均受保护

### 网关服务器 TLS (HTTPS)

启用 TLS 以通过 HTTPS 提供网关服务：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --tls-cert-path /path/to/server.crt \
  --tls-key-path /path/to/server.key
```

| 参数 | 描述 |
|------|------|
| `--tls-cert-path` | 服务器证书路径（PEM 格式） |
| `--tls-key-path` | 服务器私钥路径（PEM 格式） |

两个参数必须同时提供。网关使用 rustls 和 ring 加密提供者进行 TLS 终止。如果未配置 TLS，网关将回退到普通 HTTP。

### Worker 通信 mTLS

启用双向 TLS (mTLS) 以在 HTTP 模式下安全地与 Worker 通信：

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --client-cert-path /path/to/client.crt \
  --client-key-path /path/to/client.key \
  --ca-cert-path /path/to/ca.crt
```

| 参数 | 描述 |
|------|------|
| `--client-cert-path` | mTLS 客户端证书路径（PEM 格式） |
| `--client-key-path` | mTLS 客户端私钥路径（PEM 格式） |
| `--ca-cert-path` | 用于验证 Worker TLS 的 CA 证书路径（PEM 格式，可重复） |

**要点：**
- 客户端证书和密钥必须同时提供
- 可通过多个 `--ca-cert-path` 标志添加多个 CA 证书
- 配置 TLS 时使用 rustls 后端
- 为所有 Worker 创建单个 HTTP 客户端（假定单一安全域）
- 为长连接启用 TCP keepalive（30 秒）

### 完整 TLS 配置示例

网关 HTTPS + Worker mTLS + API 密钥认证：

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "secure-api-key" \
  --policy cache_aware
```

---

## 可观测性

### Prometheus 指标

通过 `--prometheus-host`/`--prometheus-port` 启用（默认 `0.0.0.0:29000`）。

#### 指标类别（40+ 指标）

| 层级 | 前缀 | 指标 |
|------|------|------|
| HTTP | `smg_http_*` | `requests_total`、`request_duration_seconds`、`responses_total`、`connections_active`、`rate_limit_total` |
| Router | `smg_router_*` | `requests_total`、`request_duration_seconds`、`request_errors_total`、`stage_duration_seconds`、`upstream_responses_total` |
| Inference | `smg_router_*` | `ttft_seconds`、`tpot_seconds`、`tokens_total`、`generation_duration_seconds` |
| Worker | `smg_worker_*` | `pool_size`、`connections_active`、`requests_active`、`health_checks_total`、`selection_total`、`errors_total` |
| Circuit Breaker | `smg_worker_cb_*` | `state`、`transitions_total`、`outcomes_total`、`consecutive_failures`、`consecutive_successes` |
| Retry | `smg_worker_*` | `retries_total`、`retries_exhausted_total`、`retry_backoff_seconds` |
| Discovery | `smg_discovery_*` | `registrations_total`、`deregistrations_total`、`sync_duration_seconds`、`workers_discovered` |
| MCP | `smg_mcp_*` | `tool_calls_total`、`tool_duration_seconds`、`servers_active`、`tool_iterations_total` |
| Database | `smg_db_*` | `operations_total`、`operation_duration_seconds`、`connections_active`、`items_stored` |

#### 关键推理指标（gRPC 模式）

| 指标 | 类型 | 描述 |
|------|------|------|
| `smg_router_ttft_seconds` | Histogram | 首 token 时间 |
| `smg_router_tpot_seconds` | Histogram | 每输出 token 时间 |
| `smg_router_tokens_total` | Counter | 总 token 数（输入/输出） |
| `smg_router_generation_duration_seconds` | Histogram | 端到端生成时间 |

#### 持续时间桶

1ms、5ms、10ms、25ms、50ms、100ms、250ms、500ms、1s、2.5s、5s、10s、15s、30s、45s、60s、90s、120s、180s、240s

### OpenTelemetry 追踪

通过 OTLP 导出启用分布式追踪：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317
```

#### 特性

- OTLP/gRPC 导出器（默认端口 4317）
- 用于 HTTP 和 gRPC 的 W3C Trace Context 传播
- 批量 span 处理（500ms 延迟，64 个 span 批量大小）
- 自定义过滤以减少噪音
- 将追踪上下文注入上游 Worker 请求
- 服务名称：`sgl-router`

### 日志

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

结构化追踪，可选文件输出。日志级别：`debug`、`info`、`warn`、`error`。

### 请求 ID 传播

```bash
--request-id-headers x-request-id x-trace-id x-correlation-id
```

响应包含 `x-request-id` 头用于关联。

---

## 生产环境建议

本节提供在生产环境中部署 SGLang Model Gateway 的指导。

### 安全最佳实践

**在生产环境中务必启用 TLS：**

```bash
python -m sglang_router.launch_router \
  --worker-urls https://worker1:8443 https://worker2:8443 \
  --tls-cert-path /etc/certs/server.crt \
  --tls-key-path /etc/certs/server.key \
  --client-cert-path /etc/certs/client.crt \
  --client-key-path /etc/certs/client.key \
  --ca-cert-path /etc/certs/ca.crt \
  --api-key "${ROUTER_API_KEY}"
```

**安全检查清单：**
- 启用 TLS 进行网关 HTTPS 终止
- 当 Worker 位于不可信网络时启用 mTLS 进行 Worker 通信
- 设置 `--api-key` 保护路由器端点
- 使用 Kubernetes Secrets 或密钥管理器存储凭据
- 定期轮换证书和 API 密钥
- 使用防火墙或网络策略限制网络访问

### 高可用性

**扩展策略：**

网关支持在负载均衡器后运行多个副本以实现高可用性。但有以下重要注意事项：

| 组件 | 跨副本共享 | 影响 |
|------|-----------|------|
| Worker 注册表 | 否（独立） | 每个副本独立发现 Worker |
| Radix 缓存树 | 否（独立） | 缓存命中率可能下降 10-20% |
| 熔断器状态 | 否（独立） | 每个副本独立追踪故障 |
| 限流 | 否（独立） | 限制按副本而非全局应用 |

**建议：**

1. **优先水平扩展而非垂直扩展**：部署多个较小的网关副本，而不是一个拥有过多 CPU 和内存的大实例。这提供：
   - 更好的容错性（单个副本故障不会导致网关宕机）
   - 更可预测的资源使用
   - 更容易的容量规划

2. **使用 Kubernetes 服务发现**：让网关自动发现和管理 Worker：
   ```bash
   python -m sglang_router.launch_router \
     --service-discovery \
     --selector app=sglang-worker \
     --service-discovery-namespace production
   ```

3. **接受缓存效率的权衡**：使用多个副本时，缓存感知路由策略的 radix 树不会在副本间同步。这意味着：
   - 每个副本构建自己的缓存树
   - 来自同一用户的请求可能命中不同的副本
   - 预期缓存命中率下降：**10-20%**
   - 考虑到高可用性的好处，这通常是可以接受的

4. **配置会话亲和性（可选）**：如果缓存效率至关重要，可为负载均衡器配置基于请求一致性哈希（如用户 ID 或 API 密钥）的会话亲和性。

**高可用架构示例：**
```
                    ┌─────────────────┐
                    │  负载均衡器      │
                    │   (L4/L7)       │
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │   网关     │  │   网关     │  │   网关     │
        │  副本 1    │  │  副本 2    │  │  副本 3    │
        └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
        │  Worker   │  │  Worker   │  │  Worker   │
        │  Pod 1    │  │  Pod 2    │  │  Pod N    │
        └───────────┘  └───────────┘  └───────────┘
```

### 性能

**使用 gRPC 模式获取高吞吐量：**

gRPC 模式为 SGLang Worker 提供最高性能：

```bash
# 以 gRPC 模式启动 Worker
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --grpc-mode \
  --port 20000

# 配置网关使用 gRPC
python -m sglang_router.launch_router \
  --worker-urls grpc://worker1:20000 grpc://worker2:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy cache_aware
```

**gRPC 性能优势：**
- 原生 Rust 分词（无 Python 开销）
- 更低延迟的流式传输
- 内置推理解析器执行
- 网关中的工具调用解析
- 减少序列化开销

**调优建议：**

| 参数 | 建议 | 原因 |
|------|------|------|
| `--policy` | `cache_aware` | 最适合重复提示，约 30% 延迟降低 |
| `--max-concurrent-requests` | Worker 数量的 2-4 倍 | 防止过载同时最大化吞吐量 |
| `--queue-size` | max-concurrent 的 2 倍 | 突发流量缓冲 |
| `--request-timeout-secs` | 基于最大生成长度 | 防止请求卡住 |

### Kubernetes 部署

**服务发现的 Pod 标签：**

为使网关自动发现 Worker，需一致地标注 Worker Pod：

```yaml
# Worker Deployment（常规模式）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang-worker
      component: inference
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      containers:
      - name: worker
        image: lmsysorg/sglang:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 20000
          name: grpc
```

**服务发现的网关配置：**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

**PD（预填充/解码）模式标签：**

```yaml
# 预填充 Worker
metadata:
  labels:
    app: sglang-worker
    component: prefill
  annotations:
    sglang.ai/bootstrap-port: "9001"

# 解码 Worker
metadata:
  labels:
    app: sglang-worker
    component: decode
```

**PD 发现的网关配置：**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector app=sglang-worker component=prefill \
  --decode-selector app=sglang-worker component=decode \
  --service-discovery-namespace production
```

**RBAC 要求：**

网关需要权限来监视 Pod：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-gateway
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-gateway
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-gateway
  namespace: production
roleRef:
  kind: Role
  name: sglang-gateway
  apiGroup: rbac.authorization.k8s.io
```

### 使用 PromQL 进行监控

配置 Prometheus 抓取网关指标端点（默认：`:29000/metrics`）。

**必要的仪表板：**

**1. 请求速率和延迟：**
```promql
# 按端点的请求速率
sum(rate(smg_http_requests_total[5m])) by (path, method)

# P50 延迟
histogram_quantile(0.50, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# P99 延迟
histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# 错误率
sum(rate(smg_http_responses_total{status=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

**2. Worker 健康状态：**
```promql
# 健康的 Worker 数量
sum(smg_worker_pool_size)

# 每个 Worker 的活跃连接数
smg_worker_connections_active

# Worker 健康检查失败次数
sum(rate(smg_worker_health_checks_total{result="failure"}[5m])) by (worker_id)
```

**3. 熔断器状态：**
```promql
# 熔断器状态（0=关闭, 1=断开, 2=半开）
smg_worker_cb_state

# 熔断器状态转换
sum(rate(smg_worker_cb_transitions_total[5m])) by (worker_id, from_state, to_state)

# 断路的 Worker 数量
count(smg_worker_cb_state == 1)
```

**4. 推理性能（gRPC 模式）：**
```promql
# 首 token 时间（P50）
histogram_quantile(0.50, sum(rate(smg_router_ttft_seconds_bucket[5m])) by (le, model))

# 每输出 token 时间（P99）
histogram_quantile(0.99, sum(rate(smg_router_tpot_seconds_bucket[5m])) by (le, model))

# Token 吞吐量
sum(rate(smg_router_tokens_total[5m])) by (model, direction)

# 生成时长 P95
histogram_quantile(0.95, sum(rate(smg_router_generation_duration_seconds_bucket[5m])) by (le))
```

**5. 限流和排队：**
```promql
# 限流拒绝次数
sum(rate(smg_http_rate_limit_total{decision="rejected"}[5m]))

# 队列深度（使用并发限制时）
smg_worker_requests_active

# 重试次数
sum(rate(smg_worker_retries_total[5m])) by (worker_id)

# 重试耗尽次数（所有重试后仍失败）
sum(rate(smg_worker_retries_exhausted_total[5m]))
```

**6. MCP 工具执行：**
```promql
# 工具调用速率
sum(rate(smg_mcp_tool_calls_total[5m])) by (server, tool)

# 工具延迟 P95
histogram_quantile(0.95, sum(rate(smg_mcp_tool_duration_seconds_bucket[5m])) by (le, tool))

# 活跃的 MCP 服务器连接
smg_mcp_servers_active
```

**告警规则示例：**

```yaml
groups:
- name: sglang-gateway
  rules:
  - alert: HighErrorRate
    expr: |
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "SGLang Gateway 错误率过高"

  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Worker 熔断器处于断开状态"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le)) > 30
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 延迟超过 30 秒"

  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "没有可用的健康 Worker"
```

---

## 配置参考

### 核心设置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--host` | str | 127.0.0.1 | 路由器主机 |
| `--port` | int | 30000 | 路由器端口 |
| `--worker-urls` | list | [] | Worker URL（HTTP 或 gRPC） |
| `--policy` | str | cache_aware | 路由策略 |
| `--max-concurrent-requests` | int | -1 | 并发限制（-1 禁用） |
| `--request-timeout-secs` | int | 600 | 请求超时 |
| `--max-payload-size` | int | 256MB | 最大请求负载 |

### 预填充/解码

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--pd-disaggregation` | flag | false | 启用 PD 模式 |
| `--prefill` | list | [] | 预填充 URL + 可选 bootstrap 端口 |
| `--decode` | list | [] | 解码 URL |
| `--prefill-policy` | str | None | 覆盖预填充节点的策略 |
| `--decode-policy` | str | None | 覆盖解码节点的策略 |
| `--worker-startup-timeout-secs` | int | 600 | Worker 初始化超时 |

### Kubernetes 发现

| 参数 | 类型 | 描述 |
|------|------|------|
| `--service-discovery` | flag | 启用服务发现 |
| `--selector` | list | 标签选择器（key=value） |
| `--prefill-selector` / `--decode-selector` | list | PD 模式选择器 |
| `--service-discovery-namespace` | str | 要监视的命名空间 |
| `--service-discovery-port` | int | Worker 端口（默认 80） |
| `--bootstrap-port-annotation` | str | bootstrap 端口注解 |

### TLS 配置

| 参数 | 类型 | 描述 |
|------|------|------|
| `--tls-cert-path` | str | 网关 HTTPS 的服务器证书（PEM） |
| `--tls-key-path` | str | 网关 HTTPS 的服务器私钥（PEM） |
| `--client-cert-path` | str | Worker mTLS 的客户端证书（PEM） |
| `--client-key-path` | str | Worker mTLS 的客户端私钥（PEM） |
| `--ca-cert-path` | str | 用于验证 Worker 的 CA 证书（PEM，可重复） |

---

## 故障排除

### Worker 一直未就绪

增加 `--worker-startup-timeout-secs` 或确保健康探测在路由器启动前响应。

### 负载不均衡 / 热点 Worker

检查按 Worker 划分的 `smg_router_requests_total`，并调整缓存感知阈值（`--balance-*`、`--cache-threshold`）。

### 熔断器抖动

增加 `--cb-failure-threshold` 或延长超时/窗口时长。考虑临时禁用重试。

### 队列溢出 (429)

增加 `--queue-size` 或减少客户端并发。确保 `--max-concurrent-requests` 与下游容量匹配。

### 内存增长

减小 `--max-tree-size` 或降低 `--eviction-interval-secs` 以进行更积极的缓存淘汰。

### 调试

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 \
  --log-level debug \
  --log-dir ./router_logs
```

### gRPC 连接问题

确保 Worker 以 `--grpc-mode` 启动，并验证路由器已提供 `--model-path` 或 `--tokenizer-path`。

### 分词器加载失败

检查私有模型的 HuggingFace Hub 凭据（`HF_TOKEN` 环境变量）。验证本地路径是否可访问。

---

SGLang Model Gateway 随 SGLang 运行时持续演进。在采用新功能或贡献改进时，请保持 CLI 标志、集成和文档的一致性。
