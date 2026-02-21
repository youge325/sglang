# PD 分离 (Prefill-Decode Disaggregation)

## 为什么需要 PD 分离？

大语言模型 (LLM) 推理包含两个不同的阶段：**预填充 (Prefill)** 和 **解码 (Decode)**。预填充阶段是计算密集型的，处理整个输入序列；而解码阶段是内存密集型的，管理用于 token 生成的 Key-Value (KV) 缓存。传统上这两个阶段在统一引擎中处理，但预填充和解码批次的联合调度会引入效率低下。为此，我们在 SGLang 中引入了 **PD 分离**。

### 统一调度的问题

传统的统一引擎同时处理预填充和解码批次，导致两个显著问题：

1. **预填充中断**：传入的预填充批次频繁中断正在进行的解码批次，导致 token 生成出现大幅延迟。
2. **DP 注意力不平衡**：在数据并行 (DP) 注意力中，一个 DP worker 可能处理预填充批次，另一个同时处理解码批次，导致解码延迟增加。

PD 分离通过将两个阶段分开来解决这些问题，使每个阶段都能进行有针对性的优化。

目前，我们支持 Mooncake 和 NIXL 作为传输引擎。

## 路由集成

对于大规模部署的 PD 分离，SGLang 提供了路由器支持负载均衡和故障容错。详情请参阅 [SGLang Model Gateway](../advanced_features/sgl_model_gateway.md)。

## Mooncake

### 安装要求

```bash
uv pip install mooncake-transfer-engine
```

### 使用方法

#### Llama 单节点

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

```{note}
更多 DeepSeek 多节点、NIXL 等高级部署配置请参阅 [英文文档](../../en/advanced_features/pd_disaggregation.html)。
```
