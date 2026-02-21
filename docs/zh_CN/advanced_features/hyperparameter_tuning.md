# 超参数调优

## 实现离线批量推理的高吞吐量

达到较大的批处理大小是离线批量推理中获得高吞吐量的最重要因素。
当服务器在满负载稳态运行时，请关注日志中的以下内容：

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317```

### 调整请求提交速度以控制 `#queue-req`

`#queue-req` 表示队列中的请求数量。
如果经常看到 `#queue-req: 0`，说明您的客户端代码提交请求太慢。
`#queue-req` 的健康范围是 `100 - 2000`。
但避免使 `#queue-req` 过大，因为这会增加服务器的调度开销。

### 达到较高的 `token usage`

`token usage` 表示服务器的 KV 缓存内存利用率。`token usage > 0.9` 表示良好的利用率。

如果经常看到 `token usage < 0.9` 且 `#queue-req > 0`，说明服务器对接收新请求过于保守。可以将 `--schedule-conservativeness` 降低到 0.3 等值。
当用户发送许多带有较大 `max_new_tokens` 的请求，但请求由于 EOS 或停止字符串而提前停止时，服务器可能会过于保守。

另一方面，如果看到 `token usage` 非常高且频繁出现警告如
`KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`，可以将 `--schedule-conservativeness` 增加到 1.3 等值。
如果 `KV cache pool is full. Retract requests.` 偶尔出现但不频繁（每分钟约 1 次），则没有问题。

### 调优 `--mem-fraction-static` 以增加 KV 缓存池容量
SGLang 的内存分配如下：

总内存使用 = 模型权重 + KV 缓存池 + CUDA Graph 缓冲区 + 激活值

`--mem-fraction-static` 参数决定了分配给前两个组件的内存量：

mem_fraction_static = (模型权重 + KV 缓存池) / GPU 内存容量

要支持更高的并发，应尽可能提高 `--mem-fraction-static` 以最大化 KV 缓存池容量，同时为激活值和 CUDA Graph 缓冲区保留足够的内存。

SGLang 使用简单的启发式方法设置 `--mem-fraction-static` 的默认值，但您可以根据实际使用情况进行优化。
经验法则是，为激活值保留 5-8 GB 内存通常足够。您可以通过检查服务器就绪前的日志来确认：

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

检查 `available_gpu_mem` 值：
- 如果在 5-8 GB 之间，设置合理。
- 如果太高（如 10-20 GB），增大 `--mem-fraction-static` 以分配更多内存给 KV 缓存。
- 如果太低，后续可能出现 OOM 错误，应减小 `--mem-fraction-static`。

另一种直接方法是以 0.01 的增量增加 `--mem-fraction-static`，直到您的工作负载遇到 OOM 错误。

### 通过调优 `--chunked-prefill-size`、`--mem-fraction-static` 和 `--max-running-requests` 避免内存不足错误

如果遇到内存不足（OOM）错误，可以调整以下参数：

- 如果 OOM 发生在预填充阶段，尝试将 `--chunked-prefill-size` 降低到 `4096` 或 `2048`。这会节省内存但降低长提示词的预填充速度。
- 如果 OOM 发生在解码阶段，尝试降低 `--max-running-requests`。
- 也可以将 `--mem-fraction-static` 降低到较小值，如 0.8 或 0.7。这会减少 KV 缓存内存池的内存使用，有助于防止预填充和解码阶段的 OOM 错误。但会限制最大并发数和降低峰值吞吐量。

### 调优 `--cuda-graph-max-bs`
默认情况下，CUDA Graph 仅对小 batch size（如小于 160 或 256）启用。
但对于某些模型，特别是在大张量并行度下，CUDA Graph 对 512 或 768 的 batch size 也可能有用。
因此，增大 `--cuda-graph-max-bs` 到更大值可能有益。
注意 CUDA Graph 会消耗更多内存，因此可能需要同时降低 `--mem-fraction-static`。

### 调优 `--dp-size` 和 `--tp-size`

数据并行对吞吐量更好。当 GPU 内存充足时，始终优先使用数据并行来提升吞吐量。请参阅 [SGLang Model Gateway（前身为 Router）](../advanced_features/sgl_model_gateway.md) 以获得比使用 `dp_size` 参数更好的数据并行方案。

### 尝试其他选项

- `torch.compile` 可加速小模型在小 batch size 下的推理。可通过 `--enable-torch-compile` 启用。
- 尝试其他量化方法（如使用 `--quantization fp8` 进行 FP8 量化）
- 尝试其他并行策略（如对 DeepSeek 模型使用[专家并行](https://lmsys.org/blog/2025-05-05-large-scale-ep/)  或 DP 注意力 `--enable-dp-attention --dp-size 8`）。
- 如果工作负载有大量共享前缀，尝试 `--schedule-policy lpm`。`lpm` 代表最长前缀匹配，它通过重新排序请求来鼓励更多缓存命中，但会引入更多调度开销。
