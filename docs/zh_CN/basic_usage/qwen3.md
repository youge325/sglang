# Qwen3-Next 使用指南

SGLang 自[此 PR](https://github.com/sgl-project/sglang/pull/10233) 起已支持 Qwen3-Next-80B-A3B-Instruct 和 Qwen3-Next-80B-A3B-Thinking。

## 使用 SGLang 启动 Qwen3-Next

在 4xH100/H200 GPU 上部署 Qwen3-Next 模型：

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### 配置建议
- `--max-mamba-cache-size`：调整 `--max-mamba-cache-size` 以增加 Mamba 缓存空间和最大并发请求能力。作为权衡，这会减少 KV 缓存空间。您可以根据工作负载进行调整。
- `--mamba-ssm-dtype`：可选 `bfloat16` 或 `float32`，使用 `bfloat16` 可节省 Mamba 缓存大小，使用 `float32` 可获得更精确的结果。默认设置为 `float32`。
- `--mamba-full-memory-ratio`：Mamba 状态内存与完整 KV 缓存内存的比例。默认值为 0.9。

### Mamba Radix Cache
SGLang 支持 Qwen3-Next 模型的前缀缓存功能，称为 `MambaRadixCache`，通过复用计算结果来提升推理速度。`MambaRadixCache` 有两个版本：
- `no_buffer`：默认版本，也是其他混合线性模型的选择。启用后，SGLang 会自动关闭重叠调度以保证兼容性。
- `extra_buffer`：优化版本，兼容页面大小 > 1、重叠调度和推测解码等功能。它还支持在分支位置存储 Mamba 状态。但是，每个请求需要两个额外的 Mamba 空间用于乒乓缓冲区。启用方法：在启动服务器时添加参数 `--mamba-scheduler-strategy extra_buffer`。

### EAGLE 推测解码
**描述**：SGLang 已支持 Qwen3-Next 模型的 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding)。

**用法**：
添加参数 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 来启用此功能。例如：

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

详情请参阅[此 PR](https://github.com/sgl-project/sglang/pull/10233)。
