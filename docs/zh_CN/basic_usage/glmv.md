# GLM-4.6V / GLM-4.5V 使用指南

## SGLang 启动命令

以下是针对不同硬件/精度模式的推荐启动命令。

### FP8（量化）模式

适用于高显存效率和低延迟优化的部署场景（例如在 H100、H200 上），需要 FP8 checkpoint 支持：

```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V-FP8 \
  --tp 2 \
  --ep 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --keep-mm-feature-on-device
```

### 非 FP8（BF16 / 全精度）模式
适用于在 A100/H100 上使用 BF16 的部署场景（或未使用 FP8 checkpoint）：
```bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --tp 4 \
  --ep 4 \
  --host 0.0.0.0 \
  --port 30000
```

## 硬件相关说明与建议

- 在 H100 上使用 FP8：推荐使用 FP8 checkpoint 以获得最佳显存效率。
- 在 A100 / H100 上使用 BF16（非 FP8）：建议使用 `--mm-max-concurrent-calls` 来控制图像/视频推理时的并行吞吐量和 GPU 显存使用。
- 在 H200 和 B200 上：模型可以开箱即用，支持完整上下文长度以及并发的图像 + 视频处理。

## 发送图像/视频请求

### 图像输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

### 视频输入：

```python
import requests

url = f"http://localhost:30000/v1/chat/completions"

data = {
    "model": "zai-org/GLM-4.6V",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's happening in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print(response.text)
```

## 重要的服务器参数和标志

在启动支持**多模态**的模型服务器时，可以使用以下命令行参数来调优性能和行为：

- `--mm-attention-backend`：指定多模态注意力后端。例如 `fa3`（Flash Attention 3）
- `--mm-max-concurrent-calls <value>`：指定服务器允许的**最大并发异步多模态数据处理调用数**。用于控制图像/视频推理时的并行吞吐量和 GPU 显存使用。
- `--mm-per-request-timeout <seconds>`：定义每个多模态请求的**超时时间（秒）**。如果请求超过此时间限制（例如处理非常大的视频输入），将自动终止。
- `--keep-mm-feature-on-device`：指示服务器在处理后**将多模态特征张量保留在 GPU 上**。这避免了设备到主机（D2H）的内存拷贝，提升重复或高频推理工作负载的性能。
- `--mm-enable-dp-encoder`：将 ViT 设置为数据并行，同时保持 LLM 张量并行，可以持续降低 TTFT 并提升端到端吞吐量。
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1`：基于共享内存池的 CUDA IPC 多模态数据传输，可显著改善端到端延迟。

### 使用上述优化的示例：
```bash
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
SGLANG_VLM_CACHE_SIZE_MB=0 \
python -m sglang.launch_server \
  --model-path zai-org/GLM-4.6V \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp-size 8 \
  --enable-cache-report \
  --log-level info \
  --max-running-requests 64 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 8192 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --mm-enable-dp-encoder \
  --enable-metrics
```

### GLM-4.5V / GLM-4.6V 的 Thinking Budget（思考预算）

在 SGLang 中，我们可以通过 `CustomLogitProcessor` 实现 thinking budget。

启动服务器时需开启 `--enable-custom-logit-processor` 标志。然后在请求中使用 `Glm4MoeThinkingBudgetLogitProcessor`，用法与 [glm45.md](./glm45.md) 中 `GLM-4.6` 的示例类似。
