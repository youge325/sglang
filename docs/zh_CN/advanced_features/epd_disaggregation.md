# EPD 分离部署

## 为什么需要 EPD 分离部署？什么是 EPD 分离部署？

在现代视觉语言模型（VLM）推理中，请求执行自然分解为三个不同的阶段：编码器（Encoder）、预填充（Prefill）和解码（Decode）。
编码器阶段执行视觉预处理和基于 ViT 的图像编码，计算密集度高但仅在请求初始化时需要。预填充阶段处理完整的多模态输入序列以初始化语言模型的 Key-Value（KV）缓存，而解码阶段则以内存带宽和用于自回归 token 生成的 KV 缓存访问为主。

现有部署通常将这些阶段集中在统一的执行引擎中，或者最多应用预填充-解码（PD）分离。然而，这些设计仍然将视觉编码与语言预填充紧密耦合，导致资源利用效率低下、对图像密集型工作负载的可扩展性有限，以及在高负载下调度效果不佳。

为了应对这些挑战，我们在 SGLang 中引入了编码器-预填充-解码（EPD）分离部署。EPD 进一步将视觉编码从语言处理中分离出来，实现编码器服务器的独立水平扩展、改进多模态请求的负载均衡，并与现有的 PD 分离无缝集成，形成完全解耦的三层推理架构。

### 使用方法

你可以使用 `--language-only` 启动纯语言模型，或使用 `--encoder-only` 启动纯编码器模型。
启动纯语言模型时，必须通过 `--encoder-urls` 额外指定编码器服务端点。

我们支持多种编码器传输后端，包括 zmq_to_scheduler、zmq_to_tokenizer 和 mooncake（默认为 zmq_to_scheduler）。可以使用 `--encoder-transfer-backend` 选择后端。

#### Qwen VL

- EP 分离

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# language-only server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

- EPD 分离

```bash
# encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000
# encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001
# prefill 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
# decode 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003
# router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000

```
