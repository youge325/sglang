# Bench Serving 指南

本指南介绍如何使用 `python -m sglang.bench_serving` 对在线服务的吞吐量和延迟进行基准测试。它通过 OpenAI 兼容端点和原生端点支持多种推理后端，并提供控制台指标和可选的 JSONL 输出。

### 功能概述

- 生成合成或基于数据集的提示并提交到目标服务端点
- 测量吞吐量、首 Token 时间（TTFT）、Token 间延迟（ITL）、每请求端到端延迟等指标
- 支持流式或非流式模式、速率控制和并发限制

### 支持的后端和端点

- `sglang` / `sglang-native`：`POST /generate`
- `sglang-oai`、`vllm`、`lmdeploy`：`POST /v1/completions`
- `sglang-oai-chat`、`vllm-chat`、`lmdeploy-chat`：`POST /v1/chat/completions`
- `trt`（TensorRT-LLM）：`POST /v2/models/ensemble/generate_stream`
- `gserver`：自定义服务器（此脚本中尚未实现）
- `truss`：`POST /v1/models/model:predict`

如果提供了 `--base-url`，请求将发送到该地址。否则使用 `--host` 和 `--port`。当未提供 `--model` 时，脚本将尝试查询 `GET /v1/models` 获取可用的模型 ID（OpenAI 兼容端点）。

### 前置条件

- Python 3.8+
- 此脚本通常使用的依赖包：`aiohttp`、`numpy`、`requests`、`tqdm`、`transformers`，某些数据集还需要 `datasets`、`pillow`、`pybase64`。按需安装。
- 一个正在运行且可通过上述端点访问的推理服务器
- 如果你的服务器需要身份验证，请设置环境变量 `OPENAI_API_KEY`（用作 `Authorization: Bearer <key>`）

### 快速开始

对暴露 `/generate` 的 SGLang 服务器运行基本基准测试：

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

或者，使用 OpenAI 兼容端点（completions）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### 数据集

通过 `--dataset-name` 选择：

- `sharegpt`（默认）：加载 ShareGPT 风格的对话对；可选择使用 `--sharegpt-context-len` 限制，并使用 `--sharegpt-output-len` 覆盖输出
- `random`：随机文本长度；从 ShareGPT Token 空间采样
- `random-ids`：随机 Token ID（可能产生无意义文本）
- `image`：生成图像并包装在聊天消息中；支持自定义分辨率、多种格式和不同内容类型
- `generated-shared-prefix`：具有共享长系统提示和短问题的合成数据集
- `mmmu`：从 MMMU（数学分割）采样并包含图像

常用数据集标志：

- `--num-prompts N`：请求数量
- `--random-input-len`、`--random-output-len`、`--random-range-ratio`：用于 random/random-ids/image
- `--image-count`：每个请求的图像数量（用于 `image` 数据集）。

- `--apply-chat-template`：构建提示时应用分词器的聊天模板
- `--dataset-path PATH`：ShareGPT json 的文件路径；如果为空且缺失，将自动下载并缓存

Generated Shared Prefix 标志（用于 `generated-shared-prefix`）：

- `--gsp-num-groups`
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`

图像数据集标志（用于 `image`）：

- `--image-count`：每个请求的图像数量
- `--image-resolution`：图像分辨率；支持预设值（4k、1080p、720p、360p）或自定义 '高度x宽度' 格式（例如 1080x1920、512x768）
- `--image-format`：图像格式（jpeg 或 png）
- `--image-content`：图像内容类型（random 或 blank）

### 示例

1. 要对图像数据集进行基准测试，使用每个请求 3 张图像、500 个提示、512 输入长度和 512 输出长度，可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --disable-radix-cache
```

```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name image \
    --num-prompts 500 \
    --image-count 3 \
    --image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

2. 要对随机数据集进行基准测试，使用 3000 个提示、1024 输入长度和 1024 输出长度，可以运行：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct
```

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 3000 \
    --random-input 1024 \
    --random-output 1024 \
    --random-range-ratio 0.5
```

### 选择模型和分词器

- `--model` 是必需的，除非后端暴露了 `GET /v1/models`，在这种情况下会自动选择第一个模型 ID。
- `--tokenizer` 默认为 `--model`。两者都可以是 HF 模型 ID 或本地路径。
- 对于 ModelScope 工作流，设置 `SGLANG_USE_MODELSCOPE=true` 可通过 ModelScope 获取（为提高速度会跳过权重下载）。
- 如果你的分词器缺少聊天模板，脚本会发出警告，因为对于无意义输出的 Token 计数可能不够准确。

### 速率、并发和流式

- `--request-rate`：每秒请求数。`inf` 表示立即发送所有请求（突发）。非无限速率使用泊松过程确定到达时间。
- `--max-concurrency`：无论到达速率如何，限制并发进行中请求的上限。
- `--disable-stream`：切换到非流式模式（如果支持）；此时 TTFT 等于聊天补全的总延迟。

### 其他关键选项

- `--output-file FILE.jsonl`：将 JSONL 结果追加到文件；如果未指定则自动命名
- `--output-details`：包含每请求数组（生成文本、错误、ttfts、itls、输入/输出长度）
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`：合并到请求载荷中（采样参数等）
- `--disable-ignore-eos`：传递 EOS 行为（因后端而异）
- `--warmup-requests N`：先运行短输出的预热请求（默认 1）
- `--flush-cache`：在主运行前调用 `/flush_cache`（sglang）
- `--profile`：调用 `/start_profile` 和 `/stop_profile`（需要服务器启用分析，例如设置 `SGLANG_TORCH_PROFILER_DIR`）
- `--lora-name name1 name2 ...`：每个请求随机选择一个并传递给后端（例如 sglang 的 `lora_path`）
- `--tokenize-prompt`：发送整数 ID 而非文本（目前仅支持 `--backend sglang`）

### 身份验证

如果你的目标端点需要 OpenAI 风格的身份验证，请设置：

```bash
export OPENAI_API_KEY=sk-...yourkey...
```

脚本将自动为 OpenAI 兼容路由添加 `Authorization: Bearer $OPENAI_API_KEY`。

### 指标说明

每次运行后打印：

- 请求吞吐量（req/s）
- 输入 Token 吞吐量（tok/s）- 包括文本和视觉 Token
- 输出 Token 吞吐量（tok/s）
- 总 Token 吞吐量（tok/s）- 包括文本和视觉 Token
- 总输入文本 Token 和总输入视觉 Token - 按模态分类
- 并发度：所有请求的总时间除以墙钟时间
- 端到端延迟（ms）：每请求总延迟的均值/中位数/标准差/p99
- 首 Token 时间（TTFT，ms）：流式模式的均值/中位数/标准差/p99
- Token 间延迟（ITL，ms）：Token 之间的均值/中位数/标准差/p95/p99/最大值
- TPOT（ms）：首 Token 后的 Token 处理时间，即 `(latency - ttft)/(tokens-1)`
- Accept length（仅 sglang，如果可用）：推测解码接受长度

脚本还会使用配置的分词器重新对生成文本进行分词，并报告"重新分词"的计数。

### JSONL 输出格式

当设置 `--output-file` 时，每次运行追加一个 JSON 对象。基本字段：

- 参数摘要：backend、dataset、request_rate、max_concurrency 等
- 持续时间和总计：completed、total_input_tokens、total_output_tokens、重新分词的总计
- 与控制台打印相同的吞吐量和延迟统计
- 可用时的 `accept_length`（sglang）

使用 `--output-details` 时，扩展对象还包括数组：

- `input_lens`、`output_lens`
- `ttfts`、`itls`（每请求：ITL 数组）
- `generated_texts`、`errors`

### 端到端示例

1) SGLang 原生 `/generate`（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 100 \
  --max-concurrency 512 \
  --output-file sglang_random.jsonl --output-details
```

2) OpenAI 兼容 Completions（例如 vLLM）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --sharegpt-output-len 256
```

3) OpenAI 兼容 Chat Completions（流式）：

```bash
python3 -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --apply-chat-template
```

4) 图像（VLM）使用聊天模板：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 2 \
  --image-resolution 720p \
  --random-input-len 128 --random-output-len 256 \
  --num-prompts 200 \
  --apply-chat-template
```

4a) 自定义分辨率的图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 512x768 \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

4b) 1080p PNG 格式空白内容图像：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 1080p \
  --image-format png \
  --image-content blank \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

5) Generated shared prefix（长系统提示 + 短问题）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
  --num-prompts 1024
```

6) 分词后的提示（ID）用于严格长度控制（仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.2
```

7) 性能分析和缓存清除（sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile \
  --flush-cache
```

8) TensorRT-LLM 流式端点：

```bash
python3 -m sglang.bench_serving \
  --backend trt \
  --base-url http://127.0.0.1:8000 \
  --model your-trt-llm-model \
  --dataset-name random \
  --num-prompts 100 \
  --disable-ignore-eos
```

9) 使用 mooncake trace 评估大规模 KVCache 共享（仅 sglang）：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model model-name \
  --dataset-name mooncake \
  --mooncake-slowdown-factor 1.0 \
  --mooncake-num-rounds 1000 \
  --mooncake-workload conversation|mooncake|agent|synthetic
  --use-trace-timestamps true \
  --random-output-len 256
```

### 故障排除

- 所有请求失败：验证 `--backend`、服务器 URL/端口、`--model` 和身份验证。检查脚本打印的预热错误。
- 吞吐量似乎过低：调整 `--request-rate` 和 `--max-concurrency`；验证服务器批次大小/调度；确保在适当时启用流式传输。
- Token 计数看起来异常：优先使用带有正确聊天模板的 chat/instruct 模型；否则对无意义文本的分词可能不一致。
- 图像/MMMU 数据集：确保安装了额外的依赖（`pillow`、`datasets`、`pybase64`）。
- 身份验证错误（401/403）：设置 `OPENAI_API_KEY` 或在服务器上禁用身份验证。

### 说明

- 该脚本会提升文件描述符的软限制（`RLIMIT_NOFILE`），以支持大量并发连接。
- 对于 sglang，运行后会查询 `/get_server_info` 以报告可用的推测解码接受长度。
