# 重排序模型

SGLang 通过整合优化的服务框架和灵活的编程接口，为重排序模型提供了全面的支持。此配置使跨编码器重排序任务能够高效处理，提高搜索结果排序的准确性和相关性。SGLang 的设计确保了重排序模型部署时的高吞吐量和低延迟，使其成为大规模检索系统中基于语义的结果精细化的理想选择。

```{important}
SGLang 中的重排序模型分为两类：

- **跨编码器重排序模型**：使用 `--is-embedding`（嵌入运行器）运行。
- **仅解码器重排序模型**：**不使用** `--is-embedding` 运行，使用下一个 token 的 logprob 评分（yes/no）。
  - 纯文本（例如 Qwen3-Reranker）
  - 多模态（例如 Qwen3-VL-Reranker）：也支持图像/视频内容

部分模型可能需要 `--trust-remote-code`。
```

## 支持的重排序模型

| 模型系列（重排序）                          | HuggingFace 标识符示例       | 聊天模板 | 描述                                                                                                                      |
|------------------------------------------------|--------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| **BGE-Reranker (BgeRerankModel)**              | `BAAI/bge-reranker-v2-m3`            | N/A           | 目前仅支持 `attention-backend` 为 `triton` 和 `torch_native`。BAAI 的高性能跨编码器重排序模型，适用于基于语义相关性的搜索结果重排序。   |
| **Qwen3-Reranker（仅解码器 yes/no）**       | `Qwen/Qwen3-Reranker-8B`             | `examples/chat_template/qwen3_reranker.jinja` | 使用下一个 token logprob 评分标签（yes/no）的仅解码器重排序模型。启动时**不使用** `--is-embedding`。 |
| **Qwen3-VL-Reranker（多模态 yes/no）**      | `Qwen/Qwen3-VL-Reranker-2B`          | `examples/chat_template/qwen3_vl_reranker.jinja` | 支持文本、图像和视频的多模态仅解码器重排序模型。使用 yes/no logprob 评分。启动时**不使用** `--is-embedding`。 |


## 跨编码器重排序（嵌入运行器）

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding \
  --port 30000
```

### 客户端请求示例

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ],
    "top_n": 1,
    "return_documents": True
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    if item.get("document"):
        print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    else:
        print(f"Score: {item['score']:.2f} - Index: {item['index']}")
```

**请求参数：**

- `query`（必需）：用于对文档进行排序的查询文本
- `documents`（必需）：需要排序的文档列表
- `model`（必需）：用于重排序的模型
- `top_n`（可选）：返回的最大文档数。默认返回所有文档。如果指定的值大于文档总数，则返回所有文档。
- `return_documents`（可选）：是否在响应中返回文档。默认为 `True`。

## Qwen3-Reranker（仅解码器 yes/no 重排序）

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 8001 \
  --chat-template examples/chat_template/qwen3_reranker.jinja
```

```{note}
Qwen3-Reranker 使用仅解码器的 logprob 评分（yes/no）。**不要**使用 `--is-embedding` 启动。
```

### 客户端请求示例（支持可选的 instruct、top_n 和 return_documents）

```shell
curl -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "法国首都是哪里？",
    "documents": [
      "法国的首都是巴黎。",
      "德国的首都是柏林。",
      "香蕉是黄色的水果。"
    ],
    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    "top_n": 2,
    "return_documents": true
  }'
```

**请求参数：**

- `query`（必需）：用于对文档进行排序的查询文本
- `documents`（必需）：需要排序的文档列表
- `model`（必需）：用于重排序的模型
- `instruct`（可选）：重排序器的指令文本
- `top_n`（可选）：返回的最大文档数。默认返回所有文档。如果指定的值大于文档总数，则返回所有文档。
- `return_documents`（可选）：是否在响应中返回文档。默认为 `True`。

### 响应格式

`/v1/rerank` 返回一个对象列表（按分数降序排列）：

- `score`：浮点数，越高表示越相关
- `document`：原始文档字符串（仅在 `return_documents` 为 `true` 时包含）
- `index`：输入 `documents` 中的原始索引
- `meta_info`：可选的调试/使用信息（某些模型可能存在）

返回结果的数量由 `top_n` 参数控制。如果未指定 `top_n` 或其值大于文档总数，则返回所有文档。

示例（`return_documents: true`）：

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1},
  {"score": 0.00, "document": "香蕉是黄色的水果。", "index": 2}
]
```

示例（`return_documents: false`）：

```json
[
  {"score": 0.99, "index": 0},
  {"score": 0.01, "index": 1},
  {"score": 0.00, "index": 2}
]
```

示例（`top_n: 2`）：

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1}
]
```

### 常见问题

- 如果使用 `--is-embedding` 启动 Qwen3-Reranker，`/v1/rerank` 将无法计算 yes/no logprob 分数。请**不使用** `--is-embedding` 重新启动。
- 如果看到类似 "score should be a valid number" 的验证错误且后端返回了列表，请升级到能将 `embedding[0]` 强制转换为 `score` 的版本以用于重排序响应。

## Qwen3-VL-Reranker（多模态仅解码器重排序）

Qwen3-VL-Reranker 将 Qwen3-Reranker 扩展为支持多模态内容，允许对包含文本、图像和视频的文档进行重排序。

### 启动命令

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-Reranker-2B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000 \
  --chat-template examples/chat_template/qwen3_vl_reranker.jinja
```

```{note}
Qwen3-VL-Reranker 像 Qwen3-Reranker 一样使用仅解码器的 logprob 评分（yes/no）。**不要**使用 `--is-embedding` 启动。
```

### 纯文本重排序（向后兼容）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "Qwen3-VL-Reranker-2B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        "The weather in Paris is usually mild with occasional rain.",
        "Deep learning is a subset of machine learning using neural networks with many layers.",
    ],
    "instruct": "Retrieve passages that answer the question.",
    "return_documents": True
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Score: {item['score']:.4f} - {item['document'][:60]}...")
```

### 图像重排序（文本查询，图像/混合文档）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "query": "A woman playing with her dog on a beach at sunset.",
    "documents": [
        # 文档 1：文本描述
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
        # 文档 2：图像 URL
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ],
        # 文档 3：文本 + 图像（混合）
        [
            {"type": "text", "text": "A joyful scene at the beach:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/beach_dog.jpeg"
                }
            }
        ]
    ],
    "instruct": "Retrieve images or text relevant to the user's query.",
    "return_documents": False
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### 多模态查询重排序（带图像的查询）

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    # 包含文本和图像的查询
    "query": [
        {"type": "text", "text": "Find similar images to this:"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/reference_image.jpeg"
            }
        }
    ],
    "documents": [
        "A cat sleeping on a couch.",
        "A woman and her dog enjoying the sunset at the beach.",
        "A busy city street with cars and pedestrians.",
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/similar_image.jpeg"
                }
            }
        ]
    ],
    "instruct": "Find images or descriptions similar to the query image."
}

response = requests.post(url, json=payload)
results = response.json()

for item in results:
    print(f"Index: {item['index']}, Score: {item['score']:.4f}")
```

### 请求参数（多模态）

- `query`（必需）：可以是字符串（纯文本）或内容部分列表：
  - `{"type": "text", "text": "..."}` 用于文本
  - `{"type": "image_url", "image_url": {"url": "..."}}` 用于图像
  - `{"type": "video_url", "video_url": {"url": "..."}}` 用于视频
- `documents`（必需）：列表，每个文档可以是字符串或内容部分列表（与查询格式相同）
- `instruct`（可选）：重排序器的指令文本
- `top_n`（可选）：返回的最大文档数
- `return_documents`（可选）：是否在响应中返回文档（默认：`false`）

### 常见问题

- Qwen3-VL-Reranker 始终使用 `--chat-template examples/chat_template/qwen3_vl_reranker.jinja`。
- **不要**使用 `--is-embedding` 启动。
- 为获得最佳效果，请使用 `--disable-radix-cache` 以避免多模态内容的缓存问题。
- **注意**：目前仅测试和支持 `Qwen3-VL-Reranker-2B`。8B 模型可能存在不同的行为，并且不保证能与此模板一起使用。
