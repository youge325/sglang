# 嵌入模型

SGLang 通过将高效的服务机制与灵活的编程接口相集成，为嵌入模型提供了强大的支持。这种集成使嵌入任务的处理更加流畅，有助于实现更快、更准确的检索和语义搜索操作。SGLang 的架构能够更好地利用资源，降低嵌入模型部署中的延迟。

```{important}
嵌入模型需要使用 `--is-embedding` 标志执行，部分可能需要 `--trust-remote-code`。
```

## 快速开始

### 启动服务器

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

### 客户端请求

```python
import requests

url = "http://127.0.0.1:30000"

payload = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "What is the capital of France?",
    "encoding_format": "float"
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```



## 多模态嵌入示例

对于支持文本和图像的多模态模型（如 GME）：

```shell
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
  --is-embedding \
  --chat-template gme-qwen2-vl \
  --host 0.0.0.0 \
  --port 30000
```

```python
import requests

url = "http://127.0.0.1:30000"

text_input = "Represent this image in embedding space."
image_path = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"

payload = {
    "model": "gme-qwen2-vl",
    "input": [
        {
            "text": text_input
        },
        {
            "image": image_path
        }
    ],
}

response = requests.post(url + "/v1/embeddings", json=payload).json()

print("Embeddings:", [x.get("embedding") for x in response.get("data", [])])
```

## 俄罗斯套娃嵌入示例

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) 或 [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) 是一种用于训练嵌入模型的技术。它允许用户在性能和成本之间进行权衡。

### 1. 启动支持 Matryoshka 的模型

如果模型配置中已包含 `matryoshka_dimensions` 或 `is_matryoshka`，则无需额外覆盖。否则，可以使用 `--json-model-override-args`，如下所示：

```shell
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Embedding-0.6B \
    --is-embedding \
    --host 0.0.0.0 \
    --port 30000 \
    --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

1. 设置 `"is_matryoshka": true` 允许截断到任意维度。否则，服务器将验证请求中指定的维度是否属于 `matryoshka_dimensions`。
2. 在请求中省略 `dimensions` 将返回完整向量。

### 2. 使用不同的输出维度发起请求

```python
import requests

url = "http://127.0.0.1:30000"

# 通过指定支持的维度来请求截断的（Matryoshka）嵌入。
payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Explain diffusion models simply.",
    "dimensions": 512  # 更改为 128 / 1024 / 省略以获取完整尺寸
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```


## 支持的模型

| 模型系列                               | 模型示例                          | 聊天模板 | 描述                                                                 |
| ------------------------------------------ | -------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| **E5 (基于 Llama/Mistral)**              | `intfloat/e5-mistral-7b-instruct`     | N/A           | 基于 Mistral/Llama 架构的高质量文本嵌入          |
| **GTE-Qwen2**                             | `Alibaba-NLP/gte-Qwen2-7B-instruct`   | N/A           | 阿里巴巴的文本嵌入模型，支持多语言                   |
| **Qwen3-Embedding**                       | `Qwen/Qwen3-Embedding-4B`             | N/A           | 最新的基于 Qwen3 的文本嵌入模型，用于语义表示        |
| **BGE**                                    | `BAAI/bge-large-en-v1.5`              | N/A           | BAAI 的文本嵌入（需要 `attention-backend` 为 triton/torch_native）  |
| **GME（多模态）**                      | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`| `gme-qwen2-vl`| 用于文本和图像跨模态任务的多模态嵌入                  |
| **CLIP**                                   | `openai/clip-vit-large-patch14-336`   | N/A           | OpenAI 的 CLIP，用于图像和文本嵌入                                |
