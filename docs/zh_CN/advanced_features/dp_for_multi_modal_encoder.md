# SGLang 中多模态编码器的数据并行 (DP)

典型的 VLM（视觉语言模型）架构包括两个主要组件：一个多模态编码器和一个文本解码器。

大多数 VLM 使用 Vision Transformer (ViT) 作为其多模态编码器，负责处理视觉数据、提取特征（物体、颜色、纹理等），并将其转换为模型能够理解的格式。

文本解码器基于 LLM，负责处理文本数据并根据编码后的视觉特征生成输出。

然而，由于 ViT 的规模相比语言解码器非常小，
TP（张量并行）带来的增益相对较少。另一方面，由于每层之后都需要执行 all-reduce 操作，TP 会产生显著的通信开销。

将 ViT 置于数据并行模式，同时保持 LLM 使用张量并行，可以持续降低 TTFT（首 token 时间）并提升端到端吞吐量。在这种混合布局中，视觉前端变得并行且轻量，而稀缺的互连带宽和集合操作则留给 LLM。

数据并行将整个模型复制到多组 GPU 上，并行处理不同批次的请求。

## 命令示例
您可以通过设置 `mm-enable-dp-encoder` 来启用批级 DP，例如：
```
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp 2 \
    --mm-enable-dp-encoder
```

## 已知支持的模型
- Qwen2.5-VL (<https://github.com/sgl-project/sglang/pull/13126>)
- Qwen3-VL (<https://github.com/sgl-project/sglang/pull/13724>)
- InternVL (<https://github.com/sgl-project/sglang/pull/13925>)
- GLM-4.5V & GLM-4.6V (<https://github.com/sgl-project/sglang/pull/14097>)
