# 多模态语言模型

这些模型接受多模态输入（例如图像和文本）并生成文本输出。它们通过多模态编码器增强语言模型的能力。

## 启动命令示例

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \  # HF/本地路径示例
  --host 0.0.0.0 \
  --port 30000 \
```

> 有关如何发送多模态请求，请参阅 [OpenAI API 部分](https://docs.sglang.io/basic_usage/openai_api_vision.html)。

## 支持的模型

下面以表格形式汇总了支持的模型。

如果您不确定某个特定架构是否已实现，可以通过 GitHub 搜索。例如，要搜索 `Qwen2_5_VLForConditionalGeneration`，请在 GitHub 搜索栏中使用以下表达式：

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen2_5_VLForConditionalGeneration
```


| 模型系列（变体）    | HuggingFace 标识符示例             | 描述                                                                                                                                                                                                     | 备注 |
|----------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| **Qwen-VL** | `Qwen/Qwen3-VL-235B-A22B-Instruct`              | 阿里巴巴 Qwen 的视觉语言扩展；例如 Qwen2.5-VL（7B 及更大变体）可以分析和讨论图像内容。                                                                     |  |
| **DeepSeek-VL2**           | `deepseek-ai/deepseek-vl2`                 | DeepSeek 的视觉语言变体（配备专用图像处理器），支持对图像和文本输入进行高级多模态推理。                                                                        |  |
| **DeepSeek-OCR / OCR-2**   | `deepseek-ai/DeepSeek-OCR-2`               | 面向 OCR 的 DeepSeek 模型，用于文档理解和文本提取。                                                                                                                                    | 需使用 `--trust-remote-code`。 |
| **Janus-Pro** (1B, 7B)     | `deepseek-ai/Janus-Pro-7B`                 | DeepSeek 的开源多模态模型，能够同时进行图像理解和生成。Janus-Pro 采用解耦架构，为视觉编码提供独立的路径，从而提升两项任务的性能。 |  |
| **MiniCPM-V / MiniCPM-o**  | `openbmb/MiniCPM-V-2_6`                    | MiniCPM-V（2.6，约 8B）支持图像输入，MiniCPM-o 增加了音频/视频支持；这些多模态 LLM 针对移动端/边缘设备的端侧部署进行了优化。                                                 |  |
| **Llama 3.2 Vision** (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Llama 3 的视觉增强变体（11B），接受图像输入，用于视觉问答和其他多模态任务。                                                                                     |  |
| **LLaVA** (v1.5 和 v1.6)    | *例如* `liuhaotian/llava-v1.5-13b`         | 开源视觉对话模型，为 LLaMA/Vicuna 添加图像编码器（如 LLaMA2 13B），用于遵循多模态指令提示。                                                                               |  |
| **LLaVA-NeXT** (8B, 72B)   | `lmms-lab/llava-next-72b`                  | 改进版 LLaVA 模型（含 8B Llama3 版本和 72B 版本），在多模态基准测试中提供更好的视觉指令遵循和准确性。                                                       |  |
| **LLaVA-OneVision**        | `lmms-lab/llava-onevision-qwen2-7b-ov`     | 增强型 LLaVA 变体，以 Qwen 作为骨干网络；通过 OpenAI Vision API 兼容格式支持多张图像（甚至视频帧）作为输入。                                                 |  |
| **Gemma 3（多模态）**   | `google/gemma-3-4b-it`                     | Gemma 3 的较大模型（4B、12B、27B）可在 128K token 的组合上下文中同时接受图像（每张图像编码为 256 个 token）和文本。                                                                        |  |
| **Kimi-VL** (A3B)          | `moonshotai/Kimi-VL-A3B-Instruct`          | Kimi-VL 是一个多模态模型，能够理解图像并生成文本。                                                                                                                                |  |
| **Mistral-Small-3.1-24B**  | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral 3.1 是一个多模态模型，可以从文本或图像输入生成文本。还支持工具调用和结构化输出。 |  |
| **Phi-4-multimodal-instruct**  | `microsoft/Phi-4-multimodal-instruct` | Phi-4-multimodal-instruct 是 Phi-4-mini 模型的多模态变体，通过 LoRA 增强了多模态能力。在 SGLang 中支持文本、视觉和音频模态。 |  |
| **MiMo-VL** (7B)           | `XiaomiMiMo/MiMo-VL-7B-RL`                 | 小米的紧凑但强大的视觉语言模型，采用原生分辨率 ViT 编码器捕捉细粒度视觉细节、MLP 投影器进行跨模态对齐，以及针对复杂推理任务优化的 MiMo-7B 语言模型。 |  |
| **GLM-4.5V** (106B) / **GLM-4.1V**(9B)           | `zai-org/GLM-4.5V`                   | GLM-4.5V 和 GLM-4.1V-Thinking：通过可扩展强化学习实现多功能多模态推理                                                                                                                                                                                                      | 使用 `--chat-template glm-4v` |
| **GLM-OCR**          | `zai-org/GLM-OCR`                   | GLM-OCR：快速准确的通用 OCR 模型                                                                   |  |
| **DotsVLM** (通用/OCR)  | `rednote-hilab/dots.vlm1.inst`             | 小红书的视觉语言模型，基于 1.2B 视觉编码器和 DeepSeek V3 LLM 构建，采用从头训练的 NaViT 视觉编码器，支持动态分辨率，并通过结构化图像数据训练增强了 OCR 能力。 |  |
| **DotsVLM-OCR**            | `rednote-hilab/dots.ocr`                   | DotsVLM 的 OCR 专用变体，针对光学字符识别任务进行优化，具备增强的文本提取和文档理解能力。 | 不要使用 `--trust-remote-code` |
| **NVILA** (8B, 15B, Lite-2B, Lite-8B, Lite-15B) | `Efficient-Large-Model/NVILA-8B` | `chatml` | NVILA 探索多模态设计的全栈效率，实现更低训练成本、更快部署和更好性能。 |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | NVIDIA Nemotron Nano v2 VL 支持多图推理和视频理解，以及强大的文档智能、视觉问答和摘要能力。它基于 Nemotron Nano V2（混合 Mamba-Transformer LLM）构建，以在长文档和视频场景中实现更高的推理吞吐量。 | 需使用 `--trust-remote-code`。您可能需要调整 `--max-mamba-cache-size`（默认为 512）以适应内存限制。 |
| **Ernie4.5-VL** | `baidu/ERNIE-4.5-VL-28B-A3B-PT`              | 百度的视觉语言模型（28B, 424B）。支持图像和视频理解，也支持思维链推理。                                                                     |  |
| **JetVLM** |  | JetVLM 是基于 Jet-Nemotron 构建的视觉语言模型，设计用于高性能多模态理解和生成任务。 | 即将推出 |
| **Step3-VL** (10B) | `stepfun-ai/Step3-VL-10B` | 阶跃星辰的轻量级开源 10B 参数 VLM，用于多模态智能，在视觉感知、复杂推理和人类对齐方面表现出色。 |  |
| **Qwen3-Omni** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 阿里巴巴的全模态 MoE 模型。目前支持 **Thinker** 组件（文本、图像、音频和视频的多模态理解），而 **Talker** 组件（音频生成）尚不支持。 |  |

## 视频输入支持

SGLang 支持视觉语言模型（VLM）的视频输入，支持时序推理任务，如视频问答、视频描述和整体场景理解。视频片段被解码，关键帧被采样，生成的张量与文本提示一起批处理，使多模态推理能够整合视觉和语言上下文。

| 模型系列 | 标识符示例 | 视频说明 |
|--------------|--------------------|-------------|
| **Qwen-VL** (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-Omni) | `Qwen/Qwen3-VL-235B-A22B-Instruct` | 处理器收集 `video_data`，运行 Qwen 的帧采样器，并在推理前将生成的特征与文本 token 合并。 |
| **GLM-4v** (4.5V, 4.1V, MOE) | `zai-org/GLM-4.5V` | 视频片段使用 Decord 读取，转换为张量，并与元数据一起传递给模型以处理旋转位置编码。 |
| **NVILA** (Full 和 Lite) | `Efficient-Large-Model/NVILA-8B` | 运行时每个片段采样八帧，当存在 `video_data` 时将其附加到多模态请求中。 |
| **LLaVA 视频变体** (LLaVA-NeXT-Video, LLaVA-OneVision) | `lmms-lab/LLaVA-NeXT-Video-7B` | 处理器将视频提示路由到 LlavaVid 视频架构，示例展示了如何使用 `sgl.video(...)` 片段进行查询。 |
| **NVIDIA Nemotron Nano 2.0 VL** | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | 处理器按 2 FPS 采样，最多 128 帧，与模型训练配置一致。该模型使用 [EVS](../../python/sglang/srt/multimodal/evs/README.md)，一种从视频嵌入中移除冗余 token 的剪枝方法。默认 `video_pruning_rate=0.7`。可通过提供 `--json-model-override-args '{"video_pruning_rate": 0.0}'` 来禁用 EVS。 |
| **JetVLM** |  | 运行时每个片段采样八帧，当存在 `video_data` 时将其附加到多模态请求中。 |

在构建提示时使用 `sgl.video(path, num_frames)` 来附加视频片段。

与 OpenAI 兼容的发送视频片段请求示例：

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
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

## 使用说明

### 性能优化

对于多模态模型，您可以使用 `--keep-mm-feature-on-device` 标志来优化延迟，但会增加 GPU 内存使用：

- **默认行为**：多模态特征张量在处理后移到 CPU 以节省 GPU 内存
- **使用 `--keep-mm-feature-on-device`**：特征张量保留在 GPU 上，减少设备到主机的复制开销并改善延迟，但消耗更多 GPU 内存

当您有足够的 GPU 内存并希望最小化多模态推理延迟时，请使用此标志。

### 多模态输入限制

- **使用 `--mm-process-config '{"image":{"max_pixels":1048576},"video":{"fps":3,"max_pixels":602112,"max_frames":60}}'`**：设置 `image`、`video` 和 `audio` 输入限制。

这可以减少 GPU 内存使用、提高推理速度，并有助于避免 OOM，但可能影响模型性能，因此请根据您的具体用例设置合适的值。目前仅 `qwen_vl` 支持此配置。请参阅 [qwen_vl 处理器](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/multimodal/processors/qwen_vl.py) 了解每个参数的含义。

### 多模态模型服务中的双向注意力
**关于 Gemma-3 多模态模型的服务说明**：

如 [Welcome Gemma 3: Google's all new multimodal, multilingual, long context open LLM](https://huggingface.co/blog/gemma3#multimodality) 中所述，Gemma-3 在预填充阶段在图像 token 之间采用双向注意力。目前，SGLang 仅在使用 Triton 注意力后端时支持双向注意力。但请注意，SGLang 当前的双向注意力实现与 CUDA Graph 和分块预填充不兼容。

要启用双向注意力，您可以使用 `TritonAttnBackend` 并禁用 CUDA Graph 和分块预填充。启动命令示例：
```shell
python -m sglang.launch_server \
  --model-path google/gemma-3-4b-it \
  --host 0.0.0.0 --port 30000 \
  --enable-multimodal \
  --dtype bfloat16 --triton-attention-reduce-in-fp32 \
  --attention-backend triton \ # 使用 Triton 注意力后端
  --disable-cuda-graph \ # 禁用 CUDA Graph
  --chunked-prefill-size -1 # 禁用分块预填充
```

如果需要更高的服务性能且可以接受一定程度的精度损失，您可以选择使用其他注意力后端，也可以启用 CUDA Graph 和分块预填充等功能以获得更好的性能，但请注意模型将回退到使用因果注意力而非双向注意力。
