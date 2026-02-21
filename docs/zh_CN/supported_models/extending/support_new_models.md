# 如何支持新模型

本文档说明如何在 SGLang 中添加对新语言模型和多模态大语言模型（MLLM）的支持。同时也涵盖了如何测试新模型和注册外部实现。

## 如何支持新的语言模型

要在 SGLang 中支持新模型，您只需在 [SGLang 模型目录](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models) 下添加一个文件。您可以参考现有模型实现并为您的模型创建新文件。对于大多数模型，您应该能找到一个类似的模型作为起点（例如从 Llama 开始）。也可以参考如何[从 vLLM 移植模型到 SGLang](#从-vllm-移植模型到-sglang)。

## 如何支持新的多模态大语言模型

要在 SGLang 中支持新的多模态大语言模型（MLLM），除了标准 LLM 支持外，还有几个关键组件：

1. **将新模型注册为多模态模型**：
   在 [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561) 中扩展 `is_multimodal_model`，使其对您的模型返回 `True`。

2. **注册新的聊天模板**：
   仅当您的默认聊天模板无法接受图像作为输入时：在 [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) 中注册新的聊天模板和相应的匹配函数。

3. **多模态数据处理器**：
   定义一个从 `BaseMultimodalProcessor` 继承的新 `Processor` 类，并将此处理器注册为您模型的专用处理器。
   更多详情请参见 [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)。

4. **处理多模态 Token**：
   为您的新模型实现 `pad_input_ids` 函数。在此函数中，提示中的多模态 token 应被扩展（如有必要）并用多模态数据哈希填充，以便 SGLang 能通过 `RadixAttention` 识别不同的多模态数据。

5. **处理图像特征提取**：
   为您的新模型实现 `get_image_feature` 函数，该函数从原始图像数据中提取图像特征，并将其转换为语言模型使用的嵌入。

6. **适配视觉注意力**：
   将 ViT 的多头 `Attention` 适配为 SGLang 的 `VisionAttention`。

您可以参考 [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) 或其他 MLLM 实现。这些模型展示了如何正确处理多模态和文本输入。

## 测试和调试

请在 PR 描述中记录所有测试和基准测试结果。

### 交互式调试

对于交互式调试，比较 Hugging Face/Transformers 和 SGLang 的输出。以下两个命令应给出相同的文本输出和非常相似的预填充 logits：

- 获取参考输出：
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [新模型] --model-type {text,mllm}
  ```
- 获取 SGLang 输出：
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [新模型]
  ```

### 将模型添加到测试套件

为确保新模型得到良好维护，请将其添加到测试套件中，具体方法是将其包含在 [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py) 文件的 `ALL_OTHER_MODELS` 列表中，在本地机器上测试新模型，并在 PR 中报告示范性基准测试（GSM8K、MMLU、MMMU、MMMU-Pro 等）的结果。\
对于 VLM，还需在 `test_vision_openai_server_{x}.py` 中添加测试（例如 [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py)、[test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)）。

以下是在本地机器上测试新模型的示例命令：

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### 基准测试

- **（必需）MMMU**：按照 MMMU 基准测试 [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) 获取 SGLang 与 HF Transformer 的精度对比。SGLang 运行的精度分数不应远低于 HF Transformer 运行的分数。类似地，按照 https://docs.sglang.io/developer_guide/benchmark_and_profiling.html 获取性能对比：TTFT 和吞吐量必须达到或超过基线（例如 HF Transformer）。
- **（可选）其他评估**：如果您运行了其他评估，请在 PR 描述中记录结果。

## 从 vLLM 移植模型到 SGLang

[vLLM 模型目录](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) 是一个宝贵的资源，因为 vLLM 覆盖了许多模型。SGLang 复用了 vLLM 的接口和一些层，使得从 vLLM 移植模型到 SGLang 更加容易。

要从 vLLM 移植模型：

- 比较以下两个文件作为参考：
  - [SGLang Llama 实现](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
  - [vLLM Llama 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- 主要区别包括：
  - **将 vLLM 的 `Attention` 替换为 `RadixAttention`**（确保将 `layer_id` 传递给 `RadixAttention`）。
  - **将 vLLM 的 `LogitsProcessor` 替换为 SGLang 的 `LogitsProcessor`。**
  - **将 ViT 的多头 `Attention` 替换为 SGLang 的 `VisionAttention`。**
  - **替换其他 vLLM 层**（如 `RMSNorm`、`SiluAndMul`）为 SGLang 层。
  - **移除 `Sample`。**
  - **修改 `forward()` 函数**并添加 `forward_batch()` 方法。
  - **添加 `EntryClass`** 到文件末尾。
  - **确保新实现仅使用 SGLang 组件**，不依赖任何 vLLM 组件。

注意：确保将新模型添加到支持模型文档中的支持模型列表中。

## 注册外部模型实现

除了上述方法外，您还可以在启动服务器之前使用 `ModelRegistry` 注册新模型。这允许您在不修改源代码的情况下集成模型。

例如：

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# 对于单个模型，将其添加到注册表：
ModelRegistry.models[model_name] = model_class

# 对于多个模型，可以模仿 import_model_classes() 函数：
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # 用您的新模型类填充 model_arch_name_to_cls。
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# 使用您的服务器参数启动服务器：
launch_server(server_args)
```

## 示例：实现和服务 Llama 包装模型

下面是一个入门级的分步指南，介绍如何在 SGLang 中端到端实现新模型，然后通过[离线引擎](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb)运行它。

### 实现我们的模型

为简单起见，这个新模型将是 [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 的简单包装，我们的目标只是通过对每个 logit 取平方根来偏置每次 `forward` 调用的输出 logits。

让我们首先在名为 `llama_wrapper.py` 的文件中定义模型。
第一步是从 SRT（SGLang 的内部后端）导入必要的库。

```python
# 在文件 `llama_wrapper.py` 中

import torch
from transformers import LlamaConfig
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.models.llama import LlamaForCausalLM
```

接下来，我们为模型声明一个新的 `class` 并让它继承自 `LlamaForCausalLM`，这允许我们的模型访问 `LlamaForCausalLM` 预定义的模块和层，如 `LlamaAttention` 和 `LlamaMLP`。
请注意，几乎所有模型实现的 `__init__` 方法都接受 `config` 和 `quant_config` 作为参数；`config` 和 `quant_config` 通过 [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219) 传入。
因为我们继承自 `LlamaForCausalLM`，所以可以直接将参数传递给其构造函数，它将为我们设置成员变量。

```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

现在，我们要定义 `forward` 方法，这是推理时会被调用的方法。
请注意，`forward` 的签名对于任何模型基本相同；您可以查看 [`models` 目录](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) 中定义的其他模型作为参考。
要了解 `forward` 在 SGLang 运行时内部究竟在哪里被调用，请查看 [`ModelRunner` 类](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py) 中的 [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) 和 [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724)。

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

现在我们调用 `self.model` 的 `__call__` 方法（这是 `LlamaForCausalLM` 在其 `__init__` 方法中定义的成员变量），它最终调用 `LlamaForCausalLM` 的 `forward` 方法。
之后，我们将 `hidden_states` 馈入模型的 `LogitsProcessor`（同样在 `LlamaForCausalLM` 中定义）。

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

在接收到下一个 token 的 logits 后，我们终于可以执行偏置步骤了。

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```

现在，我们的 `LlamaWrapper` 模型已创建完成，可以开始服务了！

### 通过 SGLang 的离线引擎服务我们的模型

本指南的下一步是在离线环境中托管新模型，使其可以在本地服务，无需 HTTP 服务器。

首先，创建一个名为 `run.py` 的新文件。
现在，我们必须确保 SGLang 的 `ModelRegistry` 能找到我们的模型。
为此，我们首先从 HuggingFace 下载模型的配置和权重。

```python
# 在文件 `run.py` 中

import asyncio
from functools import lru_cache
from huggingface_hub import snapshot_download
from llama_wrapper import LlamaWrapper # 确保导入我们的新模型！
import sglang as sgl
from sglang.srt.models.registry import ModelRegistry

# 确保在 HuggingFace 上请求访问此模型，然后导出您的
# `HF_TOKEN` 以下载模型快照
llama_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./llama_ckpt",
)
```

现在我们已将模型下载到磁盘上，需要通过修改 `./llama_ckpt/config.json` 中的 `architectures` 字段为 `LlamaWrapper` 来指向它。
这样，当我们将模型检查点的路径传递给 SGLang 时，它就知道我们要使用 "LlamaWrapper" 而不是 "LlamaForCausalLM" 作为模型。

```python
{
  "architectures": [
   #  "LlamaForCausalLM"
    "LlamaWrapper"
  ],
  ...
}
```

但是，如果我们不将 `LlamaWrapper` 类链接到 "LlamaWrapper" 注册关键字，SGLang 将无法找到我们的模型。
因此，要注册 `LlamaWrapper`，我们需要按照上面"注册外部模型实现"部分中的步骤操作。

```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

最后，创建 `Engine` 时，只需传入本地模型目录的路径。
然后，我们的 `LlamaWrapper` 就可以开始服务了；在本指南中，我们将使用 SGLang `Engine` 的非流式异步生成端点。

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France's capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```

现在，当我们运行 `python run.py` 时，将得到新创建模型的输出！

## 文档

将模型添加到 [generative_models.md](../text_generation/generative_models.md) 或 [multimodal_language_models.md](../text_generation/multimodal_language_models.md) 的支持模型表格中。

---

通过遵循这些指南，您可以在 SGLang 中添加对新语言模型和多模态大语言模型的支持，并确保它们经过充分测试且易于集成到系统中。
