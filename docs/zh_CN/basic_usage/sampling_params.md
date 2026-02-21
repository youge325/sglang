# 采样参数

本文档描述了 SGLang Runtime 的采样参数。这是运行时的低级端点。
如果您需要自动处理聊天模板的高级端点，请考虑使用 [OpenAI 兼容 API](openai_api_completions.ipynb)。

## `/generate` 端点

`/generate` 端点接受以下 JSON 格式的参数。详细用法请参见[原生 API 文档](native_api.ipynb)。对象定义在 `io_struct.py::GenerateReqInput` 中。您也可以阅读源代码查找更多参数和文档。

| 参数                       | 类型/默认值                                                                 | 描述                                                                                                                                                     |
|----------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text                       | `Optional[Union[List[str], str]] = None`                                     | 输入提示词。可以是单个提示词或一批提示词。                                                                                                 |
| input_ids                  | `Optional[Union[List[List[int]], List[int]]] = None`                         | 文本的 token ID；可以指定 text 或 input_ids 之一。                                                                                               |
| input_embeds               | `Optional[Union[List[List[List[float]]], List[List[float]]]] = None`         | input_ids 的嵌入向量；可以指定 text、input_ids 或 input_embeds 之一。                                                                          |
| image_data                 | `Optional[Union[List[List[ImageDataItem]], List[ImageDataItem], ImageDataItem]] = None` | 图像输入。支持三种格式：(1) **原始图像**: PIL Image、文件路径、URL 或 base64 字符串；(2) **处理器输出**: 包含 `format: "processor_output"` 的字典，包含 HuggingFace 处理器输出；(3) **预计算嵌入**: 包含 `format: "precomputed_embedding"` 和 `feature` 的字典，包含预计算的视觉嵌入。可以是单个图像、图像列表或列表的列表。详见[多模态输入格式](#multimodal-input-formats)。 |
| audio_data                 | `Optional[Union[List[AudioDataItem], AudioDataItem]] = None`                 | 音频输入。可以是文件名、URL 或 base64 编码字符串。                                                                                             |
| sampling_params            | `Optional[Union[List[Dict], Dict]] = None`                                   | 采样参数，详见下方各节描述。                                                                                                     |
| rid                        | `Optional[Union[List[str], str]] = None`                                     | 请求 ID。                                                                                                                                                 |
| return_logprob             | `Optional[Union[List[bool], bool]] = None`                                   | 是否返回 token 的对数概率。                                                                                                                 |
| logprob_start_len          | `Optional[Union[List[int], int]] = None`                                     | 如果 return_logprob 为 true，返回 logprob 的提示词起始位置。默认为 "-1"，即仅返回输出 token 的 logprob。                     |
| top_logprobs_num           | `Optional[Union[List[int], int]] = None`                                     | 如果 return_logprob 为 true，每个位置返回的 top logprob 数量。                                                                                       |
| token_ids_logprob          | `Optional[Union[List[List[int]], List[int]]] = None`                         | 如果 return_logprob 为 true，要返回 logprob 的 token ID。                                                                                                         |
| return_text_in_logprobs    | `bool = False`                                                               | 是否在返回的 logprob 中对 token 进行反分词处理。                                                                                                  |
| stream                     | `bool = False`                                                               | 是否启用流式输出。                                                                                                                       |
| lora_path                  | `Optional[Union[List[Optional[str]], Optional[str]]] = None`                 | LoRA 适配器的路径。                                                                                                                           |
| custom_logit_processor     | `Optional[Union[List[Optional[str]], str]] = None`                           | 用于高级采样控制的自定义 logit 处理器。必须是使用 `to_str()` 方法序列化的 `CustomLogitProcessor` 实例。用法详见下方。 |
| return_hidden_states       | `Union[List[bool], bool] = False`                                            | 是否返回隐藏状态。                                                                                                                |
| return_routed_experts      | `bool = False`                                                               | 是否返回 MoE 模型的路由专家。需要 `--enable-return-routed-experts` 服务器标志。返回 base64 编码的 int32 专家 ID，为扁平数组，逻辑形状为 `[num_tokens, num_layers, top_k]`。 |

## 采样参数

对象定义在 `sampling_params.py::SamplingParams` 中。您也可以阅读源代码查找更多参数和文档。

### 关于默认值的说明

默认情况下，SGLang 从模型的 `generation_config.json` 初始化多个采样参数（当服务器以 `--sampling-defaults model` 启动时，这是默认行为）。要使用 SGLang/OpenAI 常量默认值，请使用 `--sampling-defaults openai` 启动服务器。您始终可以通过 `sampling_params` 在每个请求中覆盖任何参数。

```bash
# 使用 generation_config.json 中模型提供的默认值（默认行为）
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults model

# 使用 SGLang/OpenAI 常量默认值
python -m sglang.launch_server --model-path <MODEL> --sampling-defaults openai
```

### 核心参数

| 参数            | 类型/默认值                                  | 描述                                                                                                                                    |
|-----------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| max_new_tokens  | `int = 128`                                  | 以 token 为单位的最大输出长度。                                                                                                  |
| stop            | `Optional[Union[str, List[str]]] = None`     | 一个或多个[停止词](https://platform.openai.com/docs/api-reference/chat/create#chat-create-stop)。如果采样到其中一个词，将停止生成。 |
| stop_token_ids  | `Optional[List[int]] = None`                 | 以 token ID 形式提供的停止词。如果采样到其中一个 token ID，将停止生成。                                        |
| stop_regex      | `Optional[Union[str, List[str]]] = None`     | 当匹配到此列表中的任何正则表达式模式时停止 |
| temperature     | `float（模型默认值；回退为 1.0）`           | 采样下一个 token 时的[温度](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)。`temperature = 0` 对应贪心采样，更高的温度会带来更多多样性。 |
| top_p           | `float（模型默认值；回退为 1.0）`           | [Top-p](https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p) 从累积概率超过 `top_p` 的最小排序集合中选择 token。当 `top_p = 1` 时，等同于从所有 token 中无限制采样。 |
| top_k           | `int（模型默认值；回退为 -1）`              | [Top-k](https://developer.nvidia.com/blog/how-to-get-better-outputs-from-your-large-language-model/#predictability_vs_creativity) 从概率最高的 `k` 个 token 中随机选择。 |
| min_p           | `float（模型默认值；回退为 0.0）`           | [Min-p](https://github.com/huggingface/transformers/issues/27670) 从概率大于 `min_p * 最高token概率` 的 token 中采样。 |

### 惩罚参数

| 参数               | 类型/默认值          | 描述                                                                                                                                    |
|--------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| frequency_penalty  | `float = 0.0`          | 根据 token 在目前生成中出现的频率进行惩罚。取值范围为 `-2` 到 `2`，负值鼓励重复 token，正值鼓励采样新 token。惩罚程度随 token 每次出现线性增长。 |
| presence_penalty   | `float = 0.0`          | 如果 token 在目前生成中出现过则进行惩罚。取值范围为 `-2` 到 `2`，负值鼓励重复 token，正值鼓励采样新 token。如果 token 出现过，惩罚程度恒定不变。 |
| repetition_penalty | `float = 1.0`          | 缩放已生成 token 的 logit，以抑制（值 > 1）或鼓励（值 < 1）重复。有效范围为 `[0, 2]`；`1.0` 表示概率不变。 |
| min_new_tokens     | `int = 0`              | 强制模型生成至少 `min_new_tokens` 个 token，直到采样到停止词或 EOS token。注意这可能导致意外行为，例如当分布高度偏向这些 token 时。 |

### 约束解码

有关以下参数，请参阅我们的[约束解码专用指南](../advanced_features/structured_outputs.ipynb)。

| 参数            | 类型/默认值                     | 描述                                                                                                                                    |
|-----------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| json_schema     | `Optional[str] = None`          | 用于结构化输出的 JSON Schema。                                                                                                            |
| regex           | `Optional[str] = None`          | 用于结构化输出的正则表达式。                                                                                                                  |
| ebnf            | `Optional[str] = None`          | 用于结构化输出的 EBNF。                                                                                                                   |
| structural_tag  | `Optional[str] = None`          | 用于结构化输出的结构标签。                                                                                                       |

### 其他选项

| 参数                          | 类型/默认值                     | 描述                                                                                                                                    |
|-------------------------------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| n                             | `int = 1`                       | 指定每个请求生成的输出序列数量。（不推荐在一个请求中生成多个输出(n > 1)；重复多次相同的提示词提供更好的控制和效率。） |
| ignore_eos                    | `bool = False`                  | 采样到 EOS token 时不停止生成。                                                                                               |
| skip_special_tokens           | `bool = True`                   | 解码时移除特殊 token。                                                                                                         |
| spaces_between_special_tokens | `bool = True`                   | 反分词时是否在特殊 token 之间添加空格。                                                                     |
| no_stop_trim                  | `bool = False`                  | 不从生成文本中修剪停止词或 EOS token。                                                                                    |
| custom_params                 | `Optional[List[Optional[Dict[str, Any]]]] = None` | 使用 `CustomLogitProcessor` 时使用。用法详见下方。                                                                              |

## 示例

### 普通请求

启动服务器：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

详细示例请参见[发送请求](./send_request.ipynb)。

### 流式输出

发送请求并流式获取输出：

```python
import requests, json

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"].strip()
        print(output[prev:], end="", flush=True)
        prev = len(output)
print("")
```

详细示例请参见 [OpenAI 兼容 API](openai_api_completions.ipynb)。

### 多模态

启动服务器：

```bash
python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov
```

下载图像：

```bash
curl -o example_image.png -L https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\nDescribe this image in a very short sentence.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "image_data": "example_image.png",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)
print(response.json())
```

`image_data` 可以是文件名、URL 或 base64 编码的字符串。另见 `python/sglang/srt/utils.py:load_image`。

流式输出的使用方式与[上文](#流式输出)类似。

详细示例请参见 [OpenAI API Vision](openai_api_vision.ipynb)。

### 结构化输出（JSON、正则表达式、EBNF）

您可以指定 JSON Schema、正则表达式或 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) 来约束模型输出。模型输出将保证遵循给定的约束。每个请求只能指定一个约束参数（`json_schema`、`regex` 或 `ebnf`）。

SGLang 支持两种语法后端：

- [XGrammar](https://github.com/mlc-ai/xgrammar)（默认）：支持 JSON Schema、正则表达式和 EBNF 约束。
  - XGrammar 目前使用 [GGML BNF 格式](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)。
- [Outlines](https://github.com/dottxt-ai/outlines)：支持 JSON Schema 和正则表达式约束。

如果要使用 Outlines 后端，可以使用 `--grammar-backend outlines` 标志：

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --host 0.0.0.0 --grammar-backend [xgrammar|outlines] # xgrammar 或 outlines（默认: xgrammar）
```

```python
import json
import requests

json_schema = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[\\w]+$"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
})

# JSON（Outlines 和 XGrammar 均可使用）
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of France in the JSON format.\n",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)
print(response.json())

# 正则表达式（仅 Outlines 后端）
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Paris is the capital of",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print(response.json())

# EBNF（仅 XGrammar 后端）
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Write a greeting.",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ebnf": 'root ::= "Hello" | "Hi" | "Hey"',
        },
    },
)
print(response.json())
```

详细示例请参见[结构化输出](../advanced_features/structured_outputs.ipynb)。

### 自定义 logit 处理器

使用 `--enable-custom-logit-processor` 标志启动服务器。

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 30000 \
  --enable-custom-logit-processor
```

定义一个自定义 logit 处理器，始终采样特定的 token ID。

```python
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class DeterministicLogitProcessor(CustomLogitProcessor):
    """一个简单的 logit 处理器，修改 logits 使其
    总是采样给定的 token id。
    """

    def __call__(self, logits, custom_param_list):
        # 检查 logits 数量与自定义参数数量是否匹配
        assert logits.shape[0] == len(custom_param_list)
        key = "token_id"

        for i, param_dict in enumerate(custom_param_list):
            # 屏蔽所有其他 token
            logits[i, :] = -float("inf")
            # 将最高概率分配给指定的 token
            logits[i, param_dict[key]] = 0.0
        return logits
```

发送请求：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 32,
            "custom_params": {"token_id": 5},
        },
    },
)
print(response.json())
```

发送 OpenAI 聊天补全请求：

```python
import openai
from sglang.utils import print_highlight

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0.0,
    max_tokens=32,
    extra_body={
        "custom_logit_processor": DeterministicLogitProcessor().to_str(),
        "custom_params": {"token_id": 5},
    },
)

print_highlight(f"Response: {response}")
```
