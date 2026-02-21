# SGLang 中的 Transformers 回退

`sglang` 可以回退使用 `transformers` 中可用的模型。这适用于大多数解码器风格的语言模型，对视觉语言模型的支持即将推出！

## 启动命令示例

默认情况下，如果有 SGLang 实现则使用 SGLang 实现，否则回退到 Transformers 实现。但您可以通过将 `--model-impl` 设置为 `transformers` 来切换实现。

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --model-impl transformers
```

## 支持的功能

### 量化

Transformers 回退支持 SGLang 中大多数可用的量化方式（GGUF 除外）。更多关于 SGLang 支持的量化信息，请参见[量化页面](../advanced_features/quantization.md)。

### 远程代码

此回退还意味着 Hub 上任何可以在 `transformers` 中使用 `trust_remote_code=True` 并正确实现注意力机制的模型都可以在生产环境中使用！

模型只需要满足以下两点：

```python
from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):

  def forward(self, hidden_states, **kwargs): # <- kwargs 是必需的

    ...
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
      self,
      query_states,
      key_states,
      value_states,
      **kwargs,
    )
    ...

class MyModel(PreTrainedModel):
  _supports_attention_backend = True
```

以下是后台发生的事情：

1. 加载配置
2. 从 `auto_map` 加载 `MyModel` Python 类，并检查模型是否 `_supports_attention_backend`。
3. 使用 `TransformersModel` 后端。参见 `/srt/models/transformers`，它利用 `self.config._attn_implementation = "sglang"`，因此需要使用 `ALL_ATTENTION_FUNCTIONS`。

就是这样！
