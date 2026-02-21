# 使用 SGLang 启动 GLM-4.5 / GLM-4.6 / GLM-4.7

在 8xH100/H200 GPU 上部署 GLM-4.5 / GLM-4.6 FP8 模型：

```bash
python3 -m sglang.launch_server --model zai-org/GLM-4.6-FP8 --tp 8
```

### EAGLE 推测解码

**说明**：SGLang 已支持 GLM-4.5 / GLM-4.6 模型的 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding)。

**用法**：
添加 `--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和
`--speculative-num-draft-tokens` 参数以启用此功能。例如：

``` bash
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.6-FP8 \
  --tp-size 8 \
  --tool-call-parser glm45  \
  --reasoning-parser glm45  \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3  \
  --speculative-eagle-topk 1  \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.9 \
  --served-model-name glm-4.6-fp8 \
  --enable-custom-logit-processor
```

```{tip}
要启用实验性的 EAGLE 推测解码 overlap 调度器，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。这可以通过在草稿和验证阶段之间启用重叠调度来提升性能。
```

### GLM-4.5 / GLM-4.6 的 Thinking Budget（思考预算）
**注意**：对于 GLM-4.7，`--tool-call-parser` 应设置为 `glm47`；对于 GLM-4.5 和 GLM-4.6，应设置为 `glm45`。

在 SGLang 中，我们可以通过 `CustomLogitProcessor` 实现 thinking budget。

启动服务器时需开启 `--enable-custom-logit-processor` 标志。

请求示例：

```python
import openai
from rich.pretty import pprint
from sglang.srt.sampling.custom_logit_processor import Glm4MoeThinkingBudgetLogitProcessor


client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="*")
response = client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[
        {
            "role": "user",
            "content": "Question: Is Paris the Capital of France?",
        }
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": Glm4MoeThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,
        },
    },
)
pprint(response)
```
