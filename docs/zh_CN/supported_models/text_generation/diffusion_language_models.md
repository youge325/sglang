# 扩散语言模型

扩散语言模型在具有并行解码能力的非自回归文本生成方面展现了良好前景。与自回归语言模型不同，不同的扩散语言模型需要不同的解码策略。

## 启动命令示例

SGLang 支持不同的 DLLM 算法，如 `LowConfidence` 和 `JointThreshold`。

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # HF/本地路径示例
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # 可选。未设置时使用算法默认值。
  --host 0.0.0.0 \
  --port 30000
```

## 配置文件示例

根据选择的算法不同，配置参数也有所不同。

LowConfidence 配置：

```yaml
# 接受预测 token 的置信度阈值
# - 较高值：更保守，质量更好但速度更慢
# - 较低值：更激进，速度更快但质量可能较低
# 范围：0.0 - 1.0
threshold: 0.95

# 默认值：32，用于 LLaDA2MoeModelLM
block_size: 32
```

JointThreshold 配置：

```yaml
# Mask-to-Token (M2T) 阶段的解码阈值
# - 较高值：更保守，质量更好但速度更慢
# - 较低值：更激进，速度更快但质量可能较低
# 范围：0.0 - 1.0
threshold: 0.5
# Token-to-Token (T2T) 阶段的解码阈值
# 范围：0.0 - 1.0
# 设置为 0.0 允许完全编辑（大多数情况下推荐）。
edit_threshold: 0.0
# 所有掩码移除后的最大额外 T2T 步数。防止无限循环。
max_post_edit_steps: 16
# 2-gram 重复惩罚（默认为 0）。
# 经验值 3 通常足以缓解大多数重复。
penalty_lambda: 0
```

## 客户端代码示例

与其他支持的模型一样，扩散语言模型可以通过 REST API 或 Python 客户端使用。

Python 客户端向已启动的服务器发起生成请求的示例：

```python
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="inclusionAI/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidence",
                     max_running_requests=1,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
```

Curl 向已启动的服务器发起生成请求的示例：

```bash
curl -X POST "http://127.0.0.1:30000/generate" \
     -H "Content-Type: application/json" \
     -d '{
        "text": [
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write the number from 1 to 128 <|role_end|><role>ASSISTANT</role>",
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
        ],
        "stream": true,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024
        }
    }'
```

## 支持的模型

下面以表格形式汇总了支持的模型。

| 模型系列               | 模型示例                | 描述                                                                                          |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash 是一个采用 100B 混合专家（MoE）架构的扩散语言模型。 |
| **SDAR (JetLM)**           | `JetLM/SDAR-8B-Chat`         | SDAR 系列扩散语言模型（Chat），密集架构。                                 |
| **SDAR (JetLM)**           | `JetLM/SDAR-30B-A3B-Chat`    | SDAR 系列扩散语言模型（Chat），MoE 架构。                                   |
