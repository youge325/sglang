# SGLang 中的选择方法
本文档介绍 SGLang 支持的选择方法（choices methods）。

可选的 `choices_method` 参数决定了 SGLang 的 `choices` 原语如何选择提供的选项。只有 `RuntimeEndpoint` 后端支持 `choices_method` 参数。其他后端（如 `OpenAI`）由于 API 限制使用特定的选择实现。

## 方法

### Token 长度归一化

Token 长度归一化是 SGLang 的默认选择方法。它选择所有 token 中平均 logprob 最高的选项。

使用示例（也可以直接省略 `choices_method` 参数）：
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.token_length_normalized,
        )
    )
```


如果某个选项包含很多 token，且后面的 token 基于前面的 token 以高置信度预测，则此方法可能表现不佳。例如，即使是强大的模型在指定选项为 `["Paris", "Antidisestablishmentarianism"]` 时也会在上述示例中失败。

### 贪心 Token 选择

贪心 Token 选择简单地选择初始 token logprob 最高的选项。对于重叠选项（一个选项是另一个更长选项的子集），较短选项的 logprob 将使用其平均 logprob 进行扩展，以便与较长选项进行比较。

使用示例：
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

如果某个选项因为有吸引力的初始 token 而误导模型走向错误路径，此方法可能表现不佳。例如，贪心选择在以下示例中会导致错误的响应：
```python
@sgl.function
def us_president_example(s):
    s += sgl.user("Name a US president.")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["Donald Duck", "Millard Fillmore"],
            choices_method=sgl.greedy_token_selection,
        )
    )
```

### 无条件似然归一化

无条件似然归一化选择经无条件 token logprob 归一化后平均 token logprob 最高的选项，如 [这篇 EleutherAI 博文](https://blog.eleuther.ai/multiple-choice-normalization/) 所述。此方法需要额外的一次 LLM 调用来获取无条件似然。

使用示例：
```python
@sgl.function
def example(s):
    s += sgl.user("What is the capital of France?")
    s += sgl.assistant(
        sgl.gen(
            "answer",
            choices=["London", "Paris", "Berlin"],
            choices_method=sgl.unconditional_likelihood_normalized,
        )
    )
```
