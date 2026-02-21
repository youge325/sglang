# 奖励模型

这些模型输出标量奖励分数或分类结果，通常用于强化学习或内容审核任务。

```{important}
它们需要使用 `--is-embedding` 执行，部分可能需要 `--trust-remote-code`。
```

## 启动命令示例

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \  # HF/本地路径示例
  --is-embedding \
  --host 0.0.0.0 \
  --tp-size=4 \                          # 设置张量并行度
  --port 30000 \
```

## 支持的模型

| 模型系列（奖励模型）                                                     | HuggingFace 标识符示例                              | 描述                                                                     |
|---------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|
| **Llama (3.1 Reward / `LlamaForSequenceClassification`)**                   | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`            | 基于 Llama 3.1（8B）的奖励模型（偏好分类器），用于为 RLHF 对响应进行评分和排序。  |
| **Gemma 2 (27B Reward / `Gemma2ForSequenceClassification`)**                | `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`             | 基于 Gemma-2（27B）派生的模型，为 RLHF 和多语言任务提供人类偏好评分。  |
| **InternLM 2 (Reward / `InternLM2ForRewardMode`)**                         | `internlm/internlm2-7b-reward`                       | 基于 InternLM 2（7B）的奖励模型，用于对齐流程中引导输出向偏好行为靠拢。  |
| **Qwen2.5 (Reward - Math / `Qwen2ForRewardModel`)**                         | `Qwen/Qwen2.5-Math-RM-72B`                           | 来自 Qwen2.5 系列的 72B 数学专用 RLHF 奖励模型，专为评估和优化响应而调优。  |
| **Qwen2.5 (Reward - Sequence / `Qwen2ForSequenceClassification`)**          | `jason9693/Qwen2.5-1.5B-apeach`                      | 较小的 Qwen2.5 变体，用于序列分类，提供替代的 RLHF 评分机制。  |
