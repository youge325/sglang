# Llama4 使用指南

[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) 是 Meta 最新一代的开源大语言模型，具有业界领先的性能。

SGLang 自 [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5) 起已支持 Llama 4 Scout (109B) 和 Llama 4 Maverick (400B)。

持续的优化工作在 [Roadmap](https://github.com/sgl-project/sglang/issues/5118) 中跟踪。

## 使用 SGLang 启动 Llama 4

在 8xH100/H200 GPU 上部署 Llama 4 模型：

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 8 \
  --context-length 1000000
```

### 配置建议

- **OOM 缓解**：调整 `--context-length` 以避免 GPU 内存不足问题。对于 Scout 模型，建议在 8\*H100 上将此值设置为最高 1M，在 8\*H200 上最高 2.5M。对于 Maverick 模型，在 8\*H200 上无需设置上下文长度。当启用混合 KV 缓存时，Scout 模型的 `--context-length` 在 8\*H100 上可设置为最高 5M，在 8\*H200 上最高 10M。

- **注意力后端自动选择**：SGLang 会根据您的硬件自动为 Llama 4 选择最优的注意力后端。通常无需手动指定 `--attention-backend`：
  - **Blackwell GPU (B200/GB200)**：`trtllm_mha`
  - **Hopper GPU (H100/H200)**：`fa3`
  - **AMD GPU**：`aiter`
  - **Intel XPU**：`intel_xpu`
  - **其他平台**：`triton`（回退方案）

  如需覆盖自动选择，请显式指定 `--attention-backend`，可选值为：`fa3`、`aiter`、`triton`、`trtllm_mha` 或 `intel_xpu`。

- **聊天模板**：为聊天补全任务添加 `--chat-template llama-4`。
- **启用多模态**：添加 `--enable-multimodal` 以启用多模态功能。
- **启用混合 KV 缓存**：设置 `--swa-full-tokens-ratio` 来调整 SWA 层（对于 Llama4，即局部注意力层）KV token 与全层 KV token 的比例。（默认值：0.8，范围：0-1）


### EAGLE 推测解码
**描述**：SGLang 已支持 Llama 4 Maverick (400B) 的 [EAGLE 推测解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding)。

**用法**：
添加参数 `--speculative-draft-model-path`、`--speculative-algorithm`、`--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 来启用此功能。例如：
```
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path nvidia/Llama-4-Maverick-17B-128E-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --trust-remote-code \
  --tp 8 \
  --context-length 1000000
```

- **注意** Llama 4 草稿模型 *nvidia/Llama-4-Maverick-17B-128E-Eagle3* 只能识别聊天模式下的对话。

## 基准测试结果

### 使用 `lm_eval` 进行精度测试

SGLang 上 Llama4 Scout 和 Llama4 Maverick 的精度均可匹配[官方基准测试数据](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。

在 8*H100 上的 MMLU Pro 数据集基准测试结果：
|                    | Llama-4-Scout-17B-16E-Instruct | Llama-4-Maverick-17B-128E-Instruct  |
|--------------------|--------------------------------|-------------------------------------|
| 官方基准           | 74.3                           | 80.5                                |
| SGLang             | 75.2                           | 80.7                                |

命令：

```bash
# Llama-4-Scout-17B-16E-Instruct 模型
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0

# Llama-4-Maverick-17B-128E-Instruct
python -m sglang.launch_server \
  --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --port 30000 \
  --tp 8 \
  --mem-fraction-static 0.8 \
  --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0
```

详情请参阅[此 PR](https://github.com/sgl-project/sglang/pull/5092)。
