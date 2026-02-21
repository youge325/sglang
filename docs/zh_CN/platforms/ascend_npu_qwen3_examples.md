# Qwen3 示例

### 运行 Qwen3

#### 在 1 台 Atlas 800I A3 上运行 Qwen3-32B

模型权重可在[此处](https://huggingface.co/Qwen/Qwen3-32B)获取。

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8
```

#### 在 1 台 Atlas 800I A3 上使用 Qwen3-32B-Eagle3 运行 Qwen3-32B

模型权重可在[此处](https://huggingface.co/Qwen/Qwen3-32B)获取。

投机模型权重可在[此处](https://huggingface.co/Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3)获取。

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8 \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path Qwen/Qwen3-32B-Eagle3 \
   --speculative-num-steps 1 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 2
```

#### 在 1 台 Atlas 800I A3 上运行 Qwen3-30B-A3B MOE

模型权重可在[此处](https://huggingface.co/Qwen/Qwen3-30B-A3B)获取。

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-30B-A3B \
   --mem-fraction-static 0.8
```

#### 在 1 台 Atlas 800I A3 上运行 Qwen3-235B-A22B-Instruct-2507 MOE

模型权重可在[此处](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)获取。

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
   --tp-size 16 \
   --trust-remote-code \
   --attention-backend ascend \
   --device npu \
   --watchdog-timeout 9000 \
   --mem-fraction-static 0.8
```

#### 在 1 台 Atlas 800I A3 上运行 Qwen3-VL-8B-Instruct

模型权重可在[此处](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)获取。

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --enable-multimodal \
   --attention-backend ascend \
   --mm-attention-backend ascend_attn \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-VL-8B-Instruct \
   --mem-fraction-static 0.8
```
