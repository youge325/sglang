# DeepSeek V3.2 使用指南

DeepSeek-V3.2 模型系列通过持续训练为 DeepSeek-V3.1-Terminus 配备了 DeepSeek Sparse Attention (DSA)。DSA 是一种由闪电索引器驱动的细粒度稀疏注意力机制，DeepSeek-V3.2 借此在长上下文场景中实现了效率提升。

如需报告问题或追踪即将推出的功能，请参阅此 [Roadmap](https://github.com/sgl-project/sglang/issues/11060)。

注意：本文档最初是为 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 模型的使用而编写的。[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 或 [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) 的使用方式与 DeepSeek-V3.2-Exp 相同，但 tool call parser 除外。


## 安装

### Docker

```bash
# H200/B200
docker pull lmsysorg/sglang:latest

# MI350/MI355
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi35x

# MI300
# v0.5.8-rocm700-mi30x 不包含 PR #17504。建议优先使用 Docker Hub 上最新的
# MI30x ROCm 镜像标签，或从源码构建（见下文）。
docker pull lmsysorg/sglang:v0.5.8-rocm700-mi30x


# NPU
docker pull lmsysorg/sglang:dsv32-a2
docker pull lmsysorg/sglang:dsv32-a3
```

### 从源码构建

```bash
# 安装 SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install -e "python"
```
## 使用 SGLang 启动 DeepSeek V3.2

在 8xH200/B200 GPU 上服务 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)：

```bash
# 使用 TP + DP 启动（推荐）
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention

# 使用 EP + DP 启动
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep 8 --dp 8 --enable-dp-attention

# 使用纯 TP 启动
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8

# 在 MI30x/MI35x 上使用 TP 启动
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --nsa-prefill-backend tilelang --nsa-decode-backend tilelang
```

### 配置建议
- **DP Attention（推荐）**：对于 DeepSeek V3.2 模型，内核专门针对 `dp_size=8` 的使用场景进行了定制，因此 DP attention（`--dp 8 --enable-dp-attention`）是推荐配置，可获得更好的稳定性和性能。所有测试用例默认使用此配置。
- **纯 TP 模式**：也支持使用纯 TP 启动（不带 `--dp` 和 `--enable-dp-attention`）。请注意，此模式尚未在 PD 分离部署场景中完全验证。
- **短序列 MHA 预填充（自适应）**：对于短预填充序列（默认阈值：**2048 tokens**），NSA 后端会自动使用标准 MHA（无需额外参数）。在 H200 (SM90) 上，此路径使用 FlashAttention 变长内核；在 B200 (SM100) 上使用 TRT-LLM ragged MHA。MHA 使用 `MHA_ONE_SHOT` 以获得最佳性能。`MHA_ONE_SHOT` 在单次内核调用中对所有 token（包括缓存的前缀和新扩展的 token）计算多头注意力，避免了分块 KV cache 处理的开销。这对于总序列长度在分块容量限制内的短序列可实现最优吞吐量。
- **注意力内核选择**：DeepSeek V3.2 模型的注意力后端会自动设置为 `nsa` 注意力后端。在此后端中，实现了不同的稀疏预填充/解码内核，可通过 `--nsa-prefill-backend` 和 `--nsa-decode-backend` 服务器参数指定。NSA 预填充/解码注意力内核的选项包括：
  - `flashmla_sparse`：来自 `flash_mla` 库的 `flash_mla_sparse_fwd` 内核。可在 Hopper 和 Blackwell GPU 上运行。要求 bf16 q、kv 输入。
  - `flashmla_kv`：来自 `flash_mla` 库的 `flash_mla_with_kvcache` 内核。可在 Hopper 和 Blackwell GPU 上运行。要求 bf16 q、fp8 k_cache 输入。
  - `fa3`：来自 `flash_attn` 库的 `flash_attn_with_kvcache` 内核。仅可在 Hopper GPU 上运行。要求 bf16 q、kv 输入。
  - `tilelang`：可在 GPU、HPU 和 NPU 上运行的 `tilelang` 实现。
  - `aiter`：AMD HPU 上的 Aiter 内核。仅可用作解码内核。
  - `trtllm`：来自 flashinfer 库的 `trtllm-mla` 稀疏内核。仅在 Blackwell GPU 上运行。要求 QKV bf16 或 QKV fp8。
- 根据性能基准测试，H200 和 B200 的默认配置如下：
  - H200：`flashmla_sparse` 预填充注意力（短序列预填充通过 FlashAttention varlen 使用 MHA），`fa3` 解码注意力，`bf16` kv cache 数据类型。
  - B200：`flashmla_auto` 预填充注意力（短序列预填充通过 TRT-LLM ragged 使用 MHA），`flashmla_kv` 解码注意力，`fp8_e4m3` kv cache 数据类型。`flashmla_auto` 可根据 KV cache 数据类型、硬件和启发式规则自动选择 `flashmla_sparse` 或 `flashmla_kv` 内核进行预填充。当启用 FP8 KV cache 且 `total_kv_tokens < total_q_tokens * 512` 时，使用 `flashmla_sparse` 内核；否则回退到 `flashmla_kv` 内核。如果 `flashmla_sparse` 或 `flashmla_kv` 内核的性能发生显著变化，启发式规则可能需要调整。
- 在 Blackwell 平台上，以轻微的精度损失为代价，性能可提升 3x-5x
  - B200：通过为 `--nsa-prefill-backend` 和 `--nsa-decode-backend` 都选择 `trtllm`，预填充注意力对短序列和长序列都使用 TRT-LLM ragged MHA（**有精度影响**）。将 `trtllm` 与 `fp8_e4m3` kv cache 结合使用时，kv cache 维度为 `576`（kv_lora_rank + qk_rope_head_dim）（**有精度影响**），相比之下，`flashmla_auto` 与 `fp8_e4m3` kv cache 的组合维度为 `656`（kv_lora_rank + scale 存储 (kv_lora_rank // quant_block_size * 4 bytes) + rope 维度存储）。


## 多 Token 预测 (MTP)
SGLang 基于 [EAGLE 投机解码](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding) 为 DeepSeek V3.2 实现了多 Token 预测 (MTP)。通过此优化，在小批量大小下解码速度可以得到显著提升。更多信息请查看[此 PR](https://github.com/sgl-project/sglang/pull/11652)。

使用 DP Attention 的示例：
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

使用纯 TP 的示例：
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

- `--speculative-num-steps`、`--speculative-eagle-topk` 和 `--speculative-num-draft-tokens` 的最佳配置可以使用 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 脚本针对给定的批量大小进行搜索。最小配置为 `--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`，可在较大批量大小下实现加速。
- MTP 的 `--max-running-requests` 默认值设为 `48`。对于更大的批量大小，应将此值增加到超过默认值。

```{tip}
要启用 EAGLE 投机解码的实验性重叠调度器，请设置环境变量 `SGLANG_ENABLE_SPEC_V2=1`。这可以通过启用草稿和验证阶段之间的重叠调度来提升性能。
```


## 函数调用和推理解析器
函数调用和推理解析器的使用方式与 DeepSeek V3.1 相同。请参阅 [Reasoning Parser](https://docs.sglang.io/advanced_features/separate_reasoning.html) 和 [Tool Parser](https://docs.sglang.io/advanced_features/tool_parser.html) 文档。

启动带有函数调用和推理解析器的 `DeepSeek-V3.2-Exp`：
> 注意：建议指定 chat-template，确保您位于 sglang 的根目录中。
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv31 \
  --reasoning-parser deepseek-v3 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja
```

启动带有函数调用和推理解析器的 `DeepSeek-V3.2`：
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2 \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --tool-call-parser deepseekv32 \
  --reasoning-parser deepseek-v3
```

`DeepSeek-V3.2-Speciale` 不支持工具调用，因此只能使用推理解析器启动：
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Speciale \
  --trust-remote-code \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --reasoning-parser deepseek-v3
```

## NVFP4 检查点

要在 Blackwell 设备上启动 DeepSeek V3.2 [NVFP4 检查点](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)，用户需要将量化方法指定为 `modelopt_fp4`，并将 moe runner 后端设为 `flashinfer_trtllm`（推荐）、`flashinfer_cutlass` 或 `flashinfer_cutedsl` 之一。其他用法（并行配置、推理解析器等）与 FP8 检查点相同。

启动命令示例：
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32  --reasoning-parser deepseek-v3
```

## PD 分离部署

预填充命令：
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode prefill \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --disaggregation-bootstrap-port 8998 \
        --mem-fraction-static 0.9 \
```

解码命令：
```bash
python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V3.2-Exp \
        --disaggregation-mode decode \
        --host $LOCAL_IP \
        --port $PORT \
        --tp 8 \
        --dp 8 \
        --enable-dp-attention \
        --dist-init-addr ${HOST}:${DIST_PORT} \
        --trust-remote-code \
        --mem-fraction-static 0.9 \
```

路由命令：
```bash
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill $PREFILL_ADDR 8998 \
  --decode $DECODE_ADDR \
  --host 127.0.0.1 \
  --port 8000 \
```

如需更高级的部署方式或生产就绪的部署方法，如基于 RBG 或 LWS 的部署，请参阅 [references/multi_node_deployment/rbg_pd/deepseekv32_pd.md](../references/multi_node_deployment/rbg_pd/deepseekv32_pd.md)。此外，您也可以在上述文档中找到基于 DeepEP 的 EP 并行启动命令。


## 基准测试结果

### 使用 `gsm8k` 进行精度测试
可以使用 `gsm8k` 数据集进行简单的精度基准测试：
```bash
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319
```

结果为 0.956，与我们的预期一致：
```bash
Accuracy: 0.956
Invalid: 0.000
Latency: 25.109 s
Output throughput: 5226.235 token/s
```

要测试长上下文精度，使用 `--num-shots 20` 运行 gsm8k。结果与 8 shots 的结果非常接近：
```
Accuracy: 0.956
Invalid: 0.000
Latency: 29.545 s
Output throughput: 4418.617 token/s
```


### 使用 `gpqa-diamond` 进行精度测试

长上下文精度基准测试可以在 GPQA-diamond 数据集上进行，启用长输出 token 和 thinking 模式：
```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --thinking-mode deepseek-v3
```

8 次运行的平均精度为 0.797，与官方技术报告中的 0.799 相符。
```bash
Repeat: 8, mean: 0.797
Scores: ['0.808', '0.798', '0.808', '0.798', '0.783', '0.788', '0.803', '0.793']
```

对于 DeepSeek V3.2，DeepSeek 推荐将采样参数设置为 temperature = 1.0、top_p = 0.95：

```bash
python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198 --max-tokens 128000 --repeat 8 --top-p 0.95 --temperature 1.0 --thinking-mode deepseek-v3

Repeat: 8, mean: 0.840
Scores: ['0.848', '0.808', '0.848', '0.838', '0.879', '0.813', '0.838', '0.848']
```
该结果与 [DeepSeek-V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf) 中公布的官方分数 0.824 相符。

### 使用 `aime 2025` 进行精度测试

在 Docker 或您自己的虚拟环境中安装 NeMo-Skills 以准备环境：

  ```
  pip install git+https://github.com/NVIDIA/NeMo-Skills.git --ignore-installed blinker
  ```

然后启动 SGLang 服务器：
```
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp 8 --enable-dp-attention
```

**对于 `DeepSeek-V3.2` 和 `DeepSeek-V3.2-Speciale`**：

```
python3 -m sglang.launch_server   --model-path deepseek-ai/DeepSeek-V3.2   --trust-remote-code   --tp-size 8 --dp-size 8 --enable-dp-attention   --tool-call-parser deepseekv32   --reasoning-parser deepseek-v3
```

运行以下脚本评估 AIME 2025：
```
#! /bin/bash
export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1

ns prepare_data aime25

PORT=30000
BACKEND=sglang
MODEL="deepseek-ai/DeepSeek-V3.2-Exp" # 应更改为对应的模型名称
MODEL_NAME="dsv32-fp8"

echo "Starting AIME25 evaluation with model $MODEL on port $PORT using backend $BACKEND..."
ns eval \
  --benchmarks=aime25:4 \
  --server_type=$BACKEND \
  --model=$MODEL \
  --server_address=http://localhost:${PORT}/v1 \
  --output_dir=nemo_skills_aime25_${MODEL_NAME}_output_${BACKEND}_$(date +%Y%m%d_%H%M%S) \
  ++chat_template_kwargs.thinking=true \
  ++inference.temperature=1.0 \
  ++inference.top_p=0.95 \
  ++inference.tokens_to_generate=64000
  # ++inference.tokens_to_generate=120000 用于 Speciale 模型
```

测试结果（8*B200）：

DeepSeek-V3.2-Exp：

| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 15040      | 1673        | 87.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 15040      | 1673        | 90.00%                | 0.00%     |
| pass@4             | 30          | 15040      | 1673        | 90.00%                | 0.00%     |


DeepSeek-V3.2：
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 13550      | 1632        | 92.50% ± 1.67%        | 0.00%     |
| majority@4         | 30          | 13550      | 1632        | 94.71%                | 0.00%     |
| pass@4             | 30          | 13550      | 1632        | 96.67%                | 0.00%     |


DeepSeek-V3.2-Speciale：
| evaluation_mode    | num_entries | avg_tokens | gen_seconds | symbolic_correct      | no_answer |
|--------------------|-------------|------------|-------------|-----------------------|-----------|
| pass@1[avg-of-4]   | 30          | 24155      | 3583        | 95.00% ± 1.92%        | 0.00%     |
| majority@4         | 30          | 24155      | 3583        | 95.83%                | 0.00%     |
| pass@4             | 30          | 24155      | 3583        | 100.00%               | 0.00%     |



## DSA 长序列上下文并行优化（实验性）

**注意：此功能仅在 Hopper 机器上验证过**

对于 DeepSeek V3.2 模型的上下文并行 (Context Parallel)，我们提供了两种不同的 token 切分模式，可通过参数 `--nsa-prefill-cp-mode` 控制。

### 序列内切分

第一种模式可通过 `--nsa-prefill-cp-mode in-seq-split` 启用。此模式通过在上下文并行 rank 之间均匀切分序列来实现 DSA 的上下文并行。在注意力阶段，每个 cp rank 计算切分序列的索引器结果，并通过 all gather 算子收集完整的 kv cache。添加 `attn_cp_size` 用于上下文并行的通信分组。

请注意，序列内切分模式有以下限制：
- 预填充批次的批量大小限制为 1
- 尚不支持多节点/PD 分离部署
- `moe_dense_tp_size=1`、`kv_cache_dtype = "bf16"`、`moe_a2a_backend = "deepep"`
- 为确保 `cp_size > 1`，传入的 `tp_size` 必须大于 `dp_size`

更多详情请参阅 PR https://github.com/sgl-project/sglang/pull/12065。

示例：
```bash
# 使用 EP + DP 启动的序列内切分模式
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --ep 8 --dp 2 --enable-dp-attention --enable-nsa-prefill-context-parallel --attn-cp-size 4 --nsa-prefill-cp-mode in-seq-split --max-running-requests 32
```

### 轮询切分（默认设置）

此模式可通过指定参数 `--nsa-prefill-cp-mode round-robin-split` 启用，根据 `token_idx % cp_size` 在 rank 之间分配 token。

与上述方法相比，此方案额外支持融合 MoE 后端（在单机场景下融合 MoE 后端可能比 DeepEP 提供更好的性能）、FP8 KV-cache 以及多批次预填充推理。但不能与 dp attention 同时启用。

更多详情请参阅 PR https://github.com/sgl-project/sglang/pull/13959。

使用示例：
```bash
# 使用 FusedMoe + CP8 启动
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp  --tp 8 --enable-nsa-prefill-context-parallel  --attn-cp-size 8 --nsa-prefill-cp-mode round-robin-split --max-running-requests 32
```
### 流水线并行 + 上下文并行 (PP + CP)

此模式结合了流水线并行 (PP) 和上下文并行 (CP) 以实现跨多节点扩展，可获得更好的吞吐量和首 Token 生成时间 (TTFT)。请注意，此方法仅在 H20 96G 上进行过测试。

#### 标准用法

在 2 个节点上使用 PP=2 和 CP（通过 `round-robin-split` 模式）启动。此配置默认使用融合 MoE 内核，通常提供更好的性能。

相关开发详情请参阅：
- 融合 MoE + CP 支持：[PR #13959](https://github.com/sgl-project/sglang/pull/13959)
- PP + CP 支持：[Issue #15358](https://github.com/sgl-project/sglang/issues/15358) 和 [PR #16380](https://github.com/sgl-project/sglang/pull/16380)

节点 0：
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

节点 1：
```bash
export SGLANG_PP_LAYER_PARTITION=30,31
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <HEAD_NODE_IP>:62001 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 128 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 8 \
  --page-size 64 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 --port 8000 \
  --tool-call-parser deepseekv32
```

#### 使用 PP + CP 的 PD 分离部署

如果使用 PD（预填充-解码）分离部署，预填充节点可按如下方式配置 PP + CP。

预填充节点 0：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32 \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

预填充节点 1：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --served-model-name deepseek-v32-prefill \
  --nnodes 2 --node-rank 1 \
  --dist-init-addr <PREFILL_HEAD_IP>:20102 \
  --tp 8 --pp-size 2 \
  --dp-size 1 --moe-dense-tp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split  \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --trust-remote-code \
  --disable-radix-cache \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
  --context-length 131072 \
  --mem-fraction-static 0.9 \
  --page-size 64 \
  --enable-metrics \
  --collect-tokens-histogram \
  --tokenizer-worker-num 8 \
  --host 0.0.0.0 --port 30000
```

对于解码节点，建议使用 **EP 模式**。
