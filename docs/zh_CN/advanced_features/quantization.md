# 量化

SGLang 支持多种量化方法，包括离线量化和在线动态量化。

离线量化在推理过程中直接加载预量化的模型权重。这是 GPTQ 和 AWQ 等量化方法所必需的，这些方法使用校准数据集从原始权重中收集和预计算各种统计信息。

在线量化在运行时动态计算缩放参数——如模型权重的最大/最小值。
类似于 NVIDIA FP8 训练的[延迟缩放](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#Mixed-precision-training-with-FP8)机制，在线量化即时计算适当的缩放因子，将高精度权重转换为低精度格式。

**注意：为了获得更好的性能、可用性和便利性，推荐使用离线量化而非在线量化。**

如果使用预量化模型，请不要同时添加 `--quantization` 来启用在线量化。
对于流行的预量化模型，请访问 HF 上的 [Unsloth](https://huggingface.co/unsloth)、[NVIDIA ModelOpt](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer)
或 [NeuralMagic](https://huggingface.co/collections/neuralmagic) 集合，获取经过质量验证的量化模型。量化模型必须在量化后通过基准测试进行验证，以防止异常的量化损失退化。

## 离线量化

要加载已量化的模型，只需加载模型权重和配置。**再次强调，如果模型已经离线量化，启动引擎时无需添加 `--quantization` 参数。量化方法将从下载的 Hugging Face 配置中解析。例如，DeepSeek V3/R1 模型已经是 FP8 格式，不要添加冗余参数。**

```bash
python3 -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --port 30000 --host 0.0.0.0
```

注意，如果您的模型是**按通道量化（INT8 或 FP8）且使用按 token 动态量化激活**的，可以选择包含 `--quantization w8a8_int8` 或 `--quantization w8a8_fp8` 来调用 sgl-kernel 中对应的 CUTLASS int8_kernel 或 fp8_kernel。

```bash
python3 -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
    --quantization w8a8_fp8 \
    --port 30000 --host 0.0.0.0
```

### 离线模型量化示例

#### 使用 [Unsloth](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)

我们强烈建议使用 Unsloth 来量化和加载模型。请参阅 [SGLang 部署与推理指南（Unsloth）](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)。

#### 使用 [auto-round](https://github.com/intel/auto-round)

```bash
# 安装
pip install auto-round
```

- LLM 量化

```py
from auto_round import AutoRound
model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-autoround-4bit"
scheme = "W4A16"
format = "auto_round"
autoround = AutoRound(model_id, scheme=scheme)
autoround.quantize_and_save(quant_path, format=format)
```

- VLM 量化

```py
from auto_round import AutoRoundMLLM
model_name = "Qwen/Qwen2-VL-2B-Instruct"
quant_path = "Qwen2-VL-2B-Instruct-autoround-4bit"
scheme = "W4A16"
format = "auto_round"
autoround = AutoRoundMLLM(model_name, scheme)
autoround.quantize_and_save(quant_path, format=format)
```

- 命令行用法（Gaudi/CPU/Intel GPU/CUDA）

```bash
auto-round \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --bits 4 \
    --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

## 在线量化

### FP8 量化

要为模型启用 FP8 在线量化，添加 `--quantization fp8`。这会自动将 FP16/BF16 模型权重量化为 FP8。

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

### ModelOpt 量化

ModelOpt 提供了多种量化方案。要使用 ModelOpt 量化，首先安装 nvidia-modelopt：

```bash
pip install nvidia-modelopt
```

支持的量化配置包括：`fp8`、`int4_awq`、`w4a8_awq`、`nvfp4`、`nvfp4_awq`。

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --modelopt-quant fp8 \
    --port 30000 --host 0.0.0.0
```

```{note}
这是量化文档的精简中文版本。完整的量化方法列表和详细说明请参阅 [英文文档](../../en/advanced_features/quantization.html)。
```
