# SGLang Diffusion CLI 推理

SGLang-Diffusion CLI 提供了一种快速访问图像和视频生成推理流水线的方式。

## 前置条件

- 已安装 SGLang Diffusion 且 `sglang` CLI 在 `$PATH` 中可用。


## 支持的参数

### 服务器参数

- `--model-path {MODEL_PATH}`：模型路径或模型 ID
- `--vae-path {VAE_PATH}`：自定义 VAE 模型路径或 HuggingFace 模型 ID（例如 `fal/FLUX.2-Tiny-AutoEncoder`）。如未指定，VAE 将从主模型路径加载。
- `--lora-path {LORA_PATH}`：LoRA 适配器路径（本地路径或 HuggingFace 模型 ID）。如未指定，则不应用 LoRA。
- `--lora-nickname {NAME}`：LoRA 适配器的昵称。（默认值：`default`）。
- `--num-gpus {NUM_GPUS}`：使用的 GPU 数量
- `--tp-size {TP_SIZE}`：张量并行度（仅用于编码器；如果启用了文本编码器卸载，则不应大于 1，因为逐层卸载加预取更快）
- `--sp-degree {SP_SIZE}`：序列并行度（通常应与 GPU 数量一致）
- `--ulysses-degree {ULYSSES_DEGREE}`：USP 中 DeepSpeed-Ulysses 风格序列并行的并行度
- `--ring-degree {RING_DEGREE}`：USP 中 Ring Attention 风格序列并行的并行度
- `--attention-backend {BACKEND}`：使用的注意力后端。对于 SGLang 原生流水线使用 `fa`、`torch_sdpa`、`sage_attn` 等。对于 diffusers 流水线使用 diffusers 后端名称如 `flash`、`_flash_3_hub`、`sage`、`xformers`。
- `--attention-backend-config {CONFIG}`：注意力后端的配置。可以是 JSON 字符串（例如 '{"k": "v"}'）、JSON/YAML 文件的路径或键值对（例如 "k=v,k2=v2"）。
- `--cache-dit-config {PATH}`：Cache-DiT YAML/JSON 配置文件路径（仅 diffusers 后端）
- `--dit-precision {DTYPE}`：DiT 模型的精度（目前支持 fp32、fp16 和 bf16）。


### 采样参数

- `--prompt {PROMPT}`：你想生成的视频的文本描述
- `--num-inference-steps {STEPS}`：去噪步数
- `--negative-prompt {PROMPT}`：负面提示，引导生成远离某些概念
- `--seed {SEED}`：用于可复现生成的随机种子


**图像/视频配置**

- `--height {HEIGHT}`：生成输出的高度
- `--width {WIDTH}`：生成输出的宽度
- `--num-frames {NUM_FRAMES}`：生成的帧数
- `--fps {FPS}`：如果是视频生成任务，保存输出的每秒帧数


**输出选项**

- `--output-path {PATH}`：保存生成视频的目录
- `--save-output`：是否将图像/视频保存到磁盘
- `--return-frames`：是否返回原始帧

### 使用配置文件

除了在命令行指定所有参数，你还可以使用配置文件：

```bash
sglang generate --config {CONFIG_FILE_PATH}
```

配置文件应为 JSON 或 YAML 格式，参数名称与 CLI 选项相同。命令行参数优先于配置文件中的设置，允许你在保留配置文件中其余设置的同时覆盖特定值。

配置文件示例（config.json）：

```json
{
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "output_path": "outputs/",
    "num_gpus": 2,
    "sp_size": 2,
    "tp_size": 1,
    "num_frames": 45,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 6,
    "seed": 1024,
    "fps": 24,
    "precision": "bf16",
    "vae_precision": "fp16",
    "vae_tiling": true,
    "vae_sp": true,
    "vae_config": {
        "load_encoder": false,
        "load_decoder": true,
        "tile_sample_min_height": 256,
        "tile_sample_min_width": 256
    },
    "text_encoder_precisions": [
        "fp16",
        "fp16"
    ],
    "mask_strategy_file_path": null,
    "enable_torch_compile": false
}
```

或使用 YAML 格式（config.yaml）：

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A beautiful woman in a red dress walking down a street"
output_path: "outputs/"
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: "bf16"
vae_precision: "fp16"
vae_tiling: true
vae_sp: true
vae_config:
  load_encoder: false
  load_decoder: true
  tile_sample_min_height: 256
  tile_sample_min_width: 256
text_encoder_precisions:
  - "fp16"
  - "fp16"
mask_strategy_file_path: null
enable_torch_compile: false
```


要查看所有选项，可以使用 `--help` 标志：

```bash
sglang generate --help
```

## Serve

启动 SGLang Diffusion HTTP 服务器，并使用 OpenAI SDK 和 curl 与之交互。

### 启动服务器

使用以下命令启动服务器：

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**：要加载的模型。示例使用 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。
- **--port**：监听的 HTTP 端口（此处默认为 `30010`）。

有关详细的 API 用法（包括图像、视频生成和 LoRA 管理），请参阅 [OpenAI API 文档](openai_api.md)。

### 云存储支持

SGLang Diffusion 支持自动将生成的图像和视频上传到 S3 兼容的云存储（如 AWS S3、MinIO、阿里云 OSS、腾讯云 COS）。

启用后，服务器遵循**生成 -> 上传 -> 删除**的工作流程：
1. 生成产物保存到临时本地文件。
2. 文件在后台线程中立即上传到配置的 S3 存储桶。
3. 上传成功后，删除本地文件。
4. API 响应返回上传对象的公开 URL。

**配置**

云存储通过环境变量启用。注意需要单独安装 `boto3`（`pip install boto3`）才能使用此功能。

```bash
# 启用 S3 存储
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=my-bucket
export SGLANG_S3_ACCESS_KEY_ID=your-access-key
export SGLANG_S3_SECRET_ACCESS_KEY=your-secret-key

# 可选：用于 MinIO/OSS/COS 的自定义端点
export SGLANG_S3_ENDPOINT_URL=https://minio.example.com
```

更多详情请参阅[环境变量文档](../environment_variables.md)。

## Generate

运行一次性生成任务，无需启动持久化服务器。

使用时，在 `generate` 子命令后同时传入服务器参数和采样参数，例如：

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

SAMPLING_ARGS=(
  --prompt "A curious raccoon"
  --save-output
  --output-path outputs
  --output-file-name "A curious raccoon.mp4"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

# 或者，用户可以将 `SGLANG_CACHE_DIT_ENABLED` 环境变量设为 `true` 来启用缓存加速
SGLANG_CACHE_DIT_ENABLED=true sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```

生成任务完成后，服务器会自动关闭。

> [!NOTE]
> 在此子命令中，HTTP 服务器相关的参数会被忽略。

## Diffusers 后端

SGLang Diffusion 支持 **diffusers 后端**，允许你通过 SGLang 的基础设施使用原生 diffusers 流水线运行任何兼容 diffusers 的模型。这对于运行没有原生 SGLang 实现的模型或具有自定义流水线类的模型很有用。

### 参数

| 参数 | 值 | 描述 |
|----------|--------|-------------|
| `--backend` | `auto`（默认）、`sglang`、`diffusers` | `auto`：优先使用原生 SGLang，回退到 diffusers。`sglang`：强制使用原生（不可用时失败）。`diffusers`：强制使用原生 diffusers 流水线。 |
| `--diffusers-attention-backend` | `flash`、`_flash_3_hub`、`sage`、`xformers`、`native` | diffusers 流水线的注意力后端。参见 [diffusers 注意力后端](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends)。 |
| `--trust-remote-code` | 标志 | 使用自定义流水线类的模型（如 Ovis）需要此选项。 |
| `--vae-tiling` | 标志 | 启用 VAE 分块以支持大图像（逐块解码）。 |
| `--vae-slicing` | 标志 | 启用 VAE 切片以降低内存使用（逐片解码）。 |
| `--dit-precision` | `fp16`、`bf16`、`fp32` | Diffusion Transformer 的精度。 |
| `--vae-precision` | `fp16`、`bf16`、`fp32` | VAE 的精度。 |

### 示例：运行 Ovis-Image-7B

[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) 是一个 7B 参数的文生图模型，专为高质量文字渲染而优化。

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output \
  --output-path outputs \
  --output-file-name ovis_garden.png
```

### 额外的 Diffusers 参数

对于未通过 CLI 暴露的流水线特定参数，请在配置文件中使用 `diffusers_kwargs`：

```json
{
    "model_path": "AIDC-AI/Ovis-Image-7B",
    "backend": "diffusers",
    "prompt": "A beautiful landscape",
    "diffusers_kwargs": {
        "cross_attention_kwargs": {"scale": 0.5}
    }
}
```

```bash
sglang generate --config config.json
```
