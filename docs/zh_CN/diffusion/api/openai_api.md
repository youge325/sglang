# SGLang Diffusion OpenAI API

SGLang Diffusion HTTP 服务器实现了兼容 OpenAI 的 API，用于图像和视频生成以及 LoRA 适配器管理。

## 前置条件

- 如果你计划使用 OpenAI Python SDK，需要 Python 3.11+。

## Serve

使用 `sglang serve` 命令启动服务器。

### 启动服务器

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
  --port 30010
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**：模型路径或模型 ID。
- **--port**：监听的 HTTP 端口（默认：`30000`）。

**获取模型信息**

**端点：** `GET /models`

返回此服务器提供的模型信息，包括模型路径、任务类型、流水线配置和精度设置。

**Curl 示例：**

```bash
curl -sS -X GET "http://localhost:30010/models"
```

**响应示例：**

```json
{
  "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "task_type": "T2V",
  "pipeline_name": "wan_pipeline",
  "pipeline_class": "WanPipeline",
  "num_gpus": 4,
  "dit_precision": "bf16",
  "vae_precision": "fp16"
}
```

---

## 端点

### 图像生成

服务器在 `/v1/images` 命名空间下实现了兼容 OpenAI 的 Images API。

**创建图像**

**端点：** `POST /v1/images/generations`

**Python 示例（b64_json 响应）：**

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

img = client.images.generate(
    prompt="A calico cat playing a piano on stage",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

**Curl 示例：**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json"
      }'
```

> **注意**
> `POST /v1/images/generations` 不支持 `response_format=url` 选项，将返回 `400` 错误。

**编辑图像**

**端点：** `POST /v1/images/edits`

此端点接受包含输入图像和文本提示的 multipart 表单上传。服务器可以返回 base64 编码的图像或下载图像的 URL。

**Curl 示例（b64_json 响应）：**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=b64_json"
```

**Curl 示例（URL 响应）：**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=url"
```

**下载图像内容**

当 `POST /v1/images/edits` 使用 `response_format=url` 时，API 返回一个相对 URL，如 `/v1/images/<IMAGE_ID>/content`。

**端点：** `GET /v1/images/{image_id}/content`

**Curl 示例：**

```bash
curl -sS -L "http://localhost:30010/v1/images/<IMAGE_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.png
```

### 视频生成

服务器在 `/v1/videos` 命名空间下实现了 OpenAI Videos API 的子集。

**创建视频**

**端点：** `POST /v1/videos`

**Python 示例：**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

video = client.videos.create(
    prompt="A calico cat playing a piano on stage",
    size="1280x720"
)
print(f"Video ID: {video.id}, Status: {video.status}")
```

**Curl 示例：**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1280x720"
      }'
```

**列出视频**

**端点：** `GET /v1/videos`

**Python 示例：**

```python
videos = client.videos.list()
for item in videos.data:
    print(item.id, item.status)
```

**Curl 示例：**

```bash
curl -sS -X GET "http://localhost:30010/v1/videos" \
  -H "Authorization: Bearer sk-proj-1234567890"
```

**下载视频内容**

**端点：** `GET /v1/videos/{video_id}/content`

**Python 示例：**

```python
import time

# 轮询等待完成
while True:
    page = client.videos.list()
    item = next((v for v in page.data if v.id == video_id), None)
    if item and item.status == "completed":
        break
    time.sleep(5)

# 下载内容
resp = client.videos.download_content(video_id=video_id)
with open("output.mp4", "wb") as f:
    f.write(resp.read())
```

**Curl 示例：**

```bash
curl -sS -L "http://localhost:30010/v1/videos/<VIDEO_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.mp4
```

---

### LoRA 管理

服务器支持动态加载、合并和取消合并 LoRA 适配器。

**重要说明：**
- 互斥性：同一时间只能有一个 LoRA 被*合并*（激活）
- 切换：要切换 LoRA，必须先 `unmerge` 当前的，然后 `set` 新的
- 缓存：服务器在内存中缓存已加载的 LoRA 权重。切换回之前加载过的 LoRA（相同路径）几乎没有开销

**设置 LoRA 适配器**

加载一个或多个 LoRA 适配器并将其权重合并到模型中。支持单个 LoRA（向后兼容）和多个 LoRA 适配器。

**端点：** `POST /v1/set_lora`

**参数：**
- `lora_nickname`（字符串或字符串列表，必需）：LoRA 适配器的唯一标识符。可以是单个字符串或多个 LoRA 的字符串列表
- `lora_path`（字符串或字符串/None 列表，可选）：`.safetensors` 文件或 Hugging Face 仓库 ID 的路径。首次加载时必需；重新激活缓存的昵称时可选。如果是列表，长度必须与 `lora_nickname` 匹配
- `target`（字符串或字符串列表，可选）：将 LoRA 应用到哪个 transformer。如果是列表，长度必须与 `lora_nickname` 匹配。有效值：
  - `"all"`（默认）：应用到所有 transformer
  - `"transformer"`：仅应用到主 transformer（Wan2.2 的高噪声）
  - `"transformer_2"`：仅应用到 transformer_2（Wan2.2 的低噪声）
  - `"critic"`：仅应用到 critic 模型
- `strength`（浮点数或浮点数列表，可选）：合并时的 LoRA 强度，默认 1.0。如果是列表，长度必须与 `lora_nickname` 匹配。值 < 1.0 减弱效果，值 > 1.0 增强效果

**单个 LoRA 示例：**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": "lora_name",
        "lora_path": "/path/to/lora.safetensors",
        "target": "all",
        "strength": 0.8
      }'
```

**多个 LoRA 示例：**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["lora_1", "lora_2"],
        "lora_path": ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"],
        "target": ["transformer", "transformer_2"],
        "strength": [0.8, 1.0]
      }'
```

**多个 LoRA 应用到相同目标：**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["style_lora", "character_lora"],
        "lora_path": ["/path/to/style.safetensors", "/path/to/character.safetensors"],
        "target": "all",
        "strength": [0.7, 0.9]
      }'
```

> [!NOTE]
> 使用多个 LoRA 时：
> - 所有列表参数（`lora_nickname`、`lora_path`、`target`、`strength`）必须具有相同的长度
> - 如果 `target` 或 `strength` 是单个值，它将被应用到所有 LoRA
> - 应用到相同目标的多个 LoRA 将按顺序合并


**合并 LoRA 权重**

手动将当前设置的 LoRA 权重合并到基础模型中。

> [!NOTE]
> `set_lora` 会自动执行合并，因此通常只在手动 unmerge 后想要重新应用相同 LoRA 而不再调用 `set_lora` 时才需要此操作。

**端点：** `POST /v1/merge_lora_weights`

**参数：**
- `target`（字符串，可选）：要合并到哪个 transformer。可选值为 "all"（默认）、"transformer"、"transformer_2"、"critic"
- `strength`（浮点数，可选）：合并时的 LoRA 强度，默认 1.0。值 < 1.0 减弱效果，值 > 1.0 增强效果

**Curl 示例：**

```bash
curl -X POST http://localhost:30010/v1/merge_lora_weights \
  -H "Content-Type: application/json" \
  -d '{"strength": 0.8}'
```


**取消合并 LoRA 权重**

从基础模型中取消合并当前激活的 LoRA 权重，将模型恢复到原始状态。在设置不同的 LoRA 之前**必须**调用此操作。

**端点：** `POST /v1/unmerge_lora_weights`

**Curl 示例：**

```bash
curl -X POST http://localhost:30010/v1/unmerge_lora_weights \
  -H "Content-Type: application/json"
```

**列出 LoRA 适配器**

返回已加载的 LoRA 适配器和每个模块的当前应用状态。

**端点：** `GET /v1/list_loras`

**Curl 示例：**

```bash
curl -sS -X GET "http://localhost:30010/v1/list_loras"
```

**响应示例：**

```json
{
  "loaded_adapters": [
    { "nickname": "lora_a", "path": "/weights/lora_a.safetensors" },
    { "nickname": "lora_b", "path": "/weights/lora_b.safetensors" }
  ],
  "active": {
    "transformer": [
      {
        "nickname": "lora2",
        "path": "tarn59/pixel_art_style_lora_z_image_turbo",
        "merged": true,
        "strength": 1.0
      }
    ]
  }
}
```

注意事项：
- 如果当前流水线未启用 LoRA，服务器将返回错误。
- `num_lora_layers_with_weights` 仅统计对活跃适配器应用了 LoRA 权重的层数。

### 示例：切换 LoRA

1. 设置 LoRA A：
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_a", "lora_path": "path/to/A"}'
    ```
2. 使用 LoRA A 生成...
3. 取消合并 LoRA A：
    ```bash
    curl -X POST http://localhost:30010/v1/unmerge_lora_weights
    ```
4. 设置 LoRA B：
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_b", "lora_path": "path/to/B"}'
    ```
5. 使用 LoRA B 生成...

### 调整输出质量

服务器支持通过 `output-quality` 和 `output-compression` 参数调整图像和视频生成的输出质量和压缩级别。

#### 参数

- **`output-quality`**（字符串，可选）：预设质量级别，自动设置压缩。**默认值为 `"default"`**。有效值：
  - `"maximum"`：最高质量（100）
  - `"high"`：高质量（90）
  - `"medium"`：中等质量（55）
  - `"low"`：较低质量（35）
  - `"default"`：根据媒体类型自动调整（视频为 50，图像为 75）

- **`output-compression`**（整数，可选）：直接压缩级别覆盖（0-100）。**默认值为 `None`**。提供时（非 `None`），优先于 `output-quality`。
  - `0`：最低质量，最小文件大小
  - `100`：最高质量，最大文件大小

#### 注意事项

- **优先级**：同时提供 `output-quality` 和 `output-compression` 时，`output-compression` 优先
- **格式支持**：质量设置适用于 JPEG 和视频格式。PNG 使用无损压缩，忽略这些设置
- **文件大小与质量**：较低的压缩值（或 "low" 质量预设）生成更小的文件，但可能出现可见的伪影
