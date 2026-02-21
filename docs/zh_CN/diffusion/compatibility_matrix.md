# 兼容性矩阵

下表展示了所有支持的模型及其支持的优化选项。

表中使用的符号含义如下：

- ✅ = 完全兼容
- ❌ = 不兼容
- ⭕ = 不适用于该模型

## 模型 × 优化

`HuggingFace Model ID` 可以直接传递给 `from_pretrained()` 方法，SGLang-Diffusion 会在初始化和生成视频时使用最优的默认参数。

### 视频生成模型

| 模型名称                   | Hugging Face Model ID                             | 分辨率         | TeaCache | Sliding Tile Attn | Sage Attn | Video Sparse Attention (VSA) | Sparse Linear Attention (SLA) | Sage Sparse Linear Attention (SageSLA) | Sparse Video Gen 2 (SVG2) |
|:-----------------------------|:--------------------------------------------------|:--------------------|:--------:|:-----------------:|:---------:|:----------------------------:|:----------------------------:|:-----------------------------------------------:|:----------------------------------:|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`         | 480p                |    ⭕     |         ⭕         |      ⭕     |              ✅               |              ❌               |              ❌               |    ❌     |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p                |    ⭕     |         ⭕         |     ⭕     |              ✅               |              ❌               |              ❌               |    ❌     |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                 | 720p                |    ⭕     |         ⭕         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`             | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                 | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                 | 480p, 720p          |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`            | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`            | 720p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| TurboWan2.1 T2V 1.3B         | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`      | 480p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.1 T2V 14B          | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`       | 480p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.1 T2V 14B 720P     | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers`  | 720p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.2 I2V A14B         | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers`      | 720p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |

**注意**：
1. Wan2.2 TI2V 5B 在执行图生视频（I2V）生成时存在一些质量问题。我们正在修复此问题。
2. SageSLA 基于 SpargeAttn。请先使用 `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation` 安装。

### 图像生成模型

| 模型名称       | HuggingFace Model ID                    | 分辨率    |
|:-----------------|:----------------------------------------|:---------------|
| FLUX.1-dev       | `black-forest-labs/FLUX.1-dev`          | 任意分辨率 |
| FLUX.2-dev       | `black-forest-labs/FLUX.2-dev`          | 任意分辨率 |
| FLUX.2-Klein     | `black-forest-labs/FLUX.2-klein-4B`     | 任意分辨率 |
| Z-Image-Turbo    | `Tongyi-MAI/Z-Image-Turbo`              | 任意分辨率 |
| GLM-Image        | `zai-org/GLM-Image`                     | 任意分辨率 |
| Qwen Image       | `Qwen/Qwen-Image`                       | 任意分辨率 |
| Qwen Image 2512  | `Qwen/Qwen-Image-2512`                  | 任意分辨率 |
| Qwen Image Edit  | `Qwen/Qwen-Image-Edit`                  | 任意分辨率 |

## 已验证的 LoRA 示例

本节列出了已在 **SGLang Diffusion** 流水线中与各基础模型进行过明确测试和验证的 LoRA 示例。

> 重要提示：
> 未在此列出的 LoRA 并不一定不兼容。
> 实际上，大多数标准 LoRA 都可以正常工作，尤其是遵循常见 Diffusers 或 SD 风格规范的 LoRA。
> 以下条目仅反映了 SGLang 团队已手动验证的配置。

### 按基础模型分类的已验证 LoRA

| 基础模型       | 支持的 LoRA |
|:-----------------|:----------------|
| Wan2.2           | `lightx2v/Wan2.2-Distill-Loras`<br>`Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
| Wan2.1           | `lightx2v/Wan2.1-Distill-Loras` |
| Z-Image-Turbo    | `tarn59/pixel_art_style_lora_z_image_turbo`<br>`wcde/Z-Image-Turbo-DeJPEG-Lora` |
| Qwen-Image       | `lightx2v/Qwen-Image-Lightning`<br>`flymy-ai/qwen-image-realism-lora`<br>`prithivMLmods/Qwen-Image-HeadshotX`<br>`starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit  | `ostris/qwen_image_edit_inpainting`<br>`lightx2v/Qwen-Image-Edit-2511-Lightning` |
| Flux             | `dvyio/flux-lora-simple-illustration`<br>`XLabs-AI/flux-furry-lora`<br>`XLabs-AI/flux-RealismLora` |

## 特殊要求

### Sliding Tile Attention

- 目前仅支持 Hopper GPU（H100 系列）。
