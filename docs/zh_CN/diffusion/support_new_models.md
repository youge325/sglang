# 如何支持新的扩散模型

本文档介绍如何在 SGLang Diffusion 中添加对新扩散模型的支持。

## 架构概述

SGLang Diffusion 兼顾性能和灵活性，基于模块化流水线架构构建。这种设计允许开发者通过组合和复用不同组件，轻松构建适用于各种扩散模型的复杂自定义流水线。

其核心围绕两个关键概念展开，详见我们的[博客文章](https://lmsys.org/blog/2025-11-07-sglang-diffusion/#architecture)：

- **`ComposedPipeline`**：该类编排一系列 `PipelineStage`，定义特定模型的完整生成流程。它作为模型的主入口点，管理扩散过程不同阶段之间的数据流。
- **`PipelineStage`**：每个阶段是一个模块化组件，封装了扩散过程中的一个常见功能。例如提示编码、去噪循环或 VAE 解码。这些阶段被设计为自包含且可在不同流水线中复用。

## 实现的关键组件

要添加对新扩散模型的支持，你主要需要定义或配置以下组件：

1. **`PipelineConfig`**：这是一个数据类，保存模型流水线的所有静态配置。它包括模型组件的路径（如 UNet、VAE、文本编码器）、精度设置（如 `fp16`、`bf16`）和其他模型特定的架构参数。每个模型通常有自己的 `PipelineConfig` 子类。

2. **`SamplingParams`**：这个数据类定义了在运行时控制生成过程的参数。这些是用户为生成请求提供的输入，如 `prompt`、`negative_prompt`、`guidance_scale`、`num_inference_steps`、`seed`、输出尺寸（`height`、`width`）等。

3. **`ComposedPipeline`（非配置类）**：这是一个核心类，你在其中定义模型生成流水线的结构。你需要创建一个继承自 `ComposedPipelineBase` 的新类，并在其中按正确顺序实例化和串联必要的 `PipelineStage`。参见 `ComposedPipelineBase` 和 `PipelineStage` 的基类定义：
    - [`ComposedPipelineBase`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/composed_pipeline_base.py)
    - [`PipelineStage`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/stages/base.py)
    - [中央注册表（模型/配置映射）](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)

4. **模块（流水线引用的组件）**：每个流水线引用一组从模型仓库加载的模块（如 Diffusers 的 `model_index.json`），并通过注册表/加载器进行组装。常见模块包括：
    - `text_encoder`：将文本提示编码为嵌入向量
    - `tokenizer`：为文本编码器对原始文本输入进行分词
    - `processor`：预处理图像并提取特征；常用于图生图任务
    - `image_encoder`：专用的图像特征提取器（可能与 `processor` 独立或合并）
    - `dit/transformer`：核心去噪网络（DiT/UNet 架构），在潜空间中运行
    - `scheduler`：控制推理过程中的时间步调度和去噪动态
    - `vae`：变分自编码器，用于在像素空间和潜空间之间编码/解码

## 可用的流水线阶段

你可以根据需要组合以下可用阶段来构建自定义 `ComposedPipeline`。每个阶段负责生成过程的特定部分。

| 阶段类                      | 描述                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `InputValidationStage`           | 在启动流水线之前验证用户提供的 `SamplingParams` 是否正确。     |
| `TextEncodingStage`              | 使用一个或多个文本编码器将文本提示编码为嵌入向量。                                   |
| `ImageEncodingStage`             | 将输入图像编码为嵌入向量，常用于图生图任务。                               |
| `ImageVAEEncodingStage`          | 专门使用变分自编码器 (VAE) 将输入图像编码到潜空间。        |
| `ConditioningStage`              | 为去噪循环准备条件张量（如来自文本或图像嵌入的张量）。         |
| `TimestepPreparationStage`       | 为扩散过程准备调度器的时间步。                                           |
| `LatentPreparationStage`         | 创建将被去噪的初始噪声潜变量张量。                                          |
| `DenoisingStage`                 | 执行主去噪循环，迭代应用模型（如 UNet）来精炼潜变量。    |
| `DecodingStage`                  | 使用 VAE 将去噪循环中的最终潜变量张量解码回像素空间（如图像）。 |
| `DmdDenoisingStage`              | 用于特定模型架构的专用去噪阶段。                                          |
| `CausalDMDDenoisingStage`        | 用于特定视频模型的专用因果去噪阶段。                                         |

## 示例：实现 `Qwen-Image-Edit`

为了说明这个过程，让我们看看 `Qwen-Image-Edit` 是如何实现的。典型的实现顺序是：

1. **分析所需模块**：
    - 通过检查目标模型的 `model_index.json` 或 Diffusers 实现来研究其组件，识别所需模块：
      - `processor`：图像预处理和特征提取
      - `scheduler`：扩散时间步调度
      - `text_encoder`：文本到嵌入向量的转换
      - `tokenizer`：为编码器进行文本分词
      - `transformer`：核心 DiT 去噪网络
      - `vae`：用于潜空间编码/解码的变分自编码器

2. **创建配置**：
    - **PipelineConfig**：[`QwenImageEditPipelineConfig`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipelines/qwen_image.py) 定义了模型特定的参数、精度设置、预处理函数和潜空间形状计算。
    - **SamplingParams**：[`QwenImageSamplingParams`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/sample/qwenimage.py) 设置运行时默认值，如 `num_frames=1`、`guidance_scale=4.0`、`num_inference_steps=50`。

3. **实现模型组件**：
    - 在相应的目录中适配或实现特定的模型组件：
      - **DiT/Transformer**：在 [`runtime/models/dits/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/) 中实现 - 例如，[`qwen_image.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py) 用于 Qwen 的 DiT 架构
      - **编码器**：在 [`runtime/models/encoders/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/) 中实现 - 例如，文本编码器 [`qwen2_5vl.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py)
      - **VAE**：在 [`runtime/models/vaes/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/) 中实现 - 例如，[`autoencoder_kl_qwenimage.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/autoencoder_kl_qwenimage.py)
      - **调度器**：如需要，在 [`runtime/models/schedulers/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/) 中实现
    - 这些组件处理目标扩散模型特定的核心模型逻辑、注意力机制和数据变换。

4. **定义流水线类**：
    - [`QwenImageEditPipeline`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/architectures/basic/qwen_image/qwen_image.py) 类继承自 `ComposedPipelineBase`，按顺序编排各阶段。
    - 通过 `_required_config_modules` 声明所需模块，并实现流水线阶段：

    ```python
    class QwenImageEditPipeline(ComposedPipelineBase):
        pipeline_name = "QwenImageEditPipeline"  # 与 Diffusers 的 model_index.json 匹配
        _required_config_modules = ["processor", "scheduler", "text_encoder", "tokenizer", "transformer", "vae"]

        def create_pipeline_stages(self, server_args: ServerArgs):
            """按顺序设置流水线阶段。"""
            self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
            self.add_stage(stage_name="prompt_encoding_stage_primary", stage=ImageEncodingStage(...))
            self.add_stage(stage_name="image_encoding_stage_primary", stage=ImageVAEEncodingStage(...))
            self.add_stage(stage_name="timestep_preparation_stage", stage=TimestepPreparationStage(...))
            self.add_stage(stage_name="latent_preparation_stage", stage=LatentPreparationStage(...))
            self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
            self.add_stage(stage_name="denoising_stage", stage=DenoisingStage(...))
            self.add_stage(stage_name="decoding_stage", stage=DecodingStage(...))
    ```
    流水线通过按顺序添加阶段来构建。`Qwen-Image-Edit` 使用 `ImageEncodingStage`（用于提示和图像处理）和 `ImageVAEEncodingStage`（用于潜变量提取），然后是标准的去噪和解码阶段。

5. **注册配置**：
    - 在中央注册表（[`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)）中通过 `_register_configs` 注册配置，以启用模型的自动加载和实例化。模块会根据配置和仓库结构自动加载和注入。

遵循这种定义配置和组合流水线的模式，你可以轻松地将新的扩散模型集成到 SGLang 中。
