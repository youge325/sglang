# SGLang Diffusion

SGLang Diffusion 是一个用于加速扩散模型图像和视频生成的推理框架。它提供了端到端的统一流水线，集成了优化内核和高效的调度循环。

## 主要特性

- **广泛的模型支持**：Wan 系列、FastWan 系列、Hunyuan、Qwen-Image、Qwen-Image-Edit、Flux、Z-Image、GLM-Image 等
- **快速推理**：优化内核、高效调度循环和 Cache-DiT 加速
- **易于使用**：兼容 OpenAI 的 API、CLI 和 Python SDK
- **多平台支持**：NVIDIA GPU（H100、H200、A100、B200、4090）、AMD GPU（MI300X、MI325X）和 Ascend NPU（A2、A3）

---

## 快速开始

### 安装

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

更多安装方式和 ROCm 相关说明请参阅[安装指南](installation.md)。

### 基本用法

使用 CLI 生成图像：

```bash
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains" \
    --save-output
```

或者启动一个兼容 OpenAI API 的服务器：

```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

---

## 文档目录

### 入门指南

- **[安装](installation.md)** - 通过 pip、uv、Docker 或从源码安装 SGLang Diffusion
- **[兼容性矩阵](compatibility_matrix.md)** - 支持的模型和优化兼容性

### 使用方法

- **[CLI 文档](api/cli.md)** - `sglang generate` 和 `sglang serve` 的命令行接口
- **[OpenAI API](api/openai_api.md)** - 用于图像/视频生成和 LoRA 管理的 OpenAI 兼容 API

### 性能优化

- **[性能概述](performance/index.md)** - 所有性能优化策略概述
- **[注意力后端](performance/attention_backends.md)** - 可用的注意力后端（FlashAttention、SageAttention 等）
- **[缓存策略](performance/cache/)** - Cache-DiT 和 TeaCache 加速
- **[性能分析](performance/profiling.md)** - 使用 PyTorch Profiler 和 Nsight Systems 进行性能分析

### 参考资料

- **[环境变量](environment_variables.md)** - 通过环境变量进行配置
- **[支持新模型](support_new_models.md)** - 添加新扩散模型的指南
- **[贡献指南](contributing.md)** - 贡献指南和提交信息规范
- **[CI 性能](ci_perf.md)** - 性能基线生成脚本

---

## CLI 快速参考

### Generate（一次性生成）

```bash
sglang generate --model-path <MODEL> --prompt "<PROMPT>" --save-output
```

### Serve（HTTP 服务器）

```bash
sglang serve --model-path <MODEL> --port 30010
```

### 启用 Cache-DiT 加速

```bash
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path <MODEL> --prompt "<PROMPT>"
```

---

## 参考链接

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
