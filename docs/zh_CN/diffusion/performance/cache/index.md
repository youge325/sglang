# 扩散模型的缓存加速

SGLang 为 Diffusion Transformer (DiT) 模型提供了多种缓存加速策略。这些策略可以通过跳过冗余计算来显著减少推理时间。

## 概述

SGLang 支持两种互补的缓存方法：

| 策略 | 范围 | 机制 | 最适用于 |
|----------|-------|-----------|----------|
| **Cache-DiT** | 块级 | 动态跳过单个 transformer 块 | 高级场景，更高加速比 |
| **TeaCache** | 时间步级 | 基于 L1 相似度跳过整个去噪步骤 | 简单场景，内置支持 |



## Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) 提供带有 DBCache 和 TaylorSeer 等高级策略的块级缓存。最高可实现 **1.69 倍加速**。

详细配置请参阅 [cache_dit.md](cache_dit.md)。

### 快速开始

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### 主要特性

- **DBCache**：基于残差差异的动态块级缓存
- **TaylorSeer**：基于 Taylor 展开的校准，优化缓存决策
- **SCM**：步级计算掩码，提供额外加速

## TeaCache

TeaCache（基于时间相似度的缓存）通过检测连续去噪步骤之间的相似性来跳过计算，从而加速扩散推理。

详细文档请参阅 [teacache.md](teacache.md)。

### 简要概述

- 跟踪跨时间步的调制输入之间的 L1 距离
- 当累积距离低于阈值时，复用缓存的残差
- 支持 CFG，具有独立的正向/负向缓存

### 支持的模型

- Wan（wan2.1、wan2.2）
- Hunyuan（HunyuanVideo）
- Z-Image

对于 Flux 和 Qwen 模型，启用 CFG 时 TeaCache 会自动禁用。

## 参考链接

- [Cache-DiT 仓库](https://github.com/vipshop/cache-dit)
- [TeaCache 论文](https://arxiv.org/abs/2411.14324)
