# 性能优化

SGLang-Diffusion 提供了多种性能优化策略来加速推理。本节涵盖所有可用的性能调优选项。

## 概述

| 优化策略 | 类型 | 描述 |
|--------------|------|-------------|
| **Cache-DiT** | 缓存 | 支持 DBCache、TaylorSeer 和 SCM 的块级缓存 |
| **TeaCache** | 缓存 | 基于 L1 相似度的时间步级缓存 |
| **注意力后端** | 内核 | 优化的注意力实现（FlashAttention、SageAttention 等） |
| **性能分析** | 诊断 | PyTorch Profiler 和 Nsight Systems 指南 |

## 缓存策略

SGLang 支持两种互补的缓存方法：

### Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) 提供带有高级策略的块级缓存。最高可实现 **1.69 倍加速**。

**快速开始：**
```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

**主要特性：**
- **DBCache**：基于残差差异的动态块级缓存
- **TaylorSeer**：基于 Taylor 展开的校准，优化缓存决策
- **SCM**：步级计算掩码，提供额外加速

详细配置请参阅 [Cache-DiT 文档](cache/cache_dit.md)。

### TeaCache

TeaCache（基于时间相似度的缓存）通过检测连续去噪步骤之间的相似性来跳过计算，从而加速扩散推理。

**简要概述：**
- 跟踪跨时间步的调制输入之间的 L1 距离
- 当累积距离低于阈值时，复用缓存的残差
- 支持 CFG，具有独立的正向/负向缓存

**支持的模型：** Wan（wan2.1、wan2.2）、Hunyuan（HunyuanVideo）、Z-Image

详细配置请参阅 [TeaCache 文档](cache/teacache.md)。

## 注意力后端

不同的注意力后端根据你的硬件和模型提供不同的性能特征：

- **FlashAttention**：在使用 fp16/bf16 的 NVIDIA GPU 上最快
- **SageAttention**：替代的优化实现
- **xformers**：内存高效的注意力
- **SDPA**：PyTorch 原生的缩放点积注意力

平台支持和配置选项请参阅[注意力后端](attention_backends.md)。

## 性能分析

为了诊断性能瓶颈，SGLang-Diffusion 支持以下分析工具：

- **PyTorch Profiler**：内置 Python 性能分析
- **Nsight Systems**：GPU 内核级分析

详细说明请参阅[性能分析指南](profiling.md)。

## 参考链接

- [Cache-DiT 仓库](https://github.com/vipshop/cache-dit)
- [TeaCache 论文](https://arxiv.org/abs/2411.14324)
