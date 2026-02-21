# TeaCache 加速

> **注意**：这是 SGLang 中两种可用缓存策略之一。
> 有关所有缓存选项的概述，请参阅[缓存](../index.md)。

TeaCache（基于时间相似度的缓存）通过检测连续去噪步骤之间的相似性来跳过计算，从而加速扩散推理。

## 概述

TeaCache 的工作原理：
1. 跟踪连续时间步之间调制输入的 L1 距离
2. 累积经过重新缩放的 L1 距离
3. 当累积距离低于阈值时，复用缓存的残差
4. 支持 CFG（Classifier-Free Guidance），具有独立的正向/负向缓存

## 工作原理

### L1 距离跟踪

在每个去噪步骤中，TeaCache 计算当前与前一个调制输入之间的相对 L1 距离：

```
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

然后使用多项式系数对该距离进行重新缩放并累积：

```
accumulated += poly(coefficients)(rel_l1)
```

### 缓存决策

- 如果 `accumulated >= threshold`：强制计算，重置累积器
- 如果 `accumulated < threshold`：跳过计算，使用缓存的残差

### CFG 支持

对于支持 CFG 缓存分离的模型（Wan、Hunyuan、Z-Image），TeaCache 为正向和负向分支维护独立的缓存：
- `previous_modulated_input` / `previous_residual` 用于正向分支
- `previous_modulated_input_negative` / `previous_residual_negative` 用于负向分支

对于不支持 CFG 分离的模型（Flux、Qwen），启用 CFG 时 TeaCache 会自动禁用。

## 配置

TeaCache 通过采样参数中的 `TeaCacheParams` 进行配置：

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    teacache_thresh=0.1,           # 累积 L1 距离的阈值
    coefficients=[1.0, 0.0, 0.0],  # L1 重新缩放的多项式系数
)
```

### 参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `teacache_thresh` | float | 累积 L1 距离的阈值。较低的值 = 更多缓存，更快但可能质量较低 |
| `coefficients` | list[float] | L1 重新缩放的多项式系数。需要针对特定模型调优 |

### 模型特定配置

不同模型可能有不同的最优配置。系数通常按模型调优，以平衡速度和质量。

## 支持的模型

TeaCache 内置于以下模型系列：

| 模型系列 | CFG 缓存分离 | 说明 |
|--------------|---------------------|-------|
| Wan（wan2.1、wan2.2） | 是 | 完全支持 |
| Hunyuan（HunyuanVideo） | 是 | 待支持 |
| Z-Image | 是 | 待支持 |
| Flux | 否 | 待支持 |
| Qwen | 否 | 待支持 |


## 参考链接

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
