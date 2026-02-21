# SGLang 中多模态编码器的 CUDA Graph

## 动机

在多模态推理服务中，视觉编码器（ViT / Vision Transformer）通常具有以下几个特性：

层数多，算子碎片化：每层包含 LN、QKV 投影、注意力、MLP、残差连接等，导致内核启动极其频繁。

服务端"小批量/低延迟"场景常见：批量大小非常小（有时"展平"批量后看起来只有 1），因此内核启动开销占端到端延迟的很大比例。

输入 token 数（patch 数量）频繁变化：不同的图像/视频分辨率和不同的批量组成导致不同的序列长度 S——而这正是 CUDA Graph 的最大障碍（形状不稳定）。

CUDA Graph 的价值：它将一长串具有固定形状和固定内存地址的 GPU 内核捕获到一个图中；之后对于相同的形状，可以直接重放该图，从而大幅减少启动开销，使 GPU 调度更加紧凑。

这促使我们为 ViT 寻求一个支持 CUDA Graph 的功能，以提升 ViT 的性能。

## 设计与限制

新的支持 CUDA Graph 的 ViT 逻辑构建在 ViTCudaGraphRunner 之上。该 runner 将视觉转换器的"blocks + merger + deepstack merger（可选）"部分捕获到 CUDA Graph 中，并对相同形状进行重放。以下是更多设计考量和限制的详细说明。

### 动态输入以适配 CUDA Graph 的静态约束

可变序列长度 S 在 ViT 中非常常见。而 CUDA Graph 要求固定形状。解决方案是按 S 构建图缓存（例如 graph_key = S）。第一次遇到新的 S 时创建并捕获一个图；之后直接重放。

如果存在许多不同的 S 值，则需要增加 VRAM 使用量，因为需要为多个图分配图私有内存池。

### 稳定的地址

所有"参数类"数据都变成静态缓冲区：

- block_input / block_ws / block_output
- cu_full_len / cu_window_len 及其 kk 变体
- sin_cos_ws

这样做是为了满足底层要求：重放期间不允许交换张量，只能修改张量内容。

### 注意力后端参数
注意力后端参数在图内部是固定的：

TritonAttn 需要 [cu_seqlens, cu_seqlens_kk, max_len]
FA3 需要 [cu_seqlens, max_len]

max_len 作为整数常量被冻结。
cu_seqlens 在 create_graph() 期间被缓存到字典中，后续重放时其内容不会更新。

对于相同的 graph_key = S，不仅要求输入形状匹配，还要求 cu_seqlens（及窗口 seqlens）中的分段模式完全相同。否则，注意力将错误地分割序列。

### 旋转缓冲区管理
该功能会在 seq_len 增加时重新分配更大的 sin_cos_ws。
max_content_len 用于确保分配的旋转缓冲区的最大大小。


## 命令示例
您可以通过设置环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 来启用 ViT 的 CUDA Graph，例如：
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct
```
或者您可以同时设置环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 和参数 `--enable-piecewise-cuda-graph`，将 ViT 的 CUDA Graph 与分段 CUDA Graph (Piecewise CUDA Graph) 功能一起运行，例如：
```
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --piecewise-cuda-graph-max-tokens 4096 \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-compiler eager
```

## 已知支持的模型
- Qwen2.5-VL (https://github.com/sgl-project/sglang/pull/14422)
- Qwen3-VL (https://github.com/sgl-project/sglang/pull/15320)
