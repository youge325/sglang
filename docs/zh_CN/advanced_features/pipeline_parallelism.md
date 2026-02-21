# 长上下文的流水线并行

## 为什么需要流水线并行？

随着大语言模型（LLM）向万亿参数架构和"无限"上下文窗口发展，底层服务基础设施必须朝着更细粒度的跨节点并行化策略演进。虽然 KV 缓存技术可以有效减少冗余计算，但无法规避超长序列在极大初始输入 Token 长度（ITL）下固有的首 Token 生成时间（TTFT）开销。尽管张量并行（TP）仍然是节点内扩展的传统方法，但在多节点部署中经常遇到通信瓶颈。另一方面，流水线并行仅需要在每个流水线阶段边界进行跨节点通信，与大规模 TP 相比可以实现更好的计算-通信重叠。因此，它也是提高吞吐量的一种有前景的并行化策略。

详细分析请参阅此 [博客](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)。

## 基于异步通信的实现重构
借助动态分块预填充，流水线并行有可能降低长上下文输入的 TTFT。对于每个请求，其输入 Token 可以被划分为多个块，每个块不超过分块预填充大小。同一请求的不同块可以由不同节点同时处理，从而并行化处理过程并降低 TTFT。SGLang 已经支持流水线并行（#5724）一段时间，并使其与 PD 分离特性（#8846）兼容，但实现并不完善，在性能提升方面仍有很大空间。

为了消除这一性能隐患，SGLang 实现了一个微批次事件循环，使用非阻塞异步点对点（P2P）通信来重叠 GPU 计算与 CPU 元数据处理和 PP 通信。这确保了当一个微批次在 GPU 上计算时，下一个微批次已经在准备和传输中，有效地保持流水线尽可能饱和。这种方法最初在 #7979 中提出，已在 #11852 中重新设计并纳入。

实现的关键机制包括：

* **事件循环中解耦的同步/异步逻辑：** 调度器在 `_pp_send_pyobj_to_next_stage` 中使用 `async_send`。它不会等待传输完成，而是返回一个 `P2PWork` 句柄。实际的同步（`P2PWork.work.wait()`）被推迟到调用 `_pp_commit_comm_work` 时执行，允许 CPU 在数据传输过程中执行其他工作——如调度下一个批次或处理元数据。
* **多流执行：** 除了作为同步流的主 `default_stream`，SGLang 还使用专用的 `forward_stream` 和 `copy_stream` 分别执行前向传播 GPU 计算和设备到主机（D2H）内存传输，以实现更好的重叠。当 `_pp_launch_batch` 在 GPU 上为当前阶段执行当前微批次时，CPU 使用 `_pp_process_batch_result` 处理上一个微批次的结果。

## 动态分块指导

### 为什么需要动态分块
固定大小的分块预填充会在流水线中造成气泡，尤其是当 PP 规模较大时。这一现象背后的主要原因是，即使每个块大小相同（由 Transformer 结构带来），模型的运行时间也不均匀。前缀序列长度越大，块的运行时间越长。而且这些气泡会传播到下一个阶段，显著降低更大 PP 等级的扩展效率。

为了解决这个问题，SGLang 引入了动态分块机制，预测下一个块的最优大小，使其满足以下条件：

Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)

其中 ***L*** 表示前缀序列长度。通过分析一系列具有不同 ITL 的请求，我们将累积运行时间建模为序列长度的二次函数。使用此模型，我们可以求解给定前缀长度 ***L*** 下的最优下一块大小。由于注意力机制的计算复杂度随 ***L*** 增长，下一块大小将随着 ***L*** 的增长而逐步减小，以保持跨流水线阶段的块执行时间一致。

基于此方法，调度器可以在运行时预测并动态减小块大小，以最小化阶段错位造成的气泡。需要注意的是，调度器不使用原始预测值。为了促进高效的 KV 缓存内存管理并确保与硬件执行效率的亲和性，该值会向下对齐到 max(`--page-size`, 64) 的最近倍数。


### 分块预填充大小和平滑因子

当启用 `--enable-dynamic-chunking` 时，序列的每个块大小会根据二次模型动态确定，该模型基于初始块长度的估计运行时间预测下一个块大小。在这种情况下，我们使用 `--chunked-prefill-size` 设置初始块大小。切换到动态分块模式时，初始块大小（`--chunked-prefill-size`）应设置为与原始分块预填充大小相当的较大值，以避免产生过多的块。

**`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`** 是控制动态分块算法平滑因子的环境变量，默认值为 0.75。它决定了预填充阶段块大小的变化幅度。较大的值意味着更激进的块大小变化，可能带来更好的性能，但也会导致更大的块大小变化（末尾的块大小可能变得非常小，这可能导致性能下降）和更多的总块数。当设置为 1 时，块大小将严格按照上述预测下一块大小的二次模型进行调整。较小的值意味着更保守的块大小变化，可能导致更小的块大小变化和更少的总块数。当设置为 0 时，块大小不会动态调整，因此与传统的固定分块预填充大小方式相同。

由于硬件、模型和目标工作负载的差异，静态配置很少能在所有场景中达到最优。因此，在切换到动态分块模式时，达到峰值性能需要一定程度的超参数调优。

**动态分块预填充的调优指导**

* **第 1 步 - 迭代找到目标 PP 规模的最优固定分块预填充大小**：不同的 PP 规模对于目标 ITL 可能有不同的最优分块预填充大小。因此，用户应根据可用的扩展资源进行迭代以获得基线。
* **第 2 步 - 动态分块的初始块大小选择**：将初始大小设置为最优固定分块预填充大小的 2 倍或 3 倍。这减少了总块数，防止"尾部块"未充分利用硬件。为了保持极大输入 Token 长度（ITL）的效率，动态预测器自动确保后续块不小于初始大小的 1/4。此外，建议在这类情况下也使用更大的初始块大小（例如最优固定分块预填充大小的 4 倍）。
* **第 3 步 - 平滑因子调整**：此因子控制块大小根据二次性能拟合模型预测进行调整的严格程度。
  * 1.0：严格遵循模型。
  * **0.6 – 0.85（推荐）**：在动态扩展和硬件稳定性之间取得最佳平衡的典型范围。通过实验，我们发现 0.6 到 0.85 之间的范围通常能为动态分块带来最佳性能。
  * 0：禁用动态调整，恢复为传统的固定大小分块。
* **另一个小优化技巧：** 当层数不能在各等级之间均匀分配时，将较大的分区放在较高的 PP 等级。这可以在较高 PP 等级等待上一阶段结果时增加 GPU 利用率，从而减少较高 PP 等级上的气泡。以 DeepSeek-V3.1 为例，`SGLANG_PP_LAYER_PARTITION=15,15,15,16` 通常比 `16,15,15,15` 表现更好。

## 长上下文最佳实践

### 调优分块预填充大小
优化分块预填充大小对于平衡流水线效率和资源利用率至关重要。理想的大小取决于模型架构、硬件配置和典型输入长度等因素。我们建议从较小的块大小（如 4K）开始，逐步增加直到找到适合您特定用例的最优大小（不同的目标 ITL 和 PP 规模可能有不同的最优分块预填充大小。因此，用户应根据可用的扩展资源进行迭代以获得基线）。或者，您可以分析硬件容量并根据 roofline 模型确定最优块大小。

### 为超长 ITL 启用动态分块并调整平滑因子
SGLang 还提供了可以进一步提高性能的动态分块方案。此功能目前是实验性功能，需要一定的调优实验，可能不适用于所有工作负载。此外，微调平滑因子可以帮助优化特定工作负载和模型特性的性能。

### NVIDIA H20 案例研究

在评估固定分块预填充大小从 2K 到 16K 的流水线并行时，实验结果表明 4K 块大小为 DeepSeek-V3.1 提供了最优的预填充 TTFT 性能，6K 块大小为 Qwen3-235B-A22B-FP8 提供了最优的预填充 TTFT 性能。

启用动态分块时，我们首先将最优固定分块预填充大小乘以 3 作为初始块大小。通过实验，我们发现 2-3 的乘数提供了适当的平衡——避免过大的初始流水线气泡，同时确保后续块不会随着上下文长度增加而变得过小。使用默认的动态分块平滑因子 0.75，我们进行了参数调优，确定对于 DeepSeek-V3.1 的 12K 初始块大小，0.65 的值最优；对于 Qwen3-235B-A22B-FP8 的 18K 初始块大小，0.8 的值最优。

#### DeepSeek-V3.1 128K 输入 Token 长度
```bash
# 预填充节点 0（固定分块预填充大小）
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 4096
```

```bash
# 预填充节点 0（使用动态分块）
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 8 --pp-size 4 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 12288 --enable-dynamic-chunking
```

#### Qwen3-235B-A22B-FP8 128K 输入 Token 长度
```bash
# 预填充节点 0（固定分块预填充大小）
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 6144
```

```bash
# 预填充节点 0（使用动态分块）
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code \
  --nnodes 4 --node-rank 0 --tp 4 --pp-size 8 \
  --port 30000 --dist-init-addr <MASTER_NODE_IP> \
  --disable-radix-cache --mem-fraction-static 0.8  \
  --attention-backend fa3 --host 0.0.0.0 --watchdog-timeout 3600 \
  --max-running-requests 128 --chunked-prefill-size 18432 --enable-dynamic-chunking
```

注意：`--disable-radix-cache` 仅用于可复现的基准测试目的。不建议在生产环境中使用。

## 流水线并行与 PD 分离的最佳实践
待添加。请关注流水线并行与 PD 分离的最新更新。
