# HiCache 系统设计与优化

本文档全面介绍了 SGLang HiCache，涵盖其系统架构、工作流程和关键组件。还详细说明了配置参数、优化技术以及与各种 L3 存储后端的集成，作为用户和开发者理解和调优 HiCache 以实现高效 LLM 推理的完整参考。

## 为什么需要 HiCache？什么是 HiCache？

在大语言模型推理中，预填充阶段通常非常耗时：输入序列需要首先转换为 Key-Value 缓存（KV 缓存），以供后续解码使用。当多个请求共享相同前缀时，该前缀的 KV 缓存是完全相同的。通过缓存和复用这些共享的 KV 缓存，可以避免冗余计算。为此，SGLang 引入了 RadixAttention，利用空闲 GPU 内存缓存和复用前缀 KV 缓存，以及 **HiCache**，将这一思想扩展到主机内存和分布式存储。

受现代 CPU 经典三级缓存设计的启发，HiCache 将 GPU 内存作为 L1、主机内存作为 L2、分布式存储作为 L3。这种层次结构使 HiCache 能够充分利用 GPU 和 CPU 的"空闲"存储空间，同时集成 Mooncake、3FS、NIXL 和 AIBrix KVCache 等分布式缓存系统，实现全局 KV 缓存存储和调度。因此，HiCache 在保持强劲读取性能的同时显著扩展了 KV 缓存容量——尤其是在 KV 缓存复用频繁的多问答和长上下文推理等工作负载中。详细的基准测试结果请参见[这篇博客](https://lmsys.org/blog/2025-09-10-sglang-hicache/)。


## 系统设计

### 整体架构

在许多现代 CPU 架构中，小而快速的 L1 和 L2 缓存对每个核心是私有的，能够快速访问最热的数据，而更大的 L3 缓存则在所有核心之间共享，以显著减少缓存中的冗余。类似地，在 HiCache 中，L1 和 L2 KV 缓存对每个推理实例是私有的，而 L3 KV 缓存在集群内的所有推理实例之间共享。

### HiRadixTree：HiCache 中的元数据组织

对于 KV 缓存数据组织，HiCache 基于 RadixAttention 中引入的 RadixTree 结构，提出了 HiRadixTree。在 RadixAttention 中，RadixTree 的每个节点对应 GPU 内存中一段连续 token 的 KV 缓存。从根节点到叶节点的路径表示请求的前缀，多个请求之间共享的前缀可以复用相同的节点，从而避免冗余存储。

HiRadixTree 扩展了这一思想：每个节点对应一段连续 token 的 KV 缓存，并记录该 KV 缓存的存储位置——无论是在本地 GPU 内存、CPU 内存、L3 存储中，还是同时在多个层级中。如果存储在本地，HiRadixTree 维护精确的元数据，包括确切的存储地址。但为了减少开销，HiRadixTree 不存储或持续同步 L3 KV 缓存的元数据。相反，在访问 L3 数据时，它实时查询后端以获取必要的元数据，例如数据是否存在以及存储在哪个服务器和位置。

### 整体工作流程

HiCache 的工作流程主要涉及三个关键操作：**本地匹配**、**预取**和**写回**。当系统收到新请求时，首先在本地 L1 和 L2 缓存中搜索匹配的 KV 缓存。对于本地未找到的部分，尝试从 L3 预取。预取完成后，所有需要的 KV 缓存被加载到 GPU 进行计算。预填充计算完成后，系统考虑将新生成的数据存储到 L2 或 L3。

![HiCache 工作流程](https://lmsys.org/images/blog/hicache/hicache_overview.png)

### 本地匹配

本地匹配是 HiCache 工作流程的第一步，将传入请求的 token 与 HiRadixTree 进行匹配，以定位本地内存层（L1 GPU 内存和 L2 主机内存）中缓存的 KV 数据。

匹配算法从根节点开始遍历 HiRadixTree，沿着与 token 序列前缀匹配的子节点前进。在每个节点处，将传入的 token 序列与节点存储的 token 序列进行比较。当 `page_size > 1` 时，匹配以页面粒度执行以优化内存访问模式。如果匹配在节点存储序列的中间终止，节点会自动分裂以创建精确的边界，提高未来匹配的效率。

算法返回请求的一段连续前缀，其中前半部分位于 L1，后半部分位于 L2。

由于此过程只需要遍历本地 HiRadixTree，不涉及任何实际数据复制，本地匹配速度极快。

### 从 L3 预取

数据预取是 HiCache 的核心优化技术之一，旨在主动将 KV 缓存从 L3 存储加载到本地 L2 内存中，从而减少后续操作中的访问延迟。

**预取触发条件**：
本地匹配后，对于在 L1 或 L2 中未找到的部分，系统查询 L3 以获取下一段连续匹配 KV 缓存的元数据。如果 L3 中命中缓存的长度超过阈值（默认：256 个 token，可配置），则触发预取操作。

**预取策略**：HiCache 提供三种不同的预取终止策略以应对不同场景需求：
- **best_effort**：当 GPU 可以执行预填充计算时立即终止，无等待时间，适用于对延迟极度敏感的场景。
- **wait_complete**：必须等待所有预取操作完成，适用于需要高缓存命中率的场景。
- **timeout**：在指定时间后或完成时终止，平衡延迟和缓存命中率需求。

预取停止后，已获取的数据与本地数据一起用于预填充计算。

对于 **timeout** 策略，HiCache 引入了两个配置参数以支持对预取超时条件的细粒度控制：

* `prefetch_timeout_base`：基础超时时间，表示与 token 数量无关的开销（如调度和同步）。
* `prefetch_timeout_per_ki_token`：每千个 token 的增量超时时间。

超时计算公式为：

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```

### 数据写回

写回机制负责将频繁访问的 KV 缓存从 L1 移动到 L2 和 L3，实现更大规模和更长期的存储以及跨实例的缓存共享。

**可配置的写回策略**：HiCache 支持三种写回策略：

* **write_through**：每次访问都立即写回到下一级。当带宽充足时，此策略提供最强的缓存效果。
* **write_through_selective**：仅在访问频率超过阈值后才写回数据。此策略只备份热数据，减少 I/O 开销。
* **write_back**：仅在数据从上层被驱逐时才写回到下一级。此策略缓解存储压力，适用于存储容量有限但需要最大化内存利用率的场景。

**跨实例共享**：当数据从 L2 写回到 L3 时，只有 L3 中尚不存在的数据才会被传输。存储在 L3 中的 KV 缓存随后可以在集群中的所有 SGLang 实例之间共享（取决于 L3 后端实现），在相同的内存预算下显著提高缓存命中率。

### 多 Rank 同步

在多 GPU 并行计算（如张量并行 TP）期间，HiCache 必须确保不同 rank 之间的状态一致。因此，关键计算步骤需要使用 `all_reduce` 进行状态同步。

例如，在预取期间，使用 `all_reduce(op=min)` 确保所有 rank 获得相同数量的 L3 命中，防止对是否达到预取阈值做出不一致的判断。同样，在预取完成或终止后，再次需要 `all_reduce(op=min)` 以保证各 rank 对成功获取的 KV 缓存前缀长度达成共识。

### 数据传输优化

**零拷贝数据传输**：预取和写回都涉及大量数据移动。最小化数据复制次数可以显著提升系统性能。HiCache 支持在将数据从 L2 内存传输到 L3 后端时直接传递内存地址和大小。

**"面向批次"的数据组织**：数据读写的粒度对性能影响很大。为此，HiCache L3 以**页面**粒度存储和传输 KV 缓存数据，并支持除现有 `layer first` 方案之外的不同数据布局，包括 `page first` 和 `page first direct`。在 `page first` 和 `page first direct` 布局下，属于同一页面的所有 KV 缓存数据放置在连续内存中，允许通过零拷贝传输作为单个对象传递给 L3。

![HiCache L2 内存布局](https://lmsys.org/images/blog/hicache/hicache_layout.png)

然而，由于 GPU KV 计算天然是逐层执行的，GPU 固有地以 `layer first` 布局运行。当将 `page first` 数据从 L2 传输到 GPU 时，数据必须以每层一个 token 的粒度进行传输。`page first direct` 布局通过将给定层内一个页面的所有 token 组合在一起来缓解此问题，允许从 L2 到 GPU 的传输在页面-层级别进行聚合。

**CPU 到 GPU 传输优化**：在 HiCache 中，将数据从 CPU 内存移动到 GPU 与从 L3 预取数据到 L2 一样关键。HiCache 为此过程采用了多项优化：

* **计算-传输重叠**：在预填充阶段，当从 CPU 向 GPU 传输数据时，HiCache 通过同时加载第 N+1 层的 KV 缓存和计算第 N 层来重叠各层。这有效地隐藏了数据传输延迟。
* **GPU 辅助 I/O 内核**：在 `cudaMemcpyAsync` 的基础上，HiCache 实现了一组专门为 CPU 和 GPU 之间 KV 缓存传输优化的 GPU 辅助 I/O 内核。与基准方法相比，这些内核实现了高达 3 倍的传输速度提升。

**MLA 的写回优化**：对于多 TP 下的 MHA（多头注意力）模型，每个 rank 持有一个 token KV 数据的 `1/tp_size`。而对于 MLA（多层注意力）模型，所有 rank 持有每个 token 完整且相同的 KV 数据。HiCache 包含了针对 MLA 的专门优化：只有一个 rank 发起写回操作，确保数据不会在各 rank 之间冗余存储。

### 与 PD 分离部署模式的集成

SGLang 通过 Mooncake TransferEngine 支持 PD（预填充-解码）分离部署模式（详情请参阅[此文档](https://docs.sglang.io/advanced_features/pd_disaggregation.html)）。在 PD 分离部署模式下，可以在预填充节点和解码节点上都启用 HiCache 以优化预填充性能。如果在解码节点上启用，解码输出也将被写回到 L3。

### 统一接口和丰富的 L3 存储后端

HiCache 将对 L3 后端的所有读、写和查询操作封装在 `class HiCacheStorage(ABC)` 中，提供一组简单且一致的接口。这种设计支持广泛的 L3 存储后端，并允许用户选择最适合其特定使用场景的后端。

- **Mooncake**：Mooncake 是一个用于 LLM 推理的高性能缓存系统，利用 RDMA 和多网卡资源实现零拷贝、超快速数据传输。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/mooncake_store)试用 Mooncake。

- **DeepSeek 3FS (HF3FS)**：HF3FS 是一个基于 Kubernetes 的分布式存储解决方案，采用 operator 式部署。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/hf3fs)试用 HF3FS。

- **NIXL**：NIXL 提供了一个统一的 API 来访问各种存储插件，包括但不限于 DeepSeek 的 3FS、GPU Direct Storage（GDS）和 Amazon S3 兼容对象存储。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl)试用 NIXL。

- **AIBrix KVCache**：AIBrix KVCache 是一个生产级 KVCache 卸载框架，支持高效的内存分层和低开销的跨引擎复用。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/aibrix_kvcache)试用 AIBrix KVCache。

- **HiCacheFile**：一个用于演示目的的简单文件存储后端。

特别地，**LMCache** 是一个用于企业级 LLM 推理的高效 KV 缓存层，提供了 HiCache 的替代方案。在[此处](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/lmcache)试用 LMCache。

## 相关参数

- **`--enable-hierarchical-cache`**：启用分层缓存功能。使用 HiCache 时此参数是必需的。

- **`--hicache-ratio HICACHE_RATIO`**：主机 KV 缓存内存池大小与设备池大小的比率。例如，值为 2 表示主机内存池是设备内存池的两倍大。此参数的值必须大于 1，因为当前实现要求为 KV 缓存分配的主机内存大于为 KV 缓存分配的设备内存。

- **`--hicache-size HICACHE_SIZE`**：主机 KV 缓存内存池的大小，以 GB 为单位。如果设置，此参数会覆盖 `hicache-ratio`。例如，`--hicache-size 30` 为**每个 rank** 的主机内存池分配 30GB（1GB = 1e9 字节）。如果有 8 个 rank，则总内存大小为 240GB。与 `hicache-ratio` 一样，此参数的值必须大于为 KV 缓存分配的设备内存大小。

**注意**：`--hicache-ratio` 和 `--hicache-size` 是两个关键参数。通常，更大的 HiCache 大小会带来更高的缓存命中率，从而提升预填充性能。但缓存大小与命中率之间的关系并非线性。一旦大部分可复用的 KV 数据——尤其是热点 token——已经被缓存，进一步增大缓存可能只会带来边际性能提升。用户可以根据工作负载特征和性能需求设置这些参数。

- **`--page-size PAGE_SIZE`**：每页的 token 数。此参数决定了 KV 缓存存储和检索的粒度。更大的页面大小减少了元数据开销并提高了存储后端的 I/O 效率，但当只有部分页面与存储的 KV 缓存匹配时可能会降低缓存命中率。对于具有长公共前缀的工作负载，较大的页面可以提升性能，而前缀更多样化的工作负载可能受益于较小的页面。有关页面粒度如何影响 I/O 性能，请参阅[数据传输优化](#数据传输优化)。

- **`--hicache-storage-prefetch-policy {best_effort,wait_complete,timeout}`**：控制从存储中预取何时停止。详见[从 L3 预取](#从-l3-预取)。
  - `best_effort`：尽可能多地预取而不阻塞
  - `wait_complete`：等待预取完成后再继续
  - `timeout`：在指定时间后或完成时终止（推荐用于生产环境，设置适当的超时有助于系统满足所需的 SLO）

- **`--hicache-write-policy {write_back,write_through,write_through_selective}`**：控制数据如何从较快的内存层写入较慢的内存层。详见[数据写回](#数据写回)。
  - `write_through`：立即将数据写入所有层级（最强缓存效果）
  - `write_through_selective`：使用命中计数跟踪，仅备份频繁访问的数据
  - `write_back`：仅在需要驱逐时才将数据写回较慢的层级（减少 I/O 负载）

- **`--hicache-io-backend {direct,kernel}`**：选择 CPU 和 GPU 之间 KV 缓存传输的 I/O 后端。详见[数据传输优化](#数据传输优化)。
  - `direct`：标准 CUDA 内存复制操作
  - `kernel`：GPU 辅助 I/O 内核（推荐以获得更好的性能）

- **`--hicache-mem-layout {layer_first,page_first,page_first_direct}`**：主机内存池的内存布局。详见[数据传输优化](#数据传输优化)。
  - `layer_first`：与 GPU 计算内核兼容（GPU 内存的默认布局）
  - `page_first`：针对 I/O 效率优化
  - `page_first_direct`：将给定层内一个页面的所有 token 组合在一起，允许从 L2 到 GPU 的传输在页面-层级别进行聚合

- **`--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic}`**：选择 L3 层的存储后端。内置后端：file、mooncake、hf3fs、nixl、aibrix。对于动态后端，使用 --hicache-storage-backend-extra-config 指定：`backend_name`（自定义名称）、`module_path`（Python 模块路径）、`class_name`（后端类名）。有关可用后端，请参阅[统一接口和丰富的 L3 存储后端](#统一接口和丰富的-l3-存储后端)。

- **`--enable-lmcache`**：使用 LMCache 作为替代的分层缓存方案。

- **`--hicache-storage-backend-extra-config HICACHE_STORAGE_BACKEND_EXTRA_CONFIG`**：额外配置可以是：
  - 包含存储后端额外配置的 JSON 字符串，例如 `--hicache-storage-backend-extra-config '{"prefetch_threshold":512, "prefetch_timeout_base": 0.5, "prefetch_timeout_per_ki_token": 0.25}'`，或者
  - 指定存储后端额外配置的 TOML、JSON 或 YAML 文件（为了与 JSON 字符串输入区分，在文件名前添加 `@`），例如 `--hicache-storage-backend-extra-config "@config.toml"`，其中 `config.toml` 是包含复杂配置的配置文件。当配置包含许多或复杂的键值对时这很有用（例如，NIXL 后端的配置可能很复杂，建议使用配置文件）。
