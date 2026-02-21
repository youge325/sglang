# SGLang HiCache 最佳实践

## 为什么 HiCache 很重要

SGLang HiCache 通过三层分级 KV 缓存系统扩展了传统的 RadixAttention，在长上下文和多轮对话场景中显著提升了性能。通过在 GPU 内存、主机内存和外部存储后端之间智能管理 KV 缓存，HiCache 解决了传统系统中限制缓存命中率的根本容量瓶颈。

## 配置指南

## 核心 HiCache 参数

```bash
# Essential HiCache flags
--page-size 64                        # 缓存管理的页面大小
--enable-hierarchical-cache           # 启用 HiCache
--hicache-ratio 2                     # 主机内存比率（GPU 内存的 2 倍）
--hicache-size 100                    # 主机内存大小（GB），会覆盖上述比率
--hicache-io-backend kernel           # CPU 与 GPU 之间数据传输的 I/O 后端
--hicache-write-policy write_through  # GPU 到 CPU 的缓存写入策略
--hicache-storage-backend             # 可选的存储后端（如 hf3fs、mooncake 等）
```

注意事项：

- 除了在启动时配置 `--hicache-storage-backend`，SGLang 还支持通过 HTTP 管理端点在**运行时挂载/卸载** HiCache 存储后端（无需重启）。详见[运行时挂载/卸载 HiCache 存储后端](hicache_storage_runtime_attach_detach.md)。

## 启用存储后端时的关键配置

### 内存布局优化

```bash
# Page-first：针对 I/O 效率优化，支持零拷贝（推荐与 kernel 后端配合使用）
--hicache-mem-layout page_first
# Page-first-direct：针对直接 I/O 操作优化（兼容 fa3，与 page_first 具有相同的零拷贝性能）
--hicache-mem-layout page_first_direct
# Layer-first
--hicache-mem-layout layer_first
```
**布局兼容性：**
- `page_first`：仅与 `kernel` I/O 后端兼容，使用 `direct` 后端时会自动切换到 `layer_first`
- `page_first_direct`：专为 `direct` I/O 后端设计，具有优化的内存组织方式

### 预取策略

```bash
# Best-effort：在需要时终止预取
--hicache-storage-prefetch-policy best_effort
# Wait-complete：确保完整预取，更高的缓存复用率
--hicache-storage-prefetch-policy wait_complete
# Timeout：在完成度和尽力而为之间取得平衡
--hicache-storage-prefetch-policy timeout
```

### 与 PD 分离部署的集成

HiCache 可以与 PD 分离部署无缝配合。你可以在两种配置之间选择：

1. **仅预填充 HiCache**：仅在预填充节点启用 HiCache，允许预填充实例之间共享 KV 缓存
2. **带异步卸载的完整 HiCache**：在预填充节点启用 HiCache，并在解码节点启用异步 KV 缓存卸载，允许预填充节点在多轮对话场景中复用解码节点的 KV 缓存

```bash
# 启用 HiCache 的预填充节点，用于跨预填充共享（适用于系统提示词场景）
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake

# 启用异步卸载的解码节点，允许预填充节点复用 KV 缓存（适用于多轮对话场景）
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \  # 在解码节点启用异步 KV 缓存卸载
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake
```


### 使用 HF3FS 部署

以下是使用 HiCache-HF3FS 部署 DeepSeek-R1 的示例。更多详情请参阅 [HF3FS 文档](../../python/sglang/srt/mem_cache/storage/hf3fs/docs/README.md)。

```bash
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --log-level info \
  --tp 8 \
  --host 0.0.0.0 \
  --port 10000 \
  --enable-metrics \
  --enable-cache-report \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete \
```

### 使用 Mooncake 部署

以下是使用 Mooncake 部署 Qwen3-235B-A22B-Instruct-2507 的示例。更多详情请参阅 [Mooncake 文档](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md)。

```bash
# 设置 Mooncake 环境变量
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

# 使用 Mooncake 后端启动 SGLang 服务器
python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 8 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```


## 自定义存储后端集成

要集成新的存储后端：

1. **实现三个核心方法：**
   - `get(key)`：按键检索值
   - `exists(key)`：检查键是否存在
   - `set(key, value)`：存储键值对

2. **注册你的后端：** 将你的存储后端添加到 HiCache [BackendFactory](../../python/sglang/srt/mem_cache/storage/backend_factory.py#L188)

HiCache 控制器会自动处理所有调度和同步。

### 动态后端加载

另外，你可以使用动态加载来避免在代码仓库中硬编码你的后端：

```bash
python3 -m sglang.launch_server \
  --model-path your-model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{"backend_name":"custom_backend_name", "module_path": "your_module_path", "class_name": "YourHiCacheClassName"}'
```

**配置参数：**
- `--hicache-storage-backend`：设置为 `dynamic`
- `--hicache-storage-backend-extra-config`：JSON 配置，包含：
  - `backend_name`：自定义后端标识符
  - `module_path`：你的实现的 Python 模块路径
  - `class_name`：你的 HiCache 实现类名
  - `interface_v1`：0（禁用）或 1（启用），控制是否使用 batch_get_v1 和 batch_set_v1 方法


## 社区和支持

- **GitHub Issues**：报告 bug 和功能请求
- **Slack 频道**：在 #sgl-kv-cache-store 加入社区讨论
- **文档**：参阅特定存储后端指南

---

*本文档将根据社区反馈和新功能持续更新。欢迎贡献和建议！*
