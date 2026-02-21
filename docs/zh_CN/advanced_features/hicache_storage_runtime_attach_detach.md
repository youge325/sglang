# 运行时挂载/卸载 HiCache 存储后端（无需重启）

本文档说明如何在 **SGLang 已经运行并处理流量的情况下**，**动态挂载/卸载 HiCache L3 存储后端**（如 `mooncake` / `hf3fs` / `nixl` / `file` / `aibrix` / `eic`），无需重启进程。

出于安全和一致性考虑，当前实现**严格要求**这些操作仅在服务**空闲**时进行：

- **没有正在运行的请求**
- **没有等待/排队的请求**

如果空闲条件不满足，API 会快速失败（HTTP 400）并且**不会修改**当前的服务状态。

---

## 1. 背景和实现概述

### 1.1 架构/控制路径

控制路径如下：

1. **HTTP Server**（`python/sglang/srt/entrypoints/http_server.py`）
   - 暴露 `PUT /hicache/storage-backend`、`DELETE /hicache/storage-backend`、`GET /hicache/storage-backend`
2. **TokenizerManager**（`python/sglang/srt/managers/tokenizer_communicator_mixin.py`）
   - 通过 `_Communicator` 将请求发送到 Scheduler
3. **Scheduler**（`python/sglang/srt/managers/scheduler.py`）
   - 执行**严格的空闲检查**
   - 调用 `tree_cache.attach_storage_backend(...)` / `detach_storage_backend(...)`
4. **HiRadixCache**（`python/sglang/srt/mem_cache/hiradix_cache.py`）
   - 解析 `hicache_storage_backend_extra_config_json`（支持后端配置和预取参数）
   - 调用 `cache_controller.attach_storage_backend(...)` / `detach_storage_backend(...)`
5. **HiCacheController**（`python/sglang/srt/managers/cache_controller.py`）
   - 创建/销毁存储后端实例（通过 `StorageBackendFactory`）
   - 在运行时启动/停止后端后台线程（预取/备份）

---

## 2. 空闲状态要求（严格）

Scheduler 使用更严格的 `_is_idle_for_hicache_storage_op()`：

- `_is_no_request()` 为 true（涵盖 running/overlap/pp/disagg 及其他活跃状态）
- `waiting_queue` 为空
- `grammar_queue` 为空（如果启用了 grammar 后端）

如果条件不满足，挂载/卸载会返回如下错误：

- `Reject attach: scheduler is not idle. #queue-req=... #running-req=...`

> 提示：在切换之前，先排空上游流量并等待服务器进入空闲状态，然后再调用挂载/卸载。

### 2.1 DP（数据并行）语义

当 `dp_size > 1` 时，tokenizer 会将请求分发到**所有 DP scheduler 实例**并聚合它们的响应：

- 最终的 `success` **仅当所有 DP rank 都返回成功时**才为 true
- 最终的 `message` 会连接所有 DP rank 的消息

这旨在防止"静默部分成功"，但也意味着你可能会看到：

- 整体**失败**，即使**某些 rank 已经成功**

目前**没有自动的跨 DP rank 部分回滚**（参见代码中的 TODO）。操作建议：

- 优先保持所有 rank 的后端配置一致
- 如果挂载失败，立即调用卸载（尽力而为/幂等），修复配置，然后重试挂载

---

## 3. 使用方法（HTTP 管理 API）

以下示例假设你的 SGLang HTTP 服务器地址为 `http://127.0.0.1:30000`。

### 3.1 查询当前存储后端状态

```bash
curl -s http://127.0.0.1:30000/hicache/storage-backend
```

示例响应：

```json
{
  "hicache_storage_backend": "mooncake",
  "hicache_storage_backend_extra_config": "{\"master_server_address\":\"127.0.0.1:50051\", ...}"
}
```

### 3.2 挂载（启用）存储后端
```bash
curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "mooncake"
  }'
```

```bash
curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{
    "hicache_storage_backend": "mooncake",
    "hicache_storage_backend_extra_config_json": "{\"master_server_address\":\"127.0.0.1:50051\",\"protocol\":\"tcp\",\"global_segment_size\":\"4gb\",\"prefetch_threshold\":256}",
    "hicache_storage_prefetch_policy": "timeout"
  }'
```

注意事项：

- `hicache_storage_backend_extra_config_json` 可以同时包含：
  - **后端配置**（如 Mooncake master/metadata/protocol 等）
  - **预取配置**（`prefetch_threshold`、`prefetch_timeout_base`、`prefetch_timeout_per_ki_token`、`hicache_storage_pass_prefix_keys`）

### 3.3 卸载（禁用）存储后端

```bash
curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
```

注意事项：

- 卸载只是让 SGLang **停止使用** L3 存储后端并停止预取/备份线程
- 它**不会自动删除**存储在 Mooncake/HF3FS（或其他远程后端）中的数据

---

## 4. 行为和注意事项

- **无需重启**：挂载/卸载在运行时进程内切换
- **必须空闲**：否则请求会被拒绝以避免一致性问题
- **主机 KV 布局约束仍然适用**：例如，Mooncake 仍然要求 `page_first/page_first_direct/page_head` 等布局；如果服务器的 HiCache 主机内存布局不满足后端要求，挂载将会失败并报错
- **可观测性**：
  - 挂载后，`server_args.hicache_storage_backend*` 会在 tokenizer 和 scheduler 两侧更新
  - 如果启用了指标收集，挂载会按需在 `HiRadixCache` 中创建存储指标收集器
