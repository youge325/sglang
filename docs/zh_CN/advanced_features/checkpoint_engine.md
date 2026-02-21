# 检查点引擎集成 (Checkpoint Engine Integration)

SGLang 检查点引擎集成提供了一种使用分布式检查点加载系统高效加载模型权重的方式。此功能通过在多个进程和节点之间并行化权重加载过程，显著减少了模型加载时间，特别适用于大型模型和多节点部署场景。

## 概述

检查点引擎集成使 SGLang 能够：
- 使用多个进程并行加载模型权重
- 跨多个节点分配权重加载以提高有效磁盘带宽
- 将权重加载与其他初始化任务（如 CUDA Graph 捕获）重叠执行
- 支持单节点和多节点部署

## 安装

首先，安装检查点引擎包：

```bash
pip install 'checkpoint-engine[p2p]'
```

## 架构

系统由两个主要组件组成：

1. **SGLang 服务器**：使用 `--wait-for-initial-weights` 标志运行，等待权重加载完成后再进入就绪状态
2. **检查点引擎工作进程**：独立的进程（由 torchrun 管理），负责加载和分发模型权重

检查点引擎采用参数服务器架构，支持以下模式：
- **Broadcast 模式**：权重从加载进程广播到推理进程
- **P2P 模式**：进程之间直接进行点对点权重传输
- **All 模式**：同时使用广播和点对点两种方法

## 使用示例

### 单节点部署

**终端 1 - 启动 SGLang 服务器：**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights
```

**终端 2 - 运行检查点引擎：**

使用 sglang 入口点：
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

使用 torchrun 直接运行：
```bash
torchrun --nproc-per-node 8 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### 多节点部署（2 个节点）

**节点 0：**

启动 SGLang 服务器：
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

运行检查点引擎：

使用 sglang 入口点（推荐）：
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

使用 torchrun 直接运行：
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

**节点 1：**

启动 SGLang 服务器：
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP]
```

运行检查点引擎：

使用 sglang 入口点（推荐）：
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

使用 torchrun 直接运行：
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

### 多节点部署配合张量并行 (TP=16)

**节点 0：**

启动 SGLang 服务器：
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 0
```

运行检查点引擎：

使用 sglang 入口点（推荐）：
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

使用 torchrun 直接运行：
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

**节点 1：**

启动 SGLang 服务器：
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights \
    --host [IP] \
    --dist-init-addr [IP]:9120 \
    --nnodes 2 \
    --node-rank 1
```

运行检查点引擎：

使用 sglang 入口点（推荐）：
```bash
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

使用 torchrun 直接运行：
```bash
torchrun --nproc-per-node 8 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr [IP] \
    --master-port 29500 \
    examples/checkpoint_engine/update.py \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 16
```

## 配置选项

### SGLang 服务器选项

- `--load-format dummy`：使用虚拟格式进行初始加载（允许与其他任务重叠执行）
- `--wait-for-initial-weights`：等待检查点引擎提供权重后再进入就绪状态
- `--host`：多节点部署的主机地址
- `--dist-init-addr`：张量并行的分布式初始化地址

### 检查点引擎选项

- `--update-method`：权重更新方法（`broadcast`、`p2p` 或 `all`）
- `--checkpoint-path`：模型检查点目录路径
- `--inference-parallel-size`：推理并行进程数
- `--endpoint`：SGLang 服务器端点（默认：`http://localhost:19730`）
- `--checkpoint-name`：检查点名称（默认：`my-checkpoint-iter-0`）
- `--save-metas-file`：保存检查点元数据的文件
- `--load-metas-file`：加载检查点元数据的文件
- `--uds`：用于通信的 Unix 域套接字路径
- `--weight-version`：权重版本标识符

## 性能优势

检查点引擎在两个主要方面提供了显著的时间节省：

1. **多节点加载**：每个节点只从磁盘加载一部分权重，有效提高了磁盘带宽。参与的节点越多，加速效果越大。初步测试表明，在两个节点的 H20-3e 上加载 DeepSeek-R1 可加速 20 秒。

2. **单进程优化**：使用虚拟格式（dummy format）允许将磁盘到 CPU 的传输与 CUDA Graph 捕获和其他初始化任务重叠执行，提供额外的时间节省。

## 故障排除

- 确保已安装检查点引擎包：`pip install 'checkpoint-engine[p2p]'`
- 在多节点部署中验证节点间的网络连通性
- 检查检查点路径是否包含有效的模型文件
- 监控日志以排查 SGLang 服务器与检查点引擎之间的连接错误
- 如需调试，可使用 `--sleep-time` 参数添加延迟

## 参考资料

- [Checkpoint Engine 仓库](https://github.com/MoonshotAI/checkpoint-engine)
