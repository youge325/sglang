# MindSpore 模型

## 简介

MindSpore 是一个针对昇腾 NPU 优化的高性能 AI 框架。本文档指导用户在 SGLang 中运行 MindSpore 模型。

## 环境要求

MindSpore 目前仅支持昇腾 NPU 设备。用户需要首先安装 CANN 8.5。
CANN 软件包可从[昇腾官网](https://www.hiascend.com)下载。

## 支持的模型

目前支持以下模型：

- **Qwen3**：Dense 和 MoE 模型
- **DeepSeek V3/R1**
- *更多模型即将推出...*

## 安装

> **注意**：目前，MindSpore 模型由独立包 `sgl-mindspore` 提供。MindSpore 的支持建立在 SGLang 对昇腾 NPU 平台的现有支持之上。请先[安装 SGLang 的昇腾 NPU 版本](../../platforms/ascend_npu.md)，然后安装 `sgl-mindspore`：

```shell
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```


## 运行模型

当前 SGLang-MindSpore 支持 Qwen3 和 DeepSeek V3/R1 模型。本文档以 Qwen3-8B 为例。

### 离线推理

使用以下脚本进行离线推理：

```python
import sglang as sgl

# 使用 MindSpore 后端初始化引擎
llm = sgl.Engine(
    model_path="/path/to/your/model",  # 本地模型路径
    device="npu",                      # 使用 NPU 设备
    model_impl="mindspore",            # MindSpore 实现
    attention_backend="ascend",        # 注意力后端
    tp_size=1,                         # 张量并行大小
    dp_size=1                          # 数据并行大小
)

# 生成文本
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### 启动服务器

使用 MindSpore 后端启动服务器：

```bash
# 基本服务器启动
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

多节点分布式服务器：

```bash
# 多节点分布式服务器
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

## 故障排除

#### 调试模式

通过 log-level 参数启用 SGLang 调试日志。

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

通过设置环境变量启用 MindSpore 的 info 和 debug 日志。

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

#### 显式选择设备

使用以下环境变量显式选择要使用的设备。

```shell
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # 设置设备
```

#### 部分通信环境问题

在某些特殊通信环境中，用户需要设置一些环境变量。

```shell
export MS_ENABLE_LCCL=off # 当前 SGLang-MindSpore 不支持 LCCL 通信模式
```

#### 部分 protobuf 依赖问题

在某些特殊 protobuf 版本的环境中，用户需要设置一些环境变量以避免二进制版本不匹配。

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # 避免 protobuf 二进制版本不匹配
```

## 支持
如需 MindSpore 相关问题的帮助：

- 请参阅 [MindSpore 文档](https://www.mindspore.cn/)
