# 基准测试与性能分析

## 基准测试

- 在不启动服务器的情况下，对运行单个静态批次的延迟进行基准测试。参数与 `launch_server.py` 相同。
  请注意，这是一个没有动态批处理服务器的简化测试脚本，因此可能会因批次大小过大而耗尽内存，而实际服务器可以处理更大的批次。实际服务器会将预填充（prefill）截断为多个批次，而此简化脚本不会。
  - 不使用服务器（无需启动服务器）
    ```bash
    python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32
    ```
  - 使用服务器（请先使用 `sglang.launch_server` 启动服务器，然后运行以下命令。）
    ```bash
    python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 32 --input-len 256 --output-len 32
    ```


- 离线处理基准测试。此脚本将启动一个离线引擎并运行基准测试。

  ```bash
  python3 -m sglang.bench_offline_throughput --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --num-prompts 10
  ```

- 在线服务基准测试。请先使用 `sglang.launch_server` 启动服务器，然后运行以下命令。

  ```bash
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```

## 使用 PyTorch Profiler 进行性能分析

[Pytorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 是一个方便的基础工具，用于检查内核执行时间、调用栈以及内核重叠和占用率。

### 使用 `sglang.bench_serving` 分析服务器

```bash
# 设置追踪路径
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# 启动服务器
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# 从客户端发送性能分析请求
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

必须在服务器端和客户端都设置 `SGLANG_TORCH_PROFILER_DIR` 环境变量；否则追踪文件将无法正确生成。一种安全的方式是在 shell 的资源文件中设置（例如 bash 的 `~/.bashrc`）。

更多详情请参阅 [Bench Serving 指南](./bench_serving.md)。

### 在 PD 分离模式下进行性能分析

在 PD 分离模式下进行性能分析时，由于 torch profiler 的限制，预填充和解码工作节点**必须分别进行分析**。`bench_serving` 命令提供了专门的选项：

#### 分析预填充工作节点

```bash
# 设置追踪路径
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# 启动预填充和解码服务器（设置详情请参阅 PD 分离文档）
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode prefill
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disaggregation-mode decode --port 30001 --base-gpu-id 1

# 启动路由器
python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000

# 发送针对预填充工作节点的性能分析请求
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000
```

#### 分析解码工作节点

```bash
# 发送针对解码工作节点的性能分析请求
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001
```

#### 重要说明

- `--profile-prefill-url` 和 `--profile-decode-url` **互斥** — 不能同时分析两者
- 两个选项都支持多个工作节点 URL，用于多实例设置：
  ```bash
  # 分析多个预填充工作节点
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-prefill-url http://127.0.0.1:30000 http://127.0.0.1:30002

  # 分析多个解码工作节点
  python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --profile --pd-separated --profile-decode-url http://127.0.0.1:30001 http://127.0.0.1:30003
  ```
- 确保在启动服务器之前，在所有工作节点上设置 `SGLANG_TORCH_PROFILER_DIR`
- 有关 PD 分离设置的更多详情，请参阅 [PD 分离指南](../advanced_features/pd_disaggregation.md)

### 使用 `sglang.bench_offline_throughput` 分析服务器
```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# 使用 bench_one_batch.py 分析单个批次
# 可以通过 --batch 参数控制批次大小
python3 -m sglang.bench_one_batch --model-path meta-llama/Llama-3.1-8B-Instruct --batch 32 --input-len 1024 --output-len 10 --profile

# 使用 bench_offline_throughput.py 分析多个批次
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### 使用 `sglang.profiler` 分析服务器

当服务器正在运行时（例如处理解码请求），你可以通过向服务器发送分析请求来立即开始实时性能分析。

你可以通过运行 `python3 -m sglang.profiler` 来完成。例如：

```
# 终端 1：发送一个生成请求
python3 -m sglang.test.send_one

# 终端 2：在上述请求完成之前，在另一个终端中快速启动以下命令。
# 它将为上述请求的多个解码批次生成性能分析报告。
python3 -m sglang.profiler
```

你也可以将上述操作合并为一个命令

```
python3 -m sglang.test.send_one --profile
```

### 使用 HTTP API 端点分析服务器

SGLang 提供了 HTTP API 端点来控制运行中服务器的性能分析。这允许你以编程方式启动和停止分析，这对于捕获特定的工作负载模式非常有用。

#### 使用 `/start_profile` 端点

`/start_profile` 端点在服务器上启动性能分析。你可以使用以下参数控制分析何时开始以及运行多长时间：

**基本用法：**

```bash
# 立即开始分析 10 个步骤
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "num_steps": 10
  }'
```

**参数：**

- `output_dir`（可选）：保存分析追踪文件的目录。如果未指定，则使用 `SGLANG_TORCH_PROFILER_DIR` 环境变量，默认为 `/tmp`
- `num_steps`（可选）：要分析的步骤数。如果未指定，分析将持续进行直到使用 `/end_profile` 手动停止
- `start_step`（可选）：开始分析的步骤编号（包含该步骤）。适用于跳过预热迭代
- `activities`（可选）：要分析的活动列表，例如 `["CPU", "GPU"]`。默认为 `["CPU", "GPU"]`
- `merge_profiles`（可选）：是否合并分布式追踪。默认为 `false`

**关于步骤范围的说明：** 分析从 `start_step`（包含）开始，持续 `num_steps` 次迭代。例如，当 `start_step=3` 且 `num_steps=10` 时，分析将捕获步骤 3、4、5、6、7、8、9、10、11 和 12（从步骤 3 开始共 10 个步骤）。

**使用 `start_step` 的高级用法：**

```bash
# 等待 5 个步骤（预热），然后分析 10 个步骤
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU"]
  }'
```

**持续分析（手动停止）：**

```bash
# 不指定 num_steps 启动分析 — 必须使用 /end_profile 手动停止
curl -X POST http://127.0.0.1:30000/start_profile
```

#### 使用 `/end_profile` 端点

`/end_profile` 端点停止正在进行的分析会话并保存追踪文件。

```bash
# 停止分析并保存追踪文件
curl -X POST http://127.0.0.1:30000/end_profile
```

仅在不指定 `num_steps` 启动分析时才需要此操作。如果指定了 `num_steps`，分析将在达到指定步骤数后自动停止。

#### 示例工作流

```bash
# 终端 1：启动服务器
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# 终端 2：启动持续分析
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "start_step": 3
  }'

# 终端 3：发送请求以生成负载
python -m sglang.bench_serving --backend sglang --num-prompts 100

# 终端 2：完成后停止分析
curl -X POST http://127.0.0.1:30000/end_profile
```

### 分布式追踪的 Profiler 追踪合并器

SGLang 现在支持自动合并来自多种并行类型（TP、DP、PP、EP）分布式设置的分析追踪。此功能对于分析分布式运行中的性能特别有用。

#### 多节点分析与共享存储注意事项

单节点的 profiler 输出合并完全支持。在跨多个节点的分布式环境中进行分析时，所有节点应能访问共享存储（例如 NFS、Lustre）作为输出目录，以便合并追踪文件。

如果跨节点没有可访问的共享存储，目前尚不直接支持在分析过程中自动合并追踪文件。

#### HTTP API 用法

```bash
# 启动分析并启用自动追踪合并
curl -X POST <BASE_URL>/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles", # 存储分析追踪的位置
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true # 可选参数，用于合并分析追踪（默认=False）
  }'
```

#### 命令行用法

```bash
# 启动分析并启用合并
python -m sglang.profiler \
  --num-steps 10 \
  --cpu \
  --gpu \
  --output-dir /tmp/profiles \
  --merge-profiles # 可选参数，用于合并分析追踪（默认=False）
```

#### 输出文件

分析合并器会生成：
- 各 rank 的追踪文件：`{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- 合并后的追踪文件：`merged-{profile_id}.trace.json.gz`

### 可能的 PyTorch Bug
如果在任何情况下你遇到以下错误（例如使用 qwen 2.5 VL 时）：
```bash
RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/profiler_python.cpp":983, please report a bug to PyTorch. Python replay stack is empty.
```
这可能是在 [Bug: vLLM Profiler](https://github.com/vllm-project/vllm/issues/18240) 和 [Bug: torch.profiler.profile](https://github.com/pytorch/pytorch/issues/101632) 中报告的 PyTorch Bug。作为解决方法，你可以使用环境变量禁用 `with_stack`，如下所示：
```bash
export SGLANG_PROFILE_WITH_STACK=False
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### 查看追踪文件

追踪文件可以从以下位置加载和可视化：

1. https://ui.perfetto.dev/（任何浏览器）
2. chrome://tracing（仅限 Chrome 浏览器）

如果浏览器因追踪文件过大而无法打开，客户端可以通过控制提示数量和提示输出长度来生成较小的追踪文件（<100MB）。
例如，在分析服务器时，

```bash
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 2 --sharegpt-output-len 100 --profile
```

此命令使用 `--num-prompts` 参数将提示数量设置为 2，并使用 `--sharegpt-output-len` 参数将输出序列长度限制为 100，这可以生成较小的追踪文件，使浏览器能够顺畅打开。

此外，如果你想通过追踪中的 CUDA 内核定位 SGLang Python 源代码，需要在启动服务时禁用 CUDA Graph。可以在启动服务的命令中使用 `--disable-cuda-graph` 参数来实现。

## 使用 Nsight 进行性能分析

[Nsight systems](https://docs.nvidia.com/nsight-systems/) 是一个高级工具，能暴露更多分析细节，例如寄存器和共享内存使用情况、标注的代码区域以及底层 CUDA API 和事件。

1. 前置条件：

   使用 apt 安装，或在 [NVIDIA Docker 容器](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) 或 [SGLang Docker 容器](https://github.com/sgl-project/sglang/tree/main/docker) 中运行。

   ```bash
   # 安装 nsys
   # https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
   apt update
   apt install -y --no-install-recommends gnupg
   echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
   apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
   apt update
   apt install nsight-systems-cli
   ```

2. 要分析单个批次，使用

   ```bash
   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node python3 -m sglang.bench_one_batch --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512
   ```

3. 要分析服务器，例如

   ```bash
   # 启动服务器，根据需要设置延迟和持续时间
   # 当持续时间用完后，服务器将被 nsys 终止

   nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70 python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

   # 客户端
   python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --dataset-name random --random-input 1024 --random-output 512
   ```

   实际使用中，我们建议用户将 `--duration` 参数设置为较大的值。当用户希望服务器停止分析时，首先运行：

   ```bash
   nsys sessions list
   ```

   获取形如 `profile-XXXXX` 的会话 ID，然后运行：

   ```bash
   nsys stop --session=profile-XXXXX
   ```

   手动终止分析器并立即生成 `nsys-rep` 文件。

4. 使用 NVTX 标注代码区域，例如查看其执行时间。

   ```bash
   # 安装 nvtx
   pip install nvtx
   ```

   ```python
   # 代码片段
   import nvtx
   with nvtx.annotate("description", color="color"):
       # 一些关键代码
   ```

### 使用 Nsight Systems 进行逐层 NVTX 分析

SGLang 提供了内置的逐层 NVTX 标注，可以与 CUDA Profiler 结合使用，在 Nsight Systems 中进行详细的逐层分析。这对于在层级别识别性能瓶颈特别有用。

#### 将 `--enable-layerwise-nvtx-marker` 与 Nsight Systems 和 `/start_profile` 结合使用

`--enable-layerwise-nvtx-marker` 标志会自动为模型中的每一层添加 NVTX 标记。当与 Nsight Systems 分析结合使用时，可以看到详细的逐层性能信息，功能非常强大。

**方法 1：将 `/start_profile` 与 CUDA_PROFILER 结合使用（用于编程控制）**

此方法允许你在 Nsight Systems 运行时，通过 HTTP API 精确控制分析的启动/停止时间。

1. 在 Nsight Systems 下启动带有逐层 NVTX 的服务器：

   ```bash
   # 终端 1：使用 nsys 和 capture-range 选项启动服务器
   nsys profile --trace-fork-before-exec=true \
     --cuda-graph-trace=node \
     --capture-range=cudaProfilerApi \
     --capture-range-end=stop \
     -o layerwise_profile \
     python -m sglang.launch_server \
       --model-path meta-llama/Llama-3.1-8B-Instruct \
       --enable-layerwise-nvtx-marker \
       --disable-cuda-graph
   ```

   注意：NVTX 标记不会为 CUDA Graph 捕获的内核启动发出。使用 `--disable-cuda-graph` 以确保所有逐层 NVTX 标记都出现在追踪中。

2. 在另一个终端中，通过 `/start_profile` 使用 `CUDA_PROFILER` 活动控制分析：

   ```bash
   # 终端 2：等待服务器就绪，然后启动 CUDA 分析
   # 等待 3 个步骤用于预热，然后分析 10 个步骤
   curl -X POST http://127.0.0.1:30000/start_profile \
     -H "Content-Type: application/json" \
     -d '{
       "start_step": 3,
       "num_steps": 10,
       "activities": ["CUDA_PROFILER"]
     }'
   ```

3. 发送请求以生成负载：

   ```bash
   # 终端 3：生成工作负载
   python -m sglang.bench_serving --backend sglang --num-prompts 100
   ```

4. 分析将在 10 个步骤后自动停止（由于 `num_steps: 10`）。如果你未指定 `num_steps`，则需要手动停止：

   ```bash
   # 终端 2：仅在未指定 num_steps 时需要
   curl -X POST http://127.0.0.1:30000/end_profile
   ```

`--capture-range=cudaProfilerApi` 选项告诉 Nsight Systems 仅捕获 `cudaProfilerStart()` 和 `cudaProfilerStop()` 调用之间的数据（由 `/start_profile` 和 `/end_profile` 触发），从而减少开销和文件大小。`start_step` 参数跳过前 3 个步骤以避免捕获预热开销。

**方法 2：不使用 `/start_profile` API 的更简单方法**

对于不需要精细控制分析启动/停止的简单场景，你可以使用 Nsight Systems 捕获整个工作负载：

```bash
# 终端 1：启动带有逐层 NVTX 的服务器
# 注意：--disable-cuda-graph 确保所有 NVTX 标记都会被发出
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --enable-layerwise-nvtx-marker \
  --disable-cuda-graph

# 终端 2：分析基准测试客户端
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o layerwise_profile \
  python -m sglang.bench_serving --backend sglang --num-prompts 10
```

此方法分析整个客户端执行过程，包括所有服务器交互。逐层 NVTX 标记将在 Nsight Systems 时间线中可见。

**查看分析结果：**

使用 Nsight Systems 打开生成的 `.qdrep` 文件：

```bash
nsys-ui layerwise_profile.qdrep
```

在 Nsight Systems GUI 中，你将看到：
- **NVTX 范围**：每一层在时间线中显示为带标签的范围，标记元数据中包含详细信息
- **CUDA 内核**：所有 GPU 内核与层标注一起显示
- **层级结构**：完整的模块路径（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct.model.layers.0.self_attn.qkv_proj`）帮助识别特定层。前缀使用 `--model-path` 中的完整模型路径。
- **张量形状**：输入/输出维度和参数形状包含在 NVTX 标记数据中

**逐层 NVTX 分析的优势：**

- **精细可见性**：精确查看哪些层耗时最多
- **内存追踪**：识别内存分配较大的层
- **瓶颈识别**：快速定位低效操作
- **通信开销**：在多 GPU 设置中，查看每层的通信开销
- **开发调试**：验证模型架构更改是否具有预期的性能影响

## 其他提示

1. 你可以通过仅提供 config.json 文件来使用虚拟权重对模型进行基准测试。这允许在不进行训练的情况下快速测试模型变体。为此，在上述命令中添加 `--load-format dummy`，然后你只需要在检查点文件夹下有一个正确的 `config.json`。
2. 你可以使用 `--json-model-override-args` 来使用修改后的配置（例如更少的层数）对模型进行基准测试。例如，你可以使用以下命令对仅有 2 层和 2 个 KV 头的模型进行基准测试：

   ```bash
   python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --batch 32 --input-len 256 --output-len 32 --load-format dummy --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
   ```

3. 你可以使用 `--python-backtrace=cuda` 来查看所有 CUDA 内核的 Python 调用栈，类似于 PyTorch Profiler。（注意：这可能导致基于 CUDA 事件计时的内核运行时间不准确地偏长）
4. 更多参数请参阅 [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
