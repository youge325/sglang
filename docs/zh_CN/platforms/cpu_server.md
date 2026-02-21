# CPU 服务器

本文档介绍如何在 CPU 服务器上设置 [SGLang](https://github.com/sgl-project/sglang) 环境并运行 LLM 推理。
SGLang 已在配备 Intel® AMX® 指令集的 CPU 上启用并优化，适用于第 4 代及更新的 Intel® Xeon® 可扩展处理器。

## 优化模型列表

以下热门 LLM 已在 CPU 上经过优化，可高效运行，包括最知名的开源模型，如 Llama 系列、Qwen 系列以及 DeepSeek 系列（如 DeepSeek-R1 和 DeepSeek-V3.1-Terminus）。

| 模型名称 | BF16 | W8A8_INT8 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8) | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3.1-Terminus |   | [IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8](https://huggingface.co/IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8) |   |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8) |   |
| QwQ-32B |   | [RedHatAI/QwQ-32B-quantized.w8a8](https://huggingface.co/RedHatAI/QwQ-32B-quantized.w8a8) |   |
| DeepSeek-Distilled-Llama |   | [RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |   |
| Qwen3-235B |   |   | [Qwen/Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |

**注意：** 上表中列出的模型标识已在第 6 代 Intel® Xeon® P-core 平台上验证通过。

## 安装

### 使用 Docker 安装

建议使用 Docker 来搭建 SGLang 环境。
提供了一个 [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile) 以方便安装。
请将下方的 `<secret>` 替换为您的 [HuggingFace 访问令牌](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
# 克隆 SGLang 仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# 构建 Docker 镜像
docker build -t sglang-cpu:latest -f xeon.Dockerfile .

# 启动 Docker 容器
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:latest /bin/bash
```

### 从源码安装

如果您更倾向于在裸机环境中安装 SGLang，安装步骤如下：

如果系统中尚未安装所需的软件包和库，请提前安装。
您可以参考 [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/xeon.Dockerfile#L11) 中基于 Ubuntu 的安装命令作为指导。

1. 安装 `uv` 包管理器，然后创建并激活虚拟环境：

```bash
# 以 '/opt' 作为示例 uv 环境目录，您可以根据需要更改
cd /opt
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
```

2. 创建配置文件以指定 `torch` 相关包的安装渠道（即 index-url）：

```bash
vim .venv/uv.toml
```

按 'a' 进入 `vim` 的插入模式，将以下内容粘贴到创建的文件中

```file
[[index]]
name = "torch"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchvision"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "torchaudio"
url = "https://download.pytorch.org/whl/cpu"

[[index]]
name = "triton"
url = "https://download.pytorch.org/whl/cpu"

```

保存文件（在 `vim` 中，按 'esc' 退出插入模式，然后输入 ':x+Enter'），
并将其设置为默认的 `uv` 配置。

```bash
export UV_CONFIG_FILE=/opt/.venv/uv.toml
```

3. 克隆 `sglang` 源代码并构建软件包

```bash
# 克隆 SGLang 代码
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# 使用专用的 toml 文件
cd python
cp pyproject_cpu.toml pyproject.toml
# 安装 SGLang 依赖库，并构建 SGLang 主包
uv pip install --upgrade pip setuptools
uv pip install .

# 构建 CPU 后端内核
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
uv pip install .
```

4. 设置所需的环境变量

```bash
export SGLANG_USE_CPU_ENGINE=1

# 设置 'LD_LIBRARY_PATH' 和 'LD_PRELOAD' 以确保 sglang 进程可以加载相关库
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=${LD_PRELOAD}:/opt/.venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2
```

注意事项：

- 环境变量 `SGLANG_USE_CPU_ENGINE=1` 是启用 SGLang CPU 引擎服务所必需的。

- 如果在 `sgl-kernel` 构建过程中遇到代码编译问题，请检查您的 `gcc` 和 `g++` 版本，如果版本过旧请进行升级。
    建议使用 `gcc-13` 和 `g++-13`，这些版本已在官方 Docker 容器中验证通过。

- 系统库路径通常位于以下目录之一：
    `~/.local/lib/`、`/usr/local/lib/`、`/usr/local/lib64/`、`/usr/lib/`、`/usr/lib64/`
    和 `/usr/lib/x86_64-linux-gnu/`。在上述示例命令中使用的是 `/usr/lib/x86_64-linux-gnu`。
    请根据您的服务器配置调整路径。

- 建议将以下内容添加到 `~/.bashrc` 文件中，以避免每次打开新终端时都需要设置这些变量：

    ```bash
    source .venv/bin/activate
    export SGLANG_USE_CPU_ENGINE=1
    export LD_LIBRARY_PATH=<YOUR-SYSTEM-LIBRARY-FOLDER>
    export LD_PRELOAD=<YOUR-LIBS-PATHS>
    ```

## 启动推理服务引擎

启动 SGLang 服务的示例命令：

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --host 0.0.0.0               \
    --tp 6
```

注意事项：

1. 运行 W8A8 量化模型时，请添加 `--quantization w8a8_int8` 参数。

2. `--tp 6` 参数指定使用 6 个 rank 进行张量并行（TP6）。
    指定的 TP 数量即为执行期间使用的 TP rank 数量。
    在 CPU 平台上，一个 TP rank 对应一个子 NUMA 集群（SNC）。
    通常可以通过操作系统命令（如 `lscpu`）获取可用的 SNC 信息。

    如果指定的 TP rank 数量与总 SNC 数量不同，系统将自动使用前 `n` 个 SNC。
    注意 `n` 不能超过总 SNC 数量，否则会导致错误。

    要指定使用的核心，需要通过环境变量 `SGLANG_CPU_OMP_THREADS_BIND` 进行显式设置。
    例如，如果我们想在 Xeon® 6980P 服务器上使用每个 SNC 的前 40 个核心运行 SGLang 服务，
    该服务器的一个 socket 上有 3 个 SNC，分别有 43-43-42 个核心，则应设置：

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
    ```

    请注意，设置 SGLANG_CPU_OMP_THREADS_BIND 后，各 rank 的可用内存量可能无法预先确定。
    您可能需要设置适当的 `--max-total-tokens` 以避免内存溢出错误。

3. 要使用 torch.compile 优化解码，请添加 `--enable-torch-compile` 参数。
    要指定使用 `torch.compile` 时的最大批处理大小，请设置 `--torch-compile-max-bs` 参数。
    例如，`--enable-torch-compile --torch-compile-max-bs 4` 表示使用 `torch.compile`
    并将最大批处理大小设置为 4。目前使用 `torch.compile` 优化的最大适用批处理大小为 16。

4. 服务启动时会自动触发预热步骤。
    当您看到日志 `The server is fired up and ready to roll!` 时，表示服务已就绪。

## 使用请求进行基准测试

您可以通过 `bench_serving` 脚本对性能进行基准测试。
在另一个终端中运行以下命令。示例命令如下：

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

可通过以下命令查看详细的参数说明：

```bash
python -m sglang.bench_serving -h
```

此外，请求可以使用 [OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html) 格式，
通过命令行（例如使用 `curl`）或您自己的脚本发送。

## 示例用法命令

大语言模型的参数量从不到 10 亿到数千亿不等。
超过 20B 参数的稠密模型预计需要在旗舰版第 6 代 Intel® Xeon® 处理器上运行，
使用双路配置，共 6 个子 NUMA 集群。约 10B 参数及以下的稠密模型，
或激活参数少于 10B 的 MoE（混合专家）模型，可以在更常见的第 4 代及更新的
Intel® Xeon® 处理器上运行，或使用旗舰版第 6 代 Intel® Xeon® 处理器的单路配置。

### 示例：运行 DeepSeek-V3.1-Terminus

在 Xeon® 6980P 服务器上启动 W8A8_INT8 DeepSeek-V3.1-Terminus 服务的示例命令：

```bash
python -m sglang.launch_server                                 \
    --model IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8 \
    --trust-remote-code                                        \
    --disable-overlap-schedule                                 \
    --device cpu                                               \
    --quantization w8a8_int8                                   \
    --host 0.0.0.0                                             \
    --enable-torch-compile                                     \
    --torch-compile-max-bs 4                                   \
    --tp 6
```

类似地，启动 FP8 DeepSeek-V3.1-Terminus 服务的示例命令如下：

```bash
python -m sglang.launch_server                     \
    --model deepseek-ai/DeepSeek-V3.1-Terminus     \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 4                       \
    --tp 6
```

注意：请将 `--torch-compile-max-bs` 设置为您部署所需的最大批处理大小，最大可设为 16。
示例中的值 `4` 仅供参考。

### 示例：运行 Llama-3.2-3B

使用 BF16 精度启动 Llama-3.2-3B 服务的示例命令：

```bash
python -m sglang.launch_server                     \
    --model meta-llama/Llama-3.2-3B-Instruct       \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

启动 W8A8_INT8 版本 Llama-3.2-3B 服务的示例命令：

```bash
python -m sglang.launch_server                     \
    --model RedHatAI/Llama-3.2-3B-quantized.w8a8   \
    --trust-remote-code                            \
    --disable-overlap-schedule                     \
    --device cpu                                   \
    --quantization w8a8_int8                       \
    --host 0.0.0.0                                 \
    --enable-torch-compile                         \
    --torch-compile-max-bs 16                      \
    --tp 2
```

注意：`--torch-compile-max-bs` 和 `--tp` 的设置仅为示例，应根据您的环境进行调整。
例如，在 Intel® Xeon® 6980P 服务器上使用 1 个 socket 的 3 个子 NUMA 集群时，可使用 `--tp 3`。

服务启动后，您可以使用 `bench_serving` 命令进行测试，
或按照[基准测试示例](#使用请求进行基准测试)创建自己的命令或脚本。
