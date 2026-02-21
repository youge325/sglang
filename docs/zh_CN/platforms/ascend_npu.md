# SGLang NPU 支持安装指南

您可以使用以下任一方法安装 SGLang。请查阅 `系统设置` 部分以确保集群以最佳性能运行。如果遇到任何问题，欢迎在 [sglang 仓库](https://github.com/sgl-project/sglang/issues)提交 issue。

## SGLang 组件版本对照表
| 组件              | 版本                    | 获取方式                                                                                                                                                                                                                     |
|-------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HDK               | 25.3.RC1                  | [链接](https://hiascend.com/hardware/firmware-drivers/commercial?product=7&model=33) |
| CANN              | 8.5.0                     | [获取镜像](#obtain-cann-image)                                                                                                                                                                                          |
| Pytorch Adapter   | 7.3.0                   | [链接](https://gitcode.com/Ascend/pytorch/releases)                                                                                                                                                                          |
| MemFabric         | 1.0.5                   | `pip install memfabric-hybrid==1.0.5`                                                                                                                                                                 |
| Triton            | 3.2.0                   | `pip install triton-ascend`|
| SGLang NPU Kernel | NA                      | [链接](https://github.com/sgl-project/sgl-kernel-npu/releases)                                                                                                                                                               |

<a id="obtain-cann-image"></a>
### 获取 CANN 镜像
您可以通过镜像获取指定版本的 CANN 依赖。
```shell
# 适用于 Atlas 800I A3 和 Ubuntu 操作系统
docker pull quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11
# 适用于 Atlas 800I A2 和 Ubuntu 操作系统
docker pull quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11
```

## 准备运行环境

### 方法一：从源码安装（需预先安装依赖）

#### Python 版本

目前仅支持 `python==3.11`。如果不想影响系统预装的 Python，可以尝试使用 [conda](https://github.com/conda/conda) 安装。

```shell
conda create --name sglang_npu python=3.11
conda activate sglang_npu
```

#### CANN

在 Ascend 上使用 SGLang 之前，您需要安装 CANN Toolkit、Kernels 算子包和 NNAL（版本 8.3.RC2 或更高），请查看[安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)。

#### MemFabric-Hybrid

如果您想使用 PD 分离模式，需要安装 MemFabric-Hybrid。MemFabric-Hybrid 是 Mooncake Transfer Engine 的直接替代品，可在 Ascend NPU 集群上实现 KV cache 传输。

```shell
pip install memfabric-hybrid==1.0.5
```

#### Ascend 上的 Pytorch 和 Pytorch 框架适配器

```shell
PYTORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
TORCH_NPU_VERSION=2.8.0
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$TORCH_NPU_VERSION
```

如果您使用其他版本的 `torch` 并安装 `torch_npu`，请查看[安装指南](https://github.com/Ascend/pytorch/blob/master/README.md)。

#### Ascend 上的 Triton

我们提供了 Ascend 平台的 Triton 实现。

```shell
pip install triton-ascend
```
如需安装 Ascend 上的 Triton nightly 版本或从源码编译，请参阅[安装指南](https://gitcode.com/Ascend/triton-ascend/blob/master/docs/sources/getting-started/installation.md)。

#### SGLang Kernels NPU
我们提供了 Ascend NPU 的 SGL 内核，请查看[安装指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md)。

#### DeepEP 兼容库
我们提供了一个 DeepEP 兼容库，作为 deepseek-ai 的 DeepEP 库的直接替代品，请查看[安装指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md)。

#### 从源码安装 SGLang

```shell
# 使用最新的发布分支
git clone https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_npu.toml python/pyproject.toml
pip install -e python[all_npu]
```

### 方法二：使用 Docker 镜像
#### 获取镜像
您可以下载 SGLang 镜像或基于 Dockerfile 构建镜像以获取 Ascend NPU 镜像。
1. 下载 SGLang 镜像
```angular2html
dockerhub: docker.io/lmsysorg/sglang:$tag
# 基于 main 的标签，将 main 替换为特定版本如 v0.5.6，
# 即可获取特定版本的镜像
Atlas 800I A3 : {main}-cann8.5.0-a3
Atlas 800I A2: {main}-cann8.5.0-910b
```
2. 基于 Dockerfile 构建镜像
```shell
# 克隆 SGLang 仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# 构建 Docker 镜像
# 如果出现网络错误，请修改 Dockerfile 使用离线依赖或使用代理
docker build -t <image_name> -f npu.Dockerfile .
```

#### 创建 Docker 容器
__注意：__ RDMA 需要 `--privileged` 和 `--network=host`，这通常是 Ascend NPU 集群所需的。

__注意：__ 以下 Docker 命令基于 Atlas 800I A3 机器。如果您使用 Atlas 800I A2，请确保只将 `davinci[0-7]` 映射到容器中。

```shell

alias drun='docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc \
    --volume /usr/local/sbin:/usr/local/sbin --volume /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume /etc/ascend_install.info:/etc/ascend_install.info \
    --volume /var/queue_schedule:/var/queue_schedule --volume ~/.cache/:/root/.cache/'

# 添加 HF_TOKEN 环境变量，用于 SGLang 下载模型。
drun --env "HF_TOKEN=<secret>" \
    <image_name> \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

## 系统设置

### CPU 性能电源策略

Ascend 硬件上的默认电源策略为 `ondemand`，可能会影响性能，建议将其更改为 `performance`。

```shell
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 确保更改已成功应用
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor # 显示 performance
```

### 禁用 NUMA 均衡

```shell
sudo sysctl -w kernel.numa_balancing=0
# 检查
cat /proc/sys/kernel/numa_balancing # 显示 0
```

### 防止交换系统内存

```shell
sudo sysctl -w vm.swappiness=10

# 检查
cat /proc/sys/vm/swappiness # 显示 10
```

## 运行 SGLang 服务
### 运行大语言模型服务
#### PD 混合场景
```shell
# 启用 CPU 亲和性
export SGLANG_SET_CPU_AFFINITY=1
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --attention-backend ascend
```

#### PD 分离场景
1. 启动 Prefill 服务器
```shell
# 启用 CPU 亲和性
export SGLANG_SET_CPU_AFFINITY=1

# PIP：建议配置为第一个 Prefill 服务器 IP
# PORT：一个空闲端口
# 所有 sglang 服务器需要配置相同的 PIP 和 PORT
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# 如果您使用 Atlas 800I A2 硬件并使用 RDMA 进行 KV cache 传输，请添加此参数
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8995 \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 0 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8000
```

2. 启动 Decode 服务器
```shell
# PIP：建议配置为第一个 Prefill 服务器 IP
# PORT：一个空闲端口
# 所有 sglang 服务器需要配置相同的 PIP 和 PORT
export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
# 如果您使用 Atlas 800I A2 硬件并使用 RDMA 进行 KV cache 传输，请添加此参数
export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --device npu \
    --base-gpu-id 1 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 8001
```

3. 启动路由器
```shell
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://127.0.0.1:8000 8995 \
    --decode http://127.0.0.1:8001 \
    --host 127.0.0.1 \
    --port 6688
```

### 运行多模态语言模型服务
#### PD 混合场景
```shell
python3 -m sglang.launch_server \
    --model-path Qwen3-VL-30B-A3B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --tp 4 \
    --device npu \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --disable-radix-cache \
    --trust-remote-code \
    --enable-multimodal \
    --sampling-backend ascend
```
