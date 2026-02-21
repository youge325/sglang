# 安装 SGLang

您可以使用以下方法之一安装 SGLang。
本页面主要适用于常见的 NVIDIA GPU 平台。
对于其他或更新的平台，请参阅 [AMD GPU](../platforms/amd_gpu.md)、[Intel Xeon CPU](../platforms/cpu_server.md)、[TPU](../platforms/tpu.md)、[NVIDIA DGX Spark](https://lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/)、[NVIDIA Jetson](../platforms/nvidia_jetson.md)、[昇腾 NPU](../platforms/ascend_npu.md) 和 [Intel XPU](../platforms/xpu.md) 的专用页面。

## 方法 1：使用 pip 或 uv

推荐使用 uv 以获得更快的安装速度：

```bash
pip install --upgrade pip
pip install uv
uv pip install sglang
```

### 针对 CUDA 13

推荐使用 Docker（参见方法 3 中关于 B300/GB300/CUDA 13 的说明）。如果无法使用 Docker，请按以下步骤操作：

1. 首先安装支持 CUDA 13 的 PyTorch：
```bash
# 将 X.Y.Z 替换为您 SGLang 安装对应的版本号
uv pip install torch==X.Y.Z torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

2. 安装 sglang：
```bash
uv pip install sglang
```

3. 从 [sgl-project whl 发布页](https://github.com/sgl-project/whl/blob/gh-pages/cu130/sgl-kernel/index.html) 安装 CUDA 13 的 `sgl_kernel` 安装包。将 `X.Y.Z` 替换为您 SGLang 安装所需的 `sgl_kernel` 版本（可通过运行 `uv pip show sgl_kernel` 查看）。示例：
```bash
# x86_64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sgl_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_x86_64.whl"

# aarch64
uv pip install "https://github.com/sgl-project/whl/releases/download/vX.Y.Z/sgl_kernel-X.Y.Z+cu130-cp310-abi3-manylinux2014_aarch64.whl"
```

### **常见问题快速修复**
- 如果遇到 `OSError: CUDA_HOME environment variable is not set`，请使用以下方法之一将其设置为 CUDA 安装根目录：
  1. 使用 `export CUDA_HOME=/usr/local/cuda-<your-cuda-version>` 设置 `CUDA_HOME` 环境变量。
  2. 先按照 [FlashInfer 安装文档](https://docs.flashinfer.ai/installation.html) 安装 FlashInfer，然后按上述方法安装 SGLang。

## 方法 2：从源码安装

```bash
# 使用最新的发布分支
git clone -b v0.5.6.post2 https://github.com/sgl-project/sglang.git
cd sglang

# 安装 Python 包
pip install --upgrade pip
pip install -e "python"
```

**常见问题快速修复**

- 如果您想开发 SGLang，可以尝试使用开发 Docker 镜像。请参阅[配置 Docker 容器](../developer_guide/development_guide_using_docker.md#setup-docker-container)。Docker 镜像为 `lmsysorg/sglang:dev`。

## 方法 3：使用 Docker

Docker 镜像可在 Docker Hub 上获取：[lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags)，基于 [Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker) 构建。
请将下方的 `<secret>` 替换为您的 Hugging Face Hub [令牌](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

对于生产环境部署，推荐使用 `runtime` 变体，该镜像通过排除构建工具和开发依赖显著减小了体积（约减少 40%）：

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:latest-runtime \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

您也可以在[这里](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly)找到每日构建的 Docker 镜像。

注意：
- 在 B300/GB300（SM103）或 CUDA 13 环境中，我们推荐使用 `lmsysorg/sglang:dev-cu13` 每日构建镜像或 `lmsysorg/sglang:latest-cu130-runtime` 稳定镜像。请不要在 Docker 镜像内以可编辑模式重新安装项目，因为这会覆盖 cu13 Docker 镜像指定的库版本。

## 方法 4：使用 Kubernetes

请查看 [OME](https://github.com/sgl-project/ome)，这是一个用于大语言模型（LLM）企业级管理和服务的 Kubernetes 运算符。

<details>
<summary>更多</summary>

1. 选项 1：单节点服务（通常当模型大小能够放入单个节点的 GPU 时）

   执行命令 `kubectl apply -f docker/k8s-sglang-service.yaml`，以 llama-31-8b 为例创建 K8s Deployment 和 Service。

2. 选项 2：多节点服务（通常当大模型需要多个 GPU 节点时，如 `DeepSeek-R1`）

   根据需要修改 LLM 模型路径和参数，然后执行命令 `kubectl apply -f docker/k8s-sglang-distributed-sts.yaml`，创建两节点的 K8s StatefulSet 和服务 Service。

</details>

## 方法 5：使用 Docker Compose

<details>
<summary>更多</summary>

> 如果您计划将其作为服务部署，推荐此方法。
> 更好的方式是使用 [k8s-sglang-service.yaml](https://github.com/sgl-project/sglang/blob/main/docker/k8s-sglang-service.yaml)。

1. 将 [compose.yml](https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml) 复制到本地机器
2. 在终端中执行 `docker compose up -d` 命令。
</details>

## 方法 6：使用 SkyPilot 在 Kubernetes 或云上运行

<details>
<summary>更多</summary>

要在 Kubernetes 或 12+ 种云平台上部署，您可以使用 [SkyPilot](https://github.com/skypilot-org/skypilot)。

1. 安装 SkyPilot 并设置 Kubernetes 集群或云访问：参见 [SkyPilot 文档](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)。
2. 使用单条命令在您自己的基础设施上部署并获取 HTTP API 端点：
<details>
<summary>SkyPilot YAML：<code>sglang.yaml</code></summary>

```yaml
# sglang.yaml
envs:
  HF_TOKEN: null

resources:
  image_id: docker:lmsysorg/sglang:latest
  accelerators: A100
  ports: 30000

run: |
  conda deactivate
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

</details>

```bash
# 部署到任意云或 Kubernetes 集群。使用 --cloud <cloud> 选择特定云提供商。
HF_TOKEN=<secret> sky launch -c sglang --env HF_TOKEN sglang.yaml

# 获取 HTTP API 端点
sky status --endpoint 30000 sglang
```

3. 要进一步通过自动扩缩容和故障恢复来扩展部署，请查看 [SkyServe + SGLang 指南](https://github.com/skypilot-org/skypilot/tree/master/llm/sglang#serving-llama-2-with-sglang-for-more-traffic-using-skyserve)。

</details>

## 方法 7：在 AWS SageMaker 上运行

<details>
<summary>更多</summary>

要在 AWS SageMaker 上部署 SGLang，请查看 [AWS SageMaker Inference](https://aws.amazon.com/sagemaker/ai/deploy)

Amazon Web Services 为 SGLang 容器提供支持及常规安全补丁。可用的 SGLang 容器列表请查看 [AWS SGLang DLC](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sglang-containers)

要使用自定义容器托管模型，请按以下步骤操作：

1. 使用 [sagemaker.Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/sagemaker.Dockerfile) 和 [serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本构建 Docker 容器。
2. 将容器推送到 AWS ECR。

<details>
<summary>Dockerfile 构建脚本：<code>build-and-push.sh</code></summary>

```bash
#!/bin/bash
AWS_ACCOUNT="<YOUR_AWS_ACCOUNT>"
AWS_REGION="<YOUR_AWS_REGION>"
REPOSITORY_NAME="<YOUR_REPOSITORY_NAME>"
IMAGE_TAG="<YOUR_IMAGE_TAG>"

ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPOSITORY_NAME}:${IMAGE_TAG}"

echo "Starting build and push process..."

# 登录 ECR
echo "Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# 构建镜像
echo "Building Docker image..."
docker build -t ${IMAGE_URI} -f sagemaker.Dockerfile .

echo "Pushing ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Build and push completed successfully!"
```

</details>

3. 在 AWS SageMaker 上部署模型进行服务，参见 [deploy_and_serve_endpoint.py](https://github.com/sgl-project/sglang/blob/main/examples/sagemaker/deploy_and_serve_endpoint.py)。更多信息请查看 [sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk)。
   1. 默认情况下，SageMaker 上的模型服务器将使用以下命令运行：`python3 -m sglang.launch_server --model-path opt/ml/model --host 0.0.0.0 --port 8080`。这对于在 SageMaker 上托管自己的模型是最优的。
   2. 要修改模型服务参数，[serve](https://github.com/sgl-project/sglang/blob/main/docker/serve) 脚本允许通过指定带有前缀 `SM_SGLANG_` 的环境变量来使用 `python3 -m sglang.launch_server --help` CLI 中的所有可用选项。
   3. serve 脚本会自动将所有带有前缀 `SM_SGLANG_` 的环境变量从 `SM_SGLANG_INPUT_ARGUMENT` 转换为 `--input-argument`，传递给 `python3 -m sglang.launch_server` CLI。
   4. 例如，要运行 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 并使用推理解析器，只需添加额外的环境变量 `SM_SGLANG_MODEL_PATH=Qwen/Qwen3-0.6B` 和 `SM_SGLANG_REASONING_PARSER=qwen3`。

</details>

## 常见注意事项

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是默认的注意力计算后端。它仅支持 sm75 及以上架构。如果您在 sm75+ 设备（如 T4、A10、A100、L4、L40S、H100）上遇到任何 FlashInfer 相关问题，请通过添加 `--attention-backend triton --sampling-backend pytorch` 切换到其他内核，并在 GitHub 上提交 issue。
- 要在本地重新安装 flashinfer，使用以下命令：`pip3 install --upgrade flashinfer-python --force-reinstall --no-deps`，然后删除缓存 `rm -rf ~/.cache/flashinfer`。
- 在 B300/GB300 上遇到 `ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'` 时，请使用 `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` 修复。
