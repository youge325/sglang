# 安装 SGLang-Diffusion

你可以使用以下方法之一安装 SGLang-Diffusion。

## 标准安装（NVIDIA GPU）

### 方法一：使用 pip 或 uv

推荐使用 uv 以获得更快的安装速度：

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

### 方法二：从源码安装

```bash
# 使用最新的发布分支
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 安装 Python 包
pip install --upgrade pip
pip install -e "python[diffusion]"

# 使用 uv
uv pip install -e "python[diffusion]" --prerelease=allow
```

### 方法三：使用 Docker

Docker 镜像可在 Docker Hub 的 [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang) 获取，基于 [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile) 构建。
请将下方的 `<secret>` 替换为你的 HuggingFace Hub [token](https://huggingface.co/docs/hub/en/security-tokens)。

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    zsh -c '\
        echo "Installing diffusion dependencies..." && \
        pip install -e "python[diffusion]" && \
        echo "Starting SGLang-Diffusion..." && \
        sglang generate \
            --model-path black-forest-labs/FLUX.1-dev \
            --prompt "A logo With Bold Large text: SGL Diffusion" \
            --save-output \
    '
```

## 平台特定：ROCm（AMD GPU）

对于 AMD Instinct GPU（如 MI300X），可以使用启用了 ROCm 的 Docker 镜像：

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  lmsysorg/sglang:v0.5.5.post2-rocm700-mi30x \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```

有关 ROCm 系统配置和从源码安装的详细信息，请参阅 [AMD GPU](../../platforms/amd_gpu.md)。

## 平台特定：MUSA（摩尔线程 GPU）

对于使用 MUSA 软件栈的摩尔线程 GPU (MTGPU)：

```bash
# 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 安装 Python 包
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## 平台特定：Ascend NPU

对于 Ascend NPU，请参阅 [NPU 安装指南](../platforms/ascend_npu.md)。

快速测试：

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```
