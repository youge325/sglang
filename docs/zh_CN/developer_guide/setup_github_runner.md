# 为 GitHub Actions 设置自托管 Runner

## 添加 Runner

### 步骤 1：启动一个 Docker 容器。

**你可以挂载一个文件夹用于共享 HuggingFace 模型权重缓存。**
以下命令以 `/tmp/huggingface` 为例。

```
docker pull nvidia/cuda:12.9.1-devel-ubuntu22.04
# Nvidia
docker run --shm-size 128g -it -v /tmp/huggingface:/hf_home --gpus all nvidia/cuda:12.9.1-devel-ubuntu22.04 /bin/bash
# AMD
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.5.8-rocm700-mi30x /bin/bash
# AMD 仅使用最后 2 个 GPU
docker run --rm --device=/dev/kfd --device=/dev/dri/renderD176 --device=/dev/dri/renderD184 --group-add video --shm-size 128g -it -v /tmp/huggingface:/hf_home lmsysorg/sglang:v0.5.8-rocm700-mi30x /bin/bash
```

### 步骤 2：通过 `config.sh` 配置 Runner

在容器内运行以下命令。

```
apt update && apt install -y curl python3-pip git
pip install --upgrade pip
export RUNNER_ALLOW_RUNASROOT=1
```

然后按照 https://github.com/sgl-project/sglang/settings/actions/runners/new?arch=x64&os=linux 的说明运行 `config.sh`

**注意事项**
- 不需要指定 Runner 组
- 给它一个名称（例如 `test-sgl-gpu-0`）和一些标签（例如 `1-gpu-runner`）。标签可以稍后在 GitHub Settings 中编辑。
- 不需要更改工作文件夹。

### 步骤 3：通过 `run.sh` 运行 Runner

- 设置环境变量
```
export HF_HOME=/hf_home
export SGLANG_IS_IN_CI=true
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
export CUDA_VISIBLE_DEVICES=0
```

- 持续运行
```
while true; do ./run.sh; echo "Restarting..."; sleep 2; done
```
