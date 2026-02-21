# AMD GPU

本文档介绍如何在 AMD GPU 上运行 SGLang。如果遇到问题或有疑问，请[提交 issue](https://github.com/sgl-project/sglang/issues)。

## 系统配置

使用 AMD GPU（如 MI300X）时，某些系统级优化有助于确保稳定性能。以 MI300X 为例，AMD 提供了官方优化文档：

- [AMD MI300X 调优指南](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [AMD Instinct MI300X 上的 LLM 推理性能验证](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
- [AMD Instinct MI300X 系统优化](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)

**注意:** 我们强烈建议完整阅读这些文档以充分利用您的系统。

### 更新 GRUB 设置

在 `/etc/default/grub` 中，将以下内容追加到 `GRUB_CMDLINE_LINUX`：

```text
pci=realloc=off iommu=pt
```

之后运行 `sudo update-grub` 并重启。

### 禁用 NUMA 自动平衡

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

## 安装 SGLang

### 从源码安装

```bash
git clone -b v0.5.6.post2 https://github.com/sgl-project/sglang.git
cd sglang

# 编译 sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install

# 安装 sglang python 包
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

### 使用 Docker 安装（推荐）

Docker 镜像可在 Docker Hub 上获取：[lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags)，基于 [rocm.Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker) 构建。

1. 构建 Docker 镜像。

   ```bash
   docker build -t sglang_image -f rocm.Dockerfile .
   ```

2. 创建便捷别名。

   ```bash
   alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
       --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       -v $HOME/dockerx:/dockerx \
       -v /data:/data'
   ```

3. 启动服务器。

   ```bash
   drun -p 30000:30000 \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       --env "HF_TOKEN=<secret>" \
       sglang_image \
       python3 -m sglang.launch_server \
       --model-path NousResearch/Meta-Llama-3.1-8B \
       --host 0.0.0.0 \
       --port 30000
   ```

```{note}
更多 AMD GPU 特定的故障排除和高级配置请参阅 [英文文档](../../en/platforms/amd_gpu.html)。
```
