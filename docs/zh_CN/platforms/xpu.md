# XPU

本文档介绍如何设置 [SGLang](https://github.com/sgl-project/sglang) 环境并在 Intel GPU 上运行 LLM 推理，[参阅 PyTorch 生态系统中 Intel GPU 支持的更多信息](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)。

具体而言，SGLang 针对 [Intel® Arc™ Pro B 系列显卡](https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html) 和 [Intel® Arc™ B 系列显卡](https://www.intel.com/content/www/us/en/ark/products/series/240391/intel-arc-b-series-graphics.html) 进行了优化。

## 已优化模型列表

以下 LLM 已在 Intel GPU 上进行了优化，更多模型正在适配中：

| 模型名称 | BF16 |
|:---:|:---:|
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5-1.5B |   [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

**注意：** 上表中列出的模型标识符已在 [Intel® Arc™ B580 显卡](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html) 上完成验证。

## 安装

### 从源码安装

目前 SGLang XPU 仅支持从源码安装。请参考 ["Intel GPU 入门指南"](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) 安装 XPU 依赖。

```bash
# 创建并激活 conda 环境
conda create -n sgl-xpu python=3.12 -y
conda activate sgl-xpu

# 将 PyTorch XPU 设置为主 pip 安装渠道，避免安装更大的 CUDA 版本并防止潜在的运行时问题。
pip3 install torch==2.9.0+xpu torchao torchvision torchaudio pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu
pip3 install xgrammar --no-deps # xgrammar 会引入 CUDA 版本的 triton，可能与 XPU 冲突

# 克隆 SGLang 代码
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# 使用专用 toml 文件
cd python
cp pyproject_xpu.toml pyproject.toml
# 安装 SGLang 依赖库并构建 SGLang 主包
pip install --upgrade pip setuptools
pip install -v .
```

### 使用 Docker 安装

XPU 的 Docker 支持正在积极开发中，敬请期待。

## 启动服务引擎

启动 SGLang 服务的示例命令：

```bash
python -m sglang.launch_server       \
    --model <MODEL_ID_OR_PATH>       \
    --trust-remote-code              \
    --disable-overlap-schedule       \
    --device xpu                     \
    --host 0.0.0.0                   \
    --tp 2                           \   # 使用多 GPU
    --attention-backend intel_xpu    \   # 使用 Intel 优化的 XPU 注意力后端
    --page-size                      \   # intel_xpu 注意力后端支持 [32, 64, 128]
```

## 请求基准测试

您可以通过 `bench_serving` 脚本进行性能基准测试。
在另一个终端中运行以下命令。

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

参数的详细说明可以通过以下命令查看：

```bash
python -m sglang.bench_serving -h
```

此外，请求也可以使用
[OpenAI Completions API](https://docs.sglang.io/basic_usage/openai_api_completions.html)
格式构建，并通过命令行（如使用 `curl`）或您自己的脚本发送。
