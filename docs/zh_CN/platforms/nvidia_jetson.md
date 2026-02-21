# NVIDIA Jetson Orin

## 前置条件

在开始之前，请确保以下条件：

- [**NVIDIA Jetson AGX Orin 开发套件**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 已安装 **JetPack 6.1** 或更高版本。
- **CUDA Toolkit** 和 **cuDNN** 已安装。
- 确认 Jetson AGX Orin 处于**高性能模式**：
```bash
sudo nvpmodel -m 0
```
* * * * *
## 使用 Jetson Containers 安装和运行 SGLang
克隆 jetson-containers GitHub 仓库：
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
运行安装脚本：
```
bash jetson-containers/install.sh
```
构建容器镜像：
```
jetson-containers build sglang
```
运行容器：
```
jetson-containers run $(autotag sglang)
```
或者您也可以使用以下命令手动运行容器：
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
* * * * *

运行推理
-----------------------------------------

启动服务器：
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```
量化和有限的上下文长度（`--dtype half --context-length 8192`）是由于 [NVIDIA Jetson 套件](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) 的计算资源有限。详细说明请参考[服务器参数](../advanced_features/server_arguments.md)。

启动引擎后，请参考 [Chat completions](https://docs.sglang.io/basic_usage/openai_api_completions.html#Usage) 测试可用性。
* * * * *
使用 TorchAO 运行量化
-------------------------------------
TorchAO 推荐用于 NVIDIA Jetson Orin。
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
这启用了 TorchAO 的 int4 仅权重量化，组大小为 128。使用 `--torchao-config int4wo-128` 同样是为了提高内存效率。


* * * * *
使用 XGrammar 进行结构化输出
-------------------------------
请参考 [SGLang 结构化输出文档](../advanced_features/structured_outputs.ipynb)。
* * * * *

感谢 [Nurgaliyev Shakhizat](https://github.com/shahizat)、[Dustin Franklin](https://github.com/dusty-nv) 和 [Johnny Núñez Cano](https://github.com/johnnynunez) 的支持。

参考资料
----------
-   [NVIDIA Jetson AGX Orin 文档](https://developer.nvidia.com/embedded/jetson-agx-orin)
