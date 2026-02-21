# 使用 Docker 的开发指南

## 在远程主机上设置 VSCode
（可选 - 如果你计划在本地运行 SGLang 开发容器，可以跳过此步骤）

1. 在远程主机上，从 [Https://code.visualstudio.com/docs/?dv=linux64cli](https://code.visualstudio.com/download) 下载 `code`，然后在 shell 中运行 `code tunnel`。

示例
```bash
wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/fabdb6a30b49f79a7aba0f2ad9df9b399473380f/vscode_cli_alpine_x64_cli.tar.gz
tar xf vscode_cli_alpine_x64_cli.tar.gz

# https://code.visualstudio.com/docs/remote/tunnels
./code tunnel
```

2. 在本地机器上，在 VSCode 中按 F1 并选择 "Remote Tunnels: Connect to Tunnel"。

## 设置 Docker 容器

### 选项 1：使用 VSCode 自动启动默认开发容器
SGLang 仓库根目录中有一个 `.devcontainer` 文件夹，允许 VSCode 自动在开发容器中启动。你可以在 VSCode 官方文档 [在容器中开发](https://code.visualstudio.com/docs/devcontainers/containers) 中了解更多关于此 VSCode 扩展的信息。
![image](https://github.com/user-attachments/assets/6a245da8-2d4d-4ea8-8db1-5a05b3a66f6d)
（*图 1：来自 VSCode 官方文档 [在容器中开发](https://code.visualstudio.com/docs/devcontainers/containers) 的示意图。*）

要启用此功能，你只需要：
1. 启动 Visual Studio Code 并安装 [VSCode 开发容器扩展](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)。
2. 按 F1，输入并选择 "Dev Container: Open Folder in Container"。
3. 输入你机器上的 `sglang` 本地仓库路径并按回车。

第一次在开发容器中打开可能需要更长时间，因为需要拉取和构建 Docker 镜像。成功后，你应该在左下角的状态栏看到你正在开发容器中：

![image](https://github.com/user-attachments/assets/650bba0b-c023-455f-91f9-ab357340106b)

现在当你在 VSCode 终端中运行 `sglang.launch_server` 或使用 F5 开始调试时，SGLang 服务器将在开发容器中启动，所有本地更改会自动应用：

![image](https://github.com/user-attachments/assets/748c85ba-7f8c-465e-8599-2bf7a8dde895)


### 选项 2：手动启动容器（高级）

以下启动命令是 SGLang 团队内部开发的示例。你可以**根据需要修改或添加目录映射**，特别是对于模型权重的下载，以防止不同 Docker 容器重复下载。

❗️ **关于 RDMA 的说明**

    1. `--network host` 和 `--privileged` 是 RDMA 所需的。如果你不需要 RDMA，可以移除它们，但保留它们不会造成影响。因此，我们在下面的命令中默认启用这两个标志。
    2. 如果你使用 RoCE，可能需要设置 `NCCL_IB_GID_INDEX`，例如：`export NCCL_IB_GID_INDEX=3`。

```bash
# 将名称替换为你自己的
docker run -itd --shm-size 32g --gpus all -v <volumes-to-mount> --ipc=host --network=host --privileged --name sglang_dev lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_dev /bin/zsh
```
一些有用的挂载卷：
1. **Huggingface 模型缓存**：挂载模型缓存可以避免每次 Docker 重启时重新下载。Linux 上的默认位置是 `~/.cache/huggingface/`。
2. **SGLang 仓库**：SGLang 本地仓库中的代码更改将自动同步到开发容器中。

示例 1：挂载本地缓存文件夹 `/opt/dlami/nvme/.cache` 但不挂载 SGLang 仓库。当你更倾向于手动将本地代码更改传输到开发容器时使用此方式。
```bash
docker run -itd --shm-size 32g --gpus all -v /opt/dlami/nvme/.cache:/root/.cache --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```
示例 2：同时挂载 HuggingFace 缓存和本地 SGLang 仓库。由于 SGLang 在开发镜像中以可编辑模式安装，本地代码更改会自动同步到开发容器。
```bash
docker run -itd --shm-size 32g --gpus all -v $HOME/.cache/huggingface/:/root/.cache/huggingface -v $HOME/src/sglang:/sgl-workspace/sglang --ipc=host --network=host --privileged --name sglang_zhyncs lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_zhyncs /bin/zsh
```
## 使用 VSCode 调试器调试 SGLang
1. （如果不存在则创建）在 VSCode 中打开 `launch.json`。
2. 添加以下配置并保存。请注意，你可以根据需要编辑脚本以应用不同的参数或调试不同的程序（例如基准测试脚本）。
     ```JSON
       {
          "version": "0.2.0",
          "configurations": [
              {
                  "name": "Python Debugger: launch_server",
                  "type": "debugpy",
                  "request": "launch",
                  "module": "sglang.launch_server",
                  "console": "integratedTerminal",
                  "args": [
                      "--model-path", "meta-llama/Llama-3.2-1B",
                      "--host", "0.0.0.0",
                      "--port", "30000",
                      "--trust-remote-code",
                  ],
                  "justMyCode": false
              }
          ]
      }
    ```

3. 按 "F5" 开始。VSCode 调试器会确保程序在断点处暂停，即使程序运行在远程 SSH/Tunnel 主机 + 开发容器中。

## 性能分析

```bash
# 更改批次大小、输入、输出并添加 `disable-cuda-graph`（便于分析）
# 例如 DeepSeek V3
nsys profile -o deepseek_v3 python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 --disable-cuda-graph
```

## 评估

```bash
# 例如 gsm8k 8-shot
python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8
```
