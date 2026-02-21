# 使用 ModelScope 中的模型

要使用 [ModelScope](https://www.modelscope.cn) 中的模型，请设置环境变量 `SGLANG_USE_MODELSCOPE`。

```bash
export SGLANG_USE_MODELSCOPE=true
```

我们以 [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) 为例。

启动服务器：
```bash
python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

或通过 Docker 启动：

```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

请注意，ModelScope 使用与 HuggingFace 不同的缓存目录。您可能需要手动设置以避免磁盘空间不足。
