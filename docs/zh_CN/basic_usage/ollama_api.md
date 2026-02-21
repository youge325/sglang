# Ollama 兼容 API

SGLang 提供 Ollama API 兼容性，允许您使用 Ollama CLI 和 Python 库以 SGLang 作为推理后端。

## 前提条件

```bash
# 安装 Ollama Python 库（用于 Python 客户端）
pip install ollama
```

> **注意**: 您无需安装 Ollama 服务器 - SGLang 充当后端。您只需要 `ollama` CLI 或 Python 库作为客户端。

## 接口端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET, HEAD | Ollama CLI 健康检查 |
| `/api/tags` | GET | 列出可用模型 |
| `/api/chat` | POST | 对话补全（流式和非流式） |
| `/api/generate` | POST | 文本生成（流式和非流式） |
| `/api/show` | POST | 模型信息 |

## 快速开始

### 1. 启动 SGLang 服务器

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

> **注意**: 使用 `ollama run` 时的模型名称必须与传递给 `--model` 的名称完全一致。

### 2. 使用 Ollama CLI

```bash
# 列出可用模型
OLLAMA_HOST=http://localhost:30001 ollama list

# 交互式对话
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

如果连接到防火墙后的远程服务器：

```bash
# SSH 隧道
ssh -L 30001:localhost:30001 user@gpu-server -N &

# 然后按上述方式使用 Ollama CLI
OLLAMA_HOST=http://localhost:30001 ollama list
```

### 3. 使用 Ollama Python 库

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# 非流式
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# 流式
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': '给我讲个故事'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## 智能路由

关于在本地 Ollama（快速）和远程 SGLang（强大）之间使用 LLM 裁判进行智能路由，请参阅[智能路由文档](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md)。

## 总结

| 组件 | 用途 |
|------|------|
| **Ollama API** | 开发者熟悉的 CLI/API 接口 |
| **SGLang 后端** | 高性能推理引擎 |
| **智能路由** | 智能路由 - 简单任务使用快速的本地推理，复杂任务使用强大的远程推理 |
