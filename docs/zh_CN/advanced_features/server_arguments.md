# 服务器参数

本页列出了在部署中用于配置语言模型服务器行为和性能的命令行服务器参数。
这些参数使用户能够自定义服务器的关键方面，包括模型选择、并行策略、内存管理和优化技术。
您可以通过 `python3 -m sglang.launch_server --help` 查看所有参数。

## 常用启动命令

- 要使用配置文件，创建一个包含服务器参数的 YAML 文件，并使用 `--config` 指定。CLI 参数会覆盖配置文件中的值。

  ```bash
  # 创建 config.yaml
  cat > config.yaml << EOF
  model-path: meta-llama/Meta-Llama-3-8B-Instruct
  host: 0.0.0.0
  port: 30000
  tensor-parallel-size: 2
  enable-metrics: true
  log-requests: true
  EOF

  # 使用配置文件启动服务器
  python -m sglang.launch_server --config config.yaml
  ```

- 要启用多 GPU 张量并行，添加 `--tp 2`。如果报错 "peer access is not supported between these two devices"，请在启动命令中添加 `--enable-p2p-check`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```

- 要启用多 GPU 数据并行，添加 `--dp 2`。如果内存充足，数据并行在吞吐量方面更优。也可与张量并行一起使用。以下命令总共使用 4 个 GPU。我们推荐使用 [SGLang Model Gateway（前身为 Router）](../advanced_features/sgl_model_gateway.md) 进行数据并行。

  ```bash
  python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
  ```

- 如果在服务过程中遇到内存不足错误，请通过设置较小的 `--mem-fraction-static` 值来减少 KV 缓存池的内存使用。默认值为 `0.9`。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```

- 关于性能调优，请参见[超参数调优](hyperparameter_tuning.md)。
- 对于 Docker 和 Kubernetes 运行，需要配置共享内存用于进程间通信。参见 Docker 的 `--shm-size` 以及 Kubernetes 配置中的 `/dev/shm` 大小更新。
- 如果在长提示词预填充时遇到内存不足错误，请尝试设置较小的分块预填充大小。

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
  ```
- 要启用 fp8 权重量化，在 fp16 检查点上添加 `--quantization fp8`，或直接加载 fp8 检查点无需指定任何参数。
- 要启用 fp8 KV 缓存量化，添加 `--kv-cache-dtype fp8_e4m3` 或 `--kv-cache-dtype fp8_e5m2`。
- 要启用确定性推理和批处理无关操作，添加 `--enable-deterministic-inference`。更多细节请参见[确定性推理文档](../advanced_features/deterministic_inference.md)。
- 如果模型在 Hugging Face tokenizer 中没有聊天模板，可以指定[自定义聊天模板](../references/custom_chat_template.md)。如果 tokenizer 有多个命名模板（如 'default'、'tool_use'），可以使用 `--hf-chat-template-name tool_use` 选择。
- 要在多节点上运行张量并行，添加 `--nnodes 2`。如果有两个节点，每个节点有两个 GPU，想运行 TP=4，令 `sgl-dev-0` 为第一个节点的主机名，`50000` 为可用端口，请使用以下命令。如果遇到死锁，请尝试添加 `--disable-cuda-graph`。
- （注意：此功能已停止维护，可能会导致错误）要启用 `torch.compile` 加速，添加 `--enable-torch-compile`。它可以加速小模型在小 batch size 下的推理。默认缓存路径位于 `/tmp/torchinductor_root`，可通过环境变量 `TORCHINDUCTOR_CACHE_DIR` 自定义。详情请参阅 [PyTorch 官方文档](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) 和[启用 torch.compile 缓存](https://docs.sglang.io/references/torch_compile_cache.html)。

  ```bash
  # 节点 0
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 0

  # 节点 1
  python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 4 \
    --dist-init-addr sgl-dev-0:50000 \
    --nnodes 2 \
    --node-rank 1
  ```

请查阅下方文档和 [server_args.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) 了解启动服务器时可提供的更多参数。

## 模型和分词器
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--model-path`<br>`--model` | 模型权重路径。可以是本地文件夹或 Hugging Face 仓库 ID。 | `None` | Type: str |
| `--tokenizer-path` | 分词器路径。 | `None` | Type: str |
| `--tokenizer-mode` | 分词器模式。'auto' 将优先使用快速分词器（如可用），'slow' 将始终使用慢速分词器。 | `auto` | `auto`, `slow` |
| `--tokenizer-worker-num` | 分词器管理器的工作线程数。 | `1` | Type: int |
| `--skip-tokenizer-init` | 如果设置，跳过分词器初始化，在生成请求中传递 input_ids。 | `False` | bool flag |
| `--load-format` | 加载模型权重的格式。"auto" 将尝试以 safetensors 格式加载，不可用时回退到 pytorch bin 格式。 | `auto` | `auto`, `pt`, `safetensors`, `npcache`, `dummy`, `sharded_state`, `gguf`, `bitsandbytes`, `layered`, `flash_rl`, `remote`, `remote_instance`, `fastsafetensors`, `private` |
| `--trust-remote-code` | 是否允许 Hub 上定义的自定义模型。 | `False` | bool flag |
| `--context-length` | 模型的最大上下文长度。默认为 None（使用 config.json 中的值）。 | `None` | Type: int |
| `--is-embedding` | 是否将 CausalLM 作为嵌入模型使用。 | `False` | bool flag |
| `--enable-multimodal` | 启用模型的多模态功能。 | `None` | bool flag |
| `--revision` | 使用的模型版本（分支名、标签或 commit ID）。 | `None` | Type: str |
| `--model-impl` | 使用的模型实现。'auto' 优先使用 SGLang 实现。 | `auto` | Type: str |

## HTTP 服务器
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--host` | HTTP 服务器主机地址。 | `127.0.0.1` | Type: str |
| `--port` | HTTP 服务器端口。 | `30000` | Type: int |

## 量化和数据类型
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--dtype` | 模型权重和激活的数据类型。'auto' 对 FP32 和 FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。 | `auto` | `auto`, `half`, `float16`, `bfloat16`, `float`, `float32` |
| `--quantization` | 量化方法。 | `None` | `awq`, `fp8`, `gptq`, `marlin` 等 |
| `--kv-cache-dtype` | KV 缓存存储的数据类型。'auto' 使用模型数据类型。 | `auto` | `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `fp4_e2m1` |

## 内存和调度
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--mem-fraction-static` | 用于静态分配的内存比例（模型权重和 KV 缓存内存池）。如果遇到内存不足，请使用较小的值。 | `None` | Type: float |
| `--max-running-requests` | 最大运行请求数。 | `None` | Type: int |
| `--max-queued-requests` | 最大排队请求数。 | `None` | Type: int |
| `--chunked-prefill-size` | 分块预填充中每个块的最大 token 数。设置为 -1 禁用分块预填充。 | `None` | Type: int |
| `--schedule-policy` | 请求的调度策略。 | `fcfs` | `lpm`, `random`, `fcfs`, `dfs-weight`, `lof`, `priority`, `routing-key` |

## 运行时选项
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--device` | 使用的设备（'cuda'、'xpu'、'hpu'、'npu'、'cpu'）。 | `None` | Type: str |
| `--tensor-parallel-size`<br>`--tp-size` | 张量并行大小。 | `1` | Type: int |
| `--pipeline-parallel-size`<br>`--pp-size` | 流水线并行大小。 | `1` | Type: int |
| `--stream-interval` | 流式输出的间隔（以 token 长度计）。较小值使流式更流畅，较大值使吞吐量更高。 | `1` | Type: int |
| `--random-seed` | 随机种子。 | `None` | Type: int |

## 日志
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--log-level` | 所有日志器的日志级别。 | `info` | Type: str |
| `--log-requests` | 记录所有请求的元数据、输入、输出。 | `False` | bool flag |
| `--enable-metrics` | 启用 Prometheus 指标记录。 | `False` | bool flag |

## API 相关
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--api-key` | 设置服务器的 API 密钥。也用于 OpenAI API 兼容服务器。 | `None` | Type: str |
| `--served-model-name` | 覆盖 OpenAI API 服务器 v1/models 端点返回的模型名称。 | `None` | Type: str |
| `--chat-template` | 内置聊天模板名称或聊天模板文件路径。仅用于 OpenAI 兼容 API 服务器。 | `None` | Type: str |
| `--reasoning-parser` | 指定推理模型的解析器。支持的解析器：[deepseek-r1, deepseek-v3, glm45, gpt-oss, kimi, qwen3, qwen3-thinking, step3]。 | `None` | 见英文文档 |
| `--sampling-defaults` | 默认采样参数的来源。'model' 使用 generation_config.json，'openai' 使用 SGLang/OpenAI 默认值。 | `model` | `openai`, `model` |

## 数据并行
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--data-parallel-size`<br>`--dp-size` | 数据并行大小。 | `1` | Type: int |
| `--load-balance-method` | 数据并行的负载均衡策略。 | `auto` | `auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens` |

## 多节点分布式服务
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--dist-init-addr`<br>`--nccl-init-addr` | 初始化分布式后端的主机地址（如 `192.168.0.2:25000`）。 | `None` | Type: str |
| `--nnodes` | 节点数。 | `1` | Type: int |
| `--node-rank` | 节点排名。 | `0` | Type: int |

## LoRA
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--enable-lora` | 为模型启用 LoRA 支持。 | `False` | bool flag |
| `--lora-paths` | 要加载的 LoRA 适配器列表。 | `None` | Type: List[str] |
| `--max-loras-per-batch` | 运行批次中的最大适配器数量。 | `8` | Type: int |

## 推测解码
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--speculative-algorithm` | 推测算法。 | `None` | `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM` |
| `--speculative-draft-model-path` | 草稿模型权重路径。 | `None` | Type: str |
| `--speculative-num-steps` | 推测解码中从草稿模型采样的步数。 | `None` | Type: int |
| `--speculative-num-draft-tokens` | 推测解码中从草稿模型采样的 token 数。 | `None` | Type: int |

## 优化/调试选项
| 参数 | 描述 | 默认值 | 选项 |
| --- | --- | --- | --- |
| `--disable-radix-cache` | 禁用 RadixAttention 前缀缓存。 | `False` | bool flag |
| `--cuda-graph-max-bs` | 设置 CUDA Graph 的最大 batch size。 | `None` | Type: int |
| `--disable-cuda-graph` | 禁用 CUDA Graph。 | `False` | bool flag |
| `--enable-dp-attention` | 启用注意力的数据并行和 FFN 的张量并行。dp size 应等于 tp size。 | `False` | bool flag |
| `--enable-torch-compile` | 使用 torch.compile 优化模型。实验性功能。 | `False` | bool flag |

```{note}
这是服务器参数的精简中文版本。完整的参数列表请参阅 [英文文档](../../en/advanced_features/server_arguments.html)。
```
