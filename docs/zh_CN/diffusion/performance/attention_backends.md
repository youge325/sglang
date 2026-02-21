# 注意力后端

本文档介绍了 SGLang Diffusion（`sglang.multimodal_gen`）中可用的注意力后端及其选择方式。

## 概述

注意力后端由 `AttentionBackendEnum`（`sglang.multimodal_gen.runtime.platforms.interface.AttentionBackendEnum`）定义，通过 CLI 标志 `--attention-backend` 进行选择。

后端选择由共享注意力层（如 `sglang.multimodal_gen.runtime.layers.attention.layer` 中的 `LocalAttention` / `USPAttention` / `UlyssesAttention`）执行，因此适用于使用这些层的任何模型组件（如 Diffusion Transformer / DiT 和编码器）。

使用 diffusers 后端时，`--attention-backend` 会传递给 diffusers 的 `set_attention_backend`（例如 `flash`、`_flash_3_hub`、`sage`、`xformers`、`native`）。

- **CUDA**：优先使用 FlashAttention（FA3/FA4）（如支持）；否则回退到 PyTorch SDPA。
- **ROCm**：使用 FlashAttention（如可用）；否则回退到 PyTorch SDPA。
- **MPS**：始终使用 PyTorch SDPA。
- **NPU**：始终使用 PyTorch SDPA。

## 后端选项

对于 SGLang 原生流水线，CLI 接受 `AttentionBackendEnum` 的小写名称。下表列出了内置平台实现的后端。`fa3`/`fa4` 作为 `fa` 的别名被接受。

| CLI 值 | 枚举值 | 说明 |
|---|---|---|
| `fa` / `fa3` / `fa4` | `FA` | FlashAttention。`fa3/fa4` 在参数解析时（`ServerArgs.__post_init__`）被规范化为 `fa`。 |
| `torch_sdpa` | `TORCH_SDPA` | PyTorch `scaled_dot_product_attention`。 |
| `sliding_tile_attn` | `SLIDING_TILE_ATTN` | Sliding Tile Attention (STA)。需要 `st_attn`。通过 `--attention-backend-config` 配置。 |
| `sage_attn` | `SAGE_ATTN` | 需要 `sageattention`。上游 SageAttention CUDA 扩展目标为 SM80/SM86/SM89/SM90/SM120（计算能力 8.0/8.6/8.9/9.0/12.0）；参见上游 `setup.py`：https://github.com/thu-ml/SageAttention/blob/main/setup.py。 |
| `sage_attn_3` | `SAGE_ATTN_3` | 需要按照上游说明安装 SageAttention3。 |
| `video_sparse_attn` | `VIDEO_SPARSE_ATTN` | 需要 `vsa`。通过 `--attention-backend-config` 配置 `sparsity`。 |
| `vmoba_attn` | `VMOBA_ATTN` | 需要 `kernel.attn.vmoba_attn.vmoba`。通过 `--attention-backend-config` 配置。 |
| `aiter` | `AITER` | 需要 `aiter`。 |
| `sparse_video_gen_2_attn` | `SPARSE_VIDEO_GEN_2_ATTN` | 需要 `svg`。安装说明参见 https://github.com/svg-project/Sparse-VideoGen。 |

## 选择优先级

`runtime/layers/attention/selector.py` 中的选择顺序为：

1. `global_force_attn_backend(...)` / `global_force_attn_backend_context_manager(...)`
2. CLI `--attention-backend`（`ServerArgs.attention_backend`）
3. 自动选择（平台能力、数据类型和已安装的包）

## 配置

某些后端需要额外配置。你可以通过 `--attention-backend-config` 传递这些参数。此参数接受：
- JSON 或 YAML 配置文件的路径。
- JSON 字符串（例如 `'{"sparsity": 0.5}'`）。
- 键值对（例如 `"sparsity=0.5,enable_x=true"`）。

### 支持的配置参数

**Sliding Tile Attention (`sliding_tile_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `mask_strategy_file_path` | `str` | **必需。** mask 策略 JSON 文件的路径。 | - |
| `sta_mode` | `str` | STA 模式。 | `STA_inference` |
| `skip_time_steps` | `int` | 切换到稀疏注意力之前使用完整注意力的步数。 | `15` |

**Video Sparse Attention (`video_sparse_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `sparsity` | `float` | 验证稀疏度（0.0 - 1.0）。 | `0.0` |

**V-MoBA (`vmoba_attn`)**

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `temporal_chunk_size` | `int` | 时间维度的块大小。 | - |
| `temporal_topk` | `int` | 时间维度中选择的 Top-K token 数。 | - |
| `spatial_chunk_size` | `list[int]` | 空间维度的块大小 (H, W)。 | - |
| `spatial_topk` | `int` | 空间维度中选择的 Top-K token 数。 | - |
| `st_chunk_size` | `list[int]` | 时空维度的块大小 (T, H, W)。 | - |
| `st_topk` | `int` | 时空维度中选择的 Top-K token 数。 | - |
| `moba_select_mode` | `str` | 选择模式（例如 `threshold`）。 | `threshold` |
| `moba_threshold` | `float` | 选择的阈值。 | `0.25` |
| `moba_threshold_type` | `str` | 阈值类型（例如 `query_head`）。 | `query_head` |
| `first_full_step` | `int` | 使用完整注意力的初始步数。 | `12` |
| `first_full_layer` | `int` | 使用完整注意力的初始层数。 | `0` |
| `temporal_layer` | `int` | 时间层的数量。 | `1` |
| `spatial_layer` | `int` | 空间层的数量。 | `1` |
| `st_layer` | `int` | 时空层的数量。 | `1` |

## 平台支持矩阵

| 后端 | CUDA | ROCm | MPS | NPU | 说明 |
|---|---:|---:|---:|---:|---|
| `fa` | ✅ | ✅ | ❌ | ❌ | CUDA 需要 SM80+ 和 fp16/bf16。FlashAttention 仅在安装了所需运行时时使用；否则回退到 `torch_sdpa`。 |
| `torch_sdpa` | ✅ | ✅ | ✅ | ✅ | 跨平台兼容性最好的选项。 |
| `sliding_tile_attn` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA。需要 `st_attn`。通过 `--attention-backend-config` 配置。 |
| `sage_attn` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA（可选依赖）。 |
| `sage_attn_3` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA（可选依赖）。 |
| `video_sparse_attn` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA。需要 `vsa`。通过 `--attention-backend-config` 配置 `sparsity`。 |
| `vmoba_attn` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA。需要 `kernel.attn.vmoba_attn.vmoba`。通过 `--attention-backend-config` 配置。 |
| `aiter` | ✅ | ❌ | ❌ | ❌ | 需要 `aiter`。 |
| `sparse_video_gen_2_attn` | ✅ | ❌ | ❌ | ❌ | 仅支持 CUDA。需要 `svg`。 |

## 用法

### 通过 CLI 选择后端

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa
```

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend torch_sdpa
```

### 使用 Sliding Tile Attention (STA)

```bash
# 通过配置传递 mask 策略文件路径
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend sliding_tile_attn \
  --attention-backend-config "mask_strategy_file_path=/abs/path/to/mask_strategy.json"
```

### ROCm / MPS 注意事项

- ROCm：根据环境的可用情况使用 `--attention-backend torch_sdpa` 或 `fa`。
- MPS：平台实现始终使用 `torch_sdpa`。
