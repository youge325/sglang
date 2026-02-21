# 大语言模型

这些模型接受文本输入并生成文本输出（例如聊天补全）。它们主要是大语言模型（LLM），部分采用混合专家（MoE）架构以实现扩展。

## 启动命令示例

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \  # HF/本地路径示例
  --host 0.0.0.0 \
  --port 30000 \
```

## 支持的模型

下面以表格形式汇总了支持的模型。

如果您不确定某个特定架构是否已实现，可以通过 GitHub 搜索。例如，要搜索 `Qwen3ForCausalLM`，请在 GitHub 搜索栏中使用以下表达式：

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// Qwen3ForCausalLM
```

| 模型系列（变体）             | HuggingFace 标识符示例                     | 描述                                                                            |
|-------------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **DeepSeek** (v1, v2, v3/R1)        | `deepseek-ai/DeepSeek-R1`                        | 一系列经过强化学习训练的高级推理优化模型（包括 671B MoE），在复杂推理、数学和代码任务上表现顶尖。[SGLang 为 Deepseek v3/R1 提供模型专属优化](../basic_usage/deepseek.md)和[推理解析器](../advanced_features/separate_reasoning.ipynb)|
| **Kimi K2** (Thinking, Instruct)    | `moonshotai/Kimi-K2-Instruct`                    | 月之暗面的万亿参数 MoE 模型（32B 活跃参数），支持 128K–256K 上下文；具备最先进的智能体能力，可在 200-300 次连续工具调用中保持稳定。采用 MLA 注意力机制和原生 INT4 量化。[参见推理解析器文档](../advanced_features/separate_reasoning.ipynb)|
| **Kimi Linear** (48B-A3B)           | `moonshotai/Kimi-Linear-48B-A3B-Instruct`        | 月之暗面的混合线性注意力模型（48B 总参数，3B 活跃参数），支持 100 万 token 上下文；采用 Kimi Delta Attention (KDA)，与全注意力相比可实现最高 6 倍的解码加速和 75% 的 KV 缓存减少。 |
| **GPT-OSS**       | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`       | OpenAI 最新的 GPT-OSS 系列，适用于复杂推理、智能体任务和多功能开发者用例。|
| **Qwen** (3, 3MoE, 3Next, 2.5, 2 系列)       | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-30B-A3B` `Qwen/Qwen3-Next-80B-A3B-Instruct `      | 阿里巴巴最新的 Qwen3 系列，适用于复杂推理、语言理解和生成任务；支持 MoE 变体以及上一代 2.5、2 等版本。[SGLang 提供 Qwen3 专属推理解析器](../advanced_features/separate_reasoning.ipynb)|
| **Llama** (2, 3.x, 4 系列)        | `meta-llama/Llama-4-Scout-17B-16E-Instruct`       | Meta 的开源 LLM 系列，参数范围从 7B 到 400B（Llama 2、3 和新的 Llama 4），具有广泛认可的性能。[SGLang 提供 Llama-4 模型专属优化](../basic_usage/llama4.md)  |
| **Mistral** (Mixtral, NeMo, Small3) | `mistralai/Mistral-7B-Instruct-v0.2`             | Mistral AI 的开源 7B LLM，性能出色；扩展为 MoE（"Mixtral"）和 NeMo Megatron 变体以支持更大规模。 |
| **Gemma** (v1, v2, v3)              | `google/gemma-3-1b-it`                            | Google 的高效多语言模型系列（1B–27B）；Gemma 3 提供 128K 上下文窗口，较大变体（4B+）支持视觉输入。 |
| **Phi** (Phi-1.5, Phi-2, Phi-3, Phi-4, Phi-MoE 系列) | `microsoft/Phi-4-multimodal-instruct`, `microsoft/Phi-3.5-MoE-instruct` | 微软的 Phi 小型模型系列（1.3B–5.6B）；Phi-4-multimodal（5.6B）处理文本、图像和语音，Phi-4-mini 是高精度文本模型，Phi-3.5-MoE 是混合专家模型。 |
| **MiniCPM** (v3, 4B)               | `openbmb/MiniCPM3-4B`                            | OpenBMB 的边缘设备紧凑型 LLM 系列；MiniCPM 3（4B）在文本任务上达到 GPT-3.5 级别的效果。 |
| **OLMo** (2, 3) | `allenai/OLMo-3-1125-32B`, `allenai/OLMo-3-32B-Think`, `allenai/OLMo-2-1124-7B-Instruct` | Allen AI 的开放语言模型系列，旨在推动语言模型科学研究。 |
| **OLMoE** (Open MoE)               | `allenai/OLMoE-1B-7B-0924`                       | Allen AI 的开源混合专家模型（7B 总参数，1B 活跃参数），通过稀疏专家激活实现最先进的结果。 |
| **MiniMax-M2** (M2, M2.1)                     | `minimax/MiniMax-M2`, `minimax/MiniMax-M2.1`           | MiniMax 面向编码和智能体工作流的最先进 LLM。 |
| **StableLM** (3B, 7B)               | `stabilityai/stablelm-tuned-alpha-7b`            | StabilityAI 的早期开源 LLM（3B 和 7B），用于通用文本生成；具备基本指令遵循能力的演示模型。 |
| **Command-(R,A)** (Cohere)              | `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r7b-12-2024`, `CohereLabs/c4ai-command-a-03-2025`                 | Cohere 的开源对话 LLM（Command 系列），针对长上下文、检索增强生成和工具使用进行了优化。 |
| **DBRX** (Databricks)              | `databricks/dbrx-instruct`                       | Databricks 的 132B 参数 MoE 模型（36B 活跃参数），在 12T token 上训练；作为完全开放的基础模型与 GPT-3.5 质量相当。 |
| **Grok** (xAI)                     | `xai-org/grok-1`                                | xAI 的 grok-1 模型，以巨大规模（314B 参数）和高质量著称；集成在 SGLang 中用于高性能推理。 |
| **ChatGLM** (GLM-130B 系列)       | `THUDM/chatglm2-6b`                              | 智谱 AI 的双语聊天模型（6B），在中英文对话方面表现出色；针对对话质量和对齐进行了微调。 |
| **InternLM 2** (7B, 20B)           | `internlm/internlm2-7b`                          | 商汤科技的新一代 InternLM（7B 和 20B），具备强大的推理能力和超长上下文支持（最高 200K token）。 |
| **ExaONE 3** (韩英双语)      | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`           | LG AI Research 的韩英双语模型（7.8B），在 8T token 上训练；提供高质量的双语理解和生成。 |
| **Baichuan 2** (7B, 13B)           | `baichuan-inc/Baichuan2-13B-Chat`                | 百川智能的第二代中英文 LLM（7B/13B），性能提升并开放商用许可。 |
| **XVERSE** (MoE)                   | `xverse/XVERSE-MoE-A36B`                         | 元象科技的开源 MoE LLM（XVERSE-MoE-A36B：255B 总参数，36B 活跃参数），支持约 40 种语言；通过专家路由实现 100B+ 密集模型级别的性能。 |
| **SmolLM** (135M–1.7B)            | `HuggingFaceTB/SmolLM-1.7B`                      | Hugging Face 的超小型 LLM 系列（135M–1.7B 参数），效果出奇地好，可在移动端/边缘设备上实现高级 AI。 |
| **GLM-4** (多语言 9B)        | `ZhipuAI/glm-4-9b-chat`                          | 智谱的 GLM-4 系列（最高 9B 参数）——开源多语言模型，支持 100 万 token 上下文，以及 5.6B 多模态变体（Phi-4V）。 |
| **MiMo** (7B 系列)               | `XiaomiMiMo/MiMo-7B-RL`                         | 小米的推理优化模型系列，利用多 Token 预测实现更快推理。 |
| **ERNIE-4.5** (4.5, 4.5MoE 系列) | `baidu/ERNIE-4.5-21B-A3B-PT`                    | 百度的 ERNIE-4.5 系列，包含 47B 和 3B 活跃参数的 MoE 模型，最大模型总参数达 424B，以及 0.3B 的密集模型。 |
| **Arcee AFM-4.5B**               | `arcee-ai/AFM-4.5B-Base`                         | Arcee 面向真实世界可靠性和边缘部署的基础模型系列。 |
| **Persimmon** (8B)               | `adept/persimmon-8b-chat`                         | Adept 的开源 8B 模型，拥有 16K 上下文窗口和快速推理；基于 Apache 2.0 许可证。 |
| **Solar** (10.7B)               | `upstage/SOLAR-10.7B-Instruct-v1.0`                         | Upstage 的 10.7B 参数模型，针对指令遵循任务进行优化。该架构采用深度上扩展方法，提升模型性能。 |
| **Tele FLM** (52B-1T)               | `CofeAI/Tele-FLM`                         | BAAI 和 TeleAI 的多语言模型，提供 520 亿和 1 万亿参数变体。是一个在约 2T token 上训练的 decoder-only transformer。 |
| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI 的开源 MoE 模型。Ling-Lite 总参数 16.8B / 活跃参数 2.75B，Ling-Plus 总参数 290B / 活跃参数 28.8B。专为 NLP 和复杂推理任务的高性能表现而设计。 |
| **Granite 3.0, 3.1** (IBM)               | `ibm-granite/granite-3.1-8b-instruct`                          | IBM 的开源密集基础模型，针对推理、代码和商业 AI 用例进行了优化。与 Red Hat 和 watsonx 系统集成。 |
| **Granite 3.0 MoE** (IBM)               | `ibm-granite/granite-3.0-3b-a800m-instruct`                          | IBM 的混合专家模型，在成本效率下提供强大性能。MoE 专家路由专为企业级大规模部署设计。 |
| **GPT-J** (6B)                    | `EleutherAI/gpt-j-6b`                             | EleutherAI 的类 GPT-2 因果语言模型（6B），在 [Pile](https://pile.eleuther.ai/) 数据集上训练。 |
| **Orion** (14B)               | `OrionStarAI/Orion-14B-Base`                         | OrionStarAI 的开源多语言大语言模型系列，在 2.5T token 的多语言语料库（包括中文、英文、日文、韩文等）上预训练，在这些语言上表现出色。 |
| **Llama Nemotron Super** (v1, v1.5, NVIDIA) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型系列提供最先进的推理模型，专为企业级 AI 智能体设计。 |
| **Llama Nemotron Ultra** (v1, NVIDIA) | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型系列提供最先进的推理模型，专为企业级 AI 智能体设计。 |
| **NVIDIA Nemotron Nano 2.0** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | [NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) 多模态模型系列提供最先进的推理模型，专为企业级 AI 智能体设计。`Nemotron-Nano-9B-v2` 是一个混合 Mamba-Transformer 语言模型，旨在提高推理工作负载的吞吐量，同时在同等规模模型中达到最先进的精度。 |
| **StarCoder2** (3B-15B) | `bigcode/starcoder2-7b` | StarCoder2 是一个专门用于代码生成和理解的开源大语言模型系列。它是 StarCoder 的后继版本，由 BigCode 项目（Hugging Face、ServiceNow Research 及其他贡献者的合作项目）共同开发。 |
| **Jet-Nemotron** | `jet-ai/Jet-Nemotron-2B` | Jet-Nemotron 是一个新的混合架构语言模型系列，超越了最先进的开源全注意力语言模型，同时实现了显著的效率提升。 |
| **Trinity** (Nano, Mini) | `arcee-ai/Trinity-Mini` | Arcee 的基础 MoE Trinity 模型系列，以 Apache 2.0 许可证开放权重。 |
| **Falcon-H1** (0.5B–34B) | `tiiuae/Falcon-H1-34B-Instruct` | TII 的混合 Mamba-Transformer 架构，结合注意力机制和状态空间模型，实现高效的长上下文推理。 |
| **Hunyuan-Large** (389B, MoE) | `tencent/Tencent-Hunyuan-Large` | 腾讯的开源 MoE 模型，总参数 389B / 活跃参数 52B，采用跨层注意力（CLA）提高效率。 |
| **IBM Granite 4.0 (Hybrid, Dense)** | `ibm-granite/granite-4.0-h-micro`, `ibm-granite/granite-4.0-micro` | IBM Granite 4.0 micro 模型：混合 Mamba-MoE（`h-micro`）和密集（`micro`）变体。面向企业的推理模型。 |
