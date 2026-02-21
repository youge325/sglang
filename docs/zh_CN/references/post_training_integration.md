# 后训练集成

SGLang 已成为现代 LLM 训练框架的事实标准推理后端，为业界最先进的模型提供支持。从 GLM-4.6 到 Qwen3，领先的模型都在强化学习和后训练工作流中利用 SGLang 的高性能推理能力。

SGLang 为何在后训练中不可或缺？

- 开放易用的 Refit 功能：提供多种协同部署或分离部署的方法
- 便捷的延迟生成：支持部分 rollout 和专用 rollout 控制
- 细粒度的引擎休眠与唤醒：实现最大化的 rollout 和训练效能
- 训练-推理一致性：确保训练和推理中的性能一致性
- 负载均衡路由器：缓存感知的负载均衡，实现高吞吐量 rollout
- 确定性推理：确保 rollout 和训练之间零 KL 散度

这些能力，加上对主流框架的原生集成支持，使 SGLang 成为现代 LLM/VLM 后训练的基础设施骨干。我们也在这份幻灯片中分享了最新工作：[Optimizing Large-Scale RL with SGLang](https://gamma.app/docs/Optimizing-RL-with-SGLang-y0kqgj877k34779)。

## 应用案例

- [**Miles**](https://github.com/radixark/miles)：面向大型 MoE 模型的企业级 RL 框架，支持 SGLang 原生 rollout、投机训练和生产级稳定性
- [**slime**](https://github.com/THUDM/slime)：结合 Megatron 和 SGLang 的后训练框架，用于训练 GLM-4.6
- [**AReaL**](https://github.com/inclusionAI/AReaL)：全异步 RL 系统，使用 SGLang 后端进行连续 rollout 生成，实现 2.77 倍加速
- [**ROLL**](https://github.com/alibaba/ROLL)：ROLL 是一个高效且用户友好的 RL 库，专为利用大规模 GPU 资源的大语言模型设计
- [**verl**](https://github.com/volcengine/verl)：全栈 RLHF 框架，支持 PPO、GRPO 和 ReMax，具有模块化的 SGLang 集成
- [**Unsloth**](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide)：2 倍更快的微调，优化内核，与 SGLang 推理无缝部署
- [**LLaMA Factory**](https://github.com/hiyouga/LLaMA-Factory)：统一框架，支持使用 LoRA、QLoRA 和全量微调方法训练 100+ 种 LLM
- [**Tunix**](https://github.com/google/tunix)：Google 的 JAX 原生库，用于 LLM 后训练，支持 SFT、DPO、PPO 和 GRPO
- [**RL2**](https://github.com/ChenmienTan/RL2)：Ray Less Reinforcement Learning，一个简洁的大语言模型后训练库


## 合作

由于设计合作伙伴的隐私需求，我们无法列出采用 SGLang 进行后训练的公司。然而，如果您感兴趣，我们很乐意与您分享细节，并且可以信赖在美国和中国超过 10 家顶级公司和前沿实验室的选择。如果您有兴趣将 SGLang 与您的训练框架集成，或需要技术支持，我们随时为您提供帮助！请通过 **rl_team@lmsys.org** 联系我们，获取合作、集成指导和定制功能开发支持。
