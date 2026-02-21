SGLang 中文文档
================

.. raw:: html

  <a class="github-button" href="https://github.com/sgl-project/sglang" data-size="large" data-show-count="true" aria-label="Star sgl-project/sglang on GitHub">Star</a>
  <a class="github-button" href="https://github.com/sgl-project/sglang/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork sgl-project/sglang on GitHub">Fork</a>
  <script async defer src="https://buttons.github.io/buttons.js"></script>
  <br></br>

SGLang 是一个面向大语言模型和多模态模型的高性能推理服务框架。
它专为在各种环境下提供低延迟、高吞吐量的推理而设计，覆盖从单 GPU 到大规模分布式集群的部署场景。
其核心特性包括：

- **高性能运行时**: 提供高效推理服务，支持 RadixAttention 前缀缓存、零开销 CPU 调度器、预填充-解码分离、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）以及多 LoRA 批处理。
- **广泛的模型支持**: 支持各类语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等）、嵌入模型（e5-mistral、gte、mcdse）、奖励模型（Skywork）和扩散模型（WAN、Qwen-Image），并可轻松扩展新模型。兼容大多数 Hugging Face 模型和 OpenAI API。
- **丰富的硬件支持**: 可运行在 NVIDIA GPU（GB200/B300/H100/A100/Spark）、AMD GPU（MI355/MI300）、Intel Xeon CPU、Google TPU、昇腾 NPU 等平台。
- **活跃的社区**: SGLang 是开源项目，拥有充满活力的社区和广泛的行业应用，驱动全球超过 40 万张 GPU。
- **强化学习与后训练骨干**: SGLang 是全球领先的推理 rollout 后端，具有原生强化学习集成，并被 AReaL、Miles、slime、Tunix、verl 等知名后训练框架所采用。

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   get_started/install.md

.. toctree::
   :maxdepth: 1
   :caption: 基本用法

   basic_usage/send_request.ipynb
   basic_usage/openai_api.rst
   basic_usage/ollama_api.md
   basic_usage/offline_engine_api.ipynb
   basic_usage/native_api.ipynb
   basic_usage/sampling_params.md
   basic_usage/popular_model_usage.rst

.. toctree::
   :maxdepth: 1
   :caption: 高级特性

   advanced_features/server_arguments.md
   advanced_features/hyperparameter_tuning.md
   advanced_features/attention_backend.md
   advanced_features/speculative_decoding.md
   advanced_features/structured_outputs.ipynb
   advanced_features/structured_outputs_for_reasoning_models.ipynb
   advanced_features/tool_parser.ipynb
   advanced_features/separate_reasoning.ipynb
   advanced_features/quantization.md
   advanced_features/quantized_kv_cache.md
   advanced_features/expert_parallelism.md
   advanced_features/dp_dpa_smg_guide.md
   advanced_features/lora.ipynb
   advanced_features/pd_disaggregation.md
   advanced_features/epd_disaggregation.md
   advanced_features/pipeline_parallelism.md
   advanced_features/hicache.rst
   advanced_features/vlm_query.ipynb
   advanced_features/dp_for_multi_modal_encoder.md
   advanced_features/cuda_graph_for_multi_modal_encoder.md
   advanced_features/sgl_model_gateway.md
   advanced_features/deterministic_inference.md
   advanced_features/observability.md
   advanced_features/checkpoint_engine.md
   advanced_features/sglang_for_rl.md
   advanced_features/rfork.md
   advanced_features/forward_hooks.md

.. toctree::
   :maxdepth: 2
   :caption: 支持的模型

   supported_models/text_generation/index
   supported_models/retrieval_ranking/index
   supported_models/specialized/index
   supported_models/extending/index

.. toctree::
   :maxdepth: 2
   :caption: SGLang Diffusion

   diffusion/index
   diffusion/installation
   diffusion/compatibility_matrix
   diffusion/api/cli
   diffusion/api/openai_api
   diffusion/performance/index
   diffusion/performance/attention_backends
   diffusion/performance/profiling
   diffusion/performance/cache/index
   diffusion/performance/cache/cache_dit
   diffusion/performance/cache/teacache
   diffusion/support_new_models
   diffusion/contributing
   diffusion/ci_perf
   diffusion/environment_variables

.. toctree::
   :maxdepth: 1
   :caption: 硬件平台

   platforms/amd_gpu.md
   platforms/cpu_server.md
   platforms/tpu.md
   platforms/nvidia_jetson.md
   platforms/ascend_npu_support.rst
   platforms/xpu.md
   platforms/mthreads_gpu.md
   platforms/mindspore_backend.md

.. toctree::
   :maxdepth: 1
   :caption: 开发者指南

   developer_guide/contribution_guide.md
   developer_guide/development_guide_using_docker.md
   developer_guide/development_jit_kernel_guide.md
   developer_guide/benchmark_and_profiling.md
   developer_guide/bench_serving.md
   developer_guide/evaluating_new_models.md
   developer_guide/release_process.md
   developer_guide/setup_github_runner.md

.. toctree::
   :maxdepth: 1
   :caption: 参考

   references/faq.md
   references/environment_variables.md
   references/production_metrics.md
   references/production_request_trace.md
   references/multi_node_deployment/multi_node_index.rst
   references/custom_chat_template.md
   references/frontend/frontend_index.rst
   references/post_training_integration.md
   references/torch_compile_cache.md
   references/learn_more.md
   translation_guide.md
