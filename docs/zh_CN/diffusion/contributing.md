# 为 SGLang Diffusion 做贡献

本指南概述了为 SGLang Diffusion 模块（`sglang.multimodal_gen`）做贡献的要求。

## 提交信息规范

我们遵循结构化的提交信息格式，以保持清晰的提交历史。

**格式：**
```text
[diffusion] <scope>: <subject>
```

**示例：**
- `[diffusion] cli: add --perf-dump-path argument`
- `[diffusion] scheduler: fix deadlock in batch processing`
- `[diffusion] model: support Stable Diffusion 3.5`

**规则：**
- **前缀**：始终以 `[diffusion]` 开头。
- **范围**（可选）：`cli`、`scheduler`、`model`、`pipeline`、`docs` 等。
- **主题**：使用祈使语气，简短清晰（例如，"add feature" 而非 "added feature"）。

## 性能报告

对于影响**延迟**、**吞吐量**或**内存使用**的 PR，你**应该**提供性能对比报告。

### 如何生成报告

1. **基线**：运行基准测试（针对单次生成任务）
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json
    ```

2. **新版本**：运行相同的基准测试，不修改任何 server_args 或 sampling_params
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json
    ```

3. **对比**：运行对比脚本，它会在控制台打印一个 Markdown 表格
    ```bash
    $ python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json [new2.json ...]
    ### Performance Comparison Report
    ...
    ```
4. **粘贴**：将表格粘贴到 PR 描述中

## 基于 CI 的变更保护

请考虑将测试添加到 `pr-test` 或 `nightly-test` 套件中以保护你的变更，特别是以下类型的 PR：

- 支持新模型
    - 将新模型的测试用例添加到 `testcase_configs.py`
- 支持或修复重要功能
- 显著提升性能

请运行相应的测试用例，然后按照控制台中的说明更新/添加基线到 `perf_baselines.json`（如适用）。

请参阅 [test](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/test) 获取示例。
