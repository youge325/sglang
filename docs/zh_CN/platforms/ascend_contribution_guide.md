# 贡献指南

欢迎来到 **SGLang**！感谢您对贡献的关注。本指南简要介绍了如何设置环境、运行测试、构建文档以及提交 Pull Request (PR)。无论是修复小 bug 还是开发重要功能，我们都建议遵循以下步骤以确保贡献流程顺畅。

## 从源码安装 SGLang

### 准备环境

在贡献之前，请确保您的环境已正确设置。请按照[安装指南](../platforms/ascend_npu.md)中的步骤安装必要的依赖项。我们推荐[使用 Docker](../platforms/ascend_npu.md#method-2-using-docker-image) 来构建环境。

### Fork 并克隆仓库

**注意**：新贡献者**没有**直接向 SGLang 官方仓库推送代码的权限。请先在您的 GitHub 账户下 fork 该仓库，然后在本地克隆您的 fork。

```bash
git clone https://github.com/<your_user_name>/sglang.git
# 如果您使用 Docker，环境已经设置好了。
cd sglang
export PYTHONPATH=$PWD/python:$PYTHONPATH
```

## 使用 pre-commit 格式化代码

我们使用 [pre-commit](https://pre-commit.com/) 来保持一致的代码风格检查。在推送更改之前，请运行：

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** 手动运行所有已配置的检查，并尽可能自动修复。如果第一次运行失败，请重新运行以确保所有 lint 错误已完全解决。请确保您的代码在创建 Pull Request **之前**通过所有检查。
- **不要直接**提交到 `main` 分支。始终创建一个新分支（例如 `feature/my-new-feature`），推送更改，然后从该分支发起 PR。

## 运行和添加单元测试

如果您添加了新功能或修复了 bug，请添加相应的单元测试以确保覆盖率并防止回归。
SGLang 使用 Python 内置的 [unittest](https://docs.python.org/3/library/unittest.html) 框架。
有关运行测试和将其集成到 CI 中的详细说明，请参阅 [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)。

## 编写文档

我们建议新贡献者从编写文档开始，这有助于快速了解 SGLang 代码库。
更多详情请参阅 [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)。

## 测试准确性
如果您的代码更改了模型输出，请运行准确性测试。一个快速的健全性检查是 few-shot GSM8K。

```
# 启动服务器
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# 评估
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

请注意，上述脚本主要是健全性检查，而非严格的准确性或速度测试。
由于批处理和推理引擎的非确定性特性，此测试在准确性方面可能存在较大的方差（1%–5%）。
此外，不要依赖此脚本中的"延迟/输出吞吐量"数据，因为它并非正规的速度测试。

GSM8K 对于当前最先进的模型来说已经太简单了。请尝试您自己的更具挑战性的准确性测试。
您可以在以下位置找到更多准确性评估示例：
- [test_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_eval_accuracy_large.py)
- [test_moe_eval_accuracy_large.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_moe_eval_accuracy_large.py)

## 性能基准测试
请参阅[基准测试与性能分析](../developer_guide/benchmark_and_profiling.md)。

## 请求合并审核
您可以遵循 [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md) 中描述的 Pull Request 合并流程。
您需要与 Merge Oncall、Codeowner 以及其他审核者合作以获得他们的批准。
之后您的 PR 即可合并。

## 如何触发 CI 测试

我们有大量开放的 PR 但 CI 机器有限，因此只有顶级和受信任的贡献者才有权触发 CI 测试。
有权限的用户列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中。

要在 Pull Request 上运行 CI，该 PR 必须带有 "run-ci" 标签。授权用户可以通过在 PR 上评论以下命令来添加标签或重新运行失败的测试：

- `/tag-run-ci-label`：添加 "run-ci" 标签。之后每次提交都会触发 CI。
- `/rerun-failed-ci`：重新运行最近一次提交中失败或不稳定的测试。
- `/tag-and-rerun-ci`：同时执行 `/tag-run-ci-label` 和 `/rerun-failed-ci` 的单一命令。
- `/rerun-stage <stage-name>`：重新运行特定的测试阶段，无需等待其依赖项完成。当您想要快速验证特定测试失败的修复而不想等待约 30 分钟的前置阶段完成时，此命令非常有用。

如果您有权限，[Slash Command Handler](https://github.com/sgl-project/sglang/actions/workflows/slash-command-handler.yml) 将运行您的命令并在您的评论上添加 👍 反应。反应可能需要几分钟才能出现。这是一个使用[示例](https://github.com/sgl-project/sglang/pull/14253#issuecomment-3599509302)。

为避免在 PR 中发送过多的 `/rerun-failed-ci` 评论，您也可以通过编辑现有评论并添加任何后缀来触发命令（例如 `/rerun-failed-ci try again`）。

重新运行单个测试阶段的示例：`/rerun-stage unit-test-backend-4-gpu`。

如果您没有权限，请让维护者帮您触发 CI。

### CI 速率限制

由于 CI 调度和有限的资源，更高优先级的 PR 可能会抢占正在运行的任务。在这种情况下，您可能需要重新运行测试。

我们实施了 CI 速率限制以防止滥用并确保 CI 资源的公平使用。

每个 CI 工作流在其工作流配置文件中都有一个默认限制。例如，在 [pr-gate.yml](https://github.com/sgl-project/sglang/blob/main/.github/workflows/pr-gate.yml) 中，默认冷却期为 120 分钟，每个工作流可以通过 `cool-down-minutes` 输入参数覆盖它：

```yaml
cool-down-minutes:
  description: "Default cooldown period in minutes; 0 disables rate limiting"
  type: number
  default: 120
```

列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中的用户可能有每用户的冷却间隔。实际上，我们使用工作流默认窗口和用户特定间隔的最小值。


## 代码风格指南
- 避免代码重复。如果相同的代码片段（超过五行）出现多次，请将其提取为共享函数。
- 最小化设备同步。尽可能减少昂贵的 CPU-GPU 同步操作，如 `tensor.item()` 或 `tensor.cpu()`。使用向量化代码。
- 优先考虑极致效率。SGLang 是一个运行时系统，您的大部分代码都运行在每个请求的关键路径上。请尽可能优化所有微小开销，尤其是在模型前向传播代码中。
  - 一个常见模式是在模型前向传播中进行一些运行时检查（例如[这个](https://github.com/sgl-project/sglang/blob/f1b0eda55c2c4838e8ab90a0fac7fb1e3d7064ab/python/sglang/srt/models/deepseek_v2.py#L486-L491)）。这些检查在每一层中很可能是相同的。请尽可能将结果缓存为单个布尔值。
- 使函数尽可能纯粹。避免就地修改参数。
- 保持文件简洁。如果文件超过 2,000 行代码，请将其拆分为多个较小的文件。（例如 `scheduler.py`、`scheduler_output_processor_mixin.py`）
- 保持测试快速运行。
  - 如果单个测试文件运行超过 500 秒，请将其拆分为多个较小的文件（例如 `test_eagle_infer_a.py`、`test_eagle_infer_b.py`）。
  - 如果 GitHub 工作流中的单个任务运行超过 30 分钟，请将其拆分为较小的任务/步骤。
  - 在单元测试中重用服务器启动以加快测试运行速度。
- 在支持新硬件或功能时，请遵循以下指南：
  - 不要大幅更改现有代码。
  - 始终优先使用新文件来引入新硬件的特定组件（例如 `allocator_ascend.py`）。
  - 如果您为新功能编写了多个 if/else 分支，请确保通用路径（例如 NVIDIA 硬件或现有代码路径）是第一个分支。

## 如何更新 sgl-kernel
由于 sglang 和 sgl-kernel 是独立的 Python 包，我们当前的 GitHub CI 基础设施不支持在同一个 Pull Request 中更新内核并立即使用它。
要在 sgl-kernel 包中添加新内核或修改现有内核，您必须使用多个 PR。

请遵循以下步骤：

1. 提交一个 PR 来更新 sgl-kernel 源代码，但不在 sglang Python 包中使用它（例如 [#8884](https://github.com/sgl-project/sglang/pull/8884/files)）。
2. 升级 sgl-kernel 的版本（例如 [#9220](https://github.com/sgl-project/sglang/pull/9220/files)）。
   - 合并后，这将自动触发 sgl-kernel wheel 包发布到 PyPI。
   - 如果不紧急，您可以等待其他人发布 wheel 包。通常在一周内会发布新版本。
3. 应用更改：
   - 更新 `sglang/python/pyproject.toml` 中的 sgl-kernel 版本以使用修改后的内核。
   - 更新 sglang 中的相关调用代码以使用新内核。

## 如何更新 sgl-kernel-npu

sgl-kernel-npu 是 Ascend NPU 的内核包，维护在 [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) 仓库中。如果您想添加新内核并在 sglang 中使用它，请按照[贡献指南](https://github.com/sgl-project/sgl-kernel-npu/blob/main/docs/developer_guide/contribution_guide.md)中的步骤操作。

## 新手提示

如果您想贡献但还没有具体的想法，可以选择标记为 ["good first issue" 或 "help wanted"](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22) 的 issue。这些任务通常复杂度较低，是了解代码库的绝佳入门途径。另外，您也可以查看这个[代码走读](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through)以深入了解 SGLang 的工作流程。

如果您有任何问题或想发起讨论，请随时在我们的 [Slack 频道](https://slack.sglang.io)中提问。

感谢您对 SGLang 的关注。祝编码愉快！
