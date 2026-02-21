# 贡献指南

欢迎来到 **SGLang**！感谢您有兴趣为项目做出贡献。本指南简明概述了如何设置开发环境、运行测试、构建文档和提交 Pull Request (PR)。无论您是修复小 bug 还是开发大型功能，我们都建议遵循以下步骤以确保顺畅的贡献流程。

## 从源码安装 SGLang

### Fork 并克隆仓库

**注意**：新贡献者没有向官方 SGLang 仓库直接推送的写入权限。请在您的 GitHub 账号下 fork 该仓库，然后在本地克隆您的 fork。

```bash
git clone https://github.com/<your_user_name>/sglang.git
```

### 从源码构建

参见[从源码安装 SGLang](../get_started/install.md#方法-2从源码安装)。

## 使用 pre-commit 格式化代码

我们使用 [pre-commit](https://pre-commit.com/) 来维护一致的代码风格检查。在推送更改之前，请运行：

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** 手动运行所有配置的检查，如有可能会自动修复。如果第一次失败，请重新运行以确保 lint 错误完全解决。请确保您的代码在创建 PR **之前**通过所有检查。
- **不要直接提交**到 `main` 分支。始终创建新分支（如 `feature/my-new-feature`），推送您的更改，然后从该分支提交 PR。

## 运行和添加单元测试

如果您添加了新功能或修复了 bug，请添加相应的单元测试以确保覆盖率并防止回归。
SGLang 使用 Python 内置的 [unittest](https://docs.python.org/3/library/unittest.html) 框架。
有关运行测试和集成到 CI 的详细说明，请参阅 [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md)。

## 编写文档

我们建议新贡献者从编写文档开始，这有助于您快速了解 SGLang 代码库。
更多详情请参阅 [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md)。

## 测试准确性
如果您的代码更改了模型输出，请运行准确性测试。快速的健全性检查是 few-shot GSM8K。

```
# 启动服务器
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct

# 评估
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

请注意，上述脚本主要是健全性检查，而非严格的准确性或速度测试。
由于批处理和推理引擎的非确定性，此测试在准确性方面可能有较大的方差（1%-5%）。

## 基准测试速度
请参阅[基准测试与性能分析](../developer_guide/benchmark_and_profiling.md)。

## 请求审查合并
您可以按照 [MAINTAINER.md](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md) 中描述的 PR 合并流程进行操作。
您需要与 Merge Oncall、Codeowner 和其他审查者合作以获得他们的批准。

## 如何触发 CI 测试

我们有大量开放的 PR 但 CI 机器有限，因此只有顶级和受信任的贡献者有权限触发 CI 测试。
拥有权限的用户列在 [CI_PERMISSIONS.json](https://github.com/sgl-project/sglang/blob/main/.github/CI_PERMISSIONS.json) 中。

要在 PR 上运行 CI，它必须有 "run-ci" 标签。授权用户可以添加标签或通过以下命令评论来重新运行失败的测试：

- `/tag-run-ci-label`：添加 "run-ci" 标签。
- `/rerun-failed-ci`：重新运行最近提交中失败或不稳定的测试。
- `/tag-and-rerun-ci`：同时执行上述两个操作。
- `/rerun-stage <stage-name>`：重新运行特定测试阶段。

## 代码风格指导
- 避免代码重复。如果相同的代码片段（超过五行）出现多次，请提取为共享函数。
- 最小化设备同步。尽可能减少昂贵的 CPU-GPU 同步操作，如 `tensor.item()` 或 `tensor.cpu()`。使用向量化代码。
- 优先考虑极致效率。SGLang 是运行时系统，大部分代码都在每个请求的关键路径上运行。
- 尽量使函数纯粹。避免就地修改参数。
- 保持文件简洁。如果文件超过 2000 行代码，请拆分为多个较小的文件。
- 保持测试运行快速。如果单个测试文件运行超过 500 秒，请拆分为多个较小的文件。

## 如何更新 sgl-kernel
由于 sglang 和 sgl-kernel 是独立的 Python 包，当前 GitHub CI 基础设施不支持在同一个 PR 中更新内核并立即使用。

请按以下步骤操作：

1. 提交 PR 更新 sgl-kernel 源代码，但不在 sglang python 包中使用。
2. 升级 sgl-kernel 版本。合并后将自动触发 sgl-kernel wheel 发布到 PyPI。
3. 在 `sglang/python/pyproject.toml` 中更新 sgl-kernel 版本并更新调用代码。

## 新手提示

如果您想贡献但没有特定想法，请选择标记为 ["good first issue" 或 "help wanted"](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22) 的 issue。另请查看此[代码走查](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through)以深入了解 SGLang 的工作流程。

如有任何问题或想开始讨论，请随时在我们的 [Slack 频道](https://slack.sglang.io)中提问。

感谢您对 SGLang 的兴趣。祝编码愉快！
