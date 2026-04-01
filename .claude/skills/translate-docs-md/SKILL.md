---
name: translate-docs-md
description: "Translate technical documentation from English to Chinese (Markdown/RST/Jupyter). Use when asked to localize docs, finish remaining untranslated files, sync zh_CN structure, update toctree links, and verify Sphinx build output."
argument-hint: "Provide source and target doc roots, e.g. docs -> docs/zh_CN"
---

# Translate Documentation (EN -> ZH)

用于将技术文档从英文批量翻译为中文，覆盖 `*.md`、`*.rst`、`*.ipynb`，并完成可构建校验。

## When To Use

- 用户要求“翻译文档”“补齐中文文档”“继续翻译剩余文件”。
- 仓库已存在 `docs/zh_CN` 或等价多语言目录，需要同步英文文档结构。
- 需要处理 Notebook 文档（只翻译 markdown 单元，不修改 code 单元）。

## Inputs

- 源文档根目录（通常 `docs/`）
- 目标文档根目录（通常 `docs/zh_CN/`）
- 目标范围：全部文档或指定子目录
- 翻译风格：默认“本地化优先”，术语尽量翻成中文；必要时保留英文缩写（如 API、JSON、Sphinx）

## Procedure

1. 盘点文件与差异范围
- 枚举源目录下 `*.md`、`*.rst`、`*.ipynb`。
- 与目标目录对比，标记：缺失文件、未翻译文件、已翻译但可能过期文件。
- 优先做“剩余文件”，避免重复返工。

2. 同步目录与文件骨架
- 在目标目录创建缺失子目录。
- 对 Notebook 建议先从源目录复制同名文件到目标目录，再替换 markdown 单元内容。

3. 翻译 `md/rst`
- 保留代码块、命令行、路径、API 名称、参数名。
- 标题层级、列表层级、表格结构保持不变。
- 链接目标尽量保持；若中英文路径不同，再统一修正。

4. 翻译 `ipynb`
- 只翻译 `cell_type == "markdown"` 的 `source`。
- 禁止改动代码单元逻辑。
- 保持 notebook JSON 合法，`source` 维持字符串数组格式。

5. 更新导航与入口
- 修正 `index.rst` / 子级 `*.rst` 的 `toctree`。
- 将原先指向英文页面的临时提示替换为本地中文条目（若已完成翻译）。

6. 构建验证
- 执行 Sphinx 构建（如：`python -m sphinx -b html zh_CN _build/html/zh_CN`）。
- 检查是否成功产出，确认新增页面被收录（尤其 Notebook copying 阶段）。

7. 清理与交付
- 删除临时翻译脚本。
- 输出变更摘要：新增/更新文件、构建结果、遗留 warning 分类。

## Decision Points

- 若文件量很小：直接逐文件编辑。
- 若 Notebook markdown 单元很多：使用脚本批处理（读取 JSON -> 替换 markdown `source` -> 写回）。
- 若目标目录没有对应入口：先更新 toctree，再做构建。
- 若构建出现 warning：
  - 若为本次改动引入（路径断链、格式错误），必须修复。
  - 若为历史遗留 warning，记录并说明“非本次引入”。

## Quality Criteria

- 语义准确：技术术语一致，中文可读。
- 本地化优先：同一术语跨文档统一中文译法，首次可中英并列。
- 结构稳定：标题、列表、表格、代码块不破坏。
- Notebook 安全：代码单元零改动。
- 导航完整：入口页能访问新增中文页面。
- 构建可用：Sphinx build 成功，关键页面可生成。

## Completion Checklist

- [ ] 目标范围文件已全部翻译
- [ ] Notebook 仅 markdown 单元被改动
- [ ] toctree/入口已更新
- [ ] Sphinx 构建已执行并记录结果
- [ ] 临时脚本已清理
- [ ] 交付说明包含：文件清单、验证方式、剩余风险

## Common Pitfalls

- 把 Notebook `source` 写成单字符串而非字符串数组。
- 翻译时误改代码示例、命令参数、JSON 键名。
- 只翻译正文，忘记更新 `index.rst` 导致页面不可达。
- 构建只看“是否结束”，未确认新增 notebook 是否被复制/收录。

## Recommended Output Format

1. 已完成项（按目录分组）
2. 关键变更文件
3. 构建结果（成功/失败 + 主要 warning）
4. 下一步建议（可选）
