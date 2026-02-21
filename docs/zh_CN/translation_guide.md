# SGLang 文档翻译指南

本指南说明如何为 SGLang 文档贡献中文翻译。

## 目录结构

```
docs/
├── conf.py              # 英文 Sphinx 配置
├── index.rst            # 英文首页
├── get_started/         # 英文文档
├── basic_usage/
├── advanced_features/
├── ...
├── zh_CN/               # 中文翻译目录
│   ├── conf.py          # 中文 Sphinx 配置
│   ├── index.rst        # 中文首页
│   ├── get_started/     # 中文翻译文件
│   ├── basic_usage/
│   ├── advanced_features/
│   └── ...
├── _static/
│   └── js/
│       └── lang_switcher.js  # 语言切换器
└── Makefile             # 构建脚本（支持多语言）
```

## 如何贡献翻译

### 1. 翻译新文件

1. 在 `docs/zh_CN/` 中创建与英文文档相同路径的文件。
   例如：要翻译 `docs/basic_usage/send_request.md`，创建 `docs/zh_CN/basic_usage/send_request.md`。

2. 翻译文件内容，保留所有代码块、命令行和链接不变。

3. 在 `docs/zh_CN/index.rst` 的相应 toctree 中添加新文件。

### 2. 翻译规范

- **保留代码块**：所有 `code-block`、命令行示例保持英文原样。
- **保留链接**：URL、GitHub 链接保持不变。
- **技术术语**：常见术语如 token、batch、GPU 等可保持英文。在首次出现时可添加中文注释。
- **参数名**：所有参数名（如 `--model-path`）保持英文。
- **表格**：翻译描述列，参数名和类型保持原样。

### 3. 对于长篇参考文档

对于非常长的参考文档（如 server_arguments.md），可以：
- 翻译最常用的部分
- 在末尾添加指向英文完整版的链接

### 4. 对于 Notebook 文件（.ipynb）

Notebook 文件目前暂不翻译。中文文档中可引用英文原始 notebook，或创建 .md 版本的要点摘要。

## 构建文档

```bash
# 仅构建英文文档
make html-en

# 仅构建中文文档
make html-zh

# 构建双语文档（推荐）
make html-all

# 自动构建并实时预览
make serve
```

构建后：
- 英文文档位于 `_build/html/en/`
- 中文文档位于 `_build/html/zh_CN/`
- 根页面 `_build/html/index.html` 会自动跳转到英文版

## 语言切换

构建后的文档顶部有语言切换横幅。点击对应语言即可切换。
语言切换器会尝试在对应语言中打开相同路径的页面。
