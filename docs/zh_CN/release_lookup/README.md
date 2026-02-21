# SGLang 版本查询工具

此工具允许用户查找包含特定 PR 或提交的最早版本。
它完全在浏览器中运行，使用从 git 历史记录生成的静态 JSON 索引。

## 使用方法

1. **生成索引**：
   运行 Python 脚本，从本地 git 仓库生成 `release_index.json` 文件。

   ```bash
   python3 generate_index.py --output release_index.json
   ```

   此脚本会：
   - 查找所有匹配 `v*` 和 `gateway-v*` 的标签。
   - 按创建日期排序。
   - 遍历历史记录，查找每个提交和 PR 首次引入的版本。
   - 从提交消息中提取 PR 编号。

2. **打开工具**：
   在浏览器中打开 `index.html`。

   ```bash
   # 如果浏览器支持本地文件访问（Firefox 通常支持），可以直接打开，
   # 或者在本地启动服务：
   python3 -m http.server
   # 然后访问 http://localhost:8000/index.html
   ```

## 文件说明

- `index.html`：查询工具的用户界面。
- `generate_index.py`：构建索引的脚本。
- `release_index.json`：用户界面使用的索引文件。

## 逻辑原理

该工具根据标签创建日期确定"最早版本"。它按从旧到新的顺序遍历标签。任何从某个标签可达（且从之前的标签不可达）的提交都会被分配到该版本。
