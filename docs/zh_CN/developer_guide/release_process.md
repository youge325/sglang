# PyPI 包发布流程

## 更新代码中的版本号
在 `python/pyproject.toml` 和 `python/sglang/__init__.py` 中更新包版本号。

## 上传 PyPI 包

```
pip install build twine
```

```
cd python
bash upload_pypi.sh
```

## 在 GitHub 上创建发布
新建一个发布 https://github.com/sgl-project/sglang/releases/new。
