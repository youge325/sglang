# Moore Threads GPU

本文档描述如何在 Moore Threads GPU 上运行 SGLang。如果您遇到问题或有疑问，请[提交 issue](https://github.com/sgl-project/sglang/issues)。

## 安装 SGLang

您可以使用以下方法之一安装 SGLang。

### 从源码安装

```bash
# 使用默认分支
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 编译 sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# 安装 sglang Python 包
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```
