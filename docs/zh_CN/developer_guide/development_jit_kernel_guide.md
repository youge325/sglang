# JIT 内核开发指南

## 环境设置

我们强烈建议使用 `clangd` 作为 JIT 内核开发的语言服务器。
对于 Ubuntu/Debian，你可以从 [apt.llvm.org](https://apt.llvm.org/) 下载 clangd。
如果你使用 VS Code，我们推荐安装 `clangd` 扩展以获得更好的 IDE 集成。

所有 JIT 相关文件位于 `python/sglang/jit_kernel`。
与 `sgl-kernel`（提前编译 CUDA/C++ 二进制文件，即 AOT）不同，即时编译（JIT）内核在运行时编译。
因此，无法生成静态的 `compile_commands.json`。
要启用 `clangd` 的代码补全，请运行 `python -m sglang.jit_kernel` 在当前目录中生成 `.clangd` 配置文件。
生成文件后，重启 clangd 语言服务器。它现在应该能够识别所有 JIT 内核文件。

## 代码结构

### C++ 实现

C++ 源代码位于 `python/sglang/jit_kernel/csrc`。
可复用的函数应放在 `python/sglang/jit_kernel/include` 中。

我们使用 [tvm-ffi](https://github.com/apache/tvm-ffi) 进行高效的跨语言绑定。
请参阅[文档](https://tvm.apache.org/ffi/)了解高级用法，例如导出 C++ 对象。
通常，`tvm::ffi::TensorView` 足以从 Python 传递 PyTorch Tensor。

### Python 接口

Python 接口定义在 `python/sglang/jit_kernel` 中。
`python/sglang/jit_kernel/utils.py` 中的 `load_jit` 工具函数用于加载并返回编译后的模块。
要导出一个 C++ 函数（例如 `cpp_func`），请将 `cuda_wrappers=[("func", "cpp_func")]` 传递给 `load_jit`。
然后可以在 Python 中通过 `module.func` 调用该函数。

对于缓存编译后的模块，优先使用 `sglang.jit_kernel.utils.cache_once` 而非 `functools.lru_cache`。
`functools.lru_cache` 与 `torch.compile` 不兼容。

### C++ 工具

以下 C++ 工具可供使用：

#### 整数范围

类似于 PyTorch，我们提供了一个 `irange` 函数来表示整数范围。

```C++
#include <sgl_kernel/utils.h>

void test() {
  for (auto i : host::irange(100)) { // [0, 100)
    // 执行某些操作
  }
  for (auto i : host::irange(0, 100)) { // [0, 100)
    // 执行某些操作
  }
}

```

#### 运行时检查

`RuntimeCheck` 在运行时验证条件。它接受可选参数用于错误报告。
如果检查失败，这些参数将被输出以辅助调试。
`RuntimeDeviceCheck` 验证上一次内核启动的状态。

```C++
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

void test() {
  host::RuntimeCheck(1 + 1 == 2, 1 + 1, " != ", 2);
  host::RuntimeDeviceCheck();
  // 检查提供的 `cudaError_t`
  host::RuntimeDeviceCheck(cudaGetLastError());
}

```

#### 张量检查

`TensorMatcher` 提供了一种可读的方式来验证和提取张量形状信息。

```cpp
#include <sgl_kernel/tensor.h>

void test(const tvm::ffi::TensorView k_cache, const tvm::ffi::TensorView v_cache) {
  using namespace host;

  auto D = SymbolicSize{"D"};  // 缓存维度
  auto N = SymbolicSize{"N"};  // kvcache 步长
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};

  TensorMatcher({-1, D})  //
      .with_strides({N, 1})
      .with_dtype<int32_t, int64_t>(dtype)
      .with_device<kDLCUDA, kDLCPU>(device)
      .verify(k_cache)
      .verify(v_cache);
}
```

在验证之前，使用期望的步长、数据类型和设备属性配置 `TensorMatcher`。
- 如果省略 `with_strides`，则期望张量是连续的。
- `with_dtype` 中的模板参数限制允许的数据类型。
- `with_device` 中的模板参数限制允许的设备。
- 传递给 `with_xxx` 方法的值执行相等性检查。
- 对大小或步长传递 `-1` 允许匹配任何值。

`Symbolic` 变量在所有验证中必须解析为相同的值。
验证后使用 `.unwrap()` 获取匹配的值。

> 注意：`TensorMatcher` 是一个临时表达式，不应存储在变量中。

> 提示：在 `TensorMatcher` 链的末尾添加 `//` 以强制正确的缩进。

#### 内核启动

`LaunchKernel::resolve_device` 从 PyTorch 获取当前的 `cudaStream`。
也可以直接使用 `LaunchKernel` 启动内核。

```cpp
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

__global__ void kernel() {}

void test() {
  const auto num_blocks = 1;
  const auto num_threads = 32;
  const auto dynamic_smem = 0;

  DLDevice dev;  // 假设已正确初始化
  host::LaunchKernel(num_blocks, num_threads, dev)(kernel);

  cudaStream_t stream = host::LaunchKernel::resolve_device(dev);
  host::LaunchKernel(num_blocks, num_threads, stream, dynamic_smem)(kernel);
}

```

## 添加新内核

本节通过一个完整的端到端示例，演示如何向系统添加新的 JIT 内核。
我们使用一个简单的 add_constant 内核作为示例，它将一个常量整数值添加到输入张量的每个元素上。

概念上，Python 接口如下所示：

```python
def add_constant(src: torch.Tensor, c: int):
    return src + c
```

### 步骤 1：编写 C++ 内核

在 [jit_kernel/csrc/add_constant.cuh](../../python/sglang/jit_kernel/csrc/add_constant.cuh) 中编写 CUDA 内核。出于演示目的，我们将常量值作为模板参数传递。

```cpp
#include <sgl_kernel/tensor.h>   // 用于 TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // 用于 LaunchKernel
#include <sgl_kernel/utils.h>    // 用于 div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

// 你也可以使用带有静态方法的 struct 作为替代
template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. 验证输入张量
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})                  // 一维张量，必须连续
      .with_dtype<int32_t>()          // 必须是 int32
      .with_device<kDLCUDA>(device_)  // 必须在 CUDA 设备上
      .verify(dst)                    // 检查张量 dst
      .verify(src);                   // 检查张量 src

  // 2. 提取必需参数，准备内核启动
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  // 使用 host::RuntimeCheck 进行一些额外的运行时检查
  RuntimeCheck(num_elements > 0, "We only support non-empty tensors, got num_elements = ", num_elements);

  // 3. 启动内核。错误码将被自动检查。
  LaunchKernel(grid_size, kBlockSize, device /*, dynamic_smem*/)(
      // 内核函数
      add_constant_kernel<kConstant>,
      // 内核参数
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace

```

### 步骤 2：创建 Python 接口

接下来，通过 Python 包装器暴露内核。
在 [jit_kernel/add_constant.py](../../python/sglang/jit_kernel/add_constant.py) 创建一个新文件并暴露所需的接口。

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)  # 传递所有模板参数
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst

```

### 步骤 3：使用你的内核

最后，像普通 Python 函数一样导入和使用内核：

```python
from sglang.jit_kernel.add_constant import add_constant
```

完整的可运行示例请参阅 [test_add_constant.py](../../python/sglang/jit_kernel/tests/test_add_constant.py)。
