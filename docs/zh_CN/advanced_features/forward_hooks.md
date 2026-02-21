# 模型钩子（Model Hooks）

SGLang 支持将 PyTorch 前向钩子（forward hooks）附加到已加载模型中的特定子模块上，完全通过 `server_args` JSON 进行配置。

这对以下场景很有用：

* 记录中间层激活值
* 调试模型内部结构
* 将隐藏状态导出到外部工具

钩子在 `ModelRunner.initialize` 期间一次性附加，并在每次前向传播时运行。

---

### 配置概述

钩子通过 `ServerArgs` 字段进行配置：

```python
class ServerArgs:
    ...
    # For forward hooks
    forward_hooks: Optional[List[dict[str, Any]]] = None
```

JSON 格式的最小配置示例如下：

```jsonc
{
  "forward_hooks": [
    {
      "name": "outer_linear_hooks",
      "target_modules": ["outer.0", "outer.1"],
      "hook_factory": "my_project.hooks:dummy_hook_factory",
      "config": {
        "tag": "outer-layer"
      }
    }
  ]
}
```

#### 顶层字段

* `forward_hooks`（可选的对象列表）
  列表中的每个元素是一个钩子规范，描述：

  * 目标模块
  * 调用的 Python 工厂函数
  * 传递给工厂函数的配置

---

### 钩子规范 Schema

`forward_hooks` 中的每个条目是一个具有以下结构的 JSON 对象：

```jsonc
{
  "name": "optional-descriptive-name",
  "target_modules": ["pattern1", "pattern2", "..."],
  "hook_factory": "module.submodule:factory_name",
  "config": {
    "...": "arbitrary JSON"
  }
}
```

#### `name`（可选）

* 人类可读的名称，用于日志记录。
* 仅在日志消息中使用，例如：

  ```text
  Registered forward hook 'outer_linear_hooks' on outer.0
  ```

#### `target_modules`（必需）

* **模块名称模式**列表，用于匹配 `model.named_modules()` 中的条目。
* 模式使用 `fnmatch.fnmatch` 进行匹配，因此：

  * `"outer.0"` 精确匹配 `"outer.0"`。
  * `"outer.*"` 匹配 `"outer.0"`、`"outer.1"`、`"outer.inner"` 等。
  * `"outer.inner.*"` 匹配 `outer.inner` 下的子模块。

> 如果没有模块匹配给定的模式，钩子注册**不会**失败。
> 相反，SGLang 会记录一条警告并继续执行：
>
> ```text
> No modules matched hook spec 'name' patterns=['...']
> ```

#### `hook_factory`（必需）

* 指向创建钩子的 Python 工厂函数的字符串路径。
* 支持的格式：

  * `"package.module:factory_name"`
  * `"package.module.submodule.factory_name"`

路径通过以下方式解析：

```python
def resolve_callable(path: Optional[str]) -> Optional[Callable]:
    if path is None:
        return None

    if ":" in path:
        module_name, fn_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook callable path '{path}'. "
                "Expected 'module.submodule:factory' or 'module.submodule.factory'."
            )
        *mod_parts, fn_name = parts
        module_name = ".".join(mod_parts)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, fn_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{fn_name}' "
            f"(from hook path '{path}')"
        ) from e
```

**失败模式**：

* 如果路径格式不正确（点号不够且没有 `:`），启动时会抛出 `ValueError`。
* 如果模块导入成功但属性不存在，则会抛出带有清晰错误消息的 `AttributeError`。
* 如果钩子工厂返回 `None`，会记录一条警告，该规范不会注册任何钩子（初始化继续进行）。

前两种情况会导致初始化快速失败并给出描述性错误；最后一种是非致命的。

#### `config`（可选）

* 任意 JSON 对象。
* 作为 Python `dict` 直接传递给钩子工厂。
* 这允许你通过配置参数化钩子行为（例如标签、日志级别、采样率等）。

---

### 钩子生命周期和行为

钩子在 `ModelRunner.initialize()` 中注册：

```python
if server_args.forward_hooks:
    register_forward_hooks(self.model, server_args.forward_hooks)
```

实际的注册逻辑由 `register_forward_hooks` 实现：

```python
def register_forward_hooks(model: nn.Module, hook_specs: List[dict[str, Any]]) -> None:
    """
    hook_specs is a list of dicts from server_args.forward_hooks.
    Attaches forward hooks to the matching modules.
    """
    name_to_module = dict(model.named_modules())

    for spec in hook_specs:
        spec_name = spec.get("name", "")
        target_patterns = spec.get("target_modules", [])
        if not target_patterns:
            logger.warning(
                f"Hook spec '{spec_name}' has no 'target_modules', skipping"
            )
            continue

        hook_factory_path = spec.get("hook_factory")
        if not hook_factory_path:
            logger.warning(
                f"Hook spec '{spec_name}' has no 'hook_factory', skipping"
            )
            continue

        config = spec.get("config") or {}
        hook_factory = resolve_callable(hook_factory_path)

        hook = hook_factory(config) if hook_factory else None
        if hook is None:
            logger.warning(
                f"Hook factory '{hook_factory_path}' for spec '{spec_name}' "
                "returned None, not registering any hook"
            )
            continue

        # Resolve patterns like "model.layers.*.mlp"
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec_name}' "
                f"patterns={target_patterns}"
            )
            continue

        for module_name, module in matched:
            if hook:
                _ = module.register_forward_hook(hook)
                logger.info(
                    f"Registered forward hook '{spec_name}' "
                    f"on {module_name}"
                )
```

要点：

* 钩子仅为**前向钩子**（通过 `module.register_forward_hook`）。
* 它们在初始化时一次性附加。
* 钩子句柄当前未存储在 `ModelRunner` 上（无法通过此 API 在后续移除）。
* 未匹配到任何模块是非致命的；会记录一条警告。
* 如果钩子工厂返回 `None`，会记录警告并跳过该规范。

---

### 编写钩子工厂

钩子工厂是一个标准的 Python 函数：

* 接受一个 `config: dict` 参数（来自 JSON）
* 返回一个签名为 `(module, inputs, output)` 的前向钩子函数

示例：

```python
HOOK_CALLS = []

def dummy_hook_factory(config):
    """Factory that returns a forward hook capturing a tag from config."""
    tag = config.get("tag", "default")

    def hook(module, inputs, output):
        HOOK_CALLS.append(
            {
                "module_type": type(module).__name__,
                "tag": tag,
                "shape": tuple(output.shape),
            }
        )
        return output  # must return output if you don't want to modify the tensor

    return hook
```

JSON 配置：

```jsonc
{
  "forward_hooks": [
    {
      "name": "capture_outer",
      "target_modules": ["outer.0", "outer.1"],
      "hook_factory": "my_project.hooks:dummy_hook_factory",
      "config": {
        "tag": "outer"
      }
    }
  ]
}
```

这将会：

* 将 `my_project.hooks:dummy_hook_factory` 解析为一个 Python 可调用对象。
* 使用 `config = {"tag": "outer"}` 调用它。
* 将返回的钩子用于所有匹配 `outer.0` 和 `outer.1` 的模块。
* 将每次调用的元数据追加到 `HOOK_CALLS` 中。

---

### 总结

* 在 `ServerArgs` 中将 `forward_hooks` 定义为规范列表即可启用该功能。

* 每个规范：

  * 通过 `target_modules`（对 `model.named_modules()` 的 glob 模式）选择模块，
  * 通过 `hook_factory` 指向钩子工厂，
  * 将任意 `config` 传递给工厂。

* 钩子工厂通过 `resolve_callable` 解析，支持 `module:factory` 和 `module.submodule.factory` 格式。

* 钩子是标准的 PyTorch 前向钩子，在启动时一次性附加，并在每次前向传播时调用。

* 配置错误的处理方式：

  * **致命且明确**（路径错误/属性缺失），或
  * **非致命但有清晰警告**（无目标匹配，或工厂返回 `None`）。
