# 生产环境请求追踪

SGLang 基于 OpenTelemetry Collector 导出请求追踪数据。你可以在启动服务器时添加 `--enable-trace` 来启用追踪，并使用 `--otlp-traces-endpoint` 配置 OpenTelemetry Collector 端点。

可视化效果的示例截图请参见 https://github.com/sgl-project/sglang/issues/8965。

## 设置指南
本节介绍如何配置请求追踪并导出追踪数据。
1. 安装所需的包和工具
    * 安装 Docker 和 Docker Compose
    * 安装依赖
    ```bash
    # 进入 SGLang 根目录
    pip install -e "python[tracing]"

    # 或手动使用 pip 安装依赖
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp opentelemetry-exporter-otlp-proto-grpc
    ```

2. 启动 OpenTelemetry Collector 和 Jaeger
    ```bash
    docker compose -f examples/monitoring/tracing_compose.yaml up -d
    ```

3. 启动启用追踪的 SGLang 服务器
    ```bash
    # 设置环境变量
    export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
    export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
    # 启动 prefill 和 decode 服务器
    python -m sglang.launch_server --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    # 启动 mini lb
    python -m sglang_router.launch_router --enable-trace --otlp-traces-endpoint 0.0.0.0:4317 <other option>
    ```

    将 `0.0.0.0:4317` 替换为 OpenTelemetry Collector 的实际端点。如果你使用 tracing_compose.yaml 启动了 OpenTelemetry Collector，默认接收端口为 4317。

    要使用 HTTP/protobuf span 导出器，请设置以下环境变量并指向 HTTP 端点，例如 `http://0.0.0.0:4318/v1/traces`。
    ```bash
    export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
    ```


4. 发送一些请求
5. 观察追踪数据是否正在导出
    * 使用浏览器访问 Jaeger 的 16686 端口来可视化请求追踪。
    * OpenTelemetry Collector 还会将追踪数据以 JSON 格式导出到 /tmp/otel_trace.json。在后续的补丁中，我们将提供一个工具将此数据转换为 Perfetto 兼容格式，从而在 Perfetto UI 中可视化请求。

## 如何为你感兴趣的片段添加追踪？
我们已经在 tokenizer 和调度器主线程中插入了检测点。如果你希望追踪额外的请求执行片段或进行更细粒度的追踪，请使用下面描述的追踪包 API。

1. 初始化

    在初始化阶段，每个参与追踪的进程应执行：
    ```python
    process_tracing_init(otlp_traces_endpoint, server_name)
    ```
    otlp_traces_endpoint 从参数中获取，server_name 可以自由设置，但在所有进程之间应保持一致。

    在初始化阶段，每个参与追踪的线程应执行：
    ```python
    trace_set_thread_info("thread label", tp_rank, dp_rank)
    ```
    "thread label" 可以视为线程的名称，用于在可视化视图中区分不同的线程。

2. 标记请求的开始和结束
    ```
    trace_req_start(rid, bootstrap_room)
    trace_req_finish(rid)
    ```
    这两个 API 必须在同一进程中调用，例如在 tokenizer 中。

3. 为片段添加追踪

    * 正常添加片段追踪：
        ```python
        trace_slice_start("slice A", rid)
        trace_slice_end("slice A", rid)
        ```

    - 使用 "anonymous" 标志在片段开始时不指定片段名称，允许片段名称由 trace_slice_end 决定。
    <br>注意：匿名片段不能嵌套。
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid)
        ```

    - 在 trace_slice_end 中，使用 auto_next_anon 自动创建下一个匿名片段，可以减少所需的检测点数量。
        ```python
        trace_slice_start("", rid, anonymous = True)
        trace_slice_end("slice A", rid, auto_next_anon = True)
        trace_slice_end("slice B", rid, auto_next_anon = True)
        trace_slice_end("slice C", rid, auto_next_anon = True)
        trace_slice_end("slice D", rid)
        ```
    - 线程中最后一个片段的结束必须标记 thread_finish_flag=True；否则线程的 span 将无法正确生成。
        ```python
        trace_slice_end("slice D", rid, thread_finish_flag = True)
        ```

4. 当请求执行流转移到另一个线程时，需要显式传播追踪上下文。
    - 发送端：在通过 ZMQ 将请求发送到另一个线程之前执行以下代码
        ```python
        trace_context = trace_get_proc_propagate_context(rid)
        req.trace_context = trace_context
        ```
    - 接收端：在通过 ZMQ 接收到请求后执行以下代码
        ```python
        trace_set_proc_propagate_context(rid, req.trace_context)
        ```

5. 当请求执行流转移到另一个节点（PD 分离架构）时，需要显式传播追踪上下文。
    - 发送端：在通过 HTTP 将请求发送到节点线程之前执行以下代码
        ```python
        trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
        headers = {"trace_context": trace_context}
        session.post(url, headers=headers)
        ```
    - 接收端：在通过 HTTP 接收到请求后执行以下代码
        ```python
        trace_set_remote_propagate_context(request.headers['trace_context'])
        ```

## 如何扩展追踪框架以支持复杂的追踪场景

当前提供的追踪包仍有进一步开发的潜力。如果你希望在其基础上构建更高级的功能，必须首先理解其现有的设计原则。

追踪框架实现的核心在于 span 结构和追踪上下文的设计。为了聚合分散的片段并支持多个请求的并发追踪，我们设计了两级追踪上下文结构和四级 span 结构：`SglangTraceReqContext`、`SglangTraceThreadContext`。它们的关系如下：
```
SglangTraceReqContext (req_id="req-123")
├── SglangTraceThreadContext(thread_label="scheduler", tp_rank=0)
|
└── SglangTraceThreadContext(thread_label="scheduler", tp_rank=1)
```

每个被追踪的请求维护一个全局的 `SglangTraceReqContext`。对于每个处理该请求的线程，会记录一个对应的 `SglangTraceThreadContext` 并组合在 `SglangTraceReqContext` 中。在每个线程内，每个当前被追踪的片段（可能嵌套）存储在一个列表中。

除了上述层级关系外，每个片段还通过 Span.add_link() 记录其前一个片段，这可以用于追踪执行流。

当请求执行流转移到新线程时，需要显式传播追踪上下文。在框架中，这由 `SglangTracePropagateContext` 表示，它包含请求 span 和前一个片段 span 的上下文。


我们设计了四级 span 结构，由 `bootstrap_room_span`、`req_root_span`、`thread_span` 和 `slice_span` 组成。其中，`req_root_span` 和 `thread_span` 分别对应 `SglangTraceReqContext` 和 `SglangTraceThreadContext`，`slice_span` 存储在 `SglangTraceThreadContext` 中。`bootstrap_room_span` 的设计是为了适应 PD 分离架构的场景。在不同节点上，我们可能希望为 `req_root_span` 添加某些属性。然而，如果 `req_root_span` 在所有节点间共享，由于 OpenTelemetry 设计的约束，Prefill 和 Decode 节点将不被允许添加属性。

```
bootstrap room span
├── router req root span
|    └── router thread span
|          └── slice span
├── prefill req root span
|    ├── tokenizer thread span
|    |     └── slice span
|    └── scheduler thread span
|          └── slice span
└── decode req root span
      ├── tokenizer thread span
      |    └── slice span
      └── scheduler thread span
           └── slice span
```
