# 可观测性

## 生产环境指标
SGLang 通过 Prometheus 暴露以下指标。你可以在启动服务器时添加 `--enable-metrics` 来启用它们。
你可以通过以下方式查询：
```
curl http://localhost:30000/metrics
```

更多详情请参阅[生产环境指标](../references/production_metrics.md)和[生产环境请求追踪](../references/production_request_trace.md)。

## 日志

默认情况下，SGLang 不记录任何请求内容。你可以使用 `--log-requests` 来记录请求。
你可以使用 `--log-request-level` 来控制日志详细程度。
更多详情请参阅[日志](server_arguments.md#logging)。

## 请求转储和重放

你可以转储所有请求并在之后重放，用于基准测试或其他目的。

要开始转储，使用以下命令向服务器发送请求：
```
python3 -m sglang.srt.managers.configure_logging --url http://localhost:30000 --dump-requests-folder /tmp/sglang_request_dump --dump-requests-threshold 100
```
服务器将每 100 个请求将请求转储到一个 pickle 文件中。

要重放请求转储，使用 `scripts/playground/replay_request_dump.py`。

## 崩溃转储和重放
有时服务器可能会崩溃，你可能想要调试崩溃原因。
SGLang 支持崩溃转储，它会转储崩溃前 5 分钟内的所有请求，让你稍后可以重放请求并调试原因。

要启用崩溃转储，使用 `--crash-dump-folder /tmp/crash_dump`。
要重放崩溃转储，使用 `scripts/playground/replay_request_dump.py`。
