# 生产环境监控指标

SGLang 通过 Prometheus 暴露以下指标。你可以在启动服务器时添加 `--enable-metrics` 来启用监控指标。

监控仪表盘示例可在 [examples/monitoring/grafana.json](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json) 中查看。

以下是指标示例：

```
$ curl http://localhost:30000/metrics
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.128902e+06
# HELP sglang:generation_tokens_total Number of generation tokens processed.
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.557572e+06
# HELP sglang:token_usage The token usage
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.28
# HELP sglang:cache_hit_rate The cache hit rate
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.007507552643049313
# HELP sglang:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2.3518979474117756e+06
sglang:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.0
sglang:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.25",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:time_to_first_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 27.0
sglang:time_to_first_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
sglang:time_to_first_token_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 314.0
sglang:time_to_first_token_seconds_bucket{le="7.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 941.0
sglang:time_to_first_token_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1330.0
sglang:time_to_first_token_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1970.0
sglang:time_to_first_token_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2326.0
sglang:time_to_first_token_seconds_bucket{le="25.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2417.0
sglang:time_to_first_token_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2513.0
sglang:time_to_first_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
sglang:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11008.0
# HELP sglang:e2e_request_latency_seconds Histogram of End-to-end request latency in seconds
# TYPE sglang:e2e_request_latency_seconds histogram
sglang:e2e_request_latency_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 3.116093850019932e+06
sglang:e2e_request_latency_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
sglang:e2e_request_latency_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="0.8",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="1.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 6.0
sglang:e2e_request_latency_seconds_bucket{le="5.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.0
sglang:e2e_request_latency_seconds_bucket{le="10.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 10.0
sglang:e2e_request_latency_seconds_bucket{le="15.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11.0
sglang:e2e_request_latency_seconds_bucket{le="20.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 14.0
sglang:e2e_request_latency_seconds_bucket{le="30.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 247.0
sglang:e2e_request_latency_seconds_bucket{le="40.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 486.0
sglang:e2e_request_latency_seconds_bucket{le="50.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 845.0
sglang:e2e_request_latency_seconds_bucket{le="60.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1513.0
sglang:e2e_request_latency_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
sglang:e2e_request_latency_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 11228.0
# HELP sglang:time_per_output_token_seconds Histogram of time per output token in seconds.
# TYPE sglang:time_per_output_token_seconds histogram
sglang:time_per_output_token_seconds_sum{model_name="meta-llama/Llama-3.1-8B-Instruct"} 866964.5791549598
sglang:time_per_output_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
sglang:time_per_output_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 73.0
sglang:time_per_output_token_seconds_bucket{le="0.015",model_name="meta-llama/Llama-3.1-8B-Instruct"} 382.0
sglang:time_per_output_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 593.0
sglang:time_per_output_token_seconds_bucket{le="0.025",model_name="meta-llama/Llama-3.1-8B-Instruct"} 855.0
sglang:time_per_output_token_seconds_bucket{le="0.03",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1035.0
sglang:time_per_output_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1815.0
sglang:time_per_output_token_seconds_bucket{le="0.05",model_name="meta-llama/Llama-3.1-8B-Instruct"} 11685.0
sglang:time_per_output_token_seconds_bucket{le="0.075",model_name="meta-llama/Llama-3.1-8B-Instruct"} 433413.0
sglang:time_per_output_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.950195e+06
sglang:time_per_output_token_seconds_bucket{le="0.15",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.039435e+06
sglang:time_per_output_token_seconds_bucket{le="0.2",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.171662e+06
sglang:time_per_output_token_seconds_bucket{le="0.3",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.266055e+06
sglang:time_per_output_token_seconds_bucket{le="0.4",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.296752e+06
sglang:time_per_output_token_seconds_bucket{le="0.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.312226e+06
sglang:time_per_output_token_seconds_bucket{le="0.75",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.339675e+06
sglang:time_per_output_token_seconds_bucket{le="1.0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.357747e+06
sglang:time_per_output_token_seconds_bucket{le="2.5",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.389414e+06
sglang:time_per_output_token_seconds_bucket{le="+Inf",model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
sglang:time_per_output_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 7.400757e+06
# HELP sglang:func_latency_seconds Function latency in seconds
# TYPE sglang:func_latency_seconds histogram
sglang:func_latency_seconds_sum{name="generate_request"} 4.514771912145079
sglang:func_latency_seconds_bucket{le="0.05",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.07500000000000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.1125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.16875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.253125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.3796875",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.56953125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="0.8542968750000001",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.2814453125",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="1.9221679687500002",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="2.8832519531250003",name="generate_request"} 14006.0
sglang:func_latency_seconds_bucket{le="4.3248779296875",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="6.487316894531251",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="9.730975341796876",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="14.596463012695313",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="21.89469451904297",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="32.84204177856446",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="49.26306266784668",name="generate_request"} 14007.0
sglang:func_latency_seconds_bucket{le="+Inf",name="generate_request"} 14007.0
sglang:func_latency_seconds_count{name="generate_request"} 14007.0
# HELP sglang:num_running_reqs The number of running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 162.0
# HELP sglang:num_used_tokens The number of used tokens
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{model_name="meta-llama/Llama-3.1-8B-Instruct"} 123859.0
# HELP sglang:gen_throughput The generate throughput (token/s)
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="meta-llama/Llama-3.1-8B-Instruct"} 86.50814177726902
# HELP sglang:num_queue_reqs The number of requests in the waiting queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="meta-llama/Llama-3.1-8B-Instruct"} 2826.0
```

## 设置指南

本节介绍如何设置 `examples/monitoring` 目录中提供的监控栈（Prometheus + Grafana）。

### 前置条件

- 已安装 Docker 和 Docker Compose
- SGLang 服务器已启动并启用指标

### 使用方法

1.  **启动 SGLang 服务器并启用指标：**

    ```bash
    python -m sglang.launch_server \
      --model-path <your_model_path> \
      --port 30000 \
      --enable-metrics
    ```
    将 `<your_model_path>` 替换为你实际的模型路径（例如 `meta-llama/Meta-Llama-3.1-8B-Instruct`）。确保从监控栈可以访问到服务器（如果在 Docker 中运行，可能需要 `--host 0.0.0.0`）。默认情况下，指标端点将在 `http://<sglang_server_host>:30000/metrics` 上可用。

2.  **进入监控示例目录：**
    ```bash
    cd examples/monitoring
    ```

3.  **启动监控栈：**
    ```bash
    docker compose up -d
    ```
    此命令将在后台启动 Prometheus 和 Grafana。

4.  **访问监控界面：**
    *   **Grafana：** 打开浏览器访问 [http://localhost:3000](http://localhost:3000)。
    *   **Prometheus：** 打开浏览器访问 [http://localhost:9090](http://localhost:9090)。

5.  **登录 Grafana：**
    *   默认用户名：`admin`
    *   默认密码：`admin`
    首次登录时会提示更改密码。

6.  **查看仪表盘：**
    SGLang 仪表盘已预配置，应该会自动可用。导航至 `Dashboards` -> `Browse` -> `SGLang Monitoring` 文件夹 -> `SGLang Dashboard`。

### 故障排除

*   **端口冲突：** 如果遇到 "port is already allocated" 等错误，请检查是否有其他服务（包括之前的 Prometheus/Grafana 实例）正在使用端口 `9090` 或 `3000`。使用 `docker ps` 查找运行中的容器，使用 `docker stop <container_id>` 停止它们，或使用 `lsof -i :<port>` 查找使用该端口的其他进程。如果端口与系统上其他必要服务永久冲突，可能需要修改 `docker-compose.yaml` 文件中的端口。

要在 Docker Compose 文件中将 Grafana 的端口改为其他端口（如 3090），需要在 grafana 服务下明确指定端口映射。

    选项 1：在 environment 部分添加 GF_SERVER_HTTP_PORT：
    ```
      environment:
    - GF_AUTH_ANONYMOUS_ENABLED=true
    - GF_SERVER_HTTP_PORT=3090  # <-- 添加此行
    ```
    选项 2：使用端口映射：
    ```
    grafana:
      image: grafana/grafana:latest
      container_name: grafana
      ports:
      - "3090:3000"  # <-- 主机:容器 端口映射
    ```
*   **连接问题：**
    *   确保 Prometheus 和 Grafana 容器都在运行（`docker ps`）。
    *   验证 Grafana 中的 Prometheus 数据源配置（通常通过 `grafana/datasources/datasource.yaml` 自动配置）。前往 `Connections` -> `Data sources` -> `Prometheus`。URL 应指向 Prometheus 服务（例如 `http://prometheus:9090`）。
    *   确认 SGLang 服务器正在运行，且指标端点（`http://<sglang_server_host>:30000/metrics`）可从 Prometheus 容器*内部*访问。如果 SGLang 在主机上运行而 Prometheus 在 Docker 中，请在 `prometheus.yaml` 抓取配置中使用 `host.docker.internal`（Docker Desktop 上）或机器的网络 IP 代替 `localhost`。
*   **仪表盘无数据：**
    *   向 SGLang 服务器发送一些流量以生成指标。例如，运行基准测试：
        ```bash
        python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 100 --random-input 128 --random-output 128
        ```
    *   在 Prometheus UI（`http://localhost:9090`）中检查 `Status` -> `Targets`，查看 SGLang 端点是否成功被抓取。
    *   验证 Prometheus 指标中的 `model_name` 和 `instance` 标签是否与 Grafana 仪表盘中使用的变量匹配。可能需要调整 Grafana 仪表盘变量或 Prometheus 配置中的标签。

### 配置文件

监控设置由 `examples/monitoring` 目录中的以下文件定义：

*   `docker-compose.yaml`：定义 Prometheus 和 Grafana 服务。
*   `prometheus.yaml`：Prometheus 配置，包括抓取目标。
*   `grafana/datasources/datasource.yaml`：为 Grafana 配置 Prometheus 数据源。
*   `grafana/dashboards/config/dashboard.yaml`：告诉 Grafana 从指定路径加载仪表盘。
*   `grafana/dashboards/json/sglang-dashboard.json`：JSON 格式的实际 Grafana 仪表盘定义。

你可以通过修改这些文件来自定义设置。例如，如果 SGLang 服务器运行在不同的主机或端口上，可能需要更新 `prometheus.yaml` 中的 `static_configs` 目标。

#### 检查指标是否正在采集

运行：
```
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 3000 \
  --random-input 1024 \
  --random-output 1024 \
  --random-range-ratio 0.5
```

生成一些请求。

然后你应该可以在 Grafana 仪表盘中看到指标数据。
