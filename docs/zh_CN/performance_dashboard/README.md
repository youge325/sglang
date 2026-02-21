# SGLang 性能仪表盘

一个用于可视化 SGLang 每夜测试性能指标的 Web 仪表盘。

## 功能特性

- **性能趋势**：查看吞吐量、延迟和 TTFT 随时间的变化趋势
- **模型对比**：比较不同模型和配置之间的性能
- **筛选过滤**：按 GPU 配置、模型、变体和批大小进行筛选
- **交互式图表**：支持缩放、平移和悬停查看详细指标
- **运行历史**：查看近期基准测试运行记录及 GitHub Actions 链接

## 快速开始

### 方式一：使用本地服务器运行（推荐）

从 GitHub Actions 产物获取实时数据：

```bash
# 安装依赖
pip install requests

# 运行服务器
python server.py --fetch-on-start

# 访问 http://localhost:8000
```

服务器提供以下功能：
- 自动从 GitHub 获取指标数据
- 缓存机制以减少 API 调用
- 为前端提供 `/api/metrics` 端点

### 方式二：手动获取数据

使用获取脚本下载指标数据：

```bash
# 获取最近 30 天的指标
python fetch_metrics.py --output metrics_data.json

# 获取特定运行的数据
python fetch_metrics.py --run-id 21338741812 --output single_run.json

# 仅获取计划任务（每夜）运行的数据
python fetch_metrics.py --scheduled-only --days 7
```

## GitHub 令牌

要从 GitHub 下载产物，需要进行身份验证：

1. **使用 `gh` CLI**（推荐）：
   ```bash
   gh auth login
   ```

2. **使用环境变量**：
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

如果没有令牌，仪表盘将显示运行元数据，但无法显示详细的基准测试结果。

## 数据结构

指标 JSON 具有以下结构：

```json
{
  "run_id": "21338741812",
  "run_date": "2026-01-25T22:24:02.090218+00:00",
  "commit_sha": "5cdb391...",
  "branch": "main",
  "results": [
    {
      "gpu_config": "8-gpu-h200",
      "partition": 0,
      "model": "deepseek-ai/DeepSeek-V3.1",
      "variant": "TP8+MTP",
      "benchmarks": [
        {
          "batch_size": 1,
          "input_len": 4096,
          "output_len": 512,
          "latency_ms": 2400.72,
          "input_throughput": 21408.64,
          "output_throughput": 231.74,
          "overall_throughput": 1919.43,
          "ttft_ms": 191.32,
          "acc_length": 3.19
        }
      ]
    }
  ]
}
```

## 部署

### GitHub Pages

仪表盘可以部署到 GitHub Pages 以供公开访问：

1. 将仪表盘文件复制到 `docs/performance_dashboard/`
2. 在仓库设置中启用 GitHub Pages
3. 设置 GitHub Action 定期更新指标数据

### 自托管

使用实时数据进行自托管部署：

1. 设置运行 `server.py` 的服务器
2. 配置 cron 任务或 systemd 定时器刷新数据
3. 可选择使用 nginx/caddy 进行反向代理以启用 SSL

## 指标说明

- **总吞吐量（Overall Throughput）**：每秒处理的总令牌数（输入 + 输出）
- **输入吞吐量（Input Throughput）**：每秒处理的输入令牌数（预填充速度）
- **输出吞吐量（Output Throughput）**：每秒生成的输出令牌数（解码速度）
- **延迟（Latency）**：完成请求的端到端时间
- **首令牌时间（TTFT）**：从请求开始到生成第一个输出令牌的时间
- **接受长度（Acc Length）**：推测解码的接受长度（MTP 变体）

## 贡献

要添加对新指标或可视化的支持：

1. 如果数据采集需要更改，请更新 `fetch_metrics.py`
2. 修改 `app.js` 以添加新的图表类型或筛选器
3. 更新 `index.html` 以进行 UI 更改

## 故障排除

**没有数据显示**
- 检查浏览器控制台是否有错误
- 确认 GitHub API 可以访问
- 尝试使用 `server.py --fetch-on-start` 运行

**API 速率限制**
- 使用 GitHub 令牌以获得更高的请求限额
- 服务器会缓存数据 5 分钟

**图表未渲染**
- 确保 Chart.js 可以从 CDN 加载
- 检查控制台是否有 JavaScript 错误
