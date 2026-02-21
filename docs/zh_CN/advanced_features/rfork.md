# R-Fork

R-Fork（张量远程分叉）是一种新颖的权重加载方法，它利用高效的节点间 GPU 到 GPU 数据传输路径，将张量从运行中的 SGLang 实例零拷贝加载到新实例。它可以将模型权重加载时间从几分钟缩短到几秒钟，从而显著优化 SGLang 实例的启动时间。

了解更多关于 R-Fork 的详情，请查阅 **<a href=https://lmsys.org/blog/2025-12-10-rfork/> R-Fork 博客 </a>**

## 使用方法

| 参数     | 用途                                      |
|--------------|--------------------------------------------|
| load-format  | 设置为 `remote_instance` 以启用 R-Fork。 |
| remote-instance-weight-loader-backend | `nccl` 或 `transfer_engine`，默认值为 `nccl` |
| remote-instance-weight-loader-seed-instance-ip | 提供模型权重的种子实例的 IP 地址 |
| remote-instance-weight-loader-seed-instance-service-port | 种子实例 HTTP 服务器监听的端口 |
| remote-instance-weight-loader-send-weights-group-ports | 种子实例上用于在种子和客户端实例之间建立 NCCL 通信组的可用端口列表。此参数仅在 `nccl` 后端下需要。 |
| remote-instance-weight-loader-start-seed-via-transfer-engine | 设置此参数以启动支持 TransferEngine 后端的种子服务。在使用 `transfer_engine` 作为后端时，种子实例需要此参数。 |

### 以 NCCL 作为后端

种子实例：
```shell
python -m sglang.launch_server [args]
```

客户端实例：
```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-send-weights-group-ports [send_weights_nccl_group_ports_list]  \
  --remote-instance-weight-loader-backend nccl
```

### 以 TransferEngine 作为后端

种子实例：
```shell
python -m sglang.launch_server [args] \
  --remote-instance-weight-loader-start-seed-via-transfer-engine
```

```shell
python -m sglang.launch_server [args] \
  --load-format remote_instance \
  --remote-instance-weight-loader-seed-instance-ip [seed_instance_ip] \
  --remote-instance-weight-loader-seed-instance-service-port [seed_instance_service_port] \
  --remote-instance-weight-loader-backend transfer_engine
```
