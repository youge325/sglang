# 多节点部署

## Llama 3.1 405B

**在两个节点上运行 405B（fp16）**

```bash
# 将 172.16.4.52:20000 替换为你自己的第一个节点的 IP 地址和端口

python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 0

python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 1
```

注意，Llama 405B（fp8）也可以在单个节点上启动。

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

## DeepSeek V3/R1

请参阅 [DeepSeek 参考文档](https://docs.sglang.io/basic_usage/deepseek.html#running-examples-on-multi-node)。

## 在 SLURM 上进行多节点推理

本示例展示了如何通过 SLURM 在多个节点上部署 SGLang 服务器。将以下作业提交到 SLURM 集群。

```
#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1  # 确保每个节点 1 个任务
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB
#SBATCH --partition="lmsys.org"
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

echo "[INFO] Activating environment on node $SLURM_PROCID"
if ! source ENV_FOLDER/bin/activate; then
    echo "[ERROR] Failed to activate environment" >&2
    exit 1
fi

# 定义参数
model=MODEL_PATH
tp_size=16

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# 使用头节点的主机名设置 NCCL 初始化地址
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# 使用 SLURM 在每个节点上启动模型服务器
srun --ntasks=2 --nodes=2 --output="SLURM_Logs/%x_%j_node$SLURM_NODEID.out" \
    --error="SLURM_Logs/%x_%j_node$SLURM_NODEID.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank "$SLURM_NODEID" &

# 等待 NCCL 服务器在端口 30000 上就绪
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done

echo "[INFO] $HEAD_NODE:30000 is ready to accept connections"

# 保持脚本运行直到 SLURM 作业超时
wait
```

然后，你可以按照其他[文档](https://docs.sglang.io/basic_usage/openai_api_completions.html)发送请求来测试服务器。

感谢 [aflah02](https://github.com/aflah02) 提供此示例，基于他的[博客文章](https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang)。
