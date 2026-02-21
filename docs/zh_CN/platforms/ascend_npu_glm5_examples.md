# GLM-5

## 简介

GLM（General Language Model）系列是由清华大学 KEG 实验室和智谱 AI 联合开发的开源双语大语言模型家族。该系列模型凭借其独特的统一预训练框架和双语能力，在中文 NLP 领域表现出色。[GLM-5](https://huggingface.co/zai-org/GLM-5) 采用 DeepSeek-V3/V3.2 架构，包括稀疏注意力（DSA）和多 token 预测（MTP）。Ascend 基于 SGLang 推理框架实现了 GLM-5 的 0Day 支持，实现了低代码无缝适配，并兼容当前 SGLang 框架内的主流分布式并行能力。欢迎开发者下载体验。

## 环境准备

### 模型权重

- `GLM-5.0`（BF16 版本）：[下载模型权重](https://www.modelscope.cn/models/ZhipuAI/GLM-5)。
- `GLM-5.0-w4a8`（量化版本，不含 MTP）：[下载模型权重](https://modelers.cn/models/Eco-Tech/GLM-5-w4a8)。
- 您可以使用 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 对模型进行原生量化。


### 安装

NPU 运行环境所需的依赖已集成到 Docker 镜像中并上传至 quay.io 平台。您可以直接拉取。

```{code-block} bash
#Atlas 800 A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-glm5
#Atlas 800 A2
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-910b-glm5

#启动容器
docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0  \
--device=/dev/davinci1:/dev/avinci1  \
--device=/dev/davinci2:/dev/davinci2  \
--device=/dev/davinci3:/dev/davinci3  \
--device=/dev/davinci4:/dev/davinci4  \
--device=/dev/davinci5:/dev/davinci5  \
--device=/dev/davinci6:/dev/davinci6  \
--device=/dev/davinci7:/dev/davinci7  \
--device=/dev/davinci8:/dev/davinci8  \
--device=/dev/davinci9:/dev/davinci9  \
--device=/dev/davinci10:/dev/davinci10  \
--device=/dev/davinci11:/dev/davinci11  \
--device=/dev/davinci12:/dev/davinci12  \
--device=/dev/davinci13:/dev/davinci13  \
--device=/dev/davinci14:/dev/davinci14  \
--device=/dev/davinci15:/dev/davinci15  \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:${TAG}
```

注意：使用此镜像时，您需要将 transformers 更新到 main 分支
``` shell
# 重新安装 transformers
pip install git+https://github.com/huggingface/transformers.git
```

## 部署

### 单节点部署

- 量化模型 `glm5_w4a8` 可以部署在 1 台 Atlas 800 A3（64G × 16）上。

运行以下脚本执行在线推理。

```shell
# 高性能 CPU 设置
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑定 CPU
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# CANN 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size 16384 --max-prefill-tokens 280000 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.7 \
        --port 8000 \
        --served-model-name glm-5 \
        --cuda-graph-bs 16 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto
```

### 多节点部署

- `GLM-5-bf16`：至少需要 2 台 Atlas 800 A3（64G × 16）。

**A3 系列**

修改 2 个节点的 IP，然后在两个节点上运行相同的脚本。

**node 0/1**

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# 绑定 CPU
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# CANN 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV

# 在两个节点上运行 ifconfig 命令,找到与您的节点 IP 相同的 inet addr。这是您的公网接口，需要在此处添加
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo


P_IP=('your ip1' 'your ip2')
P_MASTER="${P_IP[0]}:your port"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "${P_IP[$i]}"
        python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 32 --nnodes 2 --node-rank $i --dist-init-addr $P_MASTER \
        --chunked-prefill-size 16384 --max-prefill-tokens 131072 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.8\
        --port 8000 \
        --served-model-name glm-5 \
        --cuda-graph-max-bs 16 \
        --disable-radix-cache
        NODE_RANK=$i
        break
    fi
done

```

### Prefill-Decode 分离部署

尚未测试。

### 使用基准测试

详情请参阅[基准测试与性能分析](../developer_guide/benchmark_and_profiling.md)。
