#!/bin/bash

# ============================================================
# Qwen VLLM 服务启动脚本
# ============================================================

# 配置参数
MODEL_PATH="/share/cuihongxi/Qwen-235B-A22B-Instruct-2507"
HOST="0.0.0.0"
PORT="8000"
GPU_MEMORY_UTIL="0.85"
MAX_MODEL_LEN="16384"
TENSOR_PARALLEL_SIZE="8"
DTYPE="auto"

# 日志配置
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/vllm_qwen_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 打印配置信息
echo "=========================================="
echo "VLLM Qwen 服务配置"
echo "=========================================="
echo "模型路径: ${MODEL_PATH}"
echo "服务地址: ${HOST}:${PORT}"
echo "GPU显存利用率: ${GPU_MEMORY_UTIL}"
echo "最大上下文长度: ${MAX_MODEL_LEN}"
echo "Tensor并行: ${TENSOR_PARALLEL_SIZE}"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# 启动 VLLM 服务
echo "正在启动 VLLM 服务..."

vllm serve ${MODEL_PATH} \
    --dtype ${DTYPE} \
    --host ${HOST} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    2>&1 | tee ${LOG_FILE}

# 服务停止时的处理
echo "VLLM 服务已停止"