#!/bin/bash

# ============================================================
# RecSelfEvolve Agent 启动脚本
# ============================================================

# ---- 基础配置 ----
PROJECT_ROOT="/home/cuihongxi/Rec_SelEnhance"
DATA_NAME="Beauty"
BACKBONE="SASRec"
GPU="0"

# ---- LLM 配置 (根据你的环境修改) ----
# LLM_URL="http://localhost:8000/v1"
# LLM_KEY="EMPTY"
# LLM_MODEL="DeepSeek-R1"

LLM_URL="http://10.82.123.22:8000/v1"
LLM_KEY="EMPTY"
LLM_MODEL="/share/cuihongxi/Qwen-235B-A22B-Instruct-2507"

# ---- 训练参数 ----
LR="0.001"
BATCH_SIZE="1024"
HIDDEN_SIZE="64"
NEG_SAMPLER="DNS"
LOSS_TYPE="BCE"
CL_TYPE="Radical"
N="200"
M="10"
EPOCHS="200"

# ---- 进化控制 ----
ITERATIONS="20"
STRATEGY="balanced"
TEMPERATURE="0.7"
TIMEOUT="7200"

Workspace="Recmodel_Relax"
# ---- 日志配置 ----
LOG_DIR="/home/cuihongxi/Rec_SelEnhance/agent_run_context_relax"
LOG_FILE="${LOG_DIR}/evolve_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 打印配置信息
echo "=========================================="
echo "RecSelfEvolve Agent 启动配置"
echo "=========================================="
echo "项目路径: ${PROJECT_ROOT}"
echo "数据集:   ${DATA_NAME}"
echo "模型:     ${BACKBONE}"
echo "GPU:      ${GPU}"
echo "LLM:      ${LLM_URL} [${LLM_MODEL}]"
echo "迭代次数: ${ITERATIONS}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="

# 启动进化
cd ${PROJECT_ROOT}

python run_evolve.py \
    --project "${PROJECT_ROOT}/${Workspace}" \
    --data "${DATA_NAME}" \
    --backbone "${BACKBONE}" \
    --gpu "${GPU}" \
    --llm-url "${LLM_URL}" \
    --llm-key "${LLM_KEY}" \
    --llm-model "${LLM_MODEL}" \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --hidden-size ${HIDDEN_SIZE} \
    --neg-sampler "${NEG_SAMPLER}" \
    --loss-type "${LOSS_TYPE}" \
    --cl-type "${CL_TYPE}" \
    --N ${N} \
    --M ${M} \
    --epochs ${EPOCHS} \
    --iterations ${ITERATIONS} \
    --strategy "${STRATEGY}" \
    --temperature ${TEMPERATURE} \
    --timeout ${TIMEOUT} \
    --log-dir "${LOG_DIR}" \
    2>&1 | tee ${LOG_FILE}

# 运行结束
echo "运行结束，日志: ${LOG_FILE}"
