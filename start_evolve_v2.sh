#!/bin/bash

# ============================================================
# RecSelfEvolve V2 — DeepEvolve-inspired 进化引擎启动脚本
#
# 相比 V1 的改进:
#   - Researcher + Coder Agent 分离
#   - Island-based Evolution 多岛并行进化
#   - Deep Research 网络搜索
#   - Reflection 反思机制
#   - Program Database 版本管理
# ============================================================

# ---- 基础配置 ----
PROJECT_ROOT="/home/cuihongxi/Rec_SelEnhance"
DATA_NAME="Beauty"
BACKBONE="SASRec"
GPU="0"

# ---- LLM 配置 (根据你的环境修改) ----
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
TIMEOUT="7200"
TEMPERATURE_RESEARCHER="0.7"
TEMPERATURE_CODER="0.4"

Workspace="Recmodel_Relax"

# ---- V2 新增: Deep Research 配置 ----
MAX_RESEARCH_REFLECT="3"
SEARCH_TIME_BIAS="0.5"
# DISABLE_SEARCH=""  # 如果不需要网络搜索，取消注释此行

# ---- V2 新增: Island Evolution 配置 ----
NUM_ISLANDS="4"
POPULATION_SIZE="20"
MIGRATION_INTERVAL="5"
MIGRATION_RATE="0.1"

# ---- V2 新增: Coder 反思配置 ----
MAX_CODING_REFLECT="3"

# ---- V2 新增: Checkpoint 配置 ----
CHECKPOINT_INTERVAL="5"
# RESUME_FROM=""  # 如果要从 checkpoint 恢复，取消注释并指定路径

# ---- 惊喜评估配置 ----
SURPRISE_TOPK="20"
NUM_WRONG_CASES="500"
NUM_TRAIN_SUBSET="500"
SURPRISE_THRESHOLD="0.5"

# ---- 日志配置 ----
LOG_DIR="/home/cuihongxi/Rec_SelEnhance/agent_run_context_v2_relax"
LOG_FILE="${LOG_DIR}/evolve_v2_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 打印配置信息
echo "=========================================="
echo "RecSelfEvolve V2 — DeepEvolve-inspired 启动配置"
echo "=========================================="
echo "项目路径:     ${PROJECT_ROOT}/${Workspace}"
echo "数据集:       ${DATA_NAME}"
echo "模型:         ${BACKBONE}"
echo "GPU:          ${GPU}"
echo "LLM:          ${LLM_URL} [${LLM_MODEL}]"
echo "迭代次数:     ${ITERATIONS}"
echo "=========================================="
echo "V2 新增配置:"
echo "Islands:      ${NUM_ISLANDS}"
echo "Population:   ${POPULATION_SIZE}"
echo "Migration:    every ${MIGRATION_INTERVAL} gen, rate ${MIGRATION_RATE}"
echo "Researcher:   temp=${TEMPERATURE_RESEARCHER}, reflect=${MAX_RESEARCH_REFLECT}"
echo "Coder:        temp=${TEMPERATURE_CODER}, reflect=${MAX_CODING_REFLECT}"
echo "Checkpoint:   every ${CHECKPOINT_INTERVAL} iterations"
echo "Deep Search:  ${DISABLE_SEARCH:-ENABLED}"
echo "=========================================="
echo "日志目录:     ${LOG_DIR}"
echo "=========================================="

# 启动进化
cd ${PROJECT_ROOT}

# 构建 DISABLE_SEARCH 参数
DISABLE_SEARCH_ARG=""
if [ -n "${DISABLE_SEARCH}" ]; then
    DISABLE_SEARCH_ARG="--disable-search"
fi

# 构建 RESUME_FROM 参数
RESUME_FROM_ARG=""
if [ -n "${RESUME_FROM}" ]; then
    RESUME_FROM_ARG="--resume-from ${RESUME_FROM}"
fi

python run_evolve_v2.py \
    --project "${PROJECT_ROOT}/${Workspace}" \
    --data "${DATA_NAME}" \
    --backbone "${BACKBONE}" \
    --gpu "${GPU}" \
    --llm-url "${LLM_URL}" \
    --llm-key "${LLM_KEY}" \
    --llm-model "${LLM_MODEL}" \
    --researcher-temp ${TEMPERATURE_RESEARCHER} \
    --coder-temp ${TEMPERATURE_CODER} \
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
    --timeout ${TIMEOUT} \
    --max-research-reflect ${MAX_RESEARCH_REFLECT} \
    --search-time-bias ${SEARCH_TIME_BIAS} \
    ${DISABLE_SEARCH_ARG} \
    --max-coding-reflect ${MAX_CODING_REFLECT} \
    --num-islands ${NUM_ISLANDS} \
    --population-size ${POPULATION_SIZE} \
    --migration-interval ${MIGRATION_INTERVAL} \
    --migration-rate ${MIGRATION_RATE} \
    --checkpoint-interval ${CHECKPOINT_INTERVAL} \
    ${RESUME_FROM_ARG} \
    --surprise-topk ${SURPRISE_TOPK} \
    --num-wrong-cases ${NUM_WRONG_CASES} \
    --num-train-subset ${NUM_TRAIN_SUBSET} \
    --surprise-threshold ${SURPRISE_THRESHOLD} \
    --log-dir "${LOG_DIR}" \
    2>&1 | tee ${LOG_FILE}

# 运行结束
echo "运行结束，日志: ${LOG_FILE}"