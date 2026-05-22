#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ———————— 可配置项 ————————
# 模型与数据集
MODELS=("SASRec")
DATASETS=("Beauty")

# start_epoch 从 0 到 70
START_EPOCHS=(0)

# 负采样器、对比学习类型等超参
NEG_SAMPLER="Uniform"
CL_TYPE="Radical"
HIDDEN_SIZE=64
N=100
M=(100)
LOSS_TYPE="InfoNCE"
TEMPERATURE=0.1


JOBS_PER_GPU=2

AVAILABLE_GPUS=(2)
NUM_GPUS=${#AVAILABLE_GPUS[@]}
if (( NUM_GPUS == 0 )); then
  echo "Error: 请在脚本中手动设置 AVAILABLE_GPUS，至少一个 GPU ID！"
  exit 1
fi
TOTAL_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))

echo "使用手动指定的 GPU 列表: ${AVAILABLE_GPUS[*]}"
echo "每 GPU 并发: $JOBS_PER_GPU，总并发: $TOTAL_SLOTS"

# 输出目录
OUTDIR="Test"
mkdir -p "$OUTDIR"

# ———————— 子进程 PID 记录 ————————
PIDS=()

# ———————— 并发控制函数 ————————
running_jobs() {
  jobs -rp | wc -l
}
wait_for_slot() {
  while (( $(running_jobs) >= TOTAL_SLOTS )); do
    wait -n || :
  done
}

# 轮询分配 GPU：循环轮询可用 slots，配合 round-robin
GPU_INDEX=0

# ———————— 退出/中断处理函数 ————————
cleanup() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在强制清理所有子进程..."
  # 先发送 SIGTERM
  kill "${PIDS[@]}" 2>/dev/null || true
  # 等待 1 秒，若仍有进程存活则发送 SIGKILL
  sleep 1
  kill -9 "${PIDS[@]}" 2>/dev/null || true
  exit 1
}

# 捕获更多信号（SIGINT: Ctrl+C, SIGTERM, SIGQUIT: Ctrl+\）
trap cleanup SIGINT SIGTERM SIGQUIT

# ———————— 主循环 ————————
for start_epoch in "${START_EPOCHS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for m in "${M[@]}"; do  # 新增扫描 M 的循环
        wait_for_slot
        gpu="${AVAILABLE_GPUS[GPU_INDEX]}"
        GPU_INDEX=$(((GPU_INDEX + 1) % NUM_GPUS))
        logfile="$OUTDIR/${ds}-${model}_m${m}.log"
        echo "[$(date '+%H:%M:%S')] 启动 $model@$ds m=$m → GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_finetune_full.py \
          --data_name="$ds" \
          --ckp=0 \
          --hidden_size=64 \
          --start_epoch="$start_epoch" \
          --loss_type="$LOSS_TYPE" \
          --temperature="$TEMPERATURE" \
          --N="$N" \
          --M="$m" \
          --neg_sampler="$NEG_SAMPLER" \
          --CL_type="$CL_TYPE" \
          --backbone="$model" \
        > "$logfile" 2>&1 &
        
        # 记录 PID
        PIDS+=($!)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志输出到 $logfile"
      done
    done
  done
done

# 等待所有子进程结束
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部任务完成！"