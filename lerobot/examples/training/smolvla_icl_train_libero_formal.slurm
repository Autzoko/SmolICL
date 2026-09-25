#!/bin/bash
#SBATCH --job-name=smolicl-libero-formal
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00

set -euo pipefail

# SmolVLA-ICL 第一次正式 LIBERO unseen-task 训练。
# 默认使用 4×V100、每卡 batch size 4，因此全局 batch size 为 16；这与已完成的
# 单 epoch 验证保持一致。冻结共享 SigLIP，仅训练 connector、Demo Expert、Action
# Expert 等下游模块，以控制训练时间和显存占用。
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/SmolICL}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH_ROOT}/LIBERO/lerobot_libero}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH_ROOT}/SmolICL_artifacts/libero_unseen_v2}"
PYTHON_ENV="${PYTHON_ENV:-${SCRATCH_ROOT}/SmolVLAi/env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/SmolICL_runs/libero_formal_frozen_siglip}"

DATASET_REVISION="${DATASET_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOCAL_VISION_BATCH_SIZE="${LOCAL_VISION_BATCH_SIZE:-8}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"

# 109662 个 train/query frame 在全局 batch size 16 下对应 6854 step/epoch。
# 默认训练 5 个 epoch，并在每个 epoch 边界执行完整 validation 和保存 checkpoint。
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-6854}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_STEPS="${TRAIN_STEPS:-$((STEPS_PER_EPOCH * TRAIN_EPOCHS))}"
EVAL_STEPS="${EVAL_STEPS:-${STEPS_PER_EPOCH}}"
SAVE_FREQ="${SAVE_FREQ:-${STEPS_PER_EPOCH}}"

# V100 采用 FP16 AMP。初始 scale=4096 已通过 50-step 梯度审计；正式训练把
# growth interval 拉长，避免 scale 很快增长回已知会 overflow 的 8192，同时仍保留
# GradScaler 遇到非有限梯度时自动 backoff 的标准行为。
FP16_INIT_SCALE="${FP16_INIT_SCALE:-4096}"
FP16_GROWTH_INTERVAL="${FP16_GROWTH_INTERVAL:-100000}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This training job must be submitted with sbatch." >&2
  exit 1
fi
if [[ "${SLURM_GPUS_ON_NODE:-${NUM_GPUS}}" != "${NUM_GPUS}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS} does not match SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/job_${SLURM_JOB_ID}}"
LOG_DIR="${ARTIFACT_ROOT}/logs"
required_paths=(
  "${REPO_ROOT}/.git"
  "${DATASET_ROOT}"
  "${PYTHON_ENV}/bin/python"
  "${ARTIFACT_ROOT}/manifest.json"
  "${ARTIFACT_ROOT}/train_stats.json"
  "${ARTIFACT_ROOT}/pairing_sidecar.json"
  "${ARTIFACT_ROOT}/global_demo_cache/cache_manifest.json"
  "${ARTIFACT_ROOT}/local_rgb_cache/cache_manifest.json"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Training output already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${ARTIFACT_ROOT}/tmp"

export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot/src"
export HF_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/torch"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/xdg"
export TMPDIR="${ARTIFACT_ROOT}/tmp"
export OMP_NUM_THREADS="$((SLURM_CPUS_PER_TASK / NUM_GPUS))"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting SmolVLA-ICL formal training"
echo "  job_id=${SLURM_JOB_ID}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  GPUs=${NUM_GPUS}, per_gpu_batch=${BATCH_SIZE}, global_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  epochs=${TRAIN_EPOCHS}, steps_per_epoch=${STEPS_PER_EPOCH}, total_steps=${TRAIN_STEPS}"
echo "  eval_steps=${EVAL_STEPS}, save_freq=${SAVE_FREQ}"
echo "  freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
echo "  fp16_init_scale=${FP16_INIT_SCALE}, fp16_growth_interval=${FP16_GROWTH_INTERVAL}"

cd "${REPO_ROOT}/lerobot"

# Local RGB 保持 CPU uint8，只按 microbatch 搬入 GPU。前十步审计所有可训练模块
# 的梯度；后续关闭逐参数扫描，避免给正式训练增加持续开销。
torchrun --standalone --nproc-per-node="${NUM_GPUS}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0 \
  --policy.type=smolvla_icl \
  --policy.pretrained_path="${BASE_MODEL}" \
  --policy.pretrained_revision="${BASE_MODEL_REVISION}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.n_action_steps=5 \
  --policy.vlm_load_dtype=float32 \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
  --policy.local_vision_encode_batch_size="${LOCAL_VISION_BATCH_SIZE}" \
  --policy.local_vision_gradient_checkpointing=true \
  --policy.data_manifest_path="${ARTIFACT_ROOT}/manifest.json" \
  --policy.training_stats_path="${ARTIFACT_ROOT}/train_stats.json" \
  --policy.pairing_sidecar_path="${ARTIFACT_ROOT}/pairing_sidecar.json" \
  --policy.training_demo_cache_dir="${ARTIFACT_ROOT}/global_demo_cache" \
  --policy.training_local_rgb_cache_dir="${ARTIFACT_ROOT}/local_rgb_cache" \
  --policy.demo_alignment.alignment_hz=2 \
  --policy.demo_alignment.window_duration_s=2 \
  --policy.demo_alignment.matching_state_excluded_indices='[-2,-1]' \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --prefetch_factor=2 \
  --persistent_workers=true \
  --steps="${TRAIN_STEPS}" \
  --log_freq=50 \
  --gradient_audit_steps=10 \
  --eval_steps="${EVAL_STEPS}" \
  --max_eval_samples=0 \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable=false \
  --accelerator.mixed_precision=fp16 \
  --accelerator.fp16_grad_scaler.init_scale="${FP16_INIT_SCALE}" \
  --accelerator.fp16_grad_scaler.growth_interval="${FP16_GROWTH_INTERVAL}" \
  --accelerator.gradient_accumulation.steps=1 \
  --parallelism.dp_replicate="${NUM_GPUS}" \
  --parallelism.dp_shard=1 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolicl_libero_formal

echo "SmolVLA-ICL formal train/validation completed: ${OUTPUT_DIR}"
