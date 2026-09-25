#!/bin/bash
#SBATCH --job-name=smolicl-libero-1ep
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

set -euo pipefail

# 在 unseen-task v2 Manifest 上完整消费一遍 train/query，并在 epoch 末完整验证
# val/query。默认配置面向单节点 8×V100；每张卡的 batch size 和 Local Vision
# microbatch 相互独立，后者只控制一次送入 SigLIP 的图像数。
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/SmolICL}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH_ROOT}/LIBERO/lerobot_libero}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH_ROOT}/SmolICL_artifacts/libero_unseen_v2}"
PYTHON_ENV="${PYTHON_ENV:-${SCRATCH_ROOT}/SmolVLAi/env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/SmolICL_runs/libero_one_epoch}"

DATASET_REVISION="${DATASET_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOCAL_VISION_BATCH_SIZE="${LOCAL_VISION_BATCH_SIZE:-8}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
# 当前 v2 manifest 的 train/query 共有 109662 帧。ceil(109662 / (8*2)) = 6854，
# 最后一个分布式 batch 只含少量 sampler padding；不会开始第二个数据 epoch。
TRAIN_STEPS="${TRAIN_STEPS:-6854}"

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

cd "${REPO_ROOT}/lerobot"

# V100 使用 FP16 AMP；模型参数保留 FP32。Local raw RGB 始终留在 pinned CPU
# uint8 内存，只有当前 microbatch 才 non-blocking 搬到各自 GPU。前五步审计
# 所有可训练分支的梯度，之后关闭扫描开销。
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
  --gradient_audit_steps=5 \
  --eval_steps="${TRAIN_STEPS}" \
  --max_eval_samples=0 \
  --env_eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=0 \
  --wandb.enable=false \
  --accelerator.mixed_precision=fp16 \
  --accelerator.gradient_accumulation.steps=1 \
  --parallelism.dp_replicate="${NUM_GPUS}" \
  --parallelism.dp_shard=1 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolicl_libero_one_epoch

echo "SmolVLA-ICL one-epoch train/validation completed: ${OUTPUT_DIR}"
