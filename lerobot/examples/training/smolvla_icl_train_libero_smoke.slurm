#!/bin/bash
#SBATCH --job-name=smolicl-train-smoke
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

# 该作业只验证真实 LIBERO 数据链上的训练闭环：
# Dataset/preflight -> collate -> Query/Global/Local 编码 -> forward/backward -> optimizer step。
# 默认运行两步且不保存大模型 checkpoint，不用于报告训练指标。
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/SmolICL}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH_ROOT}/LIBERO/lerobot_libero}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH_ROOT}/SmolICL_artifacts/libero_unseen_v2}"
PYTHON_ENV="${PYTHON_ENV:-${SCRATCH_ROOT}/SmolVLAi/env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/SmolICL_runs/libero_smoke}"

DATASET_REVISION="${DATASET_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOCAL_VISION_BATCH_SIZE="${LOCAL_VISION_BATCH_SIZE:-2}"
# false：训练 Query/Local 共享 SigLIP；true：冻结视觉路径并自动跳过
# Local vision checkpoint。Matcher snapshot 始终冻结，不受此变量影响。
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
# V100 的 FP16 动态范围较窄。较低初始 scale 避免启动期反向溢出；延长
# growth interval 防止单个 epoch 内过早升回已知不稳定的 scale。反缩放后的
# 真实梯度、gradient clipping、optimizer 和 LR schedule 均保持不变。
FP16_INIT_SCALE="${FP16_INIT_SCALE:-4096}"
FP16_GROWTH_INTERVAL="${FP16_GROWTH_INTERVAL:-10000}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This smoke test must be submitted with sbatch." >&2
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
  echo "Smoke output already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${ARTIFACT_ROOT}/tmp"

export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot/src"
export HF_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/torch"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/xdg"
export TMPDIR="${ARTIFACT_ROOT}/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO_ROOT}/lerobot"

if (( NUM_GPUS > 1 )); then
  launcher=(torchrun --standalone --nproc-per-node="${NUM_GPUS}" -m lerobot.scripts.lerobot_train)
else
  launcher=(python -m lerobot.scripts.lerobot_train)
fi

# V100 没有原生 BF16，因此参数以 FP32 保存、训练计算使用 FP16 AMP。
# Local Demo 的 48 帧仍保留在 CPU，仅按两帧 microbatch 上传 GPU；这里不缓存
# visual token。是否让 Action Loss 更新共享视觉编码器由上方开关控制。
"${launcher[@]}" \
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
  --persistent_workers=false \
  --steps="${SMOKE_STEPS}" \
  --log_freq=1 \
  --gradient_audit_steps="${SMOKE_STEPS}" \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --accelerator.mixed_precision=fp16 \
  --accelerator.fp16_grad_scaler.init_scale="${FP16_INIT_SCALE}" \
  --accelerator.fp16_grad_scaler.growth_interval="${FP16_GROWTH_INTERVAL}" \
  --accelerator.gradient_accumulation.steps=1 \
  --parallelism.dp_replicate="${NUM_GPUS}" \
  --parallelism.dp_shard=1 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolicl_libero_smoke

echo "SmolVLA-ICL LIBERO training smoke passed: ${OUTPUT_DIR}"
