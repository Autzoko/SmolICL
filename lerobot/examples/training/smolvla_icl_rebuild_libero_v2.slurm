#!/bin/bash
#SBATCH --job-name=smolicl-libero-v2
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00

set -euo pipefail

# 该作业在一个新的 artifact root 中严格按依赖顺序重建完整训练数据链。
# Builder 均支持复用身份一致的中间结果，因此作业中断后可用相同参数重提。
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
REPO_ROOT="${REPO_ROOT:-${SCRATCH_ROOT}/SmolICL}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH_ROOT}/LIBERO/lerobot_libero}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRATCH_ROOT}/SmolICL_artifacts/libero_unseen_v2}"
PYTHON_ENV="${PYTHON_ENV:-${SCRATCH_ROOT}/SmolVLAi/env}"

DATASET_REVISION="${DATASET_REVISION:-a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4}"
MATCHER_MODEL="${MATCHER_MODEL:-lerobot/smolvla_base}"
MATCHER_REVISION="${MATCHER_REVISION:-c83c3163b8ca9b7e67c509fffd9121e66cb96205}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"
QUERY_WINDOW_REPLANS="${QUERY_WINDOW_REPLANS:-4}"
SEED="${SEED:-42}"

GLOBAL_CONFIG="${GLOBAL_CONFIG:-${REPO_ROOT}/lerobot/examples/training/smolvla_icl_global_encoder.json}"
ALIGNMENT_CONFIG="${ALIGNMENT_CONFIG:-${REPO_ROOT}/lerobot/examples/training/smolvla_icl_libero_alignment.json}"

for required_path in \
  "${REPO_ROOT}/.git" \
  "${DATASET_ROOT}" \
  "${PYTHON_ENV}/bin/python" \
  "${GLOBAL_CONFIG}" \
  "${ALIGNMENT_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${ARTIFACT_ROOT}/logs" "${ARTIFACT_ROOT}/tmp"

export PATH="${PYTHON_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/lerobot/src"
export HF_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/SmolVLAi/cache/torch"
export TMPDIR="${ARTIFACT_ROOT}/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO_ROOT}/lerobot"

python -m lerobot.policies.smolvla_icl.data.libero_manifest \
  --dataset-root "${DATASET_ROOT}" \
  --output "${ARTIFACT_ROOT}/manifest.json" \
  --repo-id lerobot/libero \
  --revision "${DATASET_REVISION}" \
  --image-key observation.images.image \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --demo-ratio 0.2 \
  --seed "${SEED}"

python -m lerobot.policies.smolvla_icl.data.train_stats_builder \
  --manifest "${ARTIFACT_ROOT}/manifest.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${ARTIFACT_ROOT}/train_stats.json"

python -m lerobot.policies.smolvla_icl.data.pairing_builder \
  --manifest "${ARTIFACT_ROOT}/manifest.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output "${ARTIFACT_ROOT}/pairing_sidecar.json" \
  --matcher-cache-dir "${ARTIFACT_ROOT}/matcher_cache" \
  --train-stats "${ARTIFACT_ROOT}/train_stats.json" \
  --alignment-config "${ALIGNMENT_CONFIG}" \
  --matcher-model "${MATCHER_MODEL}" \
  --matcher-revision "${MATCHER_REVISION}" \
  --n-action-steps "${N_ACTION_STEPS}" \
  --query-window-replans "${QUERY_WINDOW_REPLANS}" \
  --num-epochs "${NUM_EPOCHS}" \
  --seed "${SEED}" \
  --device cuda \
  --video-backend pyav

python -m lerobot.policies.smolvla_icl.data.global_cache_builder \
  --manifest "${ARTIFACT_ROOT}/manifest.json" \
  --sidecar "${ARTIFACT_ROOT}/pairing_sidecar.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${ARTIFACT_ROOT}/global_demo_cache" \
  --train-stats "${ARTIFACT_ROOT}/train_stats.json" \
  --global-config "${GLOBAL_CONFIG}" \
  --device cuda \
  --video-backend pyav

python -m lerobot.policies.smolvla_icl.data.local_rgb_cache_builder \
  --manifest "${ARTIFACT_ROOT}/manifest.json" \
  --sidecar "${ARTIFACT_ROOT}/pairing_sidecar.json" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${ARTIFACT_ROOT}/local_rgb_cache" \
  --video-backend pyav

echo "SmolVLA-ICL LIBERO v2 artifacts are ready at ${ARTIFACT_ROOT}."
