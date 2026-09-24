# JUBAIL 上的 SmolVLA-ICL 数据预处理与训练

本文供后续 agent 和实验人员在 NYUAD JUBAIL 上继续本项目使用。内容于
2026-09-22 对照 NYUAD CRC 官方文档检查；集群资源和 partition 限制可能变化，
提交长任务前应重新查看官方页面。

## 1. JUBAIL 使用原则

- 登录命令是 `ssh <NetID>@jubail.abudhabi.nyu.edu`；校外访问通常还需要
  NYUAD VPN。不要在 login node 上运行训练、视频解码、SigLIP 或 S3D。
- 代码和少量配置可以放在 `/home/<NetID>`；数据、模型缓存、离线特征和作业
  输出应放在 `/scratch/<NetID>`。官方明确禁止从 `/home` 运行计算任务。
- `/scratch` 默认适合计算，但文件存在 90 天清理策略；重要的最终 checkpoint
  应转存到 `/archive` 或其他持久位置。备份由用户自己负责。
- 通过 Slurm 提交任务：`sbatch job.slurm`，用 `squeue -u <NetID>` 查看，
  用 `scancel <job_id>` 取消。
- GPU 作业使用 `nvidia` partition，并申请 `--gres=gpu:1` 或具体 GPU 类型。
  资源上限和型号可能随集群状态变化，优先使用“任意一张 GPU”完成预处理。

官方参考：

- [JUBAIL 总览与登录规范](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html)
- [系统、访问和数据传输](https://crc-docs.abudhabi.nyu.edu/hpc/system/index.html)
- [存储规则](https://crc-docs.abudhabi.nyu.edu/hpc/storage/index.html)
- [Slurm 作业与 GPU 申请](https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html)
- [HPC Miniconda](https://crc-docs.abudhabi.nyu.edu/hpc/software/hpc_miniconda.html)

## 2. 项目约定

当前 LIBERO LeRobot 数据根目录：

```text
/scratch/ll5582/LIBERO/lerobot_libero
```

建议将可重建产物集中到：

```text
/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/
├── manifest.json
├── train_stats.json
├── pairing_sidecar.json
├── matcher_cache/
├── local_rgb_cache/
│   ├── cache_manifest.json
│   └── frames/<identity>/*.npy
├── global_demo_cache/
│   ├── cache_manifest.json
│   └── <sha256(demo_id)>.pt
└── logs/
```

旧的 `/scratch/ll5582/SmolICL_artifacts/libero/` 使用 episode-level split，
只保留作历史备份，不能用于当前训练。不要在旧目录中局部替换单个 artifact；
Manifest、train stats、sidecar、Global cache 和 Local RGB cache 必须来自同一条
fingerprint 链。

JUBAIL 上可直接提交仓库内的完整重建作业：

```bash
mkdir -p /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/logs
/opt/slurm/20.11.4-13/bin/sbatch \
  --output=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/logs/%x-%j.out \
  /scratch/ll5582/SmolICL/lerobot/examples/training/smolvla_icl_rebuild_libero_v2.slurm
```

该作业使用一张 V100，严格按 Manifest v2 → train-only stats → pairing sidecar v4
→ Global cache → Local RGB cache 顺序执行。输出目录与旧 v1 artifact 隔离；中断后
使用相同命令重提即可复用身份一致的 Matcher/Global/Local cache 文件。

不要把 Hugging Face、Torch 或视频临时缓存留在容量较小的 home。作业脚本中可
把它们显式指向 `/scratch/ll5582/cache/` 下的不同子目录。

## 3. 环境

CRC 提供集中式 Miniconda。官方推荐的一次性初始化方式是：

```bash
module load miniconda
source ~/.bashrc
```

Slurm 默认不会读取交互式 shell 的 `.bashrc`。作业脚本中应显式初始化 conda，
例如：

```bash
module purge
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate smolicl
```

环境必须满足仓库的 Python 3.12、PyTorch、TorchVision、Transformers、LeRobot
及视频解码依赖。安装后清理 conda/pip 下载缓存，避免快速耗尽文件数配额。

## 4. 固定的数据构建顺序

所有命令从仓库的 `lerobot/` 目录运行。以下 `<DATASET_REVISION>` 和
`<MATCHER_REVISION>` 必须替换成固定 commit/revision，不能使用会移动的默认
分支。

### 4.1 Manifest

Manifest 默认排除 LIBERO-Long，并在 Goal/Object/Spatial 每个 suite 内把 10 个
task 按 8/1/1 固定为互斥的 train/val/test；再把每个 task 的 episode 划成
Demo/Query。因此 val/test 是 unseen-task split。Manifest v2 会拒绝旧的同任务
跨 split 文件；重新生成 Manifest 后必须依次重建 train-only stats、Matcher cache、
pairing sidecar、Local RGB cache 和 Global cache。

```bash
python -m lerobot.policies.smolvla_icl.data.libero_manifest \
  --dataset-root /scratch/ll5582/LIBERO/lerobot_libero \
  --output /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --repo-id lerobot/libero \
  --revision <DATASET_REVISION> \
  --image-key observation.images.image
```

### 4.2 Train-only State/Action stats

该产物只扫描 Manifest 中的 `train/demo + train/query` 数值列，不解码
RGB。Validation、test 和被排除的 LIBERO-Long 都不参与统计：

```bash
python -m lerobot.policies.smolvla_icl.data.train_stats_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --dataset-root /scratch/ll5582/LIBERO/lerobot_libero \
  --output /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/train_stats.json
```

### 4.3 离线 Pairing/DTW Sidecar

该步骤解码 train/val episode，并用冻结 SmolVLA SigLIP 生成 Matcher cache，
必须作为 GPU 作业运行。

```bash
python -m lerobot.policies.smolvla_icl.data.pairing_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --dataset-root /scratch/ll5582/LIBERO/lerobot_libero \
  --output /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/pairing_sidecar.json \
  --matcher-cache-dir /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/matcher_cache \
  --train-stats /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/train_stats.json \
  --matcher-model lerobot/smolvla_base \
  --matcher-revision <MATCHER_REVISION> \
  --n-action-steps 5 \
  --query-window-replans 4 \
  --num-epochs <PAIRING_EPOCHS> \
  --video-backend pyav \
  --device cuda
```

`PAIRING_EPOCHS` 应至少覆盖计划训练时希望轮换的 Demo 配对周期。Validation
配对在 Sidecar 内保持固定。Builder 会从 Dataset FPS 和 `n_action_steps`
推导 matcher 更新频率，并把完整 Alignment 配置写入 Sidecar。以 LIBERO
10 Hz、`n_action_steps=5` 为例，结果为 `alignment_hz=2`；默认四次重规划
窗口对应 `window_duration_s=2`。训练时 Policy 必须使用完全相同的配置，
否则数据工厂会在创建 DataLoader 前拒绝启动。

### 4.4 Local Demo RGB frame cache

训练期不能缓存可训练 SigLIP/connector 的 token，但应把每条 Demo 视频只解码
一次，并保存为 episode 级 CPU `uint8` NPY。训练 worker 通过 mmap 切片 Local
窗口，不再重复打开 MP4：

```bash
python -m lerobot.policies.smolvla_icl.data.local_rgb_cache_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --sidecar /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/pairing_sidecar.json \
  --dataset-root /scratch/ll5582/LIBERO/lerobot_libero \
  --output-dir /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/local_rgb_cache \
  --video-backend pyav
```

Cache manifest 绑定 Dataset revision、Manifest fingerprint、camera、Demo ID、
episode index/length、dtype 和 frame shape。它不保存任何模型视觉 token。

### 4.5 Global Demo S3D cache

`--global-config` 接收两种 JSON：直接的 `GlobalEncoderConfig` 字典，或包含
`global_encoder` 字段的完整 SmolVLA-ICL policy config。这里必须使用与训练
完全相同的配置文件。

```bash
python -m lerobot.policies.smolvla_icl.data.global_cache_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --sidecar /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/pairing_sidecar.json \
  --dataset-root /scratch/ll5582/LIBERO/lerobot_libero \
  --output-dir /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/global_demo_cache \
  --train-stats /scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/train_stats.json \
  --global-config /path/to/smolvla_icl_config.json \
  --device cuda
```

Builder 可断点复用身份一致的 `.pt` 文件。若修改了 camera、数据 revision、
clip 配置、train stats fingerprint、TorchVision/S3D 权重或旧文件身份不匹配，应显式增加
`--overwrite`。它只覆盖当前 Sidecar 引用的文件，不删除目录中的其他数据。

完成后会生成 `cache_manifest.json` 并立即执行一次全量 preflight。

### 4.6 训练

训练配置至少要指向同一组产物：

```text
policy.data_manifest_path=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json
policy.training_stats_path=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/train_stats.json
policy.pairing_sidecar_path=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/pairing_sidecar.json
policy.training_demo_cache_dir=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/global_demo_cache
policy.training_local_rgb_cache_dir=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/local_rgb_cache
dataset.root=/scratch/ll5582/LIBERO/lerobot_libero
dataset.repo_id=lerobot/libero
dataset.revision=<DATASET_REVISION>
dataset.video_backend=pyav
dataset.eval_split=0
policy.n_action_steps=5
policy.demo_alignment.alignment_hz=2
policy.demo_alignment.window_duration_s=2
```

### 4.6.1 V100 32GB 训练配置

SmolVLA-ICL 在 V100 上采用 **FP32 参数存储 + FP16 AMP**。不要使用 BF16：
V100 没有原生 BF16 Tensor Core。训练计算精度由 Accelerator 管理，和仅用于
rollout 的 `policy.use_amp` 无关。

```text
policy.vlm_load_dtype=float32
accelerator.mixed_precision=fp16
batch_size=1
policy.local_vision_encode_batch_size=2
policy.local_vision_gradient_checkpointing=true
parallelism.dp_shard=1
```

单卡将 `accelerator.gradient_accumulation.steps` 设为 8，得到 effective batch 8。
同一节点使用多张 V100 时采用 DDP；`torchrun` 会把未显式设置的
`parallelism.dp_replicate` 自动解析为进程数。为保持 effective batch 8：

| V100 数量 | 每卡 batch | gradient accumulation | effective batch |
|---:|---:|---:|---:|
| 1 | 1 | 8 | 8 |
| 2 | 1 | 4 | 8 |
| 4 | 1 | 2 | 8 |
| 8 | 1 | 1 | 8 |

多卡启动形式为：

```bash
torchrun --standalone --nproc_per_node=<GPU_COUNT> \
  -m lerobot.scripts.lerobot_train \
  ... \
  --accelerator.mixed_precision=fp16 \
  --accelerator.gradient_accumulation.steps=<8/GPU_COUNT> \
  --batch_size=1 \
  --policy.vlm_load_dtype=float32 \
  --policy.local_vision_encode_batch_size=2 \
  --policy.local_vision_gradient_checkpointing=true \
  --parallelism.dp_shard=1
```

当前 FSDP2 路径不支持 FP16，因此 V100 多卡不要设置
`parallelism.dp_shard>1`。DDP 会在每张卡保留完整模型，但能够线性扩大数据并行
batch。A100/H100 训练时可以显式改为 `vlm_load_dtype=bfloat16` 和
`accelerator.mixed_precision=bf16`。

数据工厂会在构建 Query Dataset 和 DataLoader 之前校验 Global cache：主进程
全量读取 Tensor，其他 rank 在 barrier 后复核 manifest 和文件存在性。校验覆盖
Manifest、Sidecar、camera、train stats fingerprint、S3D snapshot、clip 配置、文件身份和
Tensor shape。preflight 不解码视频，也不运行 S3D。

### 4.7 在线 LIBERO test

通用 `lerobot-eval` 不会为 ICL Policy 注册 Demo。在线 test 使用专用入口，
它按 task 从 Manifest 的 `test/demo` 中确定性选择 Demo，先调用 `set_demo()`，
再逐 episode `reset()` 并运行 simulator Query。test 不读取训练 pairing sidecar：
DTW 在 rollout 中仅使用已经到达的 Query Observation 在线推进。

```bash
python -m lerobot.scripts.lerobot_eval_smolvla_icl \
  --policy.path=/path/to/checkpoint/pretrained_model \
  --env.type=libero \
  --env.task=libero_goal,libero_object,libero_spatial \
  --env.fps=10 \
  --eval.n_episodes=10 \
  --manifest_path=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/manifest.json \
  --train_stats_path=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/train_stats.json \
  --dataset_root=/scratch/ll5582/LIBERO/lerobot_libero \
  --matcher_model=lerobot/smolvla_base \
  --matcher_revision=<MATCHER_REVISION> \
  --swap_demo_count=1 \
  --output_dir=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/eval/run_001
```

该入口固定 `batch_size=1`、同步 env 和串行 task，因为一个 Policy 只维护一份
action queue、Observation history 与 DTW 状态。即使 `env.task` 指定完整 suite，
evaluator 也只 rollout Manifest v2 中属于 test 的 unseen tasks。输出
`eval_info.json` 包含成功率、
每次重规划的对齐轨迹，以及相同 seeds 下的 Demo-swap 成功率变化。若当前
Manifest 对某个 task 只有一条 `test/demo`，swap 指标会明确记录为不可用；不要
跨 split 借 Demo 来伪造该指标。`env.fps`、Policy 的 `n_action_steps` 与
`demo_alignment` 时间配置不一致时，评测会在启动前拒绝运行。

## 5. Slurm 模板

Pairing、Global cache 和训练都应使用 Slurm。下面是单 GPU 预处理模板；CPU、
内存和 walltime 应按实测调整，不要无依据申请整台节点。

```bash
#!/bin/bash
#SBATCH --job-name=smolicl-cache
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ll5582/SmolICL_artifacts/libero_unseen_v2/logs/%x-%j.out

module purge
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate smolicl

export HF_HOME=/scratch/ll5582/cache/huggingface
export TORCH_HOME=/scratch/ll5582/cache/torch
export TMPDIR=/scratch/ll5582/cache/tmp

cd /path/to/SmolICL/lerobot

# 在这里放 pairing_builder、global_cache_builder 或 lerobot-train 命令。
```

先用一个短作业验证单 batch forward/backward，再提交长训练。若使用 preempt
partition，作业可能被终止，应确保训练 checkpoint 和 resume 已启用。

## 6. 后续 agent 操作检查表

1. 先确认当前节点；login node 只做编辑、`git`、`sbatch` 和轻量检查。
2. 运行 `myquota` 检查 home/scratch 文件空间和文件数。
3. 核对 Dataset 与 Matcher 都使用固定 revision。
4. 按 Manifest → train-only stats → Sidecar → Local RGB cache → Global cache 生成产物。
5. 不手工修改任何生成的 JSON；修改后 fingerprint 校验会失败。
6. Global cache builder 成功后检查日志中的 Demo 数量和 newly encoded 数量。
7. 训练前先做一个 batch 的 forward/backward smoke test。
8. 长训练使用 `sbatch`，用 `squeue`/日志监控；不要在 SSH 会话内直接训练。
