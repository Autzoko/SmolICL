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

当前 LIBERO 数据路径：

```text
/scratch/ll5582/LIBERO
```

建议将可重建产物集中到：

```text
/scratch/ll5582/SmolICL_artifacts/libero/
├── manifest.json
├── pairing_sidecar.json
├── matcher_cache/
├── global_demo_cache/
│   ├── cache_manifest.json
│   └── <sha256(demo_id)>.pt
└── logs/
```

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

Manifest 固定 train/val/test 和 demo/query 划分，默认排除 LIBERO-Long。

```bash
python -m lerobot.policies.smolvla_icl.data.libero_manifest \
  --dataset-root /scratch/ll5582/LIBERO \
  --output /scratch/ll5582/SmolICL_artifacts/libero/manifest.json \
  --repo-id lerobot/libero \
  --revision <DATASET_REVISION> \
  --image-key observation.images.image
```

### 4.2 离线 Pairing/DTW Sidecar

该步骤解码 train/val episode，并用冻结 SmolVLA SigLIP 生成 Matcher cache，
必须作为 GPU 作业运行。

```bash
python -m lerobot.policies.smolvla_icl.data.pairing_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero/manifest.json \
  --dataset-root /scratch/ll5582/LIBERO \
  --output /scratch/ll5582/SmolICL_artifacts/libero/pairing_sidecar.json \
  --matcher-cache-dir /scratch/ll5582/SmolICL_artifacts/libero/matcher_cache \
  --matcher-model lerobot/smolvla_base \
  --matcher-revision <MATCHER_REVISION> \
  --num-epochs <PAIRING_EPOCHS> \
  --device cuda
```

`PAIRING_EPOCHS` 应至少覆盖计划训练时希望轮换的 Demo 配对周期。Validation
配对在 Sidecar 内保持固定。

### 4.3 Global Demo S3D cache

`--global-config` 接收两种 JSON：直接的 `GlobalEncoderConfig` 字典，或包含
`global_encoder` 字段的完整 SmolVLA-ICL policy config。这里必须使用与训练
完全相同的配置文件。

```bash
python -m lerobot.policies.smolvla_icl.data.global_cache_builder \
  --manifest /scratch/ll5582/SmolICL_artifacts/libero/manifest.json \
  --sidecar /scratch/ll5582/SmolICL_artifacts/libero/pairing_sidecar.json \
  --dataset-root /scratch/ll5582/LIBERO \
  --output-dir /scratch/ll5582/SmolICL_artifacts/libero/global_demo_cache \
  --global-config /path/to/smolvla_icl_config.json \
  --device cuda
```

Builder 可断点复用身份一致的 `.pt` 文件。若修改了 camera、数据 revision、
clip 配置、State stats、TorchVision/S3D 权重或旧文件身份不匹配，应显式增加
`--overwrite`。它只覆盖当前 Sidecar 引用的文件，不删除目录中的其他数据。

完成后会生成 `cache_manifest.json` 并立即执行一次全量 preflight。

### 4.4 训练

训练配置至少要指向同一组产物：

```text
policy.data_manifest_path=/scratch/ll5582/SmolICL_artifacts/libero/manifest.json
policy.pairing_sidecar_path=/scratch/ll5582/SmolICL_artifacts/libero/pairing_sidecar.json
policy.training_demo_cache_dir=/scratch/ll5582/SmolICL_artifacts/libero/global_demo_cache
dataset.root=/scratch/ll5582/LIBERO
dataset.repo_id=lerobot/libero
dataset.revision=<DATASET_REVISION>
dataset.eval_split=0
```

数据工厂会在构建 Query Dataset 和 DataLoader 之前校验 Global cache：主进程
全量读取 Tensor，其他 rank 在 barrier 后复核 manifest 和文件存在性。校验覆盖
Manifest、Sidecar、camera、State stats、S3D snapshot、clip 配置、文件身份和
Tensor shape。preflight 不解码视频，也不运行 S3D。

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
#SBATCH --output=/scratch/ll5582/SmolICL_artifacts/libero/logs/%x-%j.out

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
4. 按 Manifest → Sidecar → Global cache 的顺序生成产物。
5. 不手工修改任何生成的 JSON；修改后 fingerprint 校验会失败。
6. Global cache builder 成功后检查日志中的 Demo 数量和 newly encoded 数量。
7. 训练前先做一个 batch 的 forward/backward smoke test。
8. 长训练使用 `sbatch`，用 `squeue`/日志监控；不要在 SSH 会话内直接训练。
