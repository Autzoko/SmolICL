# FR3 SmolVLA: collection, cleaning, fine-tuning, and deployment

## Recommended checkpoint

Use `lerobot/smolvla_base` for the first FR3 experiment, pinned to revision:

```text
d9f33c94a60fb382c90dea2164c96845bd955e28
```

This is the official general fine-tuning checkpoint. Its stored feature schema
is SO100-specific (6-D state/action and three 256x256 cameras), so FR3 training
must explicitly clear that schema and infer the 8-D state/action plus the
`wrist`, `top`, and `right` cameras from the new dataset:

```text
--policy.input_features=null --policy.output_features=null
```

Do not use `smolvla_libero` as the primary initialization. It is closer in
robot appearance but specializes the action expert for a simulated Cartesian
LIBERO action space, not the FR3 normalized joint-delta contract. It is useful
only as a controlled ablation after the base run works.

## Collection contract

- Task text: `Pick up the black cube and place it in the wooden tray.`
- Seven marked cube layouts: `L01` through `L07`.
- Ten accepted demonstrations per layout; 70 training episodes total.
- Tray pose, cameras, lighting, robot home pose, and task text remain fixed.
- Record separate evaluation rollouts; never merge evaluation attempts into
  the training project.
- Capture and align at 15 Hz. Do not relabel or duplicate frames as 30 Hz.
- State: seven joint positions in radians plus normalized observed gripper.
- Action: seven normalized realized joint deltas plus absolute gripper target.
- Joint decoding: `q_target = q_current + clip(action[:7], -1, 1) * 0.2`.
- Gripper decoding: `<=0.5` open, `>0.5` grasp.

The recorder configuration on the Franka host is:

```text
/home/franka/Viva-La-Franka/Franka-Recorder/config/collections/
smolvla_fr3_black_cube_to_wooden_tray_7layout.json
```

Start only after confirming which process owns FCI and after freeing disk:

```bash
cd /home/franka/Viva-La-Franka
./scripts/run_smolvla_collection.sh
```

The Franka host had only 36 GB free on 2026-09-17, while this profile requires
at least 50 GB, so it will currently fail closed before opening a recording.
Archive old data or write to a larger mounted volume, for example:

```bash
./scripts/run_smolvla_collection.sh \
  --output-root /path/on/large_volume/SMOLVLA_FR3_BLACK_CUBE_TRAY_7LAYOUT
```

Also coordinate a recording window for any existing RGB-D service that owns
one of the three RealSense devices. Do not kill an unidentified process; the
recorder needs exclusive camera access but uses the existing Franky state
server read-only during capture.

## Non-destructive cleaning

Cleaning is allow-list based. Raw, rejected, interrupted, and failed attempts
are never deleted or rewritten.

```bash
cd /home/franka/Viva-La-Franka/Franka-Recorder
/home/franka/conda/envs/ricl-client/bin/python audit_smolvla.py \
  /home/franka/Viva-La-Franka/data/SMOLVLA_FR3_BLACK_CUBE_TRAY_7LAYOUT \
  --report /tmp/fr3_smolvla_audit.json \
  --selection-manifest /tmp/fr3_smolvla_selected.jsonl
```

The audit rejects incomplete profiles, missing layouts, missing cameras,
camera loss, motion faults, non-finite values, implausible durations, absent
gripper transitions, or excessive action saturation. A high stationary-action
fraction is reported as a warning so a researcher can inspect it rather than
silently deleting a valid slow demonstration.

Before training, inspect per-layout counts, duration histograms, action
quantiles, gripper transitions, and a sample of synchronized videos. Exactly
ten valid episodes should remain for each of the seven layouts.

## LeRobotDataset v3 conversion

Copy the accepted source project and the selection manifest to the training
workstation. Convert offline; video encoding must not run in the robot capture
process.

```bash
cd lerobot
uv run python examples/port_datasets/port_viva_franka.py \
  --project-root /data/SMOLVLA_FR3_BLACK_CUBE_TRAY_7LAYOUT \
  --selection-manifest /data/fr3_smolvla_selected.jsonl \
  --repo-id ${HF_USER}/fr3_black_cube_tray_7layout \
  --output-root /data/lerobot/fr3_black_cube_tray_7layout \
  --image-width 640 \
  --image-height 360 \
  --dry-run
```

Remove `--dry-run` after the summary is correct. Add `--push-to-hub` only when
the local dataset has been inspected. The converter refuses to overwrite an
existing output directory.

## First fine-tuning run

The official config defaults to a 50-step chunk, and the official real-robot
examples use 30 Hz. For this 15 Hz dataset, 25 steps is a conservative first
run that approximates the same 1.67-second physical horizon; this is an
engineering choice rather than a published FR3 optimum. Compare against the
unchanged 50-step chunk after the first pipeline works. Execute only five
actions before replanning in the first real-robot test.

```bash
cd lerobot
uv run lerobot-train \
  --dataset.repo_id=${HF_USER}/fr3_black_cube_tray_7layout \
  --dataset.root=/data/lerobot/fr3_black_cube_tray_7layout \
  --policy.path=lerobot/smolvla_base \
  --policy.pretrained_revision=d9f33c94a60fb382c90dea2164c96845bd955e28 \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size=25 \
  --policy.n_action_steps=5 \
  --policy.device=cuda \
  --steps=20000 \
  --batch_size=8 \
  --save_freq=5000 \
  --output_dir=outputs/train/fr3_smolvla_cube_tray \
  --job_name=fr3_smolvla_cube_tray \
  --wandb.enable=true
```

Keep `--dataset.repo_id` even for a local-only dataset; `--dataset.root` tells
LeRobot where that local copy lives. Omit `--dataset.root` only after the
dataset has been pushed and the Hub/cache copy is the intended input.

Evaluate checkpoints at 5k, 10k, 15k, and 20k with the same fixed layouts.
Training loss alone does not select a safe or successful robot checkpoint.

## Deployment contract

Run model inference on a GPU workstation and keep FCI, safety checks, and the
final interpolation loop on the Franka NUC. The runtime observation keys must
match training exactly:

```text
observation.state          float32[8]
observation.images.wrist   RGB
observation.images.top     RGB
observation.images.right   RGB
task                       exact training prompt
```

Load preprocessing and postprocessing from the fine-tuned checkpoint, not from
`smolvla_base`; they contain FR3 normalization statistics. Convert camera BGR
to RGB and preserve aspect ratio before the policy's 512x512 padded resize.

Begin with synchronous inference, one operator, low velocity/acceleration
limits, `n_action_steps=5`, a joint-limit margin, workspace guard, stale-frame
timeout, dropped-packet hold, and an accessible emergency stop. Apply the
existing action adapter only after checkpoint postprocessing. Never interpret
the first seven outputs as rad/s. RTC can be enabled only after synchronous
rollouts are stable and action timestamps have been measured.

Use at least 20 held-out real-robot rollouts: the seven training layouts plus
nearby unseen positions. Report grasp success, final placement success,
collision/protective-stop count, intervention count, and completion time.
