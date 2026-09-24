"""为 Manifest train split 流式计算 State/Action normalization stats。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from .libero_manifest import LiberoDataManifest
from .train_stats import TrainStatsArtifact

__all__ = ["build_train_stats"]


def _batch_array(values: object, *, key: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        tensor = values
    elif isinstance(values, list) and values and all(isinstance(value, torch.Tensor) for value in values):
        tensor = torch.stack(values)
    else:
        raise TypeError(f"Train stats 字段 {key!r} 必须是 Tensor batch。")
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError(f"Train stats 字段 {key!r} 必须是二维浮点 Tensor。")
    return tensor.detach().cpu().float().numpy()


def _json_stats(stats: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in np.asarray(values).reshape(-1)]
        for name, values in stats.items()
    }


def build_train_stats(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    state_key: str = OBS_STATE,
    action_key: str = ACTION,
    batch_size: int = 8192,
) -> TrainStatsArtifact:
    """只读取 Manifest train/demo+query 的数值列并生成独立 artifact。"""
    if batch_size < 1:
        raise ValueError("Train stats batch_size 必须大于 0。")
    manifest = LiberoDataManifest.load(manifest_path)
    train_episodes = manifest.episode_indices(split="train")
    dataset = LeRobotDataset(
        manifest.repo_id,
        root=dataset_root,
        episodes=train_episodes,
        delta_timestamps=None,
        image_transforms=None,
        revision=manifest.revision,
    )
    for key in (state_key, action_key):
        feature = dataset.meta.features.get(key)
        if feature is None or feature["dtype"] in ("image", "video", "string"):
            raise KeyError(f"Dataset 缺少可统计的数值字段：{key!r}。")

    expected_frames = sum(
        episode.length
        for episode in manifest.episodes
        if episode.split == "train"
    )
    if len(dataset) != expected_frames:
        raise ValueError("Train Dataset 帧数与 Manifest train split 不一致。")

    projected = dataset.hf_dataset.select_columns([state_key, action_key])
    accumulators = {state_key: RunningQuantileStats(), action_key: RunningQuantileStats()}
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        rows = projected[indices]
        for key, accumulator in accumulators.items():
            values = _batch_array(rows[key], key=key)
            if not np.isfinite(values).all():
                raise ValueError(f"Train stats 字段 {key!r} 包含非有限值。")
            accumulator.update(values)

    stats = {
        OBS_STATE: _json_stats(accumulators[state_key].get_statistics()),
        ACTION: _json_stats(accumulators[action_key].get_statistics()),
    }
    artifact = TrainStatsArtifact.create(manifest, stats)
    artifact.save(output_path)
    logging.info(
        "Saved train-only stats: %s (%d episodes, %d frames)",
        output_path,
        len(train_episodes),
        artifact.num_frames,
    )
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Manifest train-only State/Action stats.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--action-key", default=ACTION)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_train_stats(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_path=args.output,
        state_key=args.state_key,
        action_key=args.action_key,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
