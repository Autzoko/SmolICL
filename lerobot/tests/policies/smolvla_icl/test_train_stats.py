"""Manifest train-only normalization stats artifact 测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
)
from lerobot.policies.smolvla_icl.data.factory import _install_train_stats
from lerobot.policies.smolvla_icl.data.train_stats import TrainStatsArtifact
from lerobot.utils.constants import ACTION, OBS_STATE


def _manifest() -> LiberoDataManifest:
    return LiberoDataManifest(
        repo_id="lerobot/libero",
        revision="revision",
        fps=10.0,
        image_key="observation.images.image",
        config=LiberoManifestConfig(),
        episodes=tuple(
            LiberoManifestEpisode(
                episode_index=index,
                suite="libero_goal",
                task_index=task_index,
                task=f"task {task_index}",
                length=length,
                split=split,
                role=role,
            )
            for index, task_index, length, split, role in (
                (0, 10, 3, "train", "demo"),
                (1, 10, 5, "train", "query"),
                (2, 11, 7, "val", "demo"),
                (3, 11, 11, "val", "query"),
                (4, 12, 13, "test", "demo"),
                (5, 12, 17, "test", "query"),
            )
        ),
    )


def _stats(count: int) -> dict[str, dict[str, list[float]]]:
    return {
        OBS_STATE: {
            "min": [-1.0, -2.0],
            "max": [1.0, 2.0],
            "mean": [0.0, 0.5],
            "std": [1.0, 1.5],
            "count": [float(count)],
        },
        ACTION: {
            "min": [-0.5],
            "max": [0.5],
            "mean": [0.0],
            "std": [0.25],
            "count": [float(count)],
        },
    }


def test_train_stats_only_bind_manifest_train_demo_and_query(tmp_path) -> None:
    manifest = _manifest()
    artifact = TrainStatsArtifact.create(manifest, _stats(count=8))

    assert artifact.episode_indices == (0, 1)
    assert artifact.num_frames == 8
    restored = TrainStatsArtifact.load(artifact.save(tmp_path / "train_stats.json"), manifest=manifest)
    assert restored == artifact
    assert torch.equal(restored.to_dataset_stats()[OBS_STATE]["mean"], torch.tensor([0.0, 0.5]))


def test_train_stats_rejects_tampering_and_wrong_frame_count(tmp_path) -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="count"):
        TrainStatsArtifact.create(manifest, _stats(count=9))

    path = TrainStatsArtifact.create(manifest, _stats(count=8)).save(tmp_path / "train_stats.json")
    payload = json.loads(path.read_text())
    payload["stats"][OBS_STATE]["mean"][0] = 123.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        TrainStatsArtifact.load(path)


def test_factory_replaces_full_dataset_stats_with_train_only_stats() -> None:
    artifact = TrainStatsArtifact.create(_manifest(), _stats(count=8))
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            stats={
                "raw.state": {"mean": torch.tensor([999.0])},
                "raw.action": {"mean": torch.tensor([999.0])},
                "observation.images.image": {"mean": torch.tensor([0.1, 0.2, 0.3])},
            }
        )
    )

    _install_train_stats(
        dataset,
        artifact,
        raw_state_key="raw.state",
        raw_action_key="raw.action",
    )

    assert set(dataset.meta.stats) == {"raw.state", "raw.action"}
    assert torch.equal(dataset.meta.stats["raw.state"]["mean"], torch.tensor([0.0, 0.5]))
    assert torch.equal(dataset.meta.stats["raw.action"]["mean"], torch.tensor([0.0]))
