"""Manifest train split 专用的 State/Action normalization stats artifact。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .libero_manifest import LiberoDataManifest

__all__ = ["TrainStatsArtifact"]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainStatsArtifact:
    """只由 Manifest ``train/demo + train/query`` 帧计算的统计量。"""

    manifest_fingerprint: str
    repo_id: str
    revision: str
    episode_indices: tuple[int, ...]
    num_frames: int
    stats: dict[str, dict[str, list[float]]]
    fingerprint: str

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_fingerprint", self.manifest_fingerprint),
            ("fingerprint", self.fingerprint),
        ):
            if len(value) != 64 or any(char not in string.hexdigits for char in value):
                raise ValueError(f"{name} 必须是 SHA-256 字符串。")
        if not self.repo_id or not self.revision or not self.episode_indices or self.num_frames < 1:
            raise ValueError("Train stats 的 Dataset/episode/frame 身份不能为空。")
        if tuple(sorted(set(self.episode_indices))) != self.episode_indices:
            raise ValueError("Train stats episode_indices 必须唯一且升序。")
        for key in (OBS_STATE, ACTION):
            feature_stats = self.stats.get(key)
            if feature_stats is None or "mean" not in feature_stats or "std" not in feature_stats:
                raise ValueError(f"Train stats 缺少 {key!r} mean/std。")
            dimension = len(feature_stats["mean"])
            if dimension < 1 or len(feature_stats["std"]) != dimension:
                raise ValueError(f"Train stats {key!r} mean/std 维度无效。")
            if any(value < 0 for value in feature_stats["std"]):
                raise ValueError(f"Train stats {key!r} std 不能为负数。")
            if any(
                name != "count" and len(values) != dimension
                for name, values in feature_stats.items()
            ):
                raise ValueError(f"Train stats {key!r} 各统计量维度不一致。")
            if any(not math.isfinite(value) for values in feature_stats.values() for value in values):
                raise ValueError(f"Train stats {key!r} 包含非有限值。")
            if feature_stats.get("count") != [float(self.num_frames)]:
                raise ValueError(f"Train stats {key!r} count 与 num_frames 不一致。")

    @staticmethod
    def _payload(
        *,
        manifest: LiberoDataManifest,
        episode_indices: tuple[int, ...],
        num_frames: int,
        stats: dict[str, dict[str, list[float]]],
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "manifest_fingerprint": manifest.fingerprint,
            "dataset": {"repo_id": manifest.repo_id, "revision": manifest.revision},
            "selection": {
                "split": "train",
                "roles": ["demo", "query"],
                "episode_indices": list(episode_indices),
                "num_frames": num_frames,
            },
            "stats": stats,
        }

    @classmethod
    def create(
        cls,
        manifest: LiberoDataManifest,
        stats: dict[str, dict[str, list[float]]],
    ) -> Self:
        episode_indices = tuple(manifest.episode_indices(split="train"))
        episode_map = {episode.episode_index: episode for episode in manifest.episodes}
        num_frames = sum(episode_map[index].length for index in episode_indices)
        payload = cls._payload(
            manifest=manifest,
            episode_indices=episode_indices,
            num_frames=num_frames,
            stats=stats,
        )
        return cls(
            manifest_fingerprint=manifest.fingerprint,
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            episode_indices=episode_indices,
            num_frames=num_frames,
            stats=stats,
            fingerprint=_canonical_sha256(payload),
        )

    def validate_manifest(self, manifest: LiberoDataManifest) -> None:
        expected_indices = tuple(manifest.episode_indices(split="train"))
        episode_map = {episode.episode_index: episode for episode in manifest.episodes}
        expected_frames = sum(episode_map[index].length for index in expected_indices)
        if (
            self.manifest_fingerprint != manifest.fingerprint
            or self.repo_id != manifest.repo_id
            or self.revision != manifest.revision
            or self.episode_indices != expected_indices
            or self.num_frames != expected_frames
        ):
            raise ValueError("Train stats artifact 与当前 Manifest train split 不一致。")

    def to_dataset_stats(
        self,
        *,
        state_key: str = OBS_STATE,
        action_key: str = ACTION,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """返回 LeRobot processor 和 DemoStateNormalizer 使用的 Tensor stats。"""
        return {
            output_key: {
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in self.stats[source_key].items()
            }
            for source_key, output_key in ((OBS_STATE, state_key), (ACTION, action_key))
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": 1,
            "manifest_fingerprint": self.manifest_fingerprint,
            "dataset": {"repo_id": self.repo_id, "revision": self.revision},
            "selection": {
                "split": "train",
                "roles": ["demo", "query"],
                "episode_indices": list(self.episode_indices),
                "num_frames": self.num_frames,
            },
            "stats": self.stats,
        }
        payload["fingerprint"] = self.fingerprint
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path, *, manifest: LiberoDataManifest | None = None) -> Self:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Train stats artifact 不存在：{source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("不支持的 Train stats artifact 版本。")
        dataset = payload["dataset"]
        selection = payload["selection"]
        if selection.get("split") != "train" or selection.get("roles") != ["demo", "query"]:
            raise ValueError("Train stats selection 必须是 train/demo + train/query。")
        stats = {
            str(key): {
                str(name): [float(value) for value in values]
                for name, values in feature_stats.items()
            }
            for key, feature_stats in payload["stats"].items()
        }
        artifact = cls(
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            repo_id=str(dataset["repo_id"]),
            revision=str(dataset["revision"]),
            episode_indices=tuple(int(value) for value in selection["episode_indices"]),
            num_frames=int(selection["num_frames"]),
            stats=stats,
            fingerprint=str(payload["fingerprint"]),
        )
        expected_payload = dict(payload)
        expected_payload.pop("fingerprint", None)
        if artifact.fingerprint != _canonical_sha256(expected_payload):
            raise ValueError("Train stats artifact fingerprint 不匹配。")
        if manifest is not None:
            artifact.validate_manifest(manifest)
        return artifact
