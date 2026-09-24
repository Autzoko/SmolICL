"""SmolVLA-ICL 在线 LIBERO test 的 Demo 与指标契约。"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from .data.libero_manifest import LiberoDataManifest, LiberoManifestEpisode
from .data.reader import LeRobotMatcherEpisodeReader

__all__ = [
    "AlignmentTraceCollector",
    "TestDemo",
    "TestDemoProvider",
    "select_test_demos",
    "summarize_demo_swap",
]


def _stable_seed(seed: int, task_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:test-demo:{task_index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_test_demos(
    manifest: LiberoDataManifest,
    *,
    task_index: int,
    seed: int,
    swap_demo_count: int,
) -> tuple[LiberoManifestEpisode, ...]:
    """确定性选择一条 primary Demo 和至多若干条 swap Demo。"""
    if swap_demo_count < 0:
        raise ValueError("swap_demo_count 不能为负数。")
    candidates = [
        episode
        for episode in manifest.episodes
        if episode.split == "test" and episode.role == "demo" and episode.task_index == task_index
    ]
    if not candidates:
        raise ValueError(f"task_index={task_index} 没有 test/demo episode。")
    candidates.sort(key=lambda episode: episode.episode_index)
    random.Random(_stable_seed(seed, task_index)).shuffle(candidates)
    return tuple(candidates[: 1 + swap_demo_count])


@dataclass(frozen=True, slots=True)
class TestDemo:
    """一条可直接传给 ``policy.set_demo`` 的完整 test Demo。"""

    episode: LiberoManifestEpisode
    video: Tensor
    states: Tensor
    timestamps: Tensor


class TestDemoProvider:
    """只加载 Manifest 声明的 ``test/demo``，不读取 pairing sidecar。"""

    def __init__(
        self,
        manifest: LiberoDataManifest,
        *,
        dataset_root: str | Path,
        state_key: str = OBS_STATE,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
    ) -> None:
        self.manifest = manifest
        self.state_key = state_key
        episodes = manifest.episode_indices(split="test", role="demo")
        self.dataset = LeRobotDataset(
            manifest.repo_id,
            root=dataset_root,
            episodes=episodes,
            delta_timestamps=None,
            image_transforms=None,
            revision=manifest.revision,
            video_backend=video_backend,
            return_uint8=True,
            tolerance_s=tolerance_s,
        )
        if not math.isclose(
            float(self.dataset.meta.fps),
            manifest.fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Test Demo Dataset fps 与 Manifest 不一致。")
        self.reader = LeRobotMatcherEpisodeReader(
            dataset=self.dataset,
            image_key=manifest.image_key,
            state_key=state_key,
        )
        self._episodes = {episode.episode_index: episode for episode in manifest.episodes}
        for episode_index in episodes:
            metadata = self.dataset.meta.episodes[episode_index]
            actual_length = int(metadata["dataset_to_index"]) - int(metadata["dataset_from_index"])
            if actual_length != self._episodes[episode_index].length:
                raise ValueError(f"Test Demo episode={episode_index} 长度与 Manifest 不一致。")

    def load(self, episode: LiberoManifestEpisode) -> TestDemo:
        """读取一条 Demo；RGB 仅在注册前转换一次为 ``[0,1]`` float。"""
        expected = self._episodes.get(episode.episode_index)
        if expected != episode or episode.split != "test" or episode.role != "demo":
            raise ValueError("在线 test 只能注册当前 Manifest 中的 test/demo episode。")
        data = self.reader(episode.episode_index)
        video = data.images
        if video.dtype == torch.uint8:
            video = video.float().div_(255.0)
        elif video.is_floating_point():
            video = video.float()
        else:
            raise TypeError("Test Demo RGB 必须是 uint8 或浮点 Tensor。")
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError("Test Demo RGB 必须采用 (T,3,H,W) 格式。")
        if torch.any(video < 0) or torch.any(video > 1):
            raise ValueError("Test Demo RGB 必须位于 [0,1]。")
        return TestDemo(
            episode=episode,
            video=video,
            states=data.states,
            timestamps=data.timestamps,
        )


class AlignmentTraceCollector:
    """在每个控制步采样 Policy，只保存发生变化的重规划结果。"""

    def __init__(self) -> None:
        self.control_step = 0
        self.last_replan_index: int | None = None
        self.trace: list[dict[str, int | float | None]] = []

    def __call__(self, policy: Any) -> None:
        diagnostics_fn = getattr(policy, "alignment_diagnostics", None)
        if not callable(diagnostics_fn):
            raise TypeError("SmolVLA-ICL evaluator 要求 Policy 提供 alignment_diagnostics()。")
        diagnostics = diagnostics_fn()
        if diagnostics is not None:
            replan_index = int(diagnostics["replan_index"])
            if replan_index != self.last_replan_index:
                self.trace.append({"control_step": self.control_step, **diagnostics})
                self.last_replan_index = replan_index
        self.control_step += 1


def summarize_demo_swap(
    demo_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """按相同 seed 成对比较 primary 与替换 Demo 的成功结果。"""
    if not demo_results:
        raise ValueError("demo_results 不能为空。")

    primary = demo_results[0]
    primary_by_seed = {int(item["seed"]): bool(item["success"]) for item in primary["episodes"]}
    alternatives: list[dict[str, Any]] = []
    for result in demo_results[1:]:
        alternate_by_seed = {int(item["seed"]): bool(item["success"]) for item in result["episodes"]}
        if alternate_by_seed.keys() != primary_by_seed.keys():
            raise ValueError("Demo-swap 比较必须使用完全相同的 rollout seeds。")
        seeds = sorted(primary_by_seed)
        flip_rate = sum(primary_by_seed[seed] != alternate_by_seed[seed] for seed in seeds) / len(seeds)
        primary_rate = sum(primary_by_seed.values()) / len(seeds)
        alternate_rate = sum(alternate_by_seed.values()) / len(seeds)
        alternatives.append(
            {
                "demo_episode_index": result["demo_episode_index"],
                "success_rate": alternate_rate,
                "success_rate_delta": alternate_rate - primary_rate,
                "paired_success_flip_rate": flip_rate,
            }
        )

    return {
        "available": bool(alternatives),
        "primary_demo_episode_index": primary["demo_episode_index"],
        "primary_success_rate": sum(primary_by_seed.values()) / len(primary_by_seed),
        "alternatives": alternatives,
        "mean_absolute_success_rate_delta": (
            sum(abs(item["success_rate_delta"]) for item in alternatives) / len(alternatives)
            if alternatives
            else None
        ),
        "mean_paired_success_flip_rate": (
            sum(item["paired_success_flip_rate"] for item in alternatives) / len(alternatives)
            if alternatives
            else None
        ),
    }
