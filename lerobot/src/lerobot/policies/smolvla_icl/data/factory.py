"""Manifest 驱动的 SmolVLA-ICL 训练 Dataset 组装入口。

该工厂保留 LeRobot 原生的 Dataset 读取、Query 时间窗口和图像增广，
但 episode 选择只信任 :class:`LiberoDataManifest`：

* train Dataset 只包含 ``train/query`` episode；
* validation Dataset 只包含 ``val/query`` episode；
* Demo-only Dataset 只打开 sidecar 实际引用的 Demo episode；
* ``test`` 保留在 Manifest 中，不进入训练进程。

Pairing sidecar 必须携带同一 Manifest 的指纹。每个 Query–Demo 对还会
在 DataLoader worker 启动前检查 split、task 和 role，防止跨 split 数据泄漏。
"""

from __future__ import annotations

import math
from typing import Any

import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.storage import load_dataset_metadata
from lerobot.distributed.utils import is_main_process
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import IMAGENET_STATS, OBS_STATE

from .dataset import SmolVLAICLQueryDataset
from .global_cache import preflight_global_demo_cache
from .libero_manifest import LiberoDataManifest, LiberoManifestEpisode
from .reader import LeRobotLocalDemoReader
from .sidecar import PairingSidecar, PairingSidecarResolver
from .state import DemoStateNormalizer

__all__ = ["make_smolvla_icl_train_eval_datasets"]


def _manifest_episode_map(
    manifest: LiberoDataManifest,
) -> dict[int, LiberoManifestEpisode]:
    """按 episode_index 建立 Manifest 查询表。"""
    return {episode.episode_index: episode for episode in manifest.episodes}


def _validate_data_contract(
    cfg: Any,
    manifest: LiberoDataManifest,
    sidecar: PairingSidecar,
    metadata: Any,
) -> None:
    """确认配置、Manifest、sidecar 和本地 Dataset 描述同一份数据。"""
    if cfg.dataset.repo_id != manifest.repo_id:
        raise ValueError(
            f"Dataset repo_id={cfg.dataset.repo_id!r} 与 Manifest repo_id={manifest.repo_id!r} 不一致。"
        )
    if cfg.dataset.revision is not None and cfg.dataset.revision != manifest.revision:
        raise ValueError(
            f"Dataset revision={cfg.dataset.revision!r} 与 Manifest revision={manifest.revision!r} 不一致。"
        )
    if cfg.dataset.episodes is not None or cfg.dataset.exclude_episodes is not None:
        raise ValueError(
            "SmolVLA-ICL 的 episode 选择由 data_manifest_path 唯一决定，"
            "不能同时设置 dataset.episodes/exclude_episodes。"
        )
    if cfg.dataset.eval_split != 0.0:
        raise ValueError("SmolVLA-ICL 的 validation 划分已由 Manifest 固定，dataset.eval_split 必须保持 0。")
    if metadata.repo_id != manifest.repo_id:
        raise ValueError("LeRobot metadata 与 Manifest 的 repo_id 不一致。")
    if not math.isclose(float(metadata.fps), manifest.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("LeRobot metadata 与 Manifest 的 fps 不一致。")
    if manifest.image_key not in metadata.camera_keys:
        raise KeyError(f"Manifest image_key={manifest.image_key!r} 不存在于 Dataset。")
    if manifest.image_key in metadata.depth_keys:
        raise ValueError("Manifest image_key 必须指向 RGB camera，不能指向 depth。")
    if sidecar.manifest_fingerprint != manifest.fingerprint:
        raise ValueError("pairing sidecar 与 data manifest 的 fingerprint 不一致。")
    if sidecar.image_key != manifest.image_key:
        raise ValueError("pairing sidecar 与 data manifest 的 image_key 不一致。")

    # revision 是数据身份的主要边界；长度检查则可以及时发现
    # 本地 root 被同路径的其他快照替换。
    for episode in manifest.episodes:
        try:
            raw_episode = metadata.episodes[episode.episode_index]
        except (IndexError, KeyError) as error:
            raise ValueError(f"Dataset 缺少 Manifest episode={episode.episode_index}。") from error
        actual_length = int(raw_episode["dataset_to_index"]) - int(raw_episode["dataset_from_index"])
        if actual_length != episode.length:
            raise ValueError(
                f"Manifest episode={episode.episode_index} 长度为 {episode.length}，"
                f"Dataset 中为 {actual_length}。"
            )


def _validate_pairings(
    sidecar: PairingSidecar,
    manifest: LiberoDataManifest,
    metadata: Any,
) -> None:
    """检查 sidecar 覆盖范围、Demo 边界和同 split/task 约束。"""
    episodes = _manifest_episode_map(manifest)
    expected_queries = set(manifest.episode_indices(split="train", role="query"))
    expected_queries.update(manifest.episode_indices(split="val", role="query"))
    actual_queries = set(sidecar.query_episode_indices)
    if actual_queries != expected_queries:
        raise ValueError(
            "pairing sidecar 必须精确覆盖 Manifest 的 train/val Query episodes；"
            f"missing={sorted(expected_queries - actual_queries)}, "
            f"extra={sorted(actual_queries - expected_queries)}。"
        )

    for epoch_index, epoch in enumerate(sidecar.epochs):
        for query_episode, pairing in epoch.items():
            query = episodes[query_episode]
            try:
                demo = episodes[pairing.demo_episode_index]
            except KeyError as error:
                raise ValueError(
                    f"sidecar Demo episode={pairing.demo_episode_index} 不在 Manifest 中。"
                ) from error
            if query.role != "query" or demo.role != "demo":
                raise ValueError(
                    f"sidecar episode role 错误：Query={query_episode}/{query.role}, "
                    f"Demo={pairing.demo_episode_index}/{demo.role}。"
                )
            if query.split != demo.split:
                raise ValueError(
                    f"Query episode={query_episode} 与 Demo episode="
                    f"{pairing.demo_episode_index} 不在同一 split。"
                )
            if query.task_index != demo.task_index:
                raise ValueError(
                    f"Query episode={query_episode} 与 Demo episode="
                    f"{pairing.demo_episode_index} 的 task 不一致。"
                )
            expected_demo_id = manifest.demo_id(pairing.demo_episode_index)
            if pairing.demo_id != expected_demo_id:
                raise ValueError(
                    f"Demo episode={pairing.demo_episode_index} 应使用 "
                    f"demo_id={expected_demo_id!r}，实际为 {pairing.demo_id!r}。"
                )

            query_metadata = metadata.episodes[query_episode]
            demo_metadata = metadata.episodes[pairing.demo_episode_index]
            query_length = int(query_metadata["dataset_to_index"]) - int(query_metadata["dataset_from_index"])
            demo_length = int(demo_metadata["dataset_to_index"]) - int(demo_metadata["dataset_from_index"])
            if len(pairing.local_anchors) != query_length:
                raise ValueError(
                    f"sidecar epoch={epoch_index}, Query episode={query_episode} 包含 "
                    f"{len(pairing.local_anchors)} 个 anchors，但 episode 长度为 {query_length}。"
                )
            if max(pairing.local_anchors) >= demo_length:
                raise ValueError(
                    f"sidecar epoch={epoch_index}, Query episode={query_episode} 包含越界 "
                    f"Demo anchor；Demo episode={pairing.demo_episode_index} 长度为 {demo_length}。"
                )


def _make_lerobot_dataset(
    cfg: Any,
    manifest: LiberoDataManifest,
    *,
    episodes: list[int],
    delta_timestamps: dict[str, list] | None,
    image_transforms: Any,
) -> LeRobotDataset:
    """用统一的 Manifest revision 创建一个 LeRobot episode 子集。"""
    return LeRobotDataset(
        manifest.repo_id,
        root=cfg.dataset.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=manifest.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
    )


def _add_imagenet_stats(dataset: LeRobotDataset) -> None:
    """与 LeRobot 官方 Dataset factory 保持相同的 ImageNet stats 行为。"""
    for key in dataset.meta.camera_keys:
        if key in dataset.meta.depth_keys:
            continue
        dataset.meta.stats.setdefault(key, {})
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


def _raw_feature_key(
    canonical_key: str,
    features: dict[str, Any],
    rename_map: dict[str, str],
) -> str:
    """找到 processor rename 之前的 Dataset 字段名。"""
    candidates = [key for key in features if rename_map.get(key, key) == canonical_key]
    if len(candidates) != 1:
        raise KeyError(f"无法唯一确定 {canonical_key!r} 的原始 Dataset key：{candidates}。")
    return candidates[0]


def make_smolvla_icl_train_eval_datasets(
    cfg: Any,
) -> tuple[SmolVLAICLQueryDataset, SmolVLAICLQueryDataset]:
    """根据 Manifest 和离线 DTW sidecar 构建训练/验证 Dataset。"""
    policy_cfg = cfg.trainable_config
    if cfg.dataset.streaming:
        raise NotImplementedError("SmolVLA-ICL sidecar 只支持 map-style Dataset。")
    if policy_cfg.data_manifest_path is None:
        raise ValueError("SmolVLA-ICL 训练必须配置 policy.data_manifest_path。")
    if policy_cfg.pairing_sidecar_path is None:
        raise ValueError("SmolVLA-ICL 训练必须配置 policy.pairing_sidecar_path。")
    if policy_cfg.training_demo_cache_dir is None:
        raise ValueError("SmolVLA-ICL 训练必须配置 policy.training_demo_cache_dir。")

    manifest = LiberoDataManifest.load(policy_cfg.data_manifest_path)
    sidecar = PairingSidecar.load(policy_cfg.pairing_sidecar_path)
    metadata = load_dataset_metadata(
        manifest.repo_id,
        root=cfg.dataset.root,
        revision=manifest.revision,
        repo_type=cfg.dataset.repo_type,
    )
    _validate_data_contract(cfg, manifest, sidecar, metadata)
    _validate_pairings(sidecar, manifest, metadata)

    # 在创建 Dataset/DataLoader worker 之前检查所有 Global cache。主进程
    # 完整读取 Tensor；其他 rank 经过训练入口的 barrier 后只复核 manifest
    # 和文件存在性，避免并发扫描共享存储。
    state_key = _raw_feature_key(OBS_STATE, metadata.features, cfg.rename_map)
    preflight_global_demo_cache(
        policy_cfg.training_demo_cache_dir,
        manifest=manifest,
        sidecar=sidecar,
        config=policy_cfg.global_encoder,
        state_normalizer=DemoStateNormalizer.from_dataset_stats(
            metadata.stats,
            state_key=state_key,
        ),
        state_key=state_key,
        validate_tensors=is_main_process(),
    )

    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata, cfg.rename_map)
    train_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    train_query = _make_lerobot_dataset(
        cfg,
        manifest,
        episodes=manifest.episode_indices(split="train", role="query"),
        delta_timestamps=delta_timestamps,
        image_transforms=train_transforms,
    )
    val_query = _make_lerobot_dataset(
        cfg,
        manifest,
        episodes=manifest.episode_indices(split="val", role="query"),
        delta_timestamps=delta_timestamps,
        image_transforms=None,
    )
    demo_dataset = _make_lerobot_dataset(
        cfg,
        manifest,
        episodes=sidecar.demo_episode_indices,
        delta_timestamps=None,
        image_transforms=None,
    )
    if cfg.dataset.use_imagenet_stats:
        _add_imagenet_stats(train_query)
        _add_imagenet_stats(val_query)

    local_reader = LeRobotLocalDemoReader(
        dataset=demo_dataset,
        demo_id_to_episode=sidecar.demo_id_to_episode,
        image_key=sidecar.image_key,
        state_key=state_key,
        chunk_size=policy_cfg.demo_alignment.local_chunk_size,
        anchor_position=policy_cfg.demo_alignment.local_anchor_position,
    )
    resolver = PairingSidecarResolver(sidecar)
    resolver.validate_query_episodes(train_query.episodes)
    resolver.validate_query_episodes(val_query.episodes)
    return (
        SmolVLAICLQueryDataset(train_query, resolver, local_reader),
        SmolVLAICLQueryDataset(val_query, resolver, local_reader),
    )
