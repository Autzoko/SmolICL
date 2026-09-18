"""SmolVLA-ICL 训练 Dataset 组装入口。"""

from __future__ import annotations

from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from .dataset import SmolVLAICLQueryDataset
from .reader import LeRobotLocalDemoReader
from .sidecar import PairingSidecar, PairingSidecarResolver


__all__ = ["prepare_smolvla_icl_datasets"]


def _selected_episodes(dataset: Any) -> list[int]:
    episodes = dataset.episodes
    return list(episodes) if episodes is not None else list(range(dataset.meta.total_episodes))


def _clone_query_dataset(
    source: LeRobotDataset,
    cfg: Any,
    episode_indices: list[int],
) -> LeRobotDataset:
    """保留 LeRobot Query 时间窗口/增广配置，仅收缩 episode 子集。"""
    if not episode_indices:
        raise ValueError("SmolVLA-ICL Query episode 子集不能为空。")
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episode_indices,
        delta_timestamps=source.delta_timestamps,
        image_transforms=source.image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
    )


def _validate_pairings(sidecar: PairingSidecar, metadata: Any) -> None:
    """在启动 worker 前校验 episode 长度、anchor 边界和任务一致性。"""
    for epoch_index, epoch in enumerate(sidecar.epochs):
        for query_episode, pairing in epoch.items():
            query_metadata = metadata.episodes[query_episode]
            demo_metadata = metadata.episodes[pairing.demo_episode_index]
            query_length = int(query_metadata["dataset_to_index"]) - int(
                query_metadata["dataset_from_index"]
            )
            demo_length = int(demo_metadata["dataset_to_index"]) - int(
                demo_metadata["dataset_from_index"]
            )
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
            if set(query_metadata["tasks"]).isdisjoint(demo_metadata["tasks"]):
                raise ValueError(
                    f"Query episode={query_episode} 与 Demo episode="
                    f"{pairing.demo_episode_index} 的 task 不一致。"
                )


def _raw_feature_key(canonical_key: str, features: dict[str, Any], rename_map: dict[str, str]) -> str:
    """找到 processor rename 之前的 Dataset 字段名。"""
    candidates = [key for key in features if rename_map.get(key, key) == canonical_key]
    if len(candidates) != 1:
        raise KeyError(
            f"无法唯一确定 {canonical_key!r} 的原始 Dataset key：{candidates}。"
        )
    return candidates[0]


def prepare_smolvla_icl_datasets(
    cfg: Any,
    train_dataset: LeRobotDataset,
    eval_dataset: LeRobotDataset | None,
) -> tuple[SmolVLAICLQueryDataset, SmolVLAICLQueryDataset | None]:
    """用 sidecar 和 Demo-only reader 包装标准 LeRobot Query Dataset。"""
    policy_cfg = cfg.trainable_config
    if cfg.dataset.streaming:
        raise NotImplementedError("SmolVLA-ICL pairing sidecar 当前只支持 map-style Dataset。")
    if policy_cfg.pairing_sidecar_path is None:
        raise ValueError(
            "SmolVLA-ICL 训练必须配置 policy.pairing_sidecar_path。"
        )

    sidecar = PairingSidecar.load(policy_cfg.pairing_sidecar_path)
    _validate_pairings(sidecar, train_dataset.meta)
    resolver = PairingSidecarResolver(sidecar)

    # train/eval 先按 LeRobot 原生规则划分，然后只保留 sidecar
    # 声明的 Query 子集。Demo episodes 因此不会再成为 Query 样本。
    sidecar_queries = set(sidecar.query_episode_indices)
    train_candidates = set(_selected_episodes(train_dataset))
    eval_candidates = set(_selected_episodes(eval_dataset)) if eval_dataset is not None else set()
    available_candidates = train_candidates | eval_candidates
    unknown_queries = sorted(sidecar_queries - available_candidates)
    unknown_demos = sorted(set(sidecar.demo_episode_indices) - available_candidates)
    if unknown_queries or unknown_demos:
        raise ValueError(
            "pairing sidecar 引用了 dataset.episodes 范围外的 episode："
            f"query={unknown_queries}, demo={unknown_demos}。"
        )

    train_query_episodes = sorted(sidecar_queries & train_candidates)
    train_dataset = _clone_query_dataset(train_dataset, cfg, train_query_episodes)
    resolver.validate_query_episodes(train_query_episodes)
    if eval_dataset is not None:
        eval_query_episodes = sorted(sidecar_queries & eval_candidates)
        eval_dataset = _clone_query_dataset(eval_dataset, cfg, eval_query_episodes)
        resolver.validate_query_episodes(eval_query_episodes)

    # Demo Dataset 与 Query 来自同一 LeRobot 数据集，但只打开
    # sidecar 引用的 episode，且不带 action delta 或训练图像增广。
    demo_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=sidecar.demo_episode_indices,
        delta_timestamps=None,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        depth_output_unit=cfg.dataset.depth_output_unit,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
    )
    if sidecar.image_key not in demo_dataset.meta.features:
        raise KeyError(
            f"pairing sidecar image_key={sidecar.image_key!r} 不存在于 Dataset。"
        )
    state_key = _raw_feature_key(
        OBS_STATE,
        demo_dataset.meta.features,
        cfg.rename_map,
    )
    local_reader = LeRobotLocalDemoReader(
        dataset=demo_dataset,
        demo_id_to_episode=sidecar.demo_id_to_episode,
        image_key=sidecar.image_key,
        state_key=state_key,
        chunk_size=policy_cfg.demo_alignment.local_chunk_size,
        anchor_position=policy_cfg.demo_alignment.local_anchor_position,
    )

    train = SmolVLAICLQueryDataset(train_dataset, resolver, local_reader)
    evaluation = (
        SmolVLAICLQueryDataset(eval_dataset, resolver, local_reader)
        if eval_dataset is not None
        else None
    )
    return train, evaluation
