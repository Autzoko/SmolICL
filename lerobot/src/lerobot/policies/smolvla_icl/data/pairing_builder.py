"""为 SmolVLA-ICL 离线生成 Query–Demo 配对和逐帧 DTW anchor。

Builder 严格使用 Manifest 声明的数据边界：Query 只能从同一
``(split, task)`` 的 Demo 池中选择 Demo，test split 不进入 sidecar。
离线读取完整 Query 并不改变因果约束：每个 Query chunk 由当前帧和
历史帧构成，DTW 按时间顺序逐步推进，不读取未来 Query 特征。

冻结 SigLIP+connector 的每条 episode Matcher cache 持久化到磁盘。同一
episode 只需解码和编码一次，之后更换 epoch 配对只运行轻量 DTW。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import OBS_STATE

from ..components.demo_alignment import (
    DemoEmbeddingCache,
    OnlineDTWMatcher,
    SmolVLASigLIPHandle,
    load_smolvla_siglip,
    pool_visual_tokens,
)
from ..configuration_smolvla_icl import DemoAlignmentConfig
from .libero_manifest import LiberoDataManifest, LiberoManifestEpisode
from .reader import LeRobotMatcherEpisodeReader, MatcherEpisodeData
from .sidecar import EpisodeDemoPairing, PairingSidecar
from .state import DemoStateNormalizer

EpisodeCacheLoader = Callable[[int], DemoEmbeddingCache]


def _stable_seed(seed: int, namespace: str, episode_index: int) -> int:
    """生成不受 Python hash 随机化影响的 episode 局部种子。"""
    digest = hashlib.sha256(f"{seed}:{namespace}:{episode_index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _demo_orders(
    manifest: LiberoDataManifest,
    *,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    """为每条 train/val Query 生成确定性的同 split/task Demo 顺序。"""
    demo_pools: dict[tuple[str, int], list[int]] = defaultdict(list)
    queries: list[LiberoManifestEpisode] = []
    for episode in manifest.episodes:
        if episode.split == "test":
            continue
        if episode.role == "demo":
            demo_pools[(episode.split, episode.task_index)].append(episode.episode_index)
        else:
            queries.append(episode)

    orders: dict[int, tuple[int, ...]] = {}
    for query in queries:
        candidates = sorted(demo_pools[(query.split, query.task_index)])
        if not candidates:
            raise ValueError(f"Query episode={query.episode_index} 没有同 split/task Demo 候选。")
        rng = random.Random(_stable_seed(seed, "demo-order", query.episode_index))
        rng.shuffle(candidates)
        orders[query.episode_index] = tuple(candidates)
    return orders


def align_query_to_demo(
    query_cache: DemoEmbeddingCache,
    demo_cache: DemoEmbeddingCache,
) -> tuple[int, ...]:
    """用与 rollout 相同的因果前向 DTW 生成逐原始帧 Demo anchor。"""
    if query_cache.config != demo_cache.config:
        raise ValueError("Query 与 Demo Matcher cache 必须使用同一份对齐配置。")
    if query_cache.state_normalizer.signature != demo_cache.state_normalizer.signature:
        raise ValueError("Query 与 Demo Matcher cache 必须使用同一份 State 统计量。")

    matcher = OnlineDTWMatcher(demo_cache)
    first_valid_demo = int(torch.nonzero(demo_cache.chunk_valid_mask).flatten()[0])
    current_anchor = int(demo_cache.anchor_indices[first_valid_demo])
    anchors = [current_anchor] * query_cache.num_frames

    # Query cache 的 anchor_indices 是 alignment_hz 时间网格在原始 episode
    # 中的下标。两次 Matcher 更新之间的原始帧沿用最近一次结果，
    # 与 action chunk 执行期间不重新匹配的真实推理语义一致。
    for chunk_index in range(query_cache.num_chunks):
        query_chunk = query_cache.get_chunk(chunk_index)
        if query_chunk.valid:
            current_anchor = matcher.update(query_chunk).demo_observation_index
        start = int(query_cache.anchor_indices[chunk_index])
        end = (
            int(query_cache.anchor_indices[chunk_index + 1])
            if chunk_index + 1 < query_cache.num_chunks
            else query_cache.num_frames
        )
        anchors[start:end] = [current_anchor] * (end - start)
    return tuple(anchors)


def build_pairing_sidecar(
    manifest: LiberoDataManifest,
    *,
    matcher_snapshot: str,
    episode_cache_loader: EpisodeCacheLoader,
    num_epochs: int = 1,
    seed: int = 42,
) -> PairingSidecar:
    """从已编码 episode cache 构建可直接进入训练的 sidecar。"""
    if not matcher_snapshot:
        raise ValueError("matcher_snapshot 不能为空。")
    if num_epochs < 1:
        raise ValueError("num_epochs 必须大于 0。")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数。")

    episode_by_index = {episode.episode_index: episode for episode in manifest.episodes}
    demo_orders = _demo_orders(manifest, seed=seed)
    query_indices = sorted(demo_orders)
    pair_anchors: dict[tuple[int, int], tuple[int, ...]] = {}
    fixed_val_pairings: dict[int, EpisodeDemoPairing] = {}
    epochs: list[dict[int, EpisodeDemoPairing]] = []

    for epoch_index in range(num_epochs):
        epoch: dict[int, EpisodeDemoPairing] = {}
        for query_episode in query_indices:
            query = episode_by_index[query_episode]
            candidates = demo_orders[query_episode]
            # validation 始终使用 epoch 0 的配对，保证指标可比；
            # train 按确定性乱序轮换 Demo，每个 epoch 内保持不变。
            candidate_offset = 0 if query.split == "val" else epoch_index
            demo_episode = candidates[candidate_offset % len(candidates)]

            if query.split == "val" and query_episode in fixed_val_pairings:
                epoch[query_episode] = fixed_val_pairings[query_episode]
                continue

            pair_key = (query_episode, demo_episode)
            anchors = pair_anchors.get(pair_key)
            if anchors is None:
                anchors = align_query_to_demo(
                    episode_cache_loader(query_episode),
                    episode_cache_loader(demo_episode),
                )
                pair_anchors[pair_key] = anchors
            pairing = EpisodeDemoPairing(
                demo_id=manifest.demo_id(demo_episode),
                demo_episode_index=demo_episode,
                local_anchors=anchors,
            )
            epoch[query_episode] = pairing
            if query.split == "val":
                fixed_val_pairings[query_episode] = pairing
        epochs.append(epoch)

    return PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot=matcher_snapshot,
        image_key=manifest.image_key,
        epochs=tuple(epochs),
    )


class MatcherEpisodeCacheStore:
    """按 episode 持久化冻结 Matcher 缓存，并拒绝身份不匹配的旧文件。"""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: LiberoDataManifest,
        matcher_snapshot: str,
        alignment_config: DemoAlignmentConfig,
        state_normalizer: DemoStateNormalizer,
        resize_imgs_with_padding: tuple[int, int] | None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.alignment_config = alignment_config
        self.identity: dict[str, Any] = {
            "manifest_fingerprint": manifest.fingerprint,
            "matcher_snapshot": matcher_snapshot,
            "image_key": manifest.image_key,
            "alignment_config": asdict(alignment_config),
            "state_normalization_signature": state_normalizer.signature,
            "resize_imgs_with_padding": resize_imgs_with_padding,
        }
        canonical = json.dumps(
            self.identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.directory = self.root / hashlib.sha256(canonical).hexdigest()[:20]

    def path_for(self, episode_index: int) -> Path:
        return self.directory / f"episode_{episode_index:06d}.pt"

    def load(self, episode_index: int) -> DemoEmbeddingCache | None:
        path = self.path_for(episode_index)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("version") != 1 or payload.get("identity") != self.identity:
            raise ValueError(f"Matcher cache 身份不匹配：{path}")
        if int(payload.get("episode_index", -1)) != episode_index:
            raise ValueError(f"Matcher cache episode index 不匹配：{path}")
        cache = DemoEmbeddingCache.from_serializable(payload["cache"])
        return cache.to(self.alignment_config.cache_device)

    def save(self, episode_index: int, cache: DemoEmbeddingCache) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(episode_index)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        try:
            torch.save(
                {
                    "version": 1,
                    "identity": self.identity,
                    "episode_index": episode_index,
                    "cache": cache.to_serializable(),
                },
                temporary,
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def _prepare_matcher_images(
    images: Tensor,
    *,
    device: torch.device,
    resize_imgs_with_padding: tuple[int, int] | None,
) -> Tensor:
    """复用 SmolVLA 的 RGB range、resize 和 SigLIP 归一化规则。"""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Matcher RGB 必须是 (T,3,H,W) Tensor。")
    batch = images.to(device=device)
    if batch.dtype == torch.uint8:
        batch = batch.float().div_(255.0)
    elif batch.is_floating_point():
        batch = batch.float()
    else:
        raise TypeError("Matcher RGB 必须是 uint8 或浮点 Tensor。")
    if resize_imgs_with_padding is not None:
        target_width, target_height = resize_imgs_with_padding
        batch = resize_with_pad(batch, target_height, target_width, pad_value=0)
    return batch.mul_(2.0).sub_(1.0)


@torch.inference_mode()
def _encode_episode(
    episode: MatcherEpisodeData,
    *,
    siglip: SmolVLASigLIPHandle,
    alignment_config: DemoAlignmentConfig,
    state_normalizer: DemoStateNormalizer,
    resize_imgs_with_padding: tuple[int, int] | None,
) -> DemoEmbeddingCache:
    """分批编码完整 episode，只保留 Matcher 需要的 pooled feature。"""
    feature_batches: list[Tensor] = []
    batch_size = alignment_config.demo_encode_batch_size
    for start in range(0, len(episode.images), batch_size):
        image_batch = _prepare_matcher_images(
            episode.images[start : start + batch_size],
            device=siglip.device,
            resize_imgs_with_padding=resize_imgs_with_padding,
        )
        tokens = siglip.encode_visual_tokens(image_batch)
        feature_batches.append(
            pool_visual_tokens(tokens, normalize=False, eps=alignment_config.eps).detach().cpu()
        )
    return DemoEmbeddingCache.from_embeddings(
        torch.cat(feature_batches),
        episode.states,
        episode.timestamps,
        state_normalizer=state_normalizer,
        config=alignment_config,
    )


def _load_alignment_config(path: str | Path | None) -> DemoAlignmentConfig:
    if path is None:
        return DemoAlignmentConfig()
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("alignment config 顶层必须是 JSON object。")
    if "demo_alignment" in payload:
        payload = payload["demo_alignment"]
    if not isinstance(payload, dict):
        raise TypeError("demo_alignment 必须是 JSON object。")
    for key in ("matching_state_excluded_indices", "matching_state_velocity_scale"):
        if payload.get(key) is not None:
            payload[key] = tuple(payload[key])
    return DemoAlignmentConfig(**payload)


def build_pairing_sidecar_from_dataset(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
    matcher_cache_dir: str | Path,
    matcher_model: str | Path = "lerobot/smolvla_base",
    matcher_revision: str | None = None,
    alignment_config: DemoAlignmentConfig | None = None,
    state_key: str = OBS_STATE,
    num_epochs: int = 1,
    seed: int = 42,
    device: str = "cuda",
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
    local_files_only: bool = False,
) -> PairingSidecar:
    """读取 LeRobot Dataset、编码 Matcher cache 并生成 sidecar。"""
    manifest = LiberoDataManifest.load(manifest_path)
    cfg = alignment_config or DemoAlignmentConfig()
    matcher_path = Path(matcher_model).expanduser()
    if matcher_revision is None and not matcher_path.exists():
        raise ValueError(
            "Hub Matcher 必须显式传入 matcher_revision，避免 default branch "
            "变化后生成不可复现的 DTW sidecar。"
        )
    required_episodes = manifest.episode_indices(split="train")
    required_episodes.extend(manifest.episode_indices(split="val"))
    dataset = LeRobotDataset(
        manifest.repo_id,
        root=dataset_root,
        episodes=sorted(required_episodes),
        delta_timestamps=None,
        image_transforms=None,
        revision=manifest.revision,
        video_backend=video_backend,
        return_uint8=True,
        tolerance_s=tolerance_s,
    )
    if not math.isclose(
        float(dataset.meta.fps),
        manifest.fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Dataset fps 与 Manifest 不一致。")
    manifest_episodes = {episode.episode_index: episode for episode in manifest.episodes}
    for episode_index in required_episodes:
        metadata = dataset.meta.episodes[episode_index]
        actual_length = int(metadata["dataset_to_index"]) - int(metadata["dataset_from_index"])
        if actual_length != manifest_episodes[episode_index].length:
            raise ValueError(f"Dataset episode={episode_index} 长度与 Manifest 不一致。")
    if state_key not in dataset.meta.stats:
        raise KeyError(f"Dataset stats 中缺少 State key: {state_key!r}。")
    state_normalizer = DemoStateNormalizer.from_dataset_stats(
        dataset.meta.stats,
        state_key=state_key,
    )
    reader = LeRobotMatcherEpisodeReader(
        dataset=dataset,
        image_key=manifest.image_key,
        state_key=state_key,
    )

    model_config = SmolVLAConfig.from_pretrained(
        matcher_model,
        revision=matcher_revision,
        local_files_only=local_files_only,
    )
    model_config.device = device
    siglip = load_smolvla_siglip(
        matcher_model,
        device=device,
        freeze=True,
        config=model_config,
        revision=matcher_revision,
        local_files_only=local_files_only,
    )
    resize = model_config.resize_imgs_with_padding
    matcher_snapshot = (
        f"{matcher_model}@{matcher_revision}" if matcher_revision is not None else str(matcher_path.resolve())
    )
    store = MatcherEpisodeCacheStore(
        matcher_cache_dir,
        manifest=manifest,
        matcher_snapshot=matcher_snapshot,
        alignment_config=cfg,
        state_normalizer=state_normalizer,
        resize_imgs_with_padding=resize,
    )

    encoded_count = 0

    def load_episode_cache(episode_index: int) -> DemoEmbeddingCache:
        nonlocal encoded_count
        cached = store.load(episode_index)
        if cached is not None:
            return cached
        logging.info("Encoding Matcher cache for episode %d", episode_index)
        cache = _encode_episode(
            reader(episode_index),
            siglip=siglip,
            alignment_config=cfg,
            state_normalizer=state_normalizer,
            resize_imgs_with_padding=resize,
        )
        store.save(episode_index, cache)
        encoded_count += 1
        return cache

    sidecar = build_pairing_sidecar(
        manifest,
        matcher_snapshot=matcher_snapshot,
        episode_cache_loader=load_episode_cache,
        num_epochs=num_epochs,
        seed=seed,
    )
    sidecar.save(output_path)
    logging.info(
        "Saved pairing sidecar: %s (%d epochs, %d queries/epoch, %d newly encoded episodes)",
        output_path,
        len(sidecar.epochs),
        len(sidecar.query_episode_indices),
        encoded_count,
    )
    return sidecar


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SmolVLA-ICL offline Query-Demo DTW pairing sidecar.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matcher-cache-dir", required=True)
    parser.add_argument("--matcher-model", default="lerobot/smolvla_base")
    parser.add_argument("--matcher-revision")
    parser.add_argument("--alignment-config")
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_pairing_sidecar_from_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_path=args.output,
        matcher_cache_dir=args.matcher_cache_dir,
        matcher_model=args.matcher_model,
        matcher_revision=args.matcher_revision,
        alignment_config=_load_alignment_config(args.alignment_config),
        state_key=args.state_key,
        num_epochs=args.num_epochs,
        seed=args.seed,
        device=args.device,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "MatcherEpisodeCacheStore",
    "align_query_to_demo",
    "build_pairing_sidecar",
    "build_pairing_sidecar_from_dataset",
]
