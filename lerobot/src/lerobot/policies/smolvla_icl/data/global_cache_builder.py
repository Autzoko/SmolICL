"""批量生成 SmolVLA-ICL 训练所需的冻结 Global S3D cache。

该 CLI 只处理 pairing sidecar 实际引用的唯一 Demo。每条 Demo 解码一次，
S3D 编码一次，并以 ``demo_id`` 的稳定哈希写入独立 ``.pt`` 文件；全部完成后
才原子发布目录级 ``cache_manifest.json``。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from ..components.global_encoder import GlobalDemoEncoder
from ..configuration_smolvla_icl import GlobalEncoderConfig
from .cache import CachedTrainingDemo, DemoFeatureStore, precompute_training_demo
from .global_cache import (
    GlobalDemoCacheEntry,
    GlobalDemoCacheManifest,
    global_cache_identity,
    preflight_global_demo_cache,
    validate_global_demo_cache_entry,
)
from .libero_manifest import LiberoDataManifest
from .reader import LeRobotMatcherEpisodeReader
from .sidecar import PairingSidecar
from .state import DemoStateNormalizer

__all__ = ["build_global_demo_cache"]


def _load_global_config(path: str | Path) -> GlobalEncoderConfig:
    """读取完整 Policy config 或单独的 GlobalEncoderConfig JSON。"""
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Global config 顶层必须是 JSON object。")
    if "global_encoder" in payload:
        payload = payload["global_encoder"]
    if not isinstance(payload, dict):
        raise TypeError("global_encoder 必须是 JSON object。")
    normalized = dict(payload)
    for key in ("image_size", "image_mean", "image_std"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    return GlobalEncoderConfig(**normalized)


def _float_rgb(images: torch.Tensor) -> torch.Tensor:
    """把 LeRobot ``return_uint8`` 输出转换成 S3D 约定的 ``[0,1]``。"""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Global Demo RGB 必须是 (T,3,H,W) Tensor。")
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    if not images.is_floating_point():
        raise TypeError("Global Demo RGB 必须是 uint8 或浮点 Tensor。")
    result = images.float()
    if torch.any(result < 0) or torch.any(result > 1):
        raise ValueError("浮点 Global Demo RGB 必须位于 [0,1]。")
    return result


def _entry_from_cache(
    cached: CachedTrainingDemo,
    *,
    episode_index: int,
    episode_length: int,
) -> GlobalDemoCacheEntry:
    sample = cached.global_demo
    if sample.video_features.ndim != 2:
        raise ValueError(f"Global cache feature 不是 (K,D)：{cached.demo_id}")
    return GlobalDemoCacheEntry(
        demo_id=cached.demo_id,
        episode_index=episode_index,
        episode_length=episode_length,
        num_clips=int(sample.video_features.shape[0]),
        feature_dim=int(sample.video_features.shape[1]),
    )


def build_global_demo_cache(
    *,
    manifest_path: str | Path,
    sidecar_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    global_config: GlobalEncoderConfig,
    state_key: str = OBS_STATE,
    device: str = "cuda",
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
) -> GlobalDemoCacheManifest:
    """生成 sidecar 所需的全部 Global Demo cache，并执行最终全量校验。"""
    manifest = LiberoDataManifest.load(manifest_path)
    sidecar = PairingSidecar.load(sidecar_path)
    if sidecar.manifest_fingerprint != manifest.fingerprint:
        raise ValueError("pairing sidecar 与 data manifest 的 fingerprint 不一致。")
    if sidecar.image_key != manifest.image_key:
        raise ValueError("pairing sidecar 与 data manifest 的 image_key 不一致。")
    if not global_config.freeze_video_backbone:
        raise ValueError("Global cache 只能由冻结的 S3D backbone 生成。")

    dataset = LeRobotDataset(
        manifest.repo_id,
        root=dataset_root,
        episodes=sidecar.demo_episode_indices,
        delta_timestamps=None,
        image_transforms=None,
        revision=manifest.revision,
        video_backend=video_backend,
        return_uint8=True,
        tolerance_s=tolerance_s,
    )
    if not math.isclose(float(dataset.meta.fps), manifest.fps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Dataset fps 与 Manifest 不一致。")
    if state_key not in dataset.meta.features or state_key not in dataset.meta.stats:
        raise KeyError(f"Dataset feature/stats 中缺少 State key: {state_key!r}。")

    episode_by_index = {episode.episode_index: episode for episode in manifest.episodes}
    for demo_id, episode_index in sidecar.demo_id_to_episode.items():
        episode = episode_by_index.get(episode_index)
        if episode is None or episode.role != "demo":
            raise ValueError(f"sidecar 引用了无效 Demo episode={episode_index}。")
        if manifest.demo_id(episode_index) != demo_id:
            raise ValueError(f"sidecar 的 demo_id 与 Manifest 不一致：{demo_id}")
        metadata = dataset.meta.episodes[episode_index]
        actual_length = int(metadata["dataset_to_index"]) - int(metadata["dataset_from_index"])
        if actual_length != episode.length:
            raise ValueError(f"Demo episode={episode_index} 长度与 Manifest 不一致。")

    state_normalizer = DemoStateNormalizer.from_dataset_stats(
        dataset.meta.stats,
        state_key=state_key,
    )
    identity = global_cache_identity(
        manifest,
        config=global_config,
        state_normalizer=state_normalizer,
        state_key=state_key,
    )
    reader = LeRobotMatcherEpisodeReader(
        dataset=dataset,
        image_key=manifest.image_key,
        state_key=state_key,
    )
    store = DemoFeatureStore(output_dir, memory_entries=0)

    # 只把冻结 S3D 放上目标设备；Global Encoder 的可训练模块不参与
    # 离线缓存。
    encoder = GlobalDemoEncoder(global_config)
    encoder.video_backbone.to(torch.device(device))
    encoder.eval()

    entries: list[GlobalDemoCacheEntry] = []
    encoded_count = 0
    for demo_id, episode_index in sorted(sidecar.demo_id_to_episode.items()):
        episode = episode_by_index[episode_index]
        cache_path = store.path_for(demo_id)
        cached: CachedTrainingDemo | None = None
        if cache_path.is_file() and not overwrite:
            cached = store.load(demo_id)
            if cached.cache_identity != identity:
                raise ValueError(f"已有 Global cache 身份不匹配：{cache_path}；请使用 --overwrite 重新生成。")

        if cached is None:
            logging.info(
                "Encoding Global S3D cache for episode %d (%s)",
                episode_index,
                demo_id,
            )
            raw = reader(episode_index)
            cached = precompute_training_demo(
                demo_id,
                _float_rgb(raw.images),
                raw.states,
                raw.timestamps,
                state_normalizer=state_normalizer,
                global_encoder=encoder,
                cache_identity=identity,
            )
            store.save(cached)
            encoded_count += 1
        entry = _entry_from_cache(
            cached,
            episode_index=episode_index,
            episode_length=episode.length,
        )
        # manifest 尚未发布前先验证每个文件，避免失败时留下看似完整的
        # cache 目录。
        validate_global_demo_cache_entry(
            cached,
            entry,
            config=global_config,
            expected_identity=identity,
        )
        entries.append(entry)

    cache_manifest = GlobalDemoCacheManifest.create(
        manifest,
        config=global_config,
        state_normalizer=state_normalizer,
        state_key=state_key,
        entries=tuple(entries),
    )
    manifest_file = cache_manifest.save(output_dir)
    preflight_global_demo_cache(
        output_dir,
        manifest=manifest,
        sidecar=sidecar,
        config=global_config,
        state_normalizer=state_normalizer,
        state_key=state_key,
    )
    logging.info(
        "Saved Global Demo cache manifest: %s (%d demos, %d newly encoded)",
        manifest_file,
        len(entries),
        encoded_count,
    )
    return cache_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen S3D Global Demo cache for SmolVLA-ICL training."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--global-config",
        required=True,
        help="JSON file containing GlobalEncoderConfig or a global_encoder object.",
    )
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_global_demo_cache(
        manifest_path=args.manifest,
        sidecar_path=args.sidecar,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        global_config=_load_global_config(args.global_config),
        state_key=args.state_key,
        device=args.device,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
