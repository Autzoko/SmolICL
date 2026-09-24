"""一次性解码 Demo episode，生成训练期 Local RGB uint8 frame cache。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

from .libero_manifest import LiberoDataManifest
from .local_rgb_cache import (
    LocalRGBFrameCacheManifest,
    LocalRGBFrameEntry,
    local_rgb_cache_identity,
    preflight_local_rgb_cache,
)
from .reader import LeRobotMatcherEpisodeReader
from .sidecar import PairingSidecar

__all__ = ["build_local_rgb_cache"]


def _filename(demo_id: str) -> str:
    return f"{hashlib.sha256(demo_id.encode()).hexdigest()}.npy"


def _save_npy_atomic(path: Path, images: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, images.contiguous().numpy(), allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _channel_first_image_shape(dataset_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """将 Dataset 的单帧 RGB 元数据统一为缓存使用的 ``(3,H,W)``。

    LeRobot 的视频 feature 使用 ``(H,W,C)`` 元数据，但解码器返回
    ``(T,C,H,W)``；部分 image feature 则直接声明 ``(C,H,W)``。这里只
    归一化元数据，不转置已经解码的图像，避免额外复制完整 episode。
    """
    if len(dataset_shape) != 3:
        raise ValueError("Local RGB Dataset feature shape 必须是三维单帧图像。")
    if dataset_shape[0] == 3:
        return dataset_shape
    if dataset_shape[-1] == 3:
        height, width, _ = dataset_shape
        return 3, height, width
    raise ValueError("Local RGB Dataset feature shape 必须是 RGB 的 (3,H,W) 或 (H,W,3)。")


def build_local_rgb_cache(
    *,
    manifest_path: str | Path,
    sidecar_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    state_key: str = OBS_STATE,
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
    overwrite: bool = False,
) -> LocalRGBFrameCacheManifest:
    """为 Sidecar 引用的每条 Demo 解码一次完整 RGB episode。"""
    manifest = LiberoDataManifest.load(manifest_path)
    sidecar = PairingSidecar.load(sidecar_path)
    if sidecar.manifest_fingerprint != manifest.fingerprint or sidecar.image_key != manifest.image_key:
        raise ValueError("Sidecar 与 Manifest 的 Dataset/camera 身份不一致。")

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
    reader = LeRobotMatcherEpisodeReader(
        dataset=dataset,
        image_key=manifest.image_key,
        state_key=state_key,
    )
    output = Path(output_dir).expanduser()
    identity = local_rgb_cache_identity(manifest)
    frames_dir = output / "frames" / identity[:20]
    episode_by_index = {episode.episode_index: episode for episode in manifest.episodes}
    dataset_image_shape = tuple(
        int(value) for value in dataset.meta.features[manifest.image_key]["shape"]
    )
    image_shape = _channel_first_image_shape(dataset_image_shape)
    entries: list[LocalRGBFrameEntry] = []

    for demo_id, episode_index in sorted(sidecar.demo_id_to_episode.items()):
        episode = episode_by_index[episode_index]
        path = frames_dir / _filename(demo_id)
        expected_shape = (episode.length, *image_shape)
        reusable = False
        if path.is_file() and not overwrite:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            reusable = array.dtype == np.uint8 and tuple(array.shape) == expected_shape
        if not reusable:
            logging.info("Decoding Local RGB cache for episode %d (%s)", episode_index, demo_id)
            images = reader(episode_index).images
            if images.device.type != "cpu" or images.dtype != torch.uint8:
                raise TypeError("Local RGB cache builder 必须得到 CPU uint8 图像。")
            if images.ndim != 4 or tuple(images.shape) != expected_shape:
                raise ValueError(f"Local RGB episode shape 不正确：{demo_id}")
            _save_npy_atomic(path, images)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        entries.append(
            LocalRGBFrameEntry(
                demo_id=demo_id,
                episode_index=episode_index,
                episode_length=episode.length,
                shape=tuple(int(value) for value in array.shape),
                file=str(path.relative_to(output)),
            )
        )

    cache_manifest = LocalRGBFrameCacheManifest.create(manifest, tuple(entries))
    cache_manifest.save(output)
    preflight_local_rgb_cache(output, manifest=manifest, sidecar=sidecar)
    logging.info("Saved Local RGB cache: %s (%d demos)", output, len(entries))
    return cache_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU uint8 Local Demo RGB frame cache.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-key", default=OBS_STATE)
    parser.add_argument("--video-backend")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_local_rgb_cache(
        manifest_path=args.manifest,
        sidecar_path=args.sidecar,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        state_key=args.state_key,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
