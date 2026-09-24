"""Local Demo episode 级 uint8 frame cache 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla_icl.configuration_smolvla_icl import DemoAlignmentConfig
from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
)
from lerobot.policies.smolvla_icl.data.local_rgb_cache import (
    LocalRGBFrameCacheManifest,
    LocalRGBFrameCacheStore,
    LocalRGBFrameEntry,
    local_rgb_cache_identity,
    preflight_local_rgb_cache,
)
from lerobot.policies.smolvla_icl.data.sidecar import EpisodeDemoPairing, PairingSidecar


def _manifest_and_sidecar() -> tuple[LiberoDataManifest, PairingSidecar]:
    manifest = LiberoDataManifest(
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
                length=4,
                split=split,
                role=role,
            )
            for index, task_index, split, role in (
                (0, 10, "train", "demo"),
                (1, 10, "train", "query"),
                (2, 11, "val", "demo"),
                (3, 11, "val", "query"),
                (4, 12, "test", "demo"),
                (5, 12, "test", "query"),
            )
        ),
    )
    alignment = DemoAlignmentConfig.for_action_chunking(control_hz=10.0, n_action_steps=5)
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="matcher@revision",
        image_key=manifest.image_key,
        stats_fingerprint="b" * 64,
        alignment_config=alignment,
        control_hz=10.0,
        n_action_steps=5,
        query_window_replans=4,
        epochs=(
            {
                1: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1, 2, 3)),
                3: EpisodeDemoPairing(manifest.demo_id(2), 2, (0, 1, 2, 3)),
            },
        ),
    )
    return manifest, sidecar


def test_local_rgb_cache_preflight_and_mmap_slice(tmp_path: Path) -> None:
    manifest, sidecar = _manifest_and_sidecar()
    identity = local_rgb_cache_identity(manifest)
    entries = []
    expected = {}
    for demo_id, episode_index in sorted(sidecar.demo_id_to_episode.items()):
        frames = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
        frames = frames + episode_index
        relative = Path("frames") / identity[:20] / f"episode-{episode_index}.npy"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, frames, allow_pickle=False)
        expected[demo_id] = frames
        entries.append(
            LocalRGBFrameEntry(
                demo_id=demo_id,
                episode_index=episode_index,
                episode_length=4,
                shape=(4, 3, 2, 2),
                file=str(relative),
            )
        )

    cache_manifest = LocalRGBFrameCacheManifest.create(manifest, tuple(entries))
    cache_manifest.save(tmp_path)
    restored = preflight_local_rgb_cache(tmp_path, manifest=manifest, sidecar=sidecar)
    store = LocalRGBFrameCacheStore(tmp_path, restored, open_entries=1)

    demo_id = manifest.demo_id(0)
    result = store.read(demo_id, torch.tensor([1, 3], dtype=torch.long))
    assert result.device.type == "cpu"
    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.from_numpy(expected[demo_id][[1, 3]]))
