"""离线 Pairing/DTW sidecar builder 的因果性与数据边界测试."""

from __future__ import annotations

from pathlib import Path

import torch

from lerobot.policies.smolvla_icl.components.demo_alignment import DemoEmbeddingCache
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import DemoAlignmentConfig
from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
)
from lerobot.policies.smolvla_icl.data.pairing_builder import (
    LIBERO_MATCHING_STATE_EXCLUDED_INDICES,
    MatcherEpisodeCacheStore,
    _load_alignment_config,
    align_query_to_demo,
    build_pairing_sidecar,
)
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer


def _alignment_cache() -> DemoEmbeddingCache:
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        min_valid_fraction=0.5,
        rgb_only=True,
        dtw_forward_window=4,
    )
    normalizer = DemoStateNormalizer(mean=torch.zeros(2), std=torch.ones(2))
    return DemoEmbeddingCache.from_embeddings(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
            ]
        ),
        torch.arange(4, dtype=torch.float64) / 10.0,
        state_normalizer=normalizer,
        config=config,
    )


def _manifest() -> LiberoDataManifest:
    assignments = (
        (0, 10, "train", "demo"),
        (1, 10, "train", "demo"),
        (2, 10, "train", "query"),
        (3, 11, "val", "demo"),
        (4, 11, "val", "query"),
        (5, 12, "test", "demo"),
        (6, 12, "test", "query"),
    )
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
                length=4,
                split=split,
                role=role,
            )
            for index, task_index, split, role in assignments
        ),
    )


def test_offline_alignment_returns_dense_monotonic_anchors() -> None:
    anchors = align_query_to_demo(_alignment_cache(), _alignment_cache())

    assert len(anchors) == 4
    assert list(anchors) == sorted(anchors)
    assert all(0 <= anchor < 4 for anchor in anchors)


def test_libero_pairing_default_excludes_both_gripper_state_dimensions() -> None:
    config = _load_alignment_config(None)

    assert config.matching_state_excluded_indices == LIBERO_MATCHING_STATE_EXCLUDED_INDICES


def test_builder_rotates_train_demo_and_keeps_validation_fixed() -> None:
    manifest = _manifest()
    caches = {episode.episode_index: _alignment_cache() for episode in manifest.episodes}
    sidecar = build_pairing_sidecar(
        manifest,
        matcher_snapshot="lerobot/smolvla_base@revision",
        stats_fingerprint="b" * 64,
        alignment_config=_alignment_cache().config,
        n_action_steps=1,
        query_window_replans=2,
        episode_cache_loader=caches.__getitem__,
        num_epochs=3,
        seed=7,
    )

    assert sidecar.manifest_fingerprint == manifest.fingerprint
    assert sidecar.control_hz == manifest.fps
    assert sidecar.n_action_steps == 1
    assert sidecar.query_window_replans == 2
    assert sidecar.alignment_config == _alignment_cache().config
    assert sidecar.query_episode_indices == [2, 4]
    assert sidecar.epochs[0][2].demo_episode_index != sidecar.epochs[1][2].demo_episode_index
    assert {epoch[4].demo_episode_index for epoch in sidecar.epochs} == {3}
    assert all(len(epoch[2].local_anchors) == 4 for epoch in sidecar.epochs)


def test_matcher_episode_cache_round_trip(tmp_path: Path) -> None:
    manifest = _manifest()
    cache = _alignment_cache()
    store = MatcherEpisodeCacheStore(
        tmp_path,
        manifest=manifest,
        matcher_snapshot="lerobot/smolvla_base@revision",
        alignment_config=cache.config,
        state_normalizer=cache.state_normalizer,
        stats_fingerprint="b" * 64,
        resize_imgs_with_padding=(512, 512),
    )

    store.save(2, cache)
    restored = store.load(2)

    assert restored is not None
    assert restored.config == cache.config
    assert torch.equal(restored.anchor_indices, cache.anchor_indices)
    assert torch.equal(restored.visual_chunk_embeddings, cache.visual_chunk_embeddings)
