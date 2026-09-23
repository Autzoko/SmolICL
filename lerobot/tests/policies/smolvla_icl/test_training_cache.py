"""SmolVLA-ICL 冻结 Global feature cache 和 batch 去重测试。"""

from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn

from lerobot.policies.smolvla_icl.components.global_encoder import GlobalDemoEncoder
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import GlobalEncoderConfig
from lerobot.policies.smolvla_icl.data.cache import (
    DemoFeatureStore,
    SmolVLAICLCollator,
    cache_training_demo,
)
from lerobot.policies.smolvla_icl.data.contracts import DemoSampleRef
from lerobot.policies.smolvla_icl.data.global_cache import (
    GlobalDemoCacheEntry,
    GlobalDemoCacheManifest,
    global_cache_identity,
    preflight_global_demo_cache,
)
from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
)
from lerobot.policies.smolvla_icl.data.sidecar import EpisodeDemoPairing, PairingSidecar
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer
from lerobot.policies.smolvla_icl.data.types import (
    SMOLVLA_ICL_DEMO_REF,
    SMOLVLA_ICL_GLOBAL_DEMO,
    SMOLVLA_ICL_LOCAL_DEMO,
    GlobalDemoClips,
    RawLocalDemoSample,
)


class CountingBackbone(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, 5, dtype=torch.float32))
        self.calls = 0

    def forward(self, clips: torch.Tensor, frame_valid_mask: torch.Tensor) -> torch.Tensor:
        del frame_valid_mask
        self.calls += 1
        return clips.mean(dim=(1, 2, 3, 4))[:, None] * self.scale[None, :]


def make_global_encoder() -> tuple[GlobalDemoEncoder, CountingBackbone]:
    config = GlobalEncoderConfig(
        pretrained_backbone=False,
        freeze_video_backbone=True,
        clip_length=14,
        clip_stride=14,
        state_dim=3,
        state_feature_dim=4,
        temporal_hidden_size=8,
        temporal_num_layers=1,
        temporal_num_heads=2,
        temporal_mlp_ratio=2.0,
        num_global_tokens=2,
        output_dim=6,
    )
    backbone = CountingBackbone()
    return GlobalDemoEncoder(config, video_backbone=backbone), backbone


def make_local_sample(_: DemoSampleRef) -> RawLocalDemoSample:
    return RawLocalDemoSample(
        images=torch.randint(0, 256, (5, 3, 8, 8), dtype=torch.uint8),
        states=torch.randn(5, 3),
        timestamps=torch.arange(5, dtype=torch.float64) * 0.1,
        valid_mask=torch.ones(5, dtype=torch.bool),
        anchor_position=2,
        demo_start_timestamp=0.0,
        demo_end_timestamp=1.0,
    )


def test_disk_cache_deduplicates_global_demo_and_keeps_local_rgb_raw(tmp_path: Path) -> None:
    encoder, backbone = make_global_encoder()
    clips = GlobalDemoClips(
        video=torch.rand(2, 14, 3, 4, 4),
        states=torch.rand(2, 14, 3),
        timestamps=torch.arange(28, dtype=torch.float64).reshape(2, 14) * 0.1,
        valid_mask=torch.ones(2, 14, dtype=torch.bool),
    )
    cached = cache_training_demo("demo-A", clips, encoder, cache_identity="a" * 64)
    assert backbone.calls == 1

    store = DemoFeatureStore(tmp_path)
    store.save(cached)
    local_reader = Mock(side_effect=make_local_sample)
    collator = SmolVLAICLCollator(
        tmp_path,
        local_demo_reader=local_reader,
        state_normalizer=DemoStateNormalizer(torch.zeros(3), torch.ones(3)),
        expected_state_dim=3,
    )
    samples = [
        {
            "observation.state": torch.randn(3),
            "action": torch.randn(2, 3),
            SMOLVLA_ICL_DEMO_REF: DemoSampleRef("demo-A", index, 2),
        }
        for index in range(2)
    ]
    batch = collator(samples)
    assert batch is not None
    global_batch = batch[SMOLVLA_ICL_GLOBAL_DEMO]
    local_batch = batch[SMOLVLA_ICL_LOCAL_DEMO]

    assert global_batch.video_features.shape == (1, 2, 4)
    assert global_batch.sample_to_demo.tolist() == [0, 0]
    assert local_batch.images.shape == (2, 5, 3, 8, 8)
    assert local_batch.images.dtype == torch.uint8
    assert local_reader.call_count == 1


def _cache_manifest_inputs() -> tuple[LiberoDataManifest, PairingSidecar]:
    assignments = (
        (0, "train", "demo"),
        (1, "train", "query"),
        (2, "val", "demo"),
        (3, "val", "query"),
        (4, "test", "demo"),
        (5, "test", "query"),
    )
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
                task_index=10,
                task="pick",
                length=28,
                split=split,
                role=role,
            )
            for index, split, role in assignments
        ),
    )
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="matcher@revision",
        image_key=manifest.image_key,
        epochs=(
            {
                1: EpisodeDemoPairing(
                    demo_id=manifest.demo_id(0),
                    demo_episode_index=0,
                    local_anchors=tuple(range(28)),
                )
            },
        ),
    )
    return manifest, sidecar


def test_global_cache_manifest_preflight_validates_all_files(tmp_path: Path) -> None:
    encoder, _ = make_global_encoder()
    manifest, sidecar = _cache_manifest_inputs()
    normalizer = DemoStateNormalizer(torch.zeros(3), torch.ones(3))
    identity = global_cache_identity(
        manifest,
        config=encoder.config,
        state_normalizer=normalizer,
        state_key="observation.state",
    )
    cached = cache_training_demo(
        manifest.demo_id(0),
        GlobalDemoClips(
            video=torch.rand(2, 14, 3, 4, 4),
            states=torch.rand(2, 14, 3),
            timestamps=torch.arange(28, dtype=torch.float64).reshape(2, 14) * 0.1,
            valid_mask=torch.ones(2, 14, dtype=torch.bool),
        ),
        encoder,
        cache_identity=identity,
    )
    store = DemoFeatureStore(tmp_path)
    store.save(cached)
    cache_manifest = GlobalDemoCacheManifest.create(
        manifest,
        config=encoder.config,
        state_normalizer=normalizer,
        state_key="observation.state",
        entries=(
            GlobalDemoCacheEntry(
                demo_id=cached.demo_id,
                episode_index=0,
                episode_length=28,
                num_clips=2,
                feature_dim=4,
            ),
        ),
    )
    cache_manifest.save(tmp_path)

    restored = preflight_global_demo_cache(
        tmp_path,
        manifest=manifest,
        sidecar=sidecar,
        config=encoder.config,
        state_normalizer=normalizer,
        state_key="observation.state",
    )

    assert restored.fingerprint == cache_manifest.fingerprint
