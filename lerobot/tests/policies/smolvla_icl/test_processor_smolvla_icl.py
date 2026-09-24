"""SmolVLA-ICL 当前 Global/Local 数据边界测试。"""

import torch

from lerobot.policies.smolvla_icl.components.demo_alignment import extract_state_features
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import GlobalEncoderConfig
from lerobot.policies.smolvla_icl.data.collate import (
    collate_global_demo_samples,
    collate_raw_local_demo_samples,
    get_smolvla_icl_demo_batches,
)
from lerobot.policies.smolvla_icl.data.preprocessing import build_global_demo_clips
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer
from lerobot.policies.smolvla_icl.data.types import (
    SMOLVLA_ICL_GLOBAL_DEMO,
    SMOLVLA_ICL_LOCAL_DEMO,
    GlobalDemoSample,
    RawLocalDemoSample,
)


def make_global_config() -> GlobalEncoderConfig:
    return GlobalEncoderConfig(
        pretrained_backbone=False,
        clip_length=14,
        clip_stride=8,
        state_dim=4,
    )


def test_state_normalization_happens_before_padding_and_masks_invalid_values() -> None:
    normalizer = DemoStateNormalizer(
        mean=torch.tensor([1.0, 2.0]),
        std=torch.tensor([2.0, 4.0]),
    )
    states = torch.tensor([[3.0, 6.0], [5.0, 10.0], [float("nan"), float("nan")]])
    normalized = normalizer.normalize_and_pad(
        states,
        target_dim=4,
        valid_mask=torch.tensor([True, True, False]),
    )
    torch.testing.assert_close(
        normalized,
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_global_demo_clipping_right_aligns_the_last_full_clip() -> None:
    config = make_global_config()
    num_frames = 18
    video = torch.arange(num_frames, dtype=torch.float32).view(num_frames, 1, 1, 1)
    video = video.expand(-1, 3, 2, 2) / num_frames
    timestamps = torch.arange(num_frames, dtype=torch.float64) * 0.1
    clips = build_global_demo_clips(
        video,
        torch.randn(num_frames, 2),
        timestamps,
        state_normalizer=DemoStateNormalizer(torch.zeros(2), torch.ones(2)),
        config=config,
    )

    assert clips.video.shape == (2, 14, 3, 2, 2)
    torch.testing.assert_close(clips.timestamps[1], timestamps[4:18])
    assert torch.count_nonzero(clips.states[..., 2:]) == 0


def test_global_feature_collate_deduplicates_with_inverse_index() -> None:
    sample = GlobalDemoSample(
        video_features=torch.randn(2, 6),
        states=torch.randn(2, 14, 4),
        timestamps=torch.arange(28, dtype=torch.float64).reshape(2, 14),
        valid_mask=torch.ones(2, 14, dtype=torch.bool),
    )
    batch = collate_global_demo_samples([sample], sample_to_demo=[0, 0])

    assert batch.video_features.shape == (1, 2, 6)
    assert batch.sample_to_demo.tolist() == [0, 0]


def test_raw_local_collate_keeps_rgb_online_and_builds_state_features() -> None:
    timestamps = torch.arange(4, dtype=torch.float64) * 0.1
    images = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    images[0].zero_()
    sample = RawLocalDemoSample(
        images=images,
        states=torch.randn(4, 2),
        timestamps=timestamps,
        valid_mask=torch.tensor([False, True, True, True]),
        previous_state=None,
        previous_timestamp=None,
        anchor_position=1,
        demo_start_timestamp=0.0,
        demo_end_timestamp=1.0,
    )
    local = collate_raw_local_demo_samples(
        [sample, sample],
        state_normalizer=DemoStateNormalizer(torch.zeros(2), torch.ones(2)),
        expected_state_dim=4,
    )
    global_demo = collate_global_demo_samples(
        [
            GlobalDemoSample(
                video_features=torch.randn(1, 6),
                states=torch.randn(1, 14, 4),
                timestamps=torch.arange(14, dtype=torch.float64).view(1, 14),
                valid_mask=torch.ones(1, 14, dtype=torch.bool),
            )
        ],
        sample_to_demo=[0, 0],
    )
    batch = {
        SMOLVLA_ICL_GLOBAL_DEMO: global_demo,
        SMOLVLA_ICL_LOCAL_DEMO: local,
    }

    recovered_global, recovered_local = get_smolvla_icl_demo_batches(batch)
    assert recovered_global is global_demo
    assert recovered_local.images.shape == (2, 4, 3, 8, 8)
    assert recovered_local.images.dtype == torch.uint8
    assert torch.count_nonzero(recovered_local.images[:, 0]) == 0
    assert recovered_local.state_features.shape == (2, 4, 8)

    # ``to`` 只负责轻量 metadata，不能把完整 Local RGB 搬离 CPU。
    metadata_on_meta = recovered_local.to("meta", non_blocking=True)
    assert metadata_on_meta.images is recovered_local.images
    assert metadata_on_meta.images.device.type == "cpu"
    assert metadata_on_meta.state_features.device.type == "meta"
    assert metadata_on_meta.valid_mask.device.type == "meta"

    # Global cache feature 和轻量 metadata 支持 Trainer 的异步设备搬运约定。
    global_on_meta = recovered_global.to("meta", non_blocking=True)
    assert global_on_meta.video_features.device.type == "meta"
    assert global_on_meta.sample_to_demo.device.type == "meta"


def test_raw_local_first_velocity_matches_full_episode_features() -> None:
    """窗口首帧应复用 episode 前驱，与 rollout 的完整 Demo 特征一致。"""
    full_states = torch.tensor([[0.0], [1.0], [3.0], [6.0], [10.0]])
    full_timestamps = torch.arange(5, dtype=torch.float64) * 0.1
    expected = extract_state_features(full_states, full_timestamps)
    sample = RawLocalDemoSample(
        images=torch.zeros(3, 3, 2, 2, dtype=torch.uint8),
        states=full_states[2:5],
        timestamps=full_timestamps[2:5],
        valid_mask=torch.ones(3, dtype=torch.bool),
        previous_state=full_states[1],
        previous_timestamp=float(full_timestamps[1]),
        anchor_position=1,
        demo_start_timestamp=0.0,
        demo_end_timestamp=0.4,
    )

    local = collate_raw_local_demo_samples(
        [sample],
        state_normalizer=DemoStateNormalizer(torch.zeros(1), torch.ones(1)),
        expected_state_dim=1,
    )

    torch.testing.assert_close(local.state_features[0], expected[2:5])


def test_demo_batches_pin_memory_support_dataloader_contract() -> None:
    """CUDA 可用时，自定义 batch 必须能被 DataLoader 的 pin 线程识别。"""
    if not torch.cuda.is_available():
        return

    timestamps = torch.arange(3, dtype=torch.float64) * 0.1
    global_demo = collate_global_demo_samples(
        [
            GlobalDemoSample(
                video_features=torch.randn(1, 6),
                states=torch.randn(1, 14, 4),
                timestamps=torch.arange(14, dtype=torch.float64).view(1, 14),
                valid_mask=torch.ones(1, 14, dtype=torch.bool),
            )
        ],
        sample_to_demo=[0],
    ).pin_memory()
    local = collate_raw_local_demo_samples(
        [
            RawLocalDemoSample(
                images=torch.randint(0, 256, (3, 3, 4, 4), dtype=torch.uint8),
                states=torch.randn(3, 2),
                timestamps=timestamps,
                valid_mask=torch.ones(3, dtype=torch.bool),
                previous_state=None,
                previous_timestamp=None,
                anchor_position=1,
                demo_start_timestamp=0.0,
                demo_end_timestamp=0.2,
            )
        ],
        state_normalizer=DemoStateNormalizer(torch.zeros(2), torch.ones(2)),
        expected_state_dim=4,
    ).pin_memory()

    assert local.images.is_pinned()
    assert local.state_features.is_pinned()
    assert local.valid_mask.is_pinned()
    assert global_demo.video_features.is_pinned()
    assert global_demo.sample_to_demo.is_pinned()
