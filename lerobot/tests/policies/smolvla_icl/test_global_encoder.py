"""Global Demo Encoder 的单元测试。

测试使用极小的伪视频骨干，不下载 S3D 权重，也不依赖具体
TorchVision 版本。S3D adapter 只负责骨干接口，Global Encoder 的 Mask、
State、时间编码和 Temporal Aggregator 逻辑都能在这些测试中独立验证。
"""

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla_icl.components.global_encoder import (
    GlobalDemoEncoder,
    _resample_valid_video_frames,
)
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import GlobalEncoderConfig


class TinyVideoBackbone(nn.Module):
    """为测试提供的可导 clip encoder。"""

    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.output_dim)
        self.calls = 0

    def forward(self, clips: Tensor, frame_valid_mask: Tensor) -> Tensor:
        self.calls += 1
        mask = frame_valid_mask[:, :, None, None, None].to(clips.dtype)
        summed = (clips * mask).sum(dim=(1, 3, 4))
        pixels_per_frame = clips.shape[-2] * clips.shape[-1]
        denominator = frame_valid_mask.sum(dim=1, keepdim=True).clamp_min(1) * pixels_per_frame
        pooled = summed / denominator.to(summed.dtype)
        output = self.projection(pooled)
        clip_valid = frame_valid_mask.any(dim=1)
        return torch.where(clip_valid[:, None], output, torch.zeros_like(output))


def make_config(*, freeze_video_backbone: bool = False) -> GlobalEncoderConfig:
    """构建适合 CPU 单元测试的小型配置。"""
    return GlobalEncoderConfig(
        pretrained_backbone=False,
        freeze_video_backbone=freeze_video_backbone,
        state_dim=4,
        state_feature_dim=8,
        temporal_hidden_size=16,
        temporal_num_layers=1,
        temporal_num_heads=4,
        temporal_mlp_ratio=2.0,
        num_global_tokens=3,
        output_dim=12,
        min_valid_frame_fraction=0.5,
    )


def make_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """构建两个 clip、每个 clip 三帧的小型 Demo。"""
    torch.manual_seed(0)
    video = torch.rand(2, 2, 3, 3, 8, 8)
    states = torch.randn(2, 2, 3, 4)
    timestamps = torch.tensor(
        [
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
            [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]],
        ],
        dtype=torch.float64,
    )
    valid_mask = torch.ones(2, 2, 3, dtype=torch.bool)
    return video, states, timestamps, valid_mask


def test_global_encoder_output_contract_and_gradients() -> None:
    """输出形状固定，视频、State 和时序路径都可接收梯度。"""
    config = make_config()
    backbone = TinyVideoBackbone()
    encoder = GlobalDemoEncoder(config, video_backbone=backbone)
    video, states, timestamps, valid_mask = make_inputs()

    features = encoder.encode_video_clips(video, valid_mask)
    output = encoder(features, states, timestamps, valid_mask)

    assert output.global_tokens.shape == (2, 3, 12)
    assert output.global_mask.shape == (2, 3)
    assert output.global_mask.all()

    output.global_tokens.square().mean().backward()
    assert backbone.projection.weight.grad is not None
    assert encoder.state_encoder.frame_mlp[0].weight.grad is not None
    assert encoder.task_queries.grad is not None


def test_padding_values_do_not_change_global_tokens() -> None:
    """被 valid mask 屏蔽的 RGB/State/timestamp 值不能污染 Global 表示。"""
    config = make_config()
    encoder = GlobalDemoEncoder(config, video_backbone=TinyVideoBackbone()).eval()
    video, states, timestamps, valid_mask = make_inputs()
    valid_mask[:, 1, 2] = False

    changed_video = video.clone()
    changed_states = states.clone()
    changed_timestamps = timestamps.clone()
    changed_video[:, 1, 2] = 1.0
    changed_states[:, 1, 2] = 10_000.0
    changed_timestamps[:, 1, 2] = 10_000.0

    with torch.no_grad():
        reference = encoder(
            encoder.encode_video_clips(video, valid_mask),
            states,
            timestamps,
            valid_mask,
        )
        changed = encoder(
            encoder.encode_video_clips(changed_video, valid_mask),
            changed_states,
            changed_timestamps,
            valid_mask,
        )

    torch.testing.assert_close(reference.global_tokens, changed.global_tokens)


def test_clip_phase_uses_real_timestamps() -> None:
    """clip phase 应由真实时间中心决定，而不是固定的 clip index。"""
    config = make_config()
    encoder = GlobalDemoEncoder(config, video_backbone=TinyVideoBackbone()).eval()
    timestamps = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]], dtype=torch.float64)

    valid_mask = torch.ones(1, 2, 2, dtype=torch.bool)
    phase = encoder._compute_clip_phase(timestamps, valid_mask)

    expected = torch.tensor([[1.0 / 6.0, 5.0 / 6.0]], dtype=torch.float64)
    torch.testing.assert_close(phase, expected)


def test_partial_video_clip_is_filled_only_with_valid_frames() -> None:
    """部分有效 clip 进入 S3D 前不应继续包含零 padding 帧。"""
    clips = torch.tensor([[[[[1.0]]], [[[99.0]]], [[[3.0]]], [[[99.0]]]]])
    valid_mask = torch.tensor([[True, False, True, False]])

    dense = _resample_valid_video_frames(clips, valid_mask)

    assert dense.flatten().tolist() == [1.0, 1.0, 3.0, 3.0]


def test_overlapping_clips_keep_valid_temporal_order() -> None:
    """重叠 clip 会重复 timestamp，但只要 clip 内部和时间中心有序就应接受。"""
    config = make_config()
    encoder = GlobalDemoEncoder(config, video_backbone=TinyVideoBackbone()).eval()
    timestamps = torch.tensor(
        [[[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]]],
        dtype=torch.float64,
    )

    valid_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    phase = encoder._compute_clip_phase(timestamps, valid_mask)

    torch.testing.assert_close(
        phase,
        torch.tensor([[0.3333333333333333, 0.6666666666666666]], dtype=torch.float64),
    )


def test_frozen_video_backbone_stays_in_eval_mode() -> None:
    """调用整体 ``train()`` 不得重新打开已冻结视频骨干的训练模式。"""
    config = make_config(freeze_video_backbone=True)
    backbone = TinyVideoBackbone()
    encoder = GlobalDemoEncoder(config, video_backbone=backbone)

    encoder.train()

    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_cached_video_features_skip_backbone_and_expand_unique_demo() -> None:
    """缓存 S3D feature 后只编码唯一 Demo，再映射回 query batch。"""
    config = make_config(freeze_video_backbone=True)
    backbone = TinyVideoBackbone()
    encoder = GlobalDemoEncoder(config, video_backbone=backbone)
    video, states, timestamps, valid_mask = make_inputs()

    features = encoder.encode_video_clips(video[:1], valid_mask[:1])
    assert backbone.calls == 1
    backbone.calls = 0
    output = encoder(features, states[:1], timestamps[:1], valid_mask[:1])

    assert backbone.calls == 0
    assert output.global_tokens.shape == (1, 3, 12)
    output.global_tokens.sum().backward()
    assert encoder.state_encoder.frame_mlp[0].weight.grad is not None


def test_batched_video_encoding_matches_single_forward() -> None:
    """逐批上传 clip 只改变显存峰值，不改变 feature 顺序或数值。"""
    config = make_config(freeze_video_backbone=True)
    backbone = TinyVideoBackbone()
    encoder = GlobalDemoEncoder(config, video_backbone=backbone).eval()
    video, _, _, valid_mask = make_inputs()

    expected = encoder.encode_video_clips(video, valid_mask)
    backbone.calls = 0
    actual = encoder.encode_video_clips_batched(
        video,
        valid_mask,
        encode_batch_size=1,
    )

    torch.testing.assert_close(actual, expected)
    assert backbone.calls == video.shape[0] * video.shape[1]
