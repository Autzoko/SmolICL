"""把完整 RGB+State Demo 转换为 Global Encoder 的 clip 输入。"""

from __future__ import annotations

import torch
from torch import Tensor

from ..configuration_smolvla_icl import GlobalEncoderConfig
from .state import DemoStateNormalizer
from .types import GlobalDemoClips


def _validate_full_demo_inputs(
    video: Tensor,
    states: Tensor,
    timestamps: Tensor,
    valid_mask: Tensor,
) -> None:
    """在分段前验证一条完整 Demo 的帧级契约。"""
    if video.ndim != 4 or video.shape[1] != 3 or not video.is_floating_point():
        raise ValueError("Demo video 必须是浮点 (T,3,H,W) Tensor。")
    if video.shape[0] == 0 or video.shape[-2] == 0 or video.shape[-1] == 0:
        raise ValueError("Demo video 的时间和空间尺寸都必须大于 0。")
    if states.ndim != 2 or states.shape[0] != video.shape[0]:
        raise ValueError("Demo states 必须是与 video 时间维对齐的 (T,D) Tensor。")
    if timestamps.shape != video.shape[:1] or valid_mask.shape != video.shape[:1]:
        raise ValueError("timestamps/valid_mask 必须是与 Demo 帧对齐的 (T,) Tensor。")
    if torch.any(~torch.isfinite(timestamps)):
        raise ValueError("Demo timestamps 必须只包含有限值。")
    if timestamps.numel() > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError("Demo timestamps 必须严格递增。")
    if not torch.any(valid_mask):
        raise ValueError("一条 Demo 至少需要一帧有效数据。")
    if torch.any(~torch.isfinite(video[valid_mask])):
        raise ValueError("有效 Demo RGB 帧必须只包含有限值。")
    valid_pixels = video[valid_mask]
    if torch.any(valid_pixels < 0) or torch.any(valid_pixels > 1):
        raise ValueError("Demo RGB 必须使用 [0,1] 值域。")


def build_global_demo_clips(
    video: Tensor,
    states: Tensor,
    timestamps: Tensor,
    *,
    state_normalizer: DemoStateNormalizer,
    config: GlobalEncoderConfig | None = None,
    valid_mask: Tensor | None = None,
) -> GlobalDemoClips:
    """按时间顺序把完整 Demo 切成 Global Encoder 使用的定长 clips。"""
    cfg = config or GlobalEncoderConfig()
    frame_mask = (
        torch.ones(video.shape[:1], dtype=torch.bool, device=video.device)
        if valid_mask is None
        else valid_mask.to(device=video.device, dtype=torch.bool)
    )
    times = timestamps.to(device=video.device, dtype=torch.float64)
    state_values = states.to(device=video.device)
    _validate_full_demo_inputs(video, state_values, times, frame_mask)

    normalized_states = state_normalizer.normalize_and_pad(
        state_values,
        target_dim=cfg.state_dim,
        valid_mask=frame_mask,
    )
    safe_video = torch.where(
        frame_mask[:, None, None, None],
        video,
        torch.zeros_like(video),
    )

    num_frames = int(video.shape[0])
    if num_frames <= cfg.clip_length:
        start_indices = [0]
    else:
        last_start = num_frames - cfg.clip_length
        start_indices = list(range(0, last_start + 1, cfg.clip_stride))
        if start_indices[-1] != last_start:
            start_indices.append(last_start)
    num_clips = len(start_indices)

    output_video = video.new_zeros(num_clips, cfg.clip_length, *video.shape[1:])
    output_states = normalized_states.new_zeros(
        num_clips,
        cfg.clip_length,
        cfg.state_dim,
    )
    output_times = times.new_zeros(num_clips, cfg.clip_length)
    output_mask = torch.zeros(
        num_clips,
        cfg.clip_length,
        dtype=torch.bool,
        device=video.device,
    )

    for clip_index, start in enumerate(start_indices):
        end = min(start + cfg.clip_length, num_frames)
        length = end - start
        output_video[clip_index, :length] = safe_video[start:end]
        output_states[clip_index, :length] = normalized_states[start:end]
        output_times[clip_index, :length] = times[start:end]
        output_mask[clip_index, :length] = frame_mask[start:end]

    return GlobalDemoClips(
        video=output_video,
        states=output_states,
        timestamps=output_times,
        valid_mask=output_mask,
    )


__all__ = ["build_global_demo_clips"]
