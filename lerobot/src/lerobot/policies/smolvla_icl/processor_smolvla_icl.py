"""SmolVLA-ICL Demo 输入的最小数据契约。

本模块暂不构建完整的 :class:`PolicyProcessorPipeline`，只处理当前
Global/Local Demo 路径已经确定的数据边界：

1. 使用与 SmolVLA Current State 相同的 mean/std 归一化 Demo State；
2. 将完整 RGB+State Demo 切分为 Global Encoder 使用的固定长度 clips；
3. 将变长 Demo 在 clip 维度补齐为 batch；
4. 将 Stage Matcher 已经选定的 :class:`LocalDemoChunk` 堆叠为 batch。

这里不运行 Stage Matcher，也不改变 Local Chunk 的锚点和时间范围。
Matcher 负责“选哪一段”，Processor 只负责“归一化、补齐和组 batch”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from lerobot.utils.constants import OBS_STATE

from .components.demo_alignment import LocalDemoChunk
from .configuration_smolvla_icl import GlobalEncoderConfig


__all__ = [
    "DemoStateNormalizer",
    "GlobalDemoBatch",
    "GlobalDemoSample",
    "LocalDemoBatch",
    "build_global_demo_sample",
    "collate_global_demo_samples",
    "collate_local_demo_chunks",
]


@dataclass(frozen=True, slots=True)
class DemoStateNormalizer:
    """用 SmolVLA 训练集统计量归一化 Demo State。

    SmolVLA 对 State 默认使用 ``MEAN_STD``：

    ``normalized = (state - mean) / (std + eps)``。

    ``mean`` 和 ``std`` 只覆盖机器人的真实 State 维度。归一化完成后
    再在最后一维右侧补 0，避免 padding 维度被 ``(0-mean)/std``
    变成非零值。
    """

    mean: Tensor
    std: Tensor
    eps: float = 1e-8

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean).detach().flatten().float().clone()
        std = torch.as_tensor(self.std).detach().flatten().float().clone()
        if mean.numel() == 0 or mean.shape != std.shape:
            raise ValueError("State mean/std 必须是形状相同的非空一维张量。")
        if torch.any(~torch.isfinite(mean)) or torch.any(~torch.isfinite(std)):
            raise ValueError("State mean/std 必须只包含有限值。")
        if torch.any(std < 0):
            raise ValueError("State std 不能包含负数。")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("State normalization eps 必须是有限正数。")

        # frozen dataclass 仍需要把统计量规范为独立 float32 Tensor。
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def from_dataset_stats(
        cls,
        dataset_stats: dict[str, dict[str, Any]],
        *,
        state_key: str = OBS_STATE,
        eps: float = 1e-8,
    ) -> Self:
        """从 LeRobot ``dataset_stats`` 读取 Current/Demo 共享的 State 统计量。"""
        if state_key not in dataset_stats:
            raise KeyError(f"dataset_stats 中缺少 State key: {state_key!r}。")
        stats = dataset_stats[state_key]
        if "mean" not in stats or "std" not in stats:
            raise KeyError(f"dataset_stats[{state_key!r}] 必须同时包含 mean 和 std。")
        return cls(mean=torch.as_tensor(stats["mean"]), std=torch.as_tensor(stats["std"]), eps=eps)

    @property
    def state_dim(self) -> int:
        """返回统计量覆盖的真实 State 维度。"""
        return int(self.mean.numel())

    def normalize(
        self,
        states: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """归一化 ``(...,D_raw)`` State，保留真实维度并清零无效项。

        Stage Matcher 必须使用该输出，而不是已补到 32 维的 State。
        否则 ``matching_state_excluded_indices=(-1,)`` 会排除 padding 维，
        而不是真实 State 的最后一维夹爪。
        """
        if states.ndim < 1 or not states.is_floating_point():
            raise ValueError("states 必须是至少一维的浮点 Tensor。")
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"states 最后一维必须与统计量一致：期望 {self.state_dim}，"
                f"实际 {states.shape[-1]}。"
            )
        mask = (
            torch.ones(states.shape[:-1], dtype=torch.bool, device=states.device)
            if valid_mask is None
            else valid_mask.to(device=states.device, dtype=torch.bool)
        )
        if mask.shape != states.shape[:-1]:
            raise ValueError("State valid_mask 必须与 states 除最后一维外的形状一致。")
        if torch.any(~torch.isfinite(states[mask])):
            raise ValueError("有效 State 必须只包含有限值。")

        mean = self.mean.to(device=states.device, dtype=states.dtype)
        std = self.std.to(device=states.device, dtype=states.dtype)
        normalized = (states - mean) / (std + self.eps)
        # 无效位置可能含 NaN，必须用 where 显式替换。
        return torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    def normalize_and_pad(
        self,
        states: Tensor,
        *,
        target_dim: int,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """归一化 State，再将最后一维右侧补到 ``target_dim``。"""
        if target_dim < self.state_dim:
            raise ValueError(
                f"target_dim={target_dim} 小于真实 State 维度 {self.state_dim}。"
            )
        normalized = self.normalize(states, valid_mask=valid_mask)
        return F.pad(normalized, (0, target_dim - self.state_dim))


@dataclass(frozen=True, slots=True)
class GlobalDemoSample:
    """单条 Demo 经分段后的 Global Encoder 输入，尚无 batch 维。"""

    video: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor

    @property
    def num_clips(self) -> int:
        """返回该 Demo 的 clip 数。"""
        return int(self.video.shape[0])

    def to(self, device: torch.device | str) -> Self:
        """返回所有 Tensor 已移到目标设备的新样本。"""
        target = torch.device(device)
        return type(self)(
            video=self.video.to(target),
            states=self.states.to(target),
            timestamps=self.timestamps.to(target),
            valid_mask=self.valid_mask.to(target),
        )


@dataclass(frozen=True, slots=True)
class GlobalDemoBatch:
    """Global Encoder 可直接消费的 batch。"""

    video: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    num_clips: Tensor

    def to(self, device: torch.device | str) -> Self:
        """返回所有 Tensor 已移到目标设备的新 batch。"""
        target = torch.device(device)
        return type(self)(
            video=self.video.to(target),
            states=self.states.to(target),
            timestamps=self.timestamps.to(target),
            valid_mask=self.valid_mask.to(target),
            num_clips=self.num_clips.to(target),
        )


@dataclass(frozen=True, slots=True)
class LocalDemoBatch:
    """Local Encoder 的结构化 batch，保留 Matcher 返回的对齐元数据。"""

    visual_tokens: Tensor | None
    visual_embeddings: Tensor
    states: Tensor
    state_features: Tensor
    timestamps: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    source_indices: Tensor
    anchor_positions: Tensor
    demo_anchor_indices: Tensor
    alignment_confidence: Tensor
    observation_ids: tuple[int | str | None, ...]
    observation_timestamps: tuple[float | None, ...]

    def to(self, device: torch.device | str) -> Self:
        """返回所有 Tensor 已移到目标设备的新 batch。"""
        target = torch.device(device)
        return type(self)(
            visual_tokens=(
                self.visual_tokens.to(target) if self.visual_tokens is not None else None
            ),
            visual_embeddings=self.visual_embeddings.to(target),
            states=self.states.to(target),
            state_features=self.state_features.to(target),
            timestamps=self.timestamps.to(target),
            relative_time_s=self.relative_time_s.to(target),
            relative_position=self.relative_position.to(target),
            phase=self.phase.to(target),
            valid_mask=self.valid_mask.to(target),
            source_indices=self.source_indices.to(target),
            anchor_positions=self.anchor_positions.to(target),
            demo_anchor_indices=self.demo_anchor_indices.to(target),
            alignment_confidence=self.alignment_confidence.to(target),
            observation_ids=self.observation_ids,
            observation_timestamps=self.observation_timestamps,
        )


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


def build_global_demo_sample(
    video: Tensor,
    states: Tensor,
    timestamps: Tensor,
    *,
    state_normalizer: DemoStateNormalizer,
    config: GlobalEncoderConfig | None = None,
    valid_mask: Tensor | None = None,
) -> GlobalDemoSample:
    """将一条完整 Demo 切成 Global Encoder 所需的固定长度 clips。

    所有 clip 按原时间顺序排列。``clip_stride < clip_length`` 时允许重叠；
    最后一个 clip 不足 ``clip_length`` 时右侧补零，并由 ``valid_mask``
    区分真实帧和 padding。
    """
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
        num_clips = 1
    else:
        num_clips = math.ceil((num_frames - cfg.clip_length) / cfg.clip_stride) + 1
    start_indices = [index * cfg.clip_stride for index in range(num_clips)]

    output_video = video.new_zeros(
        num_clips,
        cfg.clip_length,
        *video.shape[1:],
    )
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

    return GlobalDemoSample(
        video=output_video,
        states=output_states,
        timestamps=output_times,
        valid_mask=output_mask,
    )


def collate_global_demo_samples(samples: Sequence[GlobalDemoSample]) -> GlobalDemoBatch:
    """将变长 ``K`` 的 Global Demo 样本在 clip 维度右侧补齐。"""
    if len(samples) == 0:
        raise ValueError("Global Demo batch 至少需要一个样本。")

    reference = samples[0]
    expected_video_tail = reference.video.shape[1:]
    expected_state_tail = reference.states.shape[1:]
    for sample in samples:
        if sample.video.ndim != 5 or sample.states.ndim != 3:
            raise ValueError("GlobalDemoSample 必须使用 (K,L,3,H,W) 和 (K,L,D) 形状。")
        if sample.video.shape[1:] != expected_video_tail:
            raise ValueError("同一 Global Demo batch 的 clip/RGB 尺寸必须一致。")
        if sample.states.shape[1:] != expected_state_tail:
            raise ValueError("同一 Global Demo batch 的 clip/State 尺寸必须一致。")
        if sample.timestamps.shape != sample.video.shape[:2]:
            raise ValueError("GlobalDemoSample timestamps 形状不正确。")
        if sample.valid_mask.shape != sample.video.shape[:2]:
            raise ValueError("GlobalDemoSample valid_mask 形状不正确。")

    batch_size = len(samples)
    max_clips = max(sample.num_clips for sample in samples)
    video = reference.video.new_zeros(batch_size, max_clips, *expected_video_tail)
    states = reference.states.new_zeros(batch_size, max_clips, *expected_state_tail)
    timestamps = reference.timestamps.new_zeros(batch_size, max_clips, expected_video_tail[0])
    valid_mask = torch.zeros(
        batch_size,
        max_clips,
        expected_video_tail[0],
        dtype=torch.bool,
        device=reference.video.device,
    )
    num_clips = torch.empty(batch_size, dtype=torch.long, device=reference.video.device)

    for batch_index, sample in enumerate(samples):
        if sample.video.device != reference.video.device:
            raise ValueError("collate 前所有 Global Demo 样本必须在同一设备。")
        count = sample.num_clips
        video[batch_index, :count] = sample.video
        states[batch_index, :count] = sample.states
        timestamps[batch_index, :count] = sample.timestamps
        valid_mask[batch_index, :count] = sample.valid_mask
        num_clips[batch_index] = count

    return GlobalDemoBatch(
        video=video,
        states=states,
        timestamps=timestamps,
        valid_mask=valid_mask,
        num_clips=num_clips,
    )


def collate_local_demo_chunks(
    chunks: Sequence[LocalDemoChunk],
    *,
    expected_state_dim: int = 32,
) -> LocalDemoBatch:
    """堆叠 Stage Matcher 已经选定的固定长度 Local Chunks。

    本函数不再执行 State 归一化。``DemoEmbeddingCache`` 必须使用
    :meth:`DemoStateNormalizer.normalize` 的未补齐输出构建，保证 Matcher
    的 ``-1`` 仍表示真实 State 末维。本函数再为 Local Encoder 将 State
    补到 ``expected_state_dim``。
    """
    if len(chunks) == 0:
        raise ValueError("Local Demo batch 至少需要一个 chunk。")
    if expected_state_dim < 1:
        raise ValueError("expected_state_dim 必须大于 0。")

    reference = chunks[0]
    chunk_length = int(reference.valid_mask.shape[0])
    raw_state_dim = int(reference.states.shape[-1])
    has_visual_tokens = reference.visual_tokens is not None
    tensor_fields = (
        "visual_embeddings",
        "state_features",
        "timestamps",
        "relative_time_s",
        "relative_position",
        "phase",
        "valid_mask",
        "source_indices",
    )

    for chunk in chunks:
        if chunk.valid_mask.ndim != 1 or len(chunk.valid_mask) != chunk_length:
            raise ValueError("同一 Local Demo batch 的 chunk 长度必须一致。")
        if chunk.states.ndim != 2 or chunk.states.shape != (chunk_length, raw_state_dim):
            raise ValueError(
                "Local Demo states 必须为 "
                f"({chunk_length},{raw_state_dim})，实际为 {tuple(chunk.states.shape)}。"
            )
        if raw_state_dim > expected_state_dim:
            raise ValueError(
                f"Local Demo 真实 State 维度 {raw_state_dim} 超过 "
                f"expected_state_dim={expected_state_dim}。"
            )
        if (chunk.visual_tokens is not None) != has_visual_tokens:
            raise ValueError("同一 Local Demo batch 不能混用有/无 visual_tokens 的 chunk。")
        for field_name in tensor_fields:
            value = getattr(chunk, field_name)
            reference_value = getattr(reference, field_name)
            if value.shape != reference_value.shape:
                raise ValueError(f"Local Demo 字段 {field_name} 的形状必须在 batch 内一致。")
        if has_visual_tokens and chunk.visual_tokens is not None:
            assert reference.visual_tokens is not None
            if chunk.visual_tokens.shape != reference.visual_tokens.shape:
                raise ValueError("Local Demo visual_tokens 形状必须在 batch 内一致。")
        if not 0 <= chunk.anchor_position < chunk_length:
            raise ValueError("Local Demo anchor_position 必须位于 chunk 内。")

    def stack(field_name: str) -> Tensor:
        return torch.stack([getattr(chunk, field_name) for chunk in chunks])

    visual_tokens = None
    if has_visual_tokens:
        visual_tokens = torch.stack(
            [chunk.visual_tokens for chunk in chunks if chunk.visual_tokens is not None]
        )

    metadata_device = reference.valid_mask.device
    return LocalDemoBatch(
        visual_tokens=visual_tokens,
        visual_embeddings=stack("visual_embeddings"),
        states=F.pad(stack("states"), (0, expected_state_dim - raw_state_dim)),
        state_features=stack("state_features"),
        timestamps=stack("timestamps"),
        relative_time_s=stack("relative_time_s"),
        relative_position=stack("relative_position"),
        phase=stack("phase"),
        valid_mask=stack("valid_mask").bool(),
        source_indices=stack("source_indices").long(),
        anchor_positions=torch.tensor(
            [chunk.anchor_position for chunk in chunks],
            dtype=torch.long,
            device=metadata_device,
        ),
        demo_anchor_indices=torch.tensor(
            [chunk.demo_anchor_index for chunk in chunks],
            dtype=torch.long,
            device=metadata_device,
        ),
        alignment_confidence=torch.tensor(
            [chunk.alignment_confidence for chunk in chunks],
            dtype=torch.float32,
            device=metadata_device,
        ),
        observation_ids=tuple(chunk.observation_id for chunk in chunks),
        observation_timestamps=tuple(chunk.observation_timestamp for chunk in chunks),
    )
