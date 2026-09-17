"""SmolVLA-ICL Demo 输入的最小数据契约。

本模块复用 SmolVLA 的标准 Policy pre/post pipeline，并处理
Global/Local Demo 路径已经确定的数据边界：

1. 使用与 SmolVLA Current State 相同的 mean/std 归一化 Demo State；
2. 将完整 RGB+State Demo 切分为 Global Encoder 使用的固定长度 clips；
3. 将变长 Demo 在 clip 维度补齐为 batch；
4. 将 Stage Matcher 已经选定的 :class:`LocalDemoChunk` 堆叠为 batch。

Demo 数据函数不运行 Stage Matcher，也不改变 Local Chunk 的锚点和时间范围。
Matcher 负责“选哪一段”，Processor 只负责“归一化、补齐和组 batch”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.collate import lerobot_collate_fn

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .components.demo_alignment import LocalDemoChunk
from .components.state_normalizer import DemoStateNormalizer
from .configuration_smolvla_icl import GlobalEncoderConfig, SmolVLAICLConfig


__all__ = [
    "DemoStateNormalizer",
    "GlobalDemoBatch",
    "GlobalDemoSample",
    "LocalDemoBatch",
    "SMOLVLA_ICL_GLOBAL_DEMO",
    "SMOLVLA_ICL_LOCAL_DEMO",
    "build_global_demo_sample",
    "collate_smolvla_icl_batch",
    "collate_global_demo_samples",
    "collate_local_demo_chunks",
    "get_smolvla_icl_demo_batches",
    "make_smolvla_icl_pre_post_processors",
]


# Dataset 单样本中分别保存 GlobalDemoSample 和 LocalDemoChunk；经过下面的
# policy-specific collate 后，同名字段变成 GlobalDemoBatch 和 LocalDemoBatch。
# 使用独立命名空间，避免与普通 observation/action feature 冲突。
SMOLVLA_ICL_GLOBAL_DEMO = "smolvla_icl.global_demo"
SMOLVLA_ICL_LOCAL_DEMO = "smolvla_icl.local_demo"


def make_smolvla_icl_pre_post_processors(
    config: SmolVLAICLConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """复用 SmolVLA 的标准输入归一化与输出反归一化管线。

    推理时完整 Demo 由 :meth:`SmolVLAICLPolicy.set_demo` 单独注册；训练时
    Global/Local Demo 作为 complementary data 穿过该 pipeline，不参与普通
    feature 归一化。模型输出先在 Policy 中裁剪到真实动作维度，再由这里返回的
    postprocessor 恢复到机器人动作尺度。
    """
    return make_smolvla_pre_post_processors(config, dataset_stats)


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


def collate_smolvla_icl_batch(
    samples: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """将标准 LeRobot 样本与配对 Demo 一起组成训练 batch。

    Dataset 的每个有效样本必须额外携带：

    * ``SMOLVLA_ICL_GLOBAL_DEMO``：一条 :class:`GlobalDemoSample`；
    * ``SMOLVLA_ICL_LOCAL_DEMO``：与当前 Observation 对齐的
      :class:`LocalDemoChunk`。

    普通 Observation、Action 和语言字段继续交给 LeRobot 原 collate；Demo
    的变长 clip 和结构化元数据由本模块已有的两个 collate 函数处理。输出仍是
    单个 ``dict``，因此标准 Trainer 可以保持 ``policy(batch)`` 调用方式。
    """
    valid_samples = [sample for sample in samples if sample is not None]
    if not valid_samples:
        return None

    for key in (SMOLVLA_ICL_GLOBAL_DEMO, SMOLVLA_ICL_LOCAL_DEMO):
        if any(key not in sample for sample in valid_samples):
            raise KeyError(f"SmolVLA-ICL 训练样本缺少必需字段 {key!r}。")

    global_samples = [sample[SMOLVLA_ICL_GLOBAL_DEMO] for sample in valid_samples]
    local_chunks = [sample[SMOLVLA_ICL_LOCAL_DEMO] for sample in valid_samples]
    if not all(isinstance(sample, GlobalDemoSample) for sample in global_samples):
        raise TypeError(f"{SMOLVLA_ICL_GLOBAL_DEMO} 必须保存 GlobalDemoSample。")
    if not all(isinstance(chunk, LocalDemoChunk) for chunk in local_chunks):
        raise TypeError(f"{SMOLVLA_ICL_LOCAL_DEMO} 必须保存 LocalDemoChunk。")

    policy_samples = [
        {
            key: value
            for key, value in sample.items()
            if key not in (SMOLVLA_ICL_GLOBAL_DEMO, SMOLVLA_ICL_LOCAL_DEMO)
        }
        for sample in valid_samples
    ]
    batch = lerobot_collate_fn(policy_samples)
    if batch is None:
        return None
    batch[SMOLVLA_ICL_GLOBAL_DEMO] = collate_global_demo_samples(global_samples)
    batch[SMOLVLA_ICL_LOCAL_DEMO] = collate_local_demo_chunks(local_chunks)
    return batch


def get_smolvla_icl_demo_batches(
    batch: dict[str, Any],
) -> tuple[GlobalDemoBatch, LocalDemoBatch]:
    """从标准 Policy batch 中取出已经 collate 的两条 Demo 输入。"""
    global_demo = batch.get(SMOLVLA_ICL_GLOBAL_DEMO)
    local_demo = batch.get(SMOLVLA_ICL_LOCAL_DEMO)
    if not isinstance(global_demo, GlobalDemoBatch) or not isinstance(local_demo, LocalDemoBatch):
        raise TypeError(
            "SmolVLA-ICL 训练 batch 必须包含 collate 后的 GlobalDemoBatch 和 "
            "LocalDemoBatch；请使用 collate_smolvla_icl_batch。"
        )
    return global_demo, local_demo


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

    所有 clip 按原时间顺序排列。``clip_stride < clip_length`` 时允许重叠。
    Demo 长度达到一个 clip 时，末尾必须再生成一个右对齐的
    完整 clip，避免最后少量帧因有效比例不足被 Global Encoder 屏蔽。
    只有整条 Demo 短于 ``clip_length`` 时才在右侧 padding。
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
        start_indices = [0]
    else:
        last_start = num_frames - cfg.clip_length
        start_indices = list(range(0, last_start + 1, cfg.clip_stride))
        if start_indices[-1] != last_start:
            start_indices.append(last_start)
    num_clips = len(start_indices)

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

    本函数不再执行 State 归一化。``DemoEmbeddingCache`` 已经使用共享的
    :class:`DemoStateNormalizer` 将 raw State 归一化，并保留真实 State 宽度，
    保证 Matcher 的 ``-1`` 表示真实夹爪维。本函数再为
    Local Encoder 将 State 和 Velocity 各自补到 ``expected_state_dim``。
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
        if chunk.state_features.shape != (chunk_length, raw_state_dim * 2):
            raise ValueError(
                "Local Demo state_features 必须按 [State, Velocity] 排列为 "
                f"({chunk_length},{raw_state_dim * 2})。"
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
    states = stack("states")
    state_features = stack("state_features")
    state_values, state_velocities = state_features.split(raw_state_dim, dim=-1)
    state_padding = expected_state_dim - raw_state_dim
    return LocalDemoBatch(
        visual_tokens=visual_tokens,
        visual_embeddings=stack("visual_embeddings"),
        states=F.pad(states, (0, state_padding)),
        # State 和 Velocity 必须分别补齐后再拼回，不能直接
        # 在末尾补零，否则会破坏 [State(32), Velocity(32)] 的分段语义。
        state_features=torch.cat(
            [
                F.pad(state_values, (0, state_padding)),
                F.pad(state_velocities, (0, state_padding)),
            ],
            dim=-1,
        ),
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
