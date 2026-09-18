"""SmolVLA-ICL Demo 输入的 batch 组装函数。

Global 训练路径只组装冻结 S3D feature；Local 训练路径只组装 Matcher
选定的 raw RGB+State，视觉编码仍在 Policy forward 内执行。完整 Demo
的 raw clip 切分位于 ``preprocessing.py``，磁盘缓存和 batch 内
``demo_id`` 去重位于 ``cache.py``。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..components.demo_alignment import LocalDemoWindow, extract_state_features
from .state import DemoStateNormalizer
from .types import (
    EncodedLocalDemoBatch,
    GlobalDemoBatch,
    GlobalDemoSample,
    LocalDemoBatch,
    RawLocalDemoSample,
    SMOLVLA_ICL_DEMO_REF,
    SMOLVLA_ICL_GLOBAL_DEMO,
    SMOLVLA_ICL_LOCAL_DEMO,
)


__all__ = [
    "build_encoded_local_demo_batch",
    "collate_global_demo_samples",
    "collate_raw_local_demo_samples",
    "get_smolvla_icl_demo_batches",
    "validate_smolvla_icl_training_batch",
]


def get_smolvla_icl_demo_batches(
    batch: dict[str, Any],
) -> tuple[GlobalDemoBatch, LocalDemoBatch]:
    """从标准 Policy batch 中取出已经 collate 的两条 Demo 输入。"""
    global_demo = batch.get(SMOLVLA_ICL_GLOBAL_DEMO)
    local_demo = batch.get(SMOLVLA_ICL_LOCAL_DEMO)
    if not isinstance(global_demo, GlobalDemoBatch) or not isinstance(local_demo, LocalDemoBatch):
        raise TypeError(
            "SmolVLA-ICL 训练 batch 必须包含 collate 后的 GlobalDemoBatch 和 "
            "LocalDemoBatch；正式训练请使用 SmolVLAICLCollator。"
        )
    return global_demo, local_demo


def validate_smolvla_icl_training_batch(
    batch: dict[str, Any],
    *,
    expected_state_dim: int | None = None,
    expected_local_chunk_size: int | None = None,
) -> None:
    """检查训练 batch 的关键维度；输入类型本身已经固定数据路径。"""
    global_demo, local_demo = get_smolvla_icl_demo_batches(batch)
    if SMOLVLA_ICL_DEMO_REF in batch:
        raise ValueError("DemoSampleRef 必须由 Collator 消费，不能继续进入 Policy batch。")

    batch_size, local_chunk_size = local_demo.valid_mask.shape
    if expected_local_chunk_size is not None and local_chunk_size != expected_local_chunk_size:
        raise ValueError(
            f"Local Chunk 长度应为 {expected_local_chunk_size}，实际为 {local_chunk_size}。"
        )
    if local_demo.images.shape[:3] != (batch_size, local_chunk_size, 3):
        raise ValueError("Local images 必须为 (B,T_local,3,H,W) Tensor。")
    if local_demo.state_features.shape[:2] != (batch_size, local_chunk_size):
        raise ValueError("Local state_features 必须与 valid_mask 共享 (B,T_local) 前缀。")
    if (
        expected_state_dim is not None
        and local_demo.state_features.shape[-1] != expected_state_dim * 2
    ):
        raise ValueError(
            "Local state_features 宽度应为 "
            f"{expected_state_dim * 2}，实际为 {local_demo.state_features.shape[-1]}。"
        )
    if local_demo.anchor_positions.shape != (batch_size,):
        raise ValueError("Local anchor_positions 必须为 (B,) Tensor。")

    unique_demo_count = global_demo.states.shape[0]
    if global_demo.video_features.shape[:2] != global_demo.states.shape[:2]:
        raise ValueError("Global video_features 必须与 states 共享 (U,K) 前缀。")
    if global_demo.sample_to_demo.shape != (batch_size,):
        raise ValueError("Global sample_to_demo 必须是与 Query batch 对齐的 (B,) inverse index。")
    if torch.any(global_demo.sample_to_demo < 0) or torch.any(
        global_demo.sample_to_demo >= unique_demo_count
    ):
        raise ValueError("Global sample_to_demo 包含越界的唯一 Demo 索引。")

    for key in ("observation.state", "action"):
        value = batch.get(key)
        if isinstance(value, Tensor) and (value.ndim == 0 or value.shape[0] != batch_size):
            raise ValueError(f"Query 字段 {key!r} 的 batch 维必须等于 Local Demo batch size。")


def collate_global_demo_samples(
    samples: Sequence[GlobalDemoSample],
    *,
    sample_to_demo: Tensor | Sequence[int],
) -> GlobalDemoBatch:
    """将 ``U`` 条唯一 Demo 补齐，并保存 query 到 Demo 的 inverse index。"""
    if len(samples) == 0:
        raise ValueError("Global Demo batch 至少需要一个样本。")

    reference = samples[0]
    expected_state_tail = reference.states.shape[1:]
    expected_feature_dim = int(reference.video_features.shape[-1])
    for sample in samples:
        if sample.states.ndim != 3:
            raise ValueError("GlobalDemoSample states 必须使用 (K,L,D) 形状。")
        if sample.video_features.shape != (sample.num_clips, expected_feature_dim):
            raise ValueError("缓存的 S3D feature 必须为统一宽度的 (K,D_v) Tensor。")
        if sample.states.shape[1:] != expected_state_tail:
            raise ValueError("同一 Global Demo batch 的 clip/State 尺寸必须一致。")
        if sample.timestamps.shape != sample.states.shape[:2]:
            raise ValueError("GlobalDemoSample timestamps 形状不正确。")
        if sample.valid_mask.shape != sample.states.shape[:2]:
            raise ValueError("GlobalDemoSample valid_mask 形状不正确。")

    unique_demo_count = len(samples)
    max_clips = max(sample.num_clips for sample in samples)
    states = reference.states.new_zeros(unique_demo_count, max_clips, *expected_state_tail)
    timestamps = reference.timestamps.new_zeros(
        unique_demo_count,
        max_clips,
        expected_state_tail[0],
    )
    valid_mask = torch.zeros(
        unique_demo_count,
        max_clips,
        expected_state_tail[0],
        dtype=torch.bool,
        device=reference.states.device,
    )
    video_features = reference.video_features.new_zeros(
        unique_demo_count,
        max_clips,
        expected_feature_dim,
    )

    for batch_index, sample in enumerate(samples):
        if sample.states.device != reference.states.device:
            raise ValueError("collate 前所有 Global Demo 样本必须在同一设备。")
        count = sample.num_clips
        states[batch_index, :count] = sample.states
        timestamps[batch_index, :count] = sample.timestamps
        valid_mask[batch_index, :count] = sample.valid_mask
        video_features[batch_index, :count] = sample.video_features

    inverse = torch.as_tensor(
        sample_to_demo,
        dtype=torch.long,
        device=reference.states.device,
    )
    if inverse.ndim != 1 or inverse.numel() == 0:
        raise ValueError("sample_to_demo 必须是非空一维 inverse index。")
    if torch.any(inverse < 0) or torch.any(inverse >= unique_demo_count):
        raise ValueError("sample_to_demo 包含越界的唯一 Demo 索引。")

    return GlobalDemoBatch(
        video_features=video_features,
        states=states,
        timestamps=timestamps,
        valid_mask=valid_mask,
        sample_to_demo=inverse,
    )


def build_encoded_local_demo_batch(
    window: LocalDemoWindow,
    demo_visual_tokens: Tensor,
    demo_visual_embeddings: Tensor,
    *,
    expected_state_dim: int = 32,
) -> EncodedLocalDemoBatch:
    """用 Matcher 窗口索引从独立的 ``E_vision`` cache 构造 rollout 输入。"""
    source_indices = window.source_indices
    valid_mask = window.valid_mask
    clamped_indices = source_indices.clamp_min(0)
    visual_tokens = demo_visual_tokens[clamped_indices]
    visual_embeddings = demo_visual_embeddings[clamped_indices]
    visual_tokens = torch.where(
        valid_mask[:, None, None],
        visual_tokens,
        torch.zeros_like(visual_tokens),
    )
    visual_embeddings = torch.where(
        valid_mask[:, None],
        visual_embeddings,
        torch.zeros_like(visual_embeddings),
    )

    raw_state_dim = window.state_features.shape[-1] // 2
    state_values, state_velocities = window.state_features.split(raw_state_dim, dim=-1)
    state_padding = expected_state_dim - raw_state_dim
    if state_padding < 0:
        raise ValueError("Local Demo State 维度超过 expected_state_dim。")
    return EncodedLocalDemoBatch(
        visual_tokens=visual_tokens.unsqueeze(0),
        visual_embeddings=visual_embeddings.unsqueeze(0),
        # State 和 Velocity 必须分别补齐后再拼回，不能直接
        # 在末尾补零，否则会破坏 [State(32), Velocity(32)] 的分段语义。
        state_features=torch.cat(
            [
                F.pad(state_values, (0, state_padding)),
                F.pad(state_velocities, (0, state_padding)),
            ],
            dim=-1,
        ).unsqueeze(0),
        relative_time_s=window.relative_time_s.unsqueeze(0),
        relative_position=window.relative_position.unsqueeze(0),
        phase=window.phase.unsqueeze(0),
        valid_mask=valid_mask.unsqueeze(0),
        anchor_positions=torch.tensor(
            [window.anchor_position],
            dtype=torch.long,
            device=valid_mask.device,
        ),
    )


def collate_raw_local_demo_samples(
    samples: Sequence[RawLocalDemoSample],
    *,
    state_normalizer: DemoStateNormalizer,
    expected_state_dim: int = 32,
) -> LocalDemoBatch:
    """Collate 已选定的 raw Local 窗口，不执行任何视觉编码。

    这是训练路径的关键边界：collator 只准备 RGB/State/Time，
    ``E_vision`` 始终在 Policy forward 内执行。
    """
    if not samples:
        raise ValueError("Local Demo batch 至少需要一个样本。")
    reference = samples[0]
    chunk_length = int(reference.timestamps.shape[0])
    state_features: list[Tensor] = []
    images: list[Tensor] = []
    relative_times: list[Tensor] = []
    relative_positions: list[Tensor] = []
    phases: list[Tensor] = []

    for sample in samples:
        if sample.images.ndim != 4 or sample.images.shape[:2] != (chunk_length, 3):
            raise ValueError("Raw Local images 必须为统一尺寸的 (T,3,H,W) Tensor。")
        if sample.states.ndim != 2 or sample.states.shape[0] != chunk_length:
            raise ValueError("Raw Local states 必须为 (T,D_state) Tensor。")
        if sample.timestamps.shape != (chunk_length,) or sample.valid_mask.shape != (chunk_length,):
            raise ValueError("Raw Local timestamp/valid_mask 必须为 (T,) Tensor。")
        if not 0 <= sample.anchor_position < chunk_length:
            raise ValueError("Raw Local anchor_position 必须位于窗口内。")

        mask = sample.valid_mask.bool()
        raw_states = sample.states.float()
        normalized = state_normalizer.normalize(raw_states, valid_mask=mask)
        features = extract_state_features(
            normalized,
            sample.timestamps,
            valid_mask=mask,
        )
        state_padding = expected_state_dim - normalized.shape[-1]
        if state_padding < 0:
            raise ValueError("Local Demo State 维度超过 expected_state_dim。")
        values, velocities = features.split(normalized.shape[-1], dim=-1)
        state_features.append(
            torch.cat(
                [F.pad(values, (0, state_padding)), F.pad(velocities, (0, state_padding))],
                dim=-1,
            )
        )

        rgb = sample.images.float()
        if sample.images.dtype == torch.uint8:
            rgb = rgb / 255.0
        rgb = torch.where(mask[:, None, None, None], rgb, torch.zeros_like(rgb))
        images.append(rgb)

        anchor_time = sample.timestamps[sample.anchor_position]
        relative_times.append((sample.timestamps - anchor_time).float())
        relative_positions.append(
            (
                torch.arange(chunk_length, device=sample.timestamps.device)
                - sample.anchor_position
            ).float()
            / chunk_length
        )
        duration = max(sample.demo_end_timestamp - sample.demo_start_timestamp, 1e-6)
        phases.append(
            ((sample.timestamps.float() - sample.demo_start_timestamp) / duration).clamp(0, 1)
        )

    metadata_device = reference.valid_mask.device
    return LocalDemoBatch(
        images=torch.stack(images),
        state_features=torch.stack(state_features),
        relative_time_s=torch.stack(relative_times),
        relative_position=torch.stack(relative_positions),
        phase=torch.stack(phases),
        valid_mask=torch.stack([sample.valid_mask.bool() for sample in samples]),
        anchor_positions=torch.tensor(
            [sample.anchor_position for sample in samples],
            dtype=torch.long,
            device=metadata_device,
        ),
    )
