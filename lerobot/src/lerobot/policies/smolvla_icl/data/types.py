"""SmolVLA-ICL 数据层与模型层之间的显式输入类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor


SMOLVLA_ICL_GLOBAL_DEMO = "smolvla_icl.global_demo"
SMOLVLA_ICL_LOCAL_DEMO = "smolvla_icl.local_demo"
SMOLVLA_ICL_DEMO_REF = "smolvla_icl.demo_ref"


@dataclass(frozen=True, slots=True)
class GlobalDemoClips:
    """完整 Demo 切片后的 raw clips，仅用于离线缓存和 ``set_demo``。"""

    video: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class GlobalDemoSample:
    """单条 Demo 的冻结 S3D feature 和仍需训练的 State 输入。"""

    video_features: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor

    @property
    def num_clips(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True, slots=True)
class GlobalDemoBatch:
    """训练用 Global batch；batch 维只包含去重后的 ``U`` 条 Demo。"""

    video_features: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    sample_to_demo: Tensor

    def to(self, device: torch.device | str) -> Self:
        target = torch.device(device)
        return type(self)(
            video_features=self.video_features.to(target),
            states=self.states.to(target),
            timestamps=self.timestamps.to(target),
            valid_mask=self.valid_mask.to(target),
            sample_to_demo=self.sample_to_demo.to(target),
        )


@dataclass(frozen=True, slots=True)
class LocalDemoBatch:
    """训练用 Local batch；RGB 必须在 Policy forward 中经过可训练 ``E_vision``。"""

    images: Tensor
    state_features: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    anchor_positions: Tensor

    def to(self, device: torch.device | str) -> Self:
        target = torch.device(device)
        return type(self)(
            images=self.images.to(target),
            state_features=self.state_features.to(target),
            relative_time_s=self.relative_time_s.to(target),
            relative_position=self.relative_position.to(target),
            phase=self.phase.to(target),
            valid_mask=self.valid_mask.to(target),
            anchor_positions=self.anchor_positions.to(target),
        )


@dataclass(frozen=True, slots=True)
class EncodedLocalDemoBatch:
    """rollout 用 Local batch；视觉 token 已由当前固定的 ``E_vision`` 生成。"""

    visual_tokens: Tensor
    visual_embeddings: Tensor
    state_features: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    anchor_positions: Tensor

    def to(self, device: torch.device | str) -> Self:
        target = torch.device(device)
        return type(self)(
            visual_tokens=self.visual_tokens.to(target),
            visual_embeddings=self.visual_embeddings.to(target),
            state_features=self.state_features.to(target),
            relative_time_s=self.relative_time_s.to(target),
            relative_position=self.relative_position.to(target),
            phase=self.phase.to(target),
            valid_mask=self.valid_mask.to(target),
            anchor_positions=self.anchor_positions.to(target),
        )


@dataclass(frozen=True, slots=True)
class RawLocalDemoSample:
    """Dataset 根据 ``demo_id + local_anchor`` 读取的 raw Local 窗口。"""

    images: Tensor
    states: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    anchor_position: int
    demo_start_timestamp: float
    demo_end_timestamp: float


__all__ = [
    "EncodedLocalDemoBatch",
    "GlobalDemoBatch",
    "GlobalDemoClips",
    "GlobalDemoSample",
    "LocalDemoBatch",
    "RawLocalDemoSample",
    "SMOLVLA_ICL_DEMO_REF",
    "SMOLVLA_ICL_GLOBAL_DEMO",
    "SMOLVLA_ICL_LOCAL_DEMO",
]
