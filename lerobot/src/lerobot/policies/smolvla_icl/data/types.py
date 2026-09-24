"""SmolVLA-ICL 数据层与模型层之间的显式输入类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

SMOLVLA_ICL_GLOBAL_DEMO = "smolvla_icl.global_demo"
SMOLVLA_ICL_LOCAL_DEMO = "smolvla_icl.local_demo"
SMOLVLA_ICL_DEMO_REF = "smolvla_icl.demo_ref"


def _pin_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.pin_memory() if tensor.device.type == "cpu" else tensor


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

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        target = torch.device(device)
        return type(self)(
            video_features=self.video_features.to(target, non_blocking=non_blocking),
            states=self.states.to(target, non_blocking=non_blocking),
            timestamps=self.timestamps.to(target, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(target, non_blocking=non_blocking),
            sample_to_demo=self.sample_to_demo.to(target, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            video_features=_pin_cpu_tensor(self.video_features),
            states=_pin_cpu_tensor(self.states),
            timestamps=_pin_cpu_tensor(self.timestamps),
            valid_mask=_pin_cpu_tensor(self.valid_mask),
            sample_to_demo=_pin_cpu_tensor(self.sample_to_demo),
        )


@dataclass(frozen=True, slots=True)
class LocalDemoBatch:
    """训练用 Local batch；RGB 固定留在 CPU，其余字段可移动到模型设备。"""

    images: Tensor
    state_features: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    anchor_positions: Tensor

    def __post_init__(self) -> None:
        if self.images.device.type != "cpu" or self.images.dtype != torch.uint8:
            raise ValueError("训练期 Local RGB 必须是留在 CPU 的 uint8 Tensor。")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """只移动轻量特征和 mask；完整 Local RGB 始终保留在 CPU。"""
        target = torch.device(device)
        return type(self)(
            images=self.images,
            state_features=self.state_features.to(target, non_blocking=non_blocking),
            relative_time_s=self.relative_time_s.to(target, non_blocking=non_blocking),
            relative_position=self.relative_position.to(target, non_blocking=non_blocking),
            phase=self.phase.to(target, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(target, non_blocking=non_blocking),
            anchor_positions=self.anchor_positions.to(target, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Self:
        """供 DataLoader pin-memory 线程递归锁页整个结构化 batch。"""
        return type(self)(
            images=_pin_cpu_tensor(self.images),
            state_features=_pin_cpu_tensor(self.state_features),
            relative_time_s=_pin_cpu_tensor(self.relative_time_s),
            relative_position=_pin_cpu_tensor(self.relative_position),
            phase=_pin_cpu_tensor(self.phase),
            valid_mask=_pin_cpu_tensor(self.valid_mask),
            anchor_positions=_pin_cpu_tensor(self.anchor_positions),
        )


@dataclass(frozen=True, slots=True)
class EncodedLocalDemoBatch:
    """rollout 用 Local batch；视觉特征已在 ``set_demo`` 中完成空间池化。"""

    visual_hidden: Tensor
    state_features: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    anchor_positions: Tensor

    def to(self, device: torch.device | str) -> Self:
        target = torch.device(device)
        return type(self)(
            visual_hidden=self.visual_hidden.to(target),
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
    # 窗口左边界的前一帧只用于计算第一个 Local token 的
    # 后向差分速度，不会产生额外 token。Episode 起点两者均为 None。
    previous_state: Tensor | None
    previous_timestamp: float | None
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
