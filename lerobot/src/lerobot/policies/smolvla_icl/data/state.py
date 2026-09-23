"""SmolVLA-ICL 共享的 State 归一化契约。

Global、Local 和 Stage Matcher 都必须使用同一份训练集统计量。
将该类放在独立数据模块中，可以让 Collator 和 Matcher 共享实现，
而不需要互相导入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.constants import OBS_STATE

StateNormalizationSignature = tuple[tuple[float, ...], tuple[float, ...], float]

__all__ = ["DemoStateNormalizer", "StateNormalizationSignature"]


@dataclass(frozen=True, slots=True)
class DemoStateNormalizer:
    """用 SmolVLA 训练集统计量归一化真实维度的 State。

    SmolVLA 对 State 默认使用 ``MEAN_STD``：

    ``normalized = (state - mean) / (std + eps)``。

    ``mean`` 和 ``std`` 只覆盖机器人的真实 State 维度。这使
    Matcher 能在归一化前拒绝已补到 32 维的 State，保证
    ``matching_state_excluded_indices=(-1,)`` 始终指向真实夹爪维。
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

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def from_dataset_stats(
        cls,
        dataset_stats: dict[str, dict[str, Any]],
        *,
        state_key: str = OBS_STATE,
        eps: float = 1e-8,
    ) -> DemoStateNormalizer:
        """从 LeRobot ``dataset_stats`` 读取 Current/Demo 共享的统计量。"""
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

    @property
    def signature(self) -> StateNormalizationSignature:
        """返回可比较的统计量签名，用于校验 Demo/Query 契约一致。"""
        return (
            tuple(float(value) for value in self.mean.tolist()),
            tuple(float(value) for value in self.std.tolist()),
            float(self.eps),
        )

    def normalize(
        self,
        states: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """归一化 ``(...,D_raw)`` State，保留真实维度并清零无效项。"""
        if states.ndim < 1 or not states.is_floating_point():
            raise ValueError("states 必须是至少一维的浮点 Tensor。")
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                "Matcher/Encoder 必须接收未 padding 的 raw State："
                f"期望 {self.state_dim} 维，实际 {states.shape[-1]} 维。"
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
        return torch.where(mask.unsqueeze(-1), normalized, torch.zeros_like(normalized))

    def normalize_and_pad(
        self,
        states: Tensor,
        *,
        target_dim: int,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """先归一化真实 State，再将最后一维右侧补到 ``target_dim``。"""
        if target_dim < self.state_dim:
            raise ValueError(f"target_dim={target_dim} 小于真实 State 维度 {self.state_dim}。")
        normalized = self.normalize(states, valid_mask=valid_mask)
        return F.pad(normalized, (0, target_dim - self.state_dim))
