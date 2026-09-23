"""SmolVLA-ICL 的 Local Demo Encoder。

本模块把 Stage Matcher 已经对齐的 Local Demo Chunk 转换为
Demo Expert 的初始 Local Tokens ``L^(0)``。每个 Demo observation 产生
一个 token；跨帧 Self-Attention 由后续 Demo Expert 负责，这里不再
叠加 GRU 或 Transformer。

输入契约：

* spatial visual tokens: ``(B, N_L, P, D_v)``，先由
  :meth:`pool_spatial_tokens` 压缩；
* visual hidden: ``(B, N_L, D_pool)``；
* ``[State, dState/dt]``: ``(B, N_L, 2*D_s)``；
* relative time / relative position / global phase: ``(B, N_L)``；
* valid mask: ``(B, N_L)``；
* Local Tokens: ``(B, N_L, d_D)``。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ...common.vla_utils import create_sinusoidal_pos_embedding
from ..configuration_smolvla_icl import LocalEncoderConfig

__all__ = ["LocalDemoEncoder", "LocalEncoderOutput"]


@dataclass(frozen=True, slots=True)
class LocalEncoderOutput:
    """Local Encoder 交给 Demo Expert 的 token 和 padding mask。"""

    local_tokens: Tensor
    local_mask: Tensor


class LocalDemoEncoder(nn.Module):
    """Local RGB+State+Time -> Demo Expert Local Tokens。

    每帧空间视觉 tokens 由一个可学习 query 执行 attention pooling，
    再与独立编码的 State 在帧内融合。
    连续时间信息沿用 SmolVLA flow timestep 的正弦/余弦编码
    风格，但 relative time、relative position 和 global phase 保持独立语义。
    """

    def __init__(self, config: LocalEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalEncoderConfig()

        self.visual_projection = nn.Sequential(
            nn.Linear(
                self.config.visual_feature_dim,
                self.config.visual_projection_dim,
            ),
            nn.LayerNorm(self.config.visual_projection_dim),
            nn.GELU(),
        )
        # 单个 query 将每帧 P 个空间 token 压缩成一个视觉表示，
        # 因而不会改变 Local 序列长度和后续 Demo Expert 的 Mask 契约。
        self.spatial_query = nn.Parameter(torch.empty(self.config.visual_projection_dim))
        self.state_projection = nn.Sequential(
            nn.Linear(
                self.config.state_dim * 2,
                self.config.state_projection_dim,
            ),
            nn.LayerNorm(self.config.state_projection_dim),
            nn.GELU(),
            nn.Linear(
                self.config.state_projection_dim,
                self.config.state_projection_dim,
            ),
        )
        self.rgb_state_fusion = nn.Sequential(
            nn.Linear(
                self.config.visual_projection_dim + self.config.state_projection_dim,
                self.config.output_dim,
            ),
            nn.LayerNorm(self.config.output_dim),
            nn.GELU(),
            nn.Linear(self.config.output_dim, self.config.output_dim),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(
                self.config.temporal_embedding_dim * 3,
                self.config.output_dim,
            ),
            nn.GELU(),
            nn.Linear(self.config.output_dim, self.config.output_dim),
        )
        self.local_type_embedding = nn.Parameter(torch.empty(1, 1, self.config.output_dim))
        self.output_norm = nn.LayerNorm(self.config.output_dim)
        nn.init.normal_(self.spatial_query, mean=0.0, std=0.02)
        nn.init.normal_(self.local_type_embedding, mean=0.0, std=0.02)

    def pool_spatial_tokens(self, visual_tokens: Tensor) -> Tensor:
        """把每帧 spatial tokens 压缩为一个视觉向量。

        该方法也用于训练期的 checkpoint 单元，使 SigLIP、connector 和
        learnable spatial pooling 在 backward 时一起重算。这样计算图长期
        保留的是 ``(..., D_pool)``，而不是 ``(..., P, D_vision)``。
        """
        if (
            visual_tokens.ndim < 3
            or visual_tokens.shape[-2] == 0
            or visual_tokens.shape[-1] != self.config.visual_feature_dim
            or not visual_tokens.is_floating_point()
        ):
            raise ValueError(
                f"Local spatial tokens 必须为 (...,P,{self.config.visual_feature_dim}) 的浮点 Tensor。"
            )
        dtype = self.visual_projection[0].weight.dtype
        spatial_hidden = self.visual_projection(visual_tokens.to(dtype=dtype))
        attention_logits = torch.einsum(
            "...pd,d->...p",
            spatial_hidden,
            self.spatial_query.to(dtype=dtype),
        ) * (self.config.visual_projection_dim**-0.5)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        visual_hidden = torch.sum(
            attention_weights.unsqueeze(-1) * spatial_hidden,
            dim=-2,
        )
        return visual_hidden

    def _encode_temporal_coordinates(
        self,
        relative_time_s: Tensor,
        relative_position: Tensor,
        phase: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """将三个连续时间坐标编码并投影到 Demo Expert 宽度。"""
        embeddings = [
            create_sinusoidal_pos_embedding(
                coordinate,
                self.config.temporal_embedding_dim,
                self.config.min_period,
                self.config.max_period,
                device=coordinate.device,
            )
            for coordinate in (relative_time_s, relative_position, phase)
        ]
        temporal_input = torch.cat(embeddings, dim=-1).to(dtype=dtype)
        return self.temporal_projection(temporal_input)

    def forward(
        self,
        visual_hidden: Tensor,
        state_features: Tensor,
        relative_time_s: Tensor,
        relative_position: Tensor,
        phase: Tensor,
        valid_mask: Tensor,
    ) -> LocalEncoderOutput:
        """融合已经 spatial-pool 的视觉特征与 State/时间信息。"""
        if valid_mask.ndim != 2:
            raise ValueError("Local valid_mask 必须为 (B,N)。")
        expected_shape = (*valid_mask.shape, self.config.visual_projection_dim)
        if visual_hidden.shape != expected_shape or not visual_hidden.is_floating_point():
            raise ValueError(
                f"Local visual_hidden 必须为 (B,N,{self.config.visual_projection_dim}) 的浮点 Tensor。"
            )
        if state_features.shape != (*valid_mask.shape, self.config.state_dim * 2):
            raise ValueError(f"Local state_features 必须为 (B,N,{self.config.state_dim * 2})。")
        for name, value in (
            ("relative_time_s", relative_time_s),
            ("relative_position", relative_position),
            ("phase", phase),
        ):
            if value.shape != valid_mask.shape:
                raise ValueError(f"Local {name} 必须为 (B,N)。")
        if not state_features.is_floating_point():
            raise ValueError("Local State 特征必须是浮点 Tensor。")

        mask = valid_mask.to(device=visual_hidden.device, dtype=torch.bool)
        visual_hidden = torch.where(
            mask.unsqueeze(-1),
            visual_hidden,
            torch.zeros_like(visual_hidden),
        )
        state_values = torch.where(
            mask.unsqueeze(-1),
            state_features,
            torch.zeros_like(state_features),
        )
        relative_time_values = torch.where(mask, relative_time_s, torch.zeros_like(relative_time_s))
        relative_position_values = torch.where(
            mask,
            relative_position,
            torch.zeros_like(relative_position),
        )
        phase_values = torch.where(mask, phase, torch.zeros_like(phase))

        model_dtype = self.visual_projection[0].weight.dtype
        state_hidden = self.state_projection(state_values.to(dtype=model_dtype))
        fused_hidden = self.rgb_state_fusion(torch.cat([visual_hidden, state_hidden], dim=-1))
        temporal_hidden = self._encode_temporal_coordinates(
            relative_time_values,
            relative_position_values,
            phase_values,
            dtype=model_dtype,
        )

        local_tokens = self.output_norm(
            fused_hidden + temporal_hidden + self.local_type_embedding.to(dtype=model_dtype)
        )
        local_tokens = torch.where(
            mask.unsqueeze(-1),
            local_tokens,
            torch.zeros_like(local_tokens),
        )
        return LocalEncoderOutput(local_tokens=local_tokens, local_mask=mask)
