"""SmolVLA-ICL 的 Global Demo Encoder。

该模块只负责把一条完整 RGB+State Demo 压缩为固定数量的
Global Task Tokens ``G^(0)``。它不运行 Stage Matcher，不读取当前
Observation，也不执行 Demo Expert Transformer。这个边界使 Global
Tokens 可以在 Demo 加载时计算一次，并在后续多次重规划中缓存。

数据契约：

* RGB: ``(B, K, L, 3, H, W)``，值域为 ``[0, 1]``；
* State: ``(B, K, L, D_s)``；
* Timestamp: ``(B, K, L)``，单位为秒；
* Valid mask: ``(B, K, L)``，``True`` 表示真实 Demo 帧；
* Global Tokens: ``(B, N_G, d_D)``，``d_D`` 与 Demo Expert 宽度一致。

``K`` 是按时间排列的 clip 数，``L`` 是每个 clip 的帧数。变长
Demo 通过 ``valid_mask`` 在 batch 内补齐。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..configuration_smolvla_icl import GlobalEncoderConfig


__all__ = [
    "GlobalDemoEncoder",
    "GlobalEncoderOutput",
    "S3DVideoBackbone",
]


@dataclass(frozen=True, slots=True)
class GlobalEncoderOutput:
    """Global Encoder 的结构化输出。

    ``global_tokens`` 是交给 Demo Expert 的初始 Global 区域。
    ``clip_tokens`` 保留 Temporal Aggregator 的片段级输出，便于后续
    添加对比损失、时序辅助损失和可视化，但 Demo Expert 首版只消费
    ``global_tokens``。
    """

    global_tokens: Tensor
    global_mask: Tensor
    clip_tokens: Tensor
    clip_mask: Tensor
    clip_phase: Tensor

    def to(self, device: torch.device | str) -> Self:
        """返回所有 Tensor 已移到目标设备的新输出对象。"""
        target = torch.device(device)
        return type(self)(
            global_tokens=self.global_tokens.to(target),
            global_mask=self.global_mask.to(target),
            clip_tokens=self.clip_tokens.to(target),
            clip_mask=self.clip_mask.to(target),
            clip_phase=self.clip_phase.to(target),
        )


def _sinusoidal_phase_embedding(phase: Tensor, dimension: int) -> Tensor:
    """把连续 Demo phase 转换为固定正弦/余弦时间编码。

    这里使用实际 timestamp 得到的 ``[0, 1]`` phase，而不是简单的
    clip index。因此同一任务以不同帧率记录时，时间位置的语义仍然一致。
    """
    if dimension < 1:
        raise ValueError("phase embedding 维度必须大于 0。")

    # 使用 float32 计算三角函数，避免 bf16/fp16 在高频区间精度不足。
    phase_fp32 = phase.float()
    half_dim = max(1, dimension // 2)
    denominator = max(1, half_dim - 1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_dim, device=phase.device, dtype=torch.float32)
        / denominator
    )
    angles = 2 * math.pi * phase_fp32.unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # hidden size 为奇数时，先生成偶数维编码，再裁剪或补零。
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding[..., :dimension]


class S3DVideoBackbone(nn.Module):
    """TorchVision S3D 的 clip-level feature adapter。

    TorchVision S3D 最终的分类头会把视频特征转为 Kinetics 类别。
    Global Encoder 不需要该分类头，因此只保留 ``features``，然后对时间
    和空间维度执行 adaptive average pooling，得到每个 clip 的一个向量。

    输入 adapter 统一使用 ``(N, T, C, H, W)``，内部再转成 S3D
    要求的 ``(N, C, T, H, W)``。
    """

    def __init__(self, config: GlobalEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # torchvision 在 LeRobot 中是基础依赖，但放到构造函数内导入
        # 可以让只读配置或 Stage Matcher 的工具不必立即加载视频模型。
        try:
            from torchvision.models.video import S3D_Weights, s3d
        except ImportError as exc:  # pragma: no cover - 只在环境缺少依赖时触发
            raise ImportError("Global S3D Encoder 需要安装 torchvision。") from exc

        weights = S3D_Weights.DEFAULT if config.pretrained_backbone else None
        model = s3d(weights=weights)
        self.features = model.features
        self.preprocess = weights.transforms() if weights is not None else None
        # TorchVision 在 S3D weights metadata 中标记的最小时间长度为
        # 14。无预训练权重时网络拓扑没有变化，仍使用同一下限。
        self.min_temporal_size = int(weights.meta["min_temporal_size"]) if weights is not None else 14

        # 不在本项目中写死 S3D 最后通道数：从原分类头的第一个
        # Conv3d 反查其输入维度，以兼容 torchvision 后续的实现调整。
        classifier_conv = next(
            (module for module in model.classifier.modules() if isinstance(module, nn.Conv3d)),
            None,
        )
        if classifier_conv is None:
            raise RuntimeError("无法从 TorchVision S3D 分类头推断特征维度。")
        self.output_dim = classifier_conv.in_channels

        self.register_buffer(
            "image_mean",
            torch.tensor(config.image_mean, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(config.image_std, dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    def _preprocess_without_weights(self, clips: Tensor) -> Tensor:
        """为无预训练权重的 S3D 执行确定性 resize 和 normalize。"""
        num_clips, num_frames, channels, _, _ = clips.shape
        frames = clips.reshape(num_clips * num_frames, channels, *clips.shape[-2:])
        frames = F.interpolate(
            frames,
            size=self.config.image_size,
            mode="bilinear",
            align_corners=False,
        )
        video = frames.reshape(num_clips, num_frames, channels, *self.config.image_size)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        mean = self.image_mean.to(device=video.device, dtype=video.dtype)
        std = self.image_std.to(device=video.device, dtype=video.dtype)
        return (video - mean) / std

    def forward(self, clips: Tensor, frame_valid_mask: Tensor) -> Tensor:
        """把一批 clip 编码为 ``(N, D_v)`` 视频特征。"""
        if clips.ndim != 5 or clips.shape[2] != 3 or not clips.is_floating_point():
            raise ValueError("S3D clips 必须是浮点 (N,T,3,H,W) Tensor。")
        if frame_valid_mask.shape != clips.shape[:2]:
            raise ValueError("frame_valid_mask 必须与 clips 的 (N,T) 维度一致。")
        if clips.shape[1] < self.min_temporal_size:
            raise ValueError(
                f"S3D 每个 clip 至少需要 {self.min_temporal_size} 帧，"
                f"实际为 {clips.shape[1]} 帧。"
            )

        mask = frame_valid_mask.to(device=clips.device, dtype=torch.bool)
        safe_clips = torch.where(
            mask[:, :, None, None, None],
            clips,
            torch.zeros_like(clips),
        )

        if self.preprocess is not None:
            # TorchVision 预训练 transform 接收 (...,T,C,H,W)，输出
            # (...,C,T,H,W)，并执行与 S3D 权重匹配的 resize/crop/normalize。
            video = self.preprocess(safe_clips)
        else:
            video = self._preprocess_without_weights(safe_clips)

        if video.ndim != 5 or video.shape[:3] != (
            clips.shape[0],
            clips.shape[2],
            clips.shape[1],
        ):
            raise RuntimeError("S3D 预处理后的视频形状不符合 (N,C,T,H,W)。")

        # 黑色 padding 帧经 normalize 后不再是 0，因此必须在预处理
        # 之后再次施加 frame mask，防止 padding 进入 3D 卷积。
        video = video * mask[:, None, :, None, None].to(video.dtype)
        feature_parameter = next(self.features.parameters())
        video = video.to(dtype=feature_parameter.dtype)
        features = self.features(video)
        pooled = F.adaptive_avg_pool3d(features, output_size=1).flatten(1)

        # 全 padding clip 即使经过带 bias 的卷积也必须输出严格的零。
        clip_valid = mask.any(dim=1)
        return torch.where(clip_valid[:, None], pooled, torch.zeros_like(pooled))


class _MaskedStateEncoder(nn.Module):
    """把每个 clip 内的 State 序列压缩为一个片段特征。

    输入同时使用绝对 State 和严格向后差分 ``dState/dt``。逐帧
    Temporal MLP 后聚合均值、最新有效帧和首尾变化。这种实现不假设
    padding 一定连续出现在 clip 末尾，比直接用 packed GRU 更适合带缺帧
    的 Demo。
    """

    def __init__(self, state_dim: int, output_dim: int, eps: float) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.eps = eps
        self.frame_mlp = nn.Sequential(
            nn.Linear(state_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.clip_projection = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, states: Tensor, timestamps: Tensor, valid_mask: Tensor) -> Tensor:
        """返回 ``(N,D_state)`` clip-level State 特征。"""
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError(
                f"State Encoder 需要 (N,T,{self.state_dim})，实际为 {tuple(states.shape)}。"
            )
        if timestamps.shape != states.shape[:2] or valid_mask.shape != states.shape[:2]:
            raise ValueError("State timestamps/valid_mask 必须与 states 的 (N,T) 一致。")

        mask = valid_mask.to(device=states.device, dtype=torch.bool)
        values = torch.where(mask.unsqueeze(-1), states, torch.zeros_like(states))
        safe_times = torch.where(mask, timestamps, torch.zeros_like(timestamps))

        velocity = torch.zeros_like(values)
        if states.shape[1] > 1:
            pair_valid = mask[:, 1:] & mask[:, :-1]
            delta_t = safe_times[:, 1:] - safe_times[:, :-1]
            safe_delta_t = torch.where(
                pair_valid,
                delta_t.clamp_min(self.eps),
                torch.ones_like(delta_t),
            )
            delta_state = values[:, 1:] - values[:, :-1]
            velocity[:, 1:] = torch.where(
                pair_valid.unsqueeze(-1),
                delta_state / safe_delta_t.unsqueeze(-1).to(delta_state.dtype),
                torch.zeros_like(delta_state),
            )

        frame_input = torch.cat([values, velocity], dim=-1)
        frame_input = frame_input.to(dtype=self.frame_mlp[0].weight.dtype)
        frame_hidden = self.frame_mlp(frame_input)
        frame_hidden = frame_hidden * mask.unsqueeze(-1).to(frame_hidden.dtype)

        valid_count = mask.sum(dim=1, keepdim=True)
        masked_mean = frame_hidden.sum(dim=1) / valid_count.clamp_min(1).to(frame_hidden.dtype)

        positions = torch.arange(states.shape[1], device=states.device).unsqueeze(0)
        first_position = torch.where(mask, positions, states.shape[1]).min(dim=1).values
        last_position = torch.where(mask, positions, -torch.ones_like(positions)).max(dim=1).values
        safe_first = first_position.clamp(0, states.shape[1] - 1)
        safe_last = last_position.clamp(0, states.shape[1] - 1)
        rows = torch.arange(states.shape[0], device=states.device)
        first_hidden = frame_hidden[rows, safe_first]
        last_hidden = frame_hidden[rows, safe_last]

        clip_hidden = self.clip_projection(
            torch.cat([masked_mean, last_hidden, last_hidden - first_hidden], dim=-1)
        )
        clip_valid = valid_count.squeeze(1) > 0
        return torch.where(clip_valid[:, None], clip_hidden, torch.zeros_like(clip_hidden))


class GlobalDemoEncoder(nn.Module):
    """RGB+State Full Demo -> Global Task Tokens。

    首版结构为：

    ``S3D clip feature + State temporal feature``
    ``-> RGB/State fusion + continuous phase embedding``
    ``-> lightweight Temporal Transformer``
    ``-> learnable Task Queries``
    ``-> fixed-length Global Tokens``。

    ``video_backbone`` 参数用于测试和后续骨干消融。自定义骨干必须
    实现 ``forward(clips, frame_valid_mask)`` 并提供 ``output_dim``。
    """

    def __init__(
        self,
        config: GlobalEncoderConfig | None = None,
        *,
        video_backbone: nn.Module | None = None,
        video_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GlobalEncoderConfig()

        if video_backbone is None:
            self.video_backbone = S3DVideoBackbone(self.config)
            inferred_video_dim = self.video_backbone.output_dim
        else:
            self.video_backbone = video_backbone
            inferred_video_dim = getattr(video_backbone, "output_dim", None)
            if video_feature_dim is not None:
                inferred_video_dim = video_feature_dim
            if inferred_video_dim is None or int(inferred_video_dim) < 1:
                raise ValueError("自定义 video_backbone 必须提供有效的 output_dim。")
        self.video_feature_dim = int(inferred_video_dim)

        self.state_encoder = _MaskedStateEncoder(
            state_dim=self.config.state_dim,
            output_dim=self.config.state_feature_dim,
            eps=self.config.eps,
        )
        self.rgb_state_fusion = nn.Sequential(
            nn.Linear(
                self.video_feature_dim + self.config.state_feature_dim,
                self.config.temporal_hidden_size,
            ),
            nn.LayerNorm(self.config.temporal_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.temporal_hidden_size, self.config.temporal_hidden_size),
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.config.temporal_hidden_size,
            nhead=self.config.temporal_num_heads,
            dim_feedforward=int(
                self.config.temporal_hidden_size * self.config.temporal_mlp_ratio
            ),
            dropout=self.config.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_aggregator = nn.TransformerEncoder(
            temporal_layer,
            num_layers=self.config.temporal_num_layers,
            norm=nn.LayerNorm(self.config.temporal_hidden_size),
        )
        self.task_queries = nn.Parameter(
            torch.empty(
                1,
                self.config.num_global_tokens,
                self.config.temporal_hidden_size,
            )
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.temporal_hidden_size, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )
        nn.init.normal_(self.task_queries, mean=0.0, std=0.02)

        self._set_video_backbone_trainability()

    def _set_video_backbone_trainability(self) -> None:
        """按配置冻结视频骨干，并固定 BatchNorm/Dropout 的运行模式。"""
        requires_grad = not self.config.freeze_video_backbone
        for parameter in self.video_backbone.parameters():
            parameter.requires_grad_(requires_grad)
        if self.config.freeze_video_backbone:
            self.video_backbone.eval()

    def train(self, mode: bool = True) -> Self:
        """切换训练模式，但冻结的视频骨干始终保持 eval。"""
        super().train(mode)
        if self.config.freeze_video_backbone:
            self.video_backbone.eval()
        return self

    def _validate_inputs(
        self,
        video: Tensor,
        states: Tensor,
        timestamps: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """在进入高成本视频骨干前拒绝错误形状或非法数值。"""
        if video.ndim != 6 or video.shape[3] != 3 or not video.is_floating_point():
            raise ValueError("Global video 必须是浮点 (B,K,L,3,H,W) Tensor。")
        if states.ndim != 4 or states.shape[:3] != video.shape[:3]:
            raise ValueError("Global states 必须是与 video 时间维对齐的 (B,K,L,Ds)。")
        if states.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"Global states 最后一维必须是 {self.config.state_dim}，"
                f"实际为 {states.shape[-1]}。"
            )
        if not states.is_floating_point():
            raise ValueError("Global states 必须是浮点 Tensor。")
        if timestamps.shape != video.shape[:3] or valid_mask.shape != video.shape[:3]:
            raise ValueError("timestamps/valid_mask 必须与 video 的 (B,K,L) 一致。")
        if video.shape[0] == 0 or video.shape[1] == 0 or video.shape[2] == 0:
            raise ValueError("Global Demo 的 batch、clip 数和每 clip 帧数都必须大于 0。")

        mask = valid_mask.to(device=video.device, dtype=torch.bool)
        if torch.any(mask.flatten(1).sum(dim=1) == 0):
            raise ValueError("每条 Demo 至少需要一帧有效 RGB+State Observation。")
        if torch.any(~torch.isfinite(video[mask])):
            raise ValueError("有效 Demo RGB 必须只包含有限值。")
        if torch.any(~torch.isfinite(states[mask])):
            raise ValueError("有效 Demo State 必须只包含有限值。")
        if torch.any(~torch.isfinite(timestamps[mask])):
            raise ValueError("有效 Demo timestamp 必须只包含有限值。")

        valid_pixels = video[mask]
        if torch.any(valid_pixels < 0) or torch.any(valid_pixels > 1):
            raise ValueError(
                "Global video 必须保持原始 [0,1] 值域，"
                "不能使用 SigLIP [-1,1] 输入。"
            )

        # Demo 的时间顺序是 Global 表示的关键语义，因此对每个
        # batch 样本严格检查所有有效帧的 timestamp 递增。
        flat_times = timestamps.reshape(timestamps.shape[0], -1)
        flat_mask = mask.reshape(mask.shape[0], -1)
        for batch_index in range(timestamps.shape[0]):
            valid_times = flat_times[batch_index, flat_mask[batch_index]]
            if len(valid_times) > 1 and torch.any(valid_times[1:] <= valid_times[:-1]):
                raise ValueError("Demo 的有效 timestamps 必须按 clip/frame 顺序严格递增。")

    def _compute_clip_phase(self, timestamps: Tensor, valid_mask: Tensor) -> Tensor:
        """计算每个 clip 中心在完整 Demo 内的连续 phase。"""
        batch_size, num_clips, frames_per_clip = timestamps.shape
        mask = valid_mask.to(dtype=torch.bool)
        safe_times = torch.where(mask, timestamps, torch.zeros_like(timestamps))
        clip_count = mask.sum(dim=-1)
        clip_center = safe_times.sum(dim=-1) / clip_count.clamp_min(1).to(safe_times.dtype)

        flat_times = timestamps.reshape(batch_size, num_clips * frames_per_clip)
        flat_mask = mask.reshape(batch_size, num_clips * frames_per_clip)
        positions = torch.arange(flat_times.shape[1], device=timestamps.device).unsqueeze(0)
        first_position = torch.where(flat_mask, positions, flat_times.shape[1]).min(dim=1).values
        last_position = torch.where(flat_mask, positions, -torch.ones_like(positions)).max(dim=1).values
        rows = torch.arange(batch_size, device=timestamps.device)
        start_time = flat_times[rows, first_position]
        end_time = flat_times[rows, last_position]
        duration = (end_time - start_time).clamp_min(self.config.eps)
        phase = (clip_center - start_time[:, None]) / duration[:, None]
        phase = phase.clamp(0.0, 1.0)
        return torch.where(clip_count > 0, phase, torch.zeros_like(phase))

    def forward(
        self,
        video: Tensor,
        states: Tensor,
        timestamps: Tensor,
        valid_mask: Tensor | None = None,
    ) -> GlobalEncoderOutput:
        """编码完整 Demo，返回固定数量的 Global Task Tokens。"""
        if valid_mask is None:
            valid_mask = torch.ones(video.shape[:3], dtype=torch.bool, device=video.device)
        else:
            valid_mask = valid_mask.to(device=video.device, dtype=torch.bool)
        timestamps = timestamps.to(device=video.device, dtype=torch.float64)
        states = states.to(device=video.device)
        self._validate_inputs(video, states, timestamps, valid_mask)

        batch_size, num_clips, frames_per_clip = video.shape[:3]
        flat_video = video.reshape(batch_size * num_clips, frames_per_clip, *video.shape[3:])
        flat_states = states.reshape(batch_size * num_clips, frames_per_clip, states.shape[-1])
        flat_times = timestamps.reshape(batch_size * num_clips, frames_per_clip)
        flat_mask = valid_mask.reshape(batch_size * num_clips, frames_per_clip)

        # 冻结骨干时不构建 autograd graph，显著降低整条 Demo 编码的显存。
        backbone_context = torch.no_grad() if self.config.freeze_video_backbone else nullcontext()
        with backbone_context:
            visual_features = self.video_backbone(flat_video, flat_mask)
        if visual_features.shape != (batch_size * num_clips, self.video_feature_dim):
            raise RuntimeError(
                "video_backbone 必须输出 "
                f"({batch_size * num_clips},{self.video_feature_dim})，"
                f"实际为 {tuple(visual_features.shape)}。"
            )

        state_features = self.state_encoder(flat_states, flat_times, flat_mask)
        visual_features = visual_features.reshape(batch_size, num_clips, -1)
        state_features = state_features.reshape(batch_size, num_clips, -1)

        valid_fraction = valid_mask.float().mean(dim=-1)
        clip_mask = valid_fraction >= self.config.min_valid_frame_fraction
        if torch.any(clip_mask.sum(dim=1) == 0):
            raise ValueError(
                "某条 Demo 没有 clip 达到 min_valid_frame_fraction，无法构建 Global Tokens。"
            )

        fusion_dtype = self.rgb_state_fusion[0].weight.dtype
        fused = self.rgb_state_fusion(
            torch.cat(
                [
                    visual_features.to(dtype=fusion_dtype),
                    state_features.to(dtype=fusion_dtype),
                ],
                dim=-1,
            )
        )
        clip_phase = self._compute_clip_phase(timestamps, valid_mask)
        phase_embedding = _sinusoidal_phase_embedding(
            clip_phase,
            self.config.temporal_hidden_size,
        ).to(dtype=fused.dtype)
        clip_input = fused + phase_embedding
        clip_input = clip_input * clip_mask.unsqueeze(-1).to(clip_input.dtype)

        task_queries = self.task_queries.to(dtype=clip_input.dtype).expand(batch_size, -1, -1)
        temporal_input = torch.cat([task_queries, clip_input], dim=1)
        query_mask = torch.ones(
            batch_size,
            self.config.num_global_tokens,
            dtype=torch.bool,
            device=video.device,
        )
        temporal_valid = torch.cat([query_mask, clip_mask], dim=1)

        # TransformerEncoder 的 padding mask 语义与项目其他 valid mask 相反：
        # True 表示忽略该 Key，因此这里取反。
        temporal_output = self.temporal_aggregator(
            temporal_input,
            src_key_padding_mask=~temporal_valid,
        )
        task_output = temporal_output[:, : self.config.num_global_tokens]
        clip_output = temporal_output[:, self.config.num_global_tokens :]
        clip_output = clip_output * clip_mask.unsqueeze(-1).to(clip_output.dtype)

        global_tokens = self.output_projection(task_output)
        global_mask = torch.ones(
            batch_size,
            self.config.num_global_tokens,
            dtype=torch.bool,
            device=video.device,
        )
        return GlobalEncoderOutput(
            global_tokens=global_tokens,
            global_mask=global_mask,
            clip_tokens=clip_output,
            clip_mask=clip_mask,
            clip_phase=clip_phase,
        )
