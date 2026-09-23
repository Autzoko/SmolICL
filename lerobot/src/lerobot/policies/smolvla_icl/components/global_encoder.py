"""SmolVLA-ICL 的 Global Demo Encoder。

该模块只负责把一条完整 RGB+State Demo 压缩为固定数量的
Global Task Tokens ``G^(0)``。它不运行 Stage Matcher，不读取当前
Observation，也不执行 Demo Expert Transformer。推理时最终 Global Tokens
可按 Demo 缓存；训练时只能缓存冻结 S3D 的 clip feature，后续可训练路径
必须在每个 step 中保留计算图。

数据契约：

* 离线阶段 RGB: ``(U, K, L, 3, H, W)``，值域为 ``[0, 1]``；
* 训练阶段冻结 S3D feature: ``(U, K, D_v)``；
* State: ``(U, K, L, D_s)``；
* Timestamp: ``(U, K, L)``，单位为秒；
* Valid mask: ``(U, K, L)``，``True`` 表示真实 Demo 帧；
* Global Tokens: ``(B, N_G, d_D)``，``d_D`` 与 Demo Expert 宽度一致。

``U`` 是 batch 内唯一 Demo 数，``B`` 是 query 数；inverse index 在编码后
恢复 ``B``。``K`` 是按时间排列的 clip 数，``L`` 是每个 clip 的帧数。
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from ..configuration_smolvla_icl import GlobalEncoderConfig

__all__ = [
    "GlobalDemoEncoder",
    "GlobalEncoderOutput",
    "S3DVideoBackbone",
]


@dataclass(frozen=True, slots=True)
class GlobalEncoderOutput:
    """交给 Demo Expert 的 Global tokens 与有效 mask。"""

    global_tokens: Tensor
    global_mask: Tensor


def _sinusoidal_phase_embedding(phase: Tensor, dimension: int) -> Tensor:
    """把连续 Demo phase 转换为固定正弦/余弦时间编码。

    这里使用实际 timestamp 得到的 ``[0, 1]`` phase，而不是简单的
    clip index。因此同一任务以不同帧率记录时，时间位置的语义仍然一致。
    """
    # 使用 float32 计算三角函数，避免 bf16/fp16 在高频区间精度不足。
    phase_fp32 = phase.float()
    half_dim = max(1, dimension // 2)
    denominator = max(1, half_dim - 1)
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half_dim, device=phase.device, dtype=torch.float32) / denominator
    )
    angles = 2 * math.pi * phase_fp32.unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # hidden size 为奇数时，先生成偶数维编码，再裁剪或补零。
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding[..., :dimension]


def _resample_valid_video_frames(clips: Tensor, valid_mask: Tensor) -> Tensor:
    """把每个部分有效的 clip 用自身有效帧简单补满。

    S3D 的 3D 卷积不能直接消费 frame mask。若只把 padding 置零，卷积和
    最终池化仍会混合真实帧与 padding。这里按时间顺序把有效帧最近邻重采样
    到原 clip 长度，使送入 S3D 的非空 clip 不再包含人工零帧。
    """
    num_frames = clips.shape[1]
    dense_clips: list[Tensor] = []
    for clip, mask in zip(clips, valid_mask, strict=True):
        valid_frames = clip[mask]
        if len(valid_frames) == 0:
            dense_clips.append(torch.zeros_like(clip))
            continue
        if len(valid_frames) == num_frames:
            dense_clips.append(clip)
            continue
        indices = (
            torch.linspace(
                0,
                len(valid_frames) - 1,
                steps=num_frames,
                device=clips.device,
            )
            .round()
            .long()
        )
        dense_clips.append(valid_frames.index_select(0, indices))
    return torch.stack(dense_clips)


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
                f"S3D 每个 clip 至少需要 {self.min_temporal_size} 帧，实际为 {clips.shape[1]} 帧。"
            )

        mask = frame_valid_mask.to(device=clips.device, dtype=torch.bool)
        # S3D 内部没有 frame-level mask，因此在进入 3D 卷积前将每个
        # 部分有效 clip 用自身的有效帧补满，而不是向视频中混入零帧。
        safe_clips = _resample_valid_video_frames(clips, mask)
        clip_valid = mask.any(dim=1)

        if self.preprocess is not None:
            # TorchVision 预训练 transform 接收 (...,T,C,H,W)，输出
            # (...,C,T,H,W)，并执行与 S3D 权重匹配的 resize/crop/normalize。
            video = self.preprocess(safe_clips)
        else:
            video = self._preprocess_without_weights(safe_clips)

        # 全空 clip 经 normalize 后也可能不为零，因此在预处理后清零；
        # 部分有效 clip 已经在上方重采样为稠密视频，不再施加原始 mask。
        video = video * clip_valid[:, None, None, None, None].to(video.dtype)
        video = video.to(dtype=next(self.features.parameters()).dtype)
        features = self.features(video)
        pooled = F.adaptive_avg_pool3d(features, output_size=1).flatten(1)

        # 全 padding clip 即使经过带 bias 的卷积也必须输出严格的零。
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
        mask = valid_mask.to(device=states.device, dtype=torch.bool)
        values = torch.where(mask.unsqueeze(-1), states, torch.zeros_like(states))

        velocity = torch.zeros_like(values)
        if states.shape[1] > 1:
            pair_valid = mask[:, 1:] & mask[:, :-1]
            delta_t = timestamps[:, 1:] - timestamps[:, :-1]
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

        first_position = mask.long().argmax(dim=1)
        last_position = states.shape[1] - 1 - torch.flip(mask, dims=(1,)).long().argmax(dim=1)
        rows = torch.arange(states.shape[0], device=states.device)
        first_hidden = frame_hidden[rows, first_position]
        last_hidden = frame_hidden[rows, last_position]

        clip_hidden = self.clip_projection(
            torch.cat([masked_mean, last_hidden, last_hidden - first_hidden], dim=-1)
        )
        clip_valid = valid_count.squeeze(1) > 0
        return torch.where(clip_valid[:, None], clip_hidden, torch.zeros_like(clip_hidden))


class _LearnedTaskQueries(nn.Module):
    """保存并展开 Global Encoder 的可学习 Task Queries。

    单独封装成模块后，PEFT 可以完整训练和保存这些新增参数，而不必把冻结的
    S3D 视频骨干一并放入 ``modules_to_save``。
    """

    def __init__(self, num_queries: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_queries, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, reference: Tensor) -> Tensor:
        """按参考 clip Tensor 的 batch、device 和 dtype 展开 queries。"""
        return self.weight.to(device=reference.device, dtype=reference.dtype).expand(
            reference.shape[0], -1, -1
        )


class GlobalDemoEncoder(nn.Module):
    """RGB+State Full Demo -> Global Task Tokens。

    ``forward`` 只消费冻结 S3D 的 clip feature；raw video 必须先显式调用
    :meth:`encode_video_clips`。因此训练不会误把完整 RGB Demo 搬到 GPU。

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
    ) -> None:
        super().__init__()
        self.config = config or GlobalEncoderConfig()

        if video_backbone is None:
            self.video_backbone = S3DVideoBackbone(self.config)
            inferred_video_dim = self.video_backbone.output_dim
        else:
            self.video_backbone = video_backbone
            inferred_video_dim = getattr(video_backbone, "output_dim", None)
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
            dim_feedforward=int(self.config.temporal_hidden_size * self.config.temporal_mlp_ratio),
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
        self.task_query_bank = _LearnedTaskQueries(
            self.config.num_global_tokens,
            self.config.temporal_hidden_size,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.temporal_hidden_size, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim),
        )
        self._set_video_backbone_trainability()

    @property
    def task_queries(self) -> nn.Parameter:
        """兼容原接口，返回实际参与计算的 Task Query 参数。"""
        return self.task_query_bank.weight

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
        video_features: Tensor,
        states: Tensor,
        timestamps: Tensor,
        valid_mask: Tensor,
    ) -> None:
        """检查冻结 S3D feature 与仍需训练的 State 输入形状。"""
        if states.ndim != 4 or states.shape[-1] != self.config.state_dim:
            raise ValueError(f"Global states 必须为浮点 (B,K,L,{self.config.state_dim}) Tensor。")
        if not states.is_floating_point():
            raise ValueError("Global states 必须是浮点 Tensor。")
        if timestamps.shape != states.shape[:3] or valid_mask.shape != states.shape[:3]:
            raise ValueError("timestamps/valid_mask 必须与 states 的 (B,K,L) 一致。")
        if not all(states.shape[:3]):
            raise ValueError("Global Demo 的 batch、clip 数和每 clip 帧数都必须大于 0。")

        if (
            video_features.shape != (*states.shape[:2], self.video_feature_dim)
            or not video_features.is_floating_point()
        ):
            raise ValueError(
                f"缓存的 Global video_features 必须为浮点 (B,K,{self.video_feature_dim}) Tensor。"
            )

        if torch.any(~valid_mask.flatten(1).any(dim=1)):
            raise ValueError("每条 Demo 至少需要一帧有效 RGB+State Observation。")

    def encode_video_clips(self, video: Tensor, valid_mask: Tensor) -> Tensor:
        """只执行冻结/可训练视频骨干，返回可离线缓存的 ``(B,K,D_v)``。

        磁盘缓存必须停在这里：后续 State Encoder、RGB/State fusion、Temporal
        Aggregator 和 Task Queries 均仍属于训练图。
        """
        if video.ndim != 6 or video.shape[3] != 3 or not video.is_floating_point():
            raise ValueError("Global video 必须是浮点 (B,K,L,3,H,W) Tensor。")
        if valid_mask.shape != video.shape[:3]:
            raise ValueError("Global video valid_mask 必须为 (B,K,L)。")

        batch_size, num_clips, frames_per_clip = video.shape[:3]
        flat_video = video.reshape(batch_size * num_clips, frames_per_clip, *video.shape[3:])
        flat_mask = valid_mask.reshape(batch_size * num_clips, frames_per_clip)
        backbone_context = torch.no_grad() if self.config.freeze_video_backbone else nullcontext()
        with backbone_context:
            features = self.video_backbone(flat_video, flat_mask)
        if features.shape != (batch_size * num_clips, self.video_feature_dim):
            raise RuntimeError(
                "video_backbone 必须输出 "
                f"({batch_size * num_clips},{self.video_feature_dim})，"
                f"实际为 {tuple(features.shape)}。"
            )
        return features.reshape(batch_size, num_clips, self.video_feature_dim)

    def encode_video_clips_batched(
        self,
        video: Tensor,
        valid_mask: Tensor,
        *,
        encode_batch_size: int | None = None,
    ) -> Tensor:
        """逐批上传并编码 clip，返回与一次性编码相同的 ``(B,K,D_v)``。

        该接口用于 ``set_demo`` 和冻结 S3D 的离线缓存。输入完整视频可以
        留在 CPU；循环内只把少量 clip 搬到视频骨干所在设备。这样不会改变
        Global Encoder 的结果或训练边界，只降低 Demo 注册时的显存峰值。
        """
        if video.ndim != 6 or video.shape[3] != 3 or not video.is_floating_point():
            raise ValueError("Global video 必须是浮点 (B,K,L,3,H,W) Tensor。")
        if valid_mask.shape != video.shape[:3]:
            raise ValueError("Global video valid_mask 必须为 (B,K,L)。")

        batch_size, num_clips, frames_per_clip = video.shape[:3]
        if batch_size < 1 or num_clips < 1 or frames_per_clip < 1:
            raise ValueError("Global video 的 batch、clip 数和每 clip 帧数都必须大于 0。")
        chunk_size = self.config.clip_encode_batch_size if encode_batch_size is None else encode_batch_size
        if chunk_size < 1:
            raise ValueError("encode_batch_size 必须大于 0。")

        backbone_tensor = next(self.video_backbone.parameters(), None)
        if backbone_tensor is None:
            backbone_tensor = next(self.video_backbone.buffers(), None)
        backbone_device = video.device if backbone_tensor is None else backbone_tensor.device

        flat_video = video.reshape(batch_size * num_clips, frames_per_clip, *video.shape[3:])
        flat_mask = valid_mask.reshape(batch_size * num_clips, frames_per_clip)
        feature_batches: list[Tensor] = []
        for start in range(0, len(flat_video), chunk_size):
            end = min(start + chunk_size, len(flat_video))
            features = self.encode_video_clips(
                flat_video[start:end].unsqueeze(0).to(backbone_device),
                flat_mask[start:end].unsqueeze(0).to(backbone_device),
            )
            feature_batches.append(features.squeeze(0))
        return torch.cat(feature_batches, dim=0).reshape(
            batch_size,
            num_clips,
            self.video_feature_dim,
        )

    def _compute_clip_phase(self, timestamps: Tensor, valid_mask: Tensor) -> Tensor:
        """计算每个 clip 中心在完整 Demo 内的连续 phase。"""
        mask = valid_mask.to(dtype=torch.bool)
        safe_times = torch.where(mask, timestamps, torch.zeros_like(timestamps))
        clip_count = mask.sum(dim=-1)
        clip_center = safe_times.sum(dim=-1) / clip_count.clamp_min(1).to(safe_times.dtype)

        flat_times = timestamps.flatten(1)
        flat_mask = mask.flatten(1)
        start_time = torch.where(flat_mask, flat_times, torch.inf).amin(dim=1)
        end_time = torch.where(flat_mask, flat_times, -torch.inf).amax(dim=1)
        duration = (end_time - start_time).clamp_min(self.config.eps)
        phase = (clip_center - start_time[:, None]) / duration[:, None]
        phase = phase.clamp(0.0, 1.0)
        return torch.where(clip_count > 0, phase, torch.zeros_like(phase))

    def forward(
        self,
        video_features: Tensor,
        states: Tensor,
        timestamps: Tensor,
        valid_mask: Tensor,
    ) -> GlobalEncoderOutput:
        """从冻结 S3D feature 编码完整 Demo，返回可训练的 Global Tokens。"""
        valid_mask = valid_mask.to(device=states.device, dtype=torch.bool)
        timestamps = timestamps.to(device=states.device, dtype=torch.float64)
        self._validate_inputs(video_features, states, timestamps, valid_mask)

        batch_size, num_clips, frames_per_clip = states.shape[:3]
        flat_states = states.reshape(batch_size * num_clips, frames_per_clip, states.shape[-1])
        flat_times = timestamps.reshape(batch_size * num_clips, frames_per_clip)
        flat_mask = valid_mask.reshape(batch_size * num_clips, frames_per_clip)

        state_features = self.state_encoder(flat_states, flat_times, flat_mask)
        state_features = state_features.reshape(batch_size, num_clips, -1)

        valid_fraction = valid_mask.float().mean(dim=-1)
        clip_mask = valid_fraction >= self.config.min_valid_frame_fraction
        if torch.any(clip_mask.sum(dim=1) == 0):
            raise ValueError("某条 Demo 没有 clip 达到 min_valid_frame_fraction，无法构建 Global Tokens。")

        fusion_dtype = self.rgb_state_fusion[0].weight.dtype
        fused = self.rgb_state_fusion(
            torch.cat(
                [
                    video_features.to(dtype=fusion_dtype),
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

        task_queries = self.task_query_bank(clip_input)
        temporal_input = torch.cat([task_queries, clip_input], dim=1)
        global_mask = torch.ones(
            batch_size,
            self.config.num_global_tokens,
            dtype=torch.bool,
            device=states.device,
        )
        temporal_valid = torch.cat([global_mask, clip_mask], dim=1)

        # TransformerEncoder 的 padding mask 语义与项目其他 valid mask 相反：
        # True 表示忽略该 Key，因此这里取反。
        temporal_output = self.temporal_aggregator(
            temporal_input,
            src_key_padding_mask=~temporal_valid,
        )
        task_output = temporal_output[:, : self.config.num_global_tokens]
        global_tokens = self.output_projection(task_output)
        return GlobalEncoderOutput(
            global_tokens=global_tokens,
            global_mask=global_mask,
        )
