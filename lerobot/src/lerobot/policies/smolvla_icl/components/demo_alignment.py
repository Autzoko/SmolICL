"""SmolVLA-ICL 的 Demo–Observation 阶段对齐。

数据边界在本模块中是明确且不对称的：

- Demo 在 rollout 开始前完整编码，Matcher 可以检索整条 Demo，并在匹配后
  读取锚点前后的 Local Demo；
- Query 由 :class:`ObservationHistoryBuffer` 逐帧追加。任意一次匹配只能使用
  当前帧以及此前已经到达的帧，不能接收或读取未来 Observation。

视觉侧使用一份独立冻结的 SigLIP+connector snapshot。训练期间它只为
DTW 生成稳定特征，不向 Demo Expert 提供 Local tokens，也不接收 Action Loss。
本模块假设 Demo 与 Query 具有相同的单调阶段拓扑，只处理速度、停顿和
小范围 layout/轨迹差异；不处理可选子动作、反向运动或不同路径拓扑。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from ..configuration_smolvla_icl import DemoAlignmentConfig
from ..data.state import DemoStateNormalizer, StateNormalizationSignature

__all__ = [
    "AlignmentChunkEmbedding",
    "AlignmentResult",
    "DemoAlignmentConfig",
    "DemoEmbeddingCache",
    "LocalDemoWindow",
    "ObservationHistoryBuffer",
    "OnlineDTWMatcher",
    "SmolVLASigLIPHandle",
    "extract_matching_state_features",
    "extract_state_features",
    "load_smolvla_siglip",
    "pool_visual_tokens",
    "reuse_smolvla_siglip",
]


@dataclass(frozen=True, slots=True)
class AlignmentChunkEmbedding:
    """Matcher 使用的单个滚动窗口特征。"""

    visual: Tensor
    state: Tensor
    anchor_index: int
    timestamp: float
    valid: bool
    valid_fraction: float
    state_normalization_signature: StateNormalizationSignature
    observation_id: int | str | None = None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """一次在线 DTW 更新返回的 Demo 锚点。

    ``confidence`` 只表示当前可达搜索带内的相对集中程度，不是绝对成功
    概率。例如 Demo 末端只剩一个候选时，它会自然等于 1。
    """

    demo_chunk_index: int
    demo_observation_index: int
    demo_timestamp: float
    phase: float
    confidence: float
    local_cost: float
    accumulated_cost: float
    observation_id: int | str | None
    observation_timestamp: float


@dataclass(frozen=True, slots=True)
class LocalDemoWindow:
    """Matcher 围绕锚点截取的 State/时间窗口，不包含模型视觉特征。"""

    state_features: Tensor
    relative_time_s: Tensor
    relative_position: Tensor
    phase: Tensor
    valid_mask: Tensor
    source_indices: Tensor
    anchor_position: int

    def to(self, device: torch.device | str) -> Self:
        """返回所有 Tensor 已移动到目标设备的新对象。"""
        target = torch.device(device)
        return type(self)(
            state_features=self.state_features.to(target),
            relative_time_s=self.relative_time_s.to(target),
            relative_position=self.relative_position.to(target),
            phase=self.phase.to(target),
            valid_mask=self.valid_mask.to(target),
            source_indices=self.source_indices.to(target),
            anchor_position=self.anchor_position,
        )


def pool_visual_tokens(
    visual_tokens: Tensor,
    token_mask: Tensor | None = None,
    *,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """把 connector 的空间 tokens 池化成统一的单帧视觉特征。"""
    if visual_tokens.ndim not in (2, 3) or not visual_tokens.is_floating_point():
        raise ValueError("visual_tokens 必须是浮点 (P,D) 或 (B,P,D) Tensor。")
    if visual_tokens.shape[-2] == 0 or visual_tokens.shape[-1] == 0:
        raise ValueError("visual_tokens 的 token 数和特征维度都必须大于 0。")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps 必须是有限正数。")

    single_frame = visual_tokens.ndim == 2
    tokens = visual_tokens.unsqueeze(0) if single_frame else visual_tokens
    tokens = tokens.float()

    if token_mask is None:
        pooled = tokens.mean(dim=1)
    else:
        mask = token_mask.unsqueeze(0) if token_mask.ndim == 1 else token_mask
        if mask.shape != tokens.shape[:2]:
            raise ValueError("token_mask 必须与 visual_tokens 的前两个维度一致。")
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        count = mask.sum(dim=1, keepdim=True)
        if torch.any(count == 0):
            raise ValueError("每帧至少需要一个有效视觉 token。")
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / count.to(tokens.dtype)

    if normalize:
        pooled = F.normalize(pooled, dim=-1, eps=eps)
    return pooled[0] if single_frame else pooled


def extract_state_features(
    states: Tensor,
    timestamps: Tensor,
    *,
    valid_mask: Tensor | None = None,
    previous_state: Tensor | None = None,
    previous_timestamp: float | Tensor | None = None,
    previous_valid: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """从归一化 State 构建 ``[state, dstate/dt]`` 逐帧特征。

    ``previous_state`` 只用于滚动窗口首帧的速度，且必须来自该窗口之前，
    所以该函数不会引入未来信息。
    """
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0 or not states.is_floating_point():
        raise ValueError("states 必须是非空浮点 (T,D) Tensor。")
    if timestamps.ndim != 1 or timestamps.shape[0] != states.shape[0]:
        raise ValueError("timestamps 必须是与 states 对齐的 (T,) Tensor。")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps 必须是有限正数。")

    values = states.float()
    times = timestamps.to(device=values.device, dtype=torch.float64)
    if torch.any(~torch.isfinite(times)):
        raise ValueError("timestamps 必须只包含有限值。")
    if times.numel() > 1 and torch.any(times[1:] <= times[:-1]):
        raise ValueError("timestamps 必须严格递增。")

    mask = (
        torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
        if valid_mask is None
        else valid_mask.to(device=values.device, dtype=torch.bool)
    )
    if mask.shape != times.shape:
        raise ValueError("valid_mask 必须是与 states 对齐的 (T,) Tensor。")
    if torch.any(~torch.isfinite(values[mask])):
        raise ValueError("有效 State 帧必须只包含有限值。")

    # ``NaN * 0`` 仍然是 NaN，因此无效帧必须用 where 显式置零，
    # 不能在计算完特征后再乘 mask。
    values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))

    velocity = torch.zeros_like(values)
    if len(values) > 1:
        delta_t = (times[1:] - times[:-1]).clamp_min(eps).to(values.dtype)
        pair_valid = mask[1:] & mask[:-1]
        velocity[1:] = (values[1:] - values[:-1]) / delta_t.unsqueeze(-1)
        velocity[1:] *= pair_valid.unsqueeze(-1)

    if previous_state is not None:
        if previous_timestamp is None:
            raise ValueError("previous_state 与 previous_timestamp 必须同时提供。")
        previous = previous_state.to(device=values.device, dtype=values.dtype)
        previous_time = torch.as_tensor(previous_timestamp, device=values.device, dtype=torch.float64)
        if previous.shape != values[0].shape or previous_time.numel() != 1:
            raise ValueError("窗口前一帧的 State 或 timestamp 形状不正确。")
        if not bool(torch.isfinite(previous_time)):
            raise ValueError("previous_timestamp 必须是有限值。")
        if float(previous_time.item()) >= float(times[0].item()):
            raise ValueError("previous_timestamp 必须早于当前窗口首帧。")
        if previous_valid and bool(mask[0]):
            if torch.any(~torch.isfinite(previous)):
                raise ValueError("有效 previous_state 必须只包含有限值。")
            delta_t = (times[0] - previous_time).clamp_min(eps).to(values.dtype)
            velocity[0] = (values[0] - previous) / delta_t
    elif previous_timestamp is not None:
        raise ValueError("previous_state 与 previous_timestamp 必须同时提供。")

    return torch.cat([values, velocity], dim=-1)


def extract_matching_state_features(
    states: Tensor,
    timestamps: Tensor,
    *,
    excluded_indices: tuple[int, ...] = (-1,),
    velocity_scale: tuple[float, ...] | Tensor | None = None,
    valid_mask: Tensor | None = None,
    previous_state: Tensor | None = None,
    previous_timestamp: float | Tensor | None = None,
    previous_valid: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """构建只供 Stage Match 使用的因果运动特征。

    Matcher 不直接使用绝对 State，因为不同物体 layout 会让同一任务阶段对应
    不同的机械臂姿态。这里仅使用 ``s_t - s_(t-1)`` 得到的已观测速度，再
    压缩成 ``log(1 + speed)``。整体运动强度用于区分静止与运动阶段，同时不
    要求两个 layout 下的关节位置或关节运动方向相同。

    ``excluded_indices`` 默认去掉最后一维夹爪 State。``velocity_scale`` 是
    排除这些维度后的固定关节速度尺度，Demo 和 Query 必须使用同一份。函数
    只读取当前帧、历史帧以及显式传入的窗口前一帧，不接收 action 或预测
    State，因此不会通过 SmolVLA 尚未执行的 action chunk 泄露未来信息。
    """
    full_features = extract_state_features(
        states,
        timestamps,
        valid_mask=valid_mask,
        previous_state=previous_state,
        previous_timestamp=previous_timestamp,
        previous_valid=previous_valid,
        eps=eps,
    )
    state_dim = states.shape[-1]

    resolved_excluded: set[int] = set()
    for index in excluded_indices:
        resolved = index + state_dim if index < 0 else index
        if not 0 <= resolved < state_dim:
            raise ValueError(f"State 排除索引 {index} 超出有效范围 [-{state_dim}, {state_dim - 1}]。")
        resolved_excluded.add(resolved)

    kept_indices = [index for index in range(state_dim) if index not in resolved_excluded]
    if not kept_indices:
        raise ValueError("排除指定 State 维度后没有可用于匹配的运动特征。")

    # ``extract_state_features`` 的后半部分是严格后向差分速度。先剔除夹爪等
    # 不允许参与对齐的维度，再计算模长，保证夹爪事件连 speed 都不会影响。
    velocity = full_features[:, state_dim:][:, kept_indices]
    if velocity_scale is not None:
        scale = torch.as_tensor(velocity_scale, device=velocity.device, dtype=velocity.dtype)
        if scale.ndim != 1 or scale.shape[0] != velocity.shape[1]:
            raise ValueError("velocity_scale 必须是一维 Tensor，长度等于排除指定维度后的 State 维数。")
        if torch.any(~torch.isfinite(scale)) or torch.any(scale <= 0):
            raise ValueError("velocity_scale 必须只包含有限正数。")
        velocity = velocity / scale
    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    return torch.log1p(speed)


def _build_causal_chunk_embeddings(
    frame_features: Tensor,
    valid_mask: Tensor,
    config: DemoAlignmentConfig,
    *,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """为每个时刻构建只包含该时刻及其历史的窗口特征。

    Demo 虽然可完整访问，但每个 Demo 锚点也采用相同的因果描述符，保证
    Query 和 Demo 的比较语义一致。完整 Demo 的未来仍可供其他锚点以及
    匹配后的 Local Demo 使用。
    """
    if frame_features.ndim != 2 or frame_features.shape[0] == 0:
        raise ValueError("frame_features 必须是非空 (T,D) Tensor。")
    if valid_mask.shape != frame_features.shape[:1]:
        raise ValueError("valid_mask 必须与 frame_features 的时间维一致。")

    features = frame_features.float()
    mask = valid_mask.to(device=features.device, dtype=torch.bool)
    window_size = config.window_size

    # 只在左侧补零。输出第 t 行的右边界永远是 t，因此不会看到 t+1。
    padded_features = F.pad(features, (0, 0, window_size - 1, 0))
    padded_mask = F.pad(mask, (window_size - 1, 0), value=False)
    windows = padded_features.unfold(0, window_size, 1).transpose(1, 2)
    window_masks = padded_mask.unfold(0, window_size, 1)

    valid_count = window_masks.sum(dim=1)
    safe_count = valid_count.clamp_min(1).to(features.dtype)
    mean_feature = (windows * window_masks.unsqueeze(-1)).sum(dim=1) / safe_count.unsqueeze(-1)

    # 每个窗口拼接均值、最新有效帧和窗口位移，
    # 兼顾外观、当前状态和运动趋势。
    first_position = window_masks.long().argmax(dim=1)
    last_position = window_size - 1 - torch.flip(window_masks, dims=(1,)).long().argmax(dim=1)
    rows = torch.arange(features.shape[0], device=features.device)
    first_feature = windows[rows, first_position]
    last_feature = windows[rows, last_position]
    chunks = torch.cat([mean_feature, last_feature, last_feature - first_feature], dim=-1)
    chunks *= (valid_count > 0).unsqueeze(-1)
    if normalize:
        chunks = F.normalize(chunks, dim=-1, eps=config.eps)

    valid_fraction = valid_count.float() / window_size
    minimum_count = max(1, math.ceil(window_size * config.min_valid_fraction))
    # 锚点自身必须有效，不能用历史帧替代一个缺失的当前 Observation。
    chunk_valid = (valid_count >= minimum_count) & window_masks[:, -1]
    return chunks, chunk_valid, valid_fraction


def _select_alignment_indices(timestamps: Tensor, alignment_hz: float, eps: float) -> Tensor:
    """从完整 Demo 时间轴选择最接近 Matcher 更新频率的因果锚点。

    输入 Demo 可以保留控制频率，Local Demo 因而仍能按 action horizon 读取
    稠密帧。这里只为 DTW 建立一个较稀疏的索引视图，不复制原始缓存。
    """
    if timestamps.ndim != 1 or timestamps.numel() == 0:
        raise ValueError("timestamps 必须是非空一维 Tensor。")

    period_s = 1.0 / alignment_hz
    next_target = float(timestamps[0])
    selected: list[int] = []
    for index, timestamp in enumerate(timestamps):
        current = float(timestamp)
        if current + eps < next_target:
            continue
        selected.append(index)
        # target 沿固定时间网格前进，避免使用当前帧时间重新起算产生累计漂移。
        while next_target <= current + eps:
            next_target += period_s
    return torch.tensor(selected, dtype=torch.long, device=timestamps.device)


def _first_parameter(module: nn.Module) -> nn.Parameter:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("SigLIP 模块没有参数，无法确定 device 和 dtype。")
    return parameter


def _resolve_vlm_with_expert(smolvla: Any) -> nn.Module:
    """兼容 Policy、VLAFlowMatching 和 SmolVLMWithExpertModel 三个层级。"""
    policy_model = getattr(smolvla, "model", None)
    if getattr(policy_model, "vlm_with_expert", None) is not None:
        return policy_model.vlm_with_expert
    if getattr(smolvla, "vlm_with_expert", None) is not None:
        return smolvla.vlm_with_expert
    if callable(getattr(smolvla, "get_vlm_model", None)):
        return smolvla
    raise TypeError("无法从传入对象中找到 SmolVLA 的 vlm_with_expert。")


@dataclass
class SmolVLASigLIPHandle:
    """对 SmolVLA 内部 SigLIP 与 connector 的轻量引用。"""

    source: Any
    vlm_with_expert: nn.Module
    vision_model: nn.Module
    connector: nn.Module
    frozen: bool = False

    @property
    def device(self) -> torch.device:
        return _first_parameter(self.vision_model).device

    @property
    def dtype(self) -> torch.dtype:
        return _first_parameter(self.vision_model).dtype

    def freeze(self) -> None:
        """冻结视觉侧，保证提前计算的 Demo cache 不会失效。"""
        for module in (self.vision_model, self.connector):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.frozen = True

    def encode_siglip_patch_tokens(self, preprocessed_images: Tensor) -> Tensor:
        """提取 connector 之前的 SigLIP patch tokens。"""
        self._validate_images(preprocessed_images)
        if self.frozen:
            self.vision_model.eval()
        images = preprocessed_images.to(device=self.device, dtype=self.dtype)
        return self.vision_model(pixel_values=images, patch_attention_mask=None).last_hidden_state

    def encode_visual_tokens(self, preprocessed_images: Tensor) -> Tensor:
        """提取 connector 输出 tokens，供 Matcher 池化和后续 Demo Expert 使用。"""
        if self.frozen:
            self.connector.eval()
        return self.connector(self.encode_siglip_patch_tokens(preprocessed_images))

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if (
            images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1] != 3
            or images.shape[2] == 0
            or images.shape[3] == 0
            or not images.is_floating_point()
        ):
            raise ValueError("SigLIP 输入必须是浮点 (B,3,H,W) Tensor。")


def reuse_smolvla_siglip(smolvla: Any, *, freeze: bool = True) -> SmolVLASigLIPHandle:
    """引用已有 SmolVLA 的视觉模块，不复制权重。"""
    vlm_with_expert = _resolve_vlm_with_expert(smolvla)
    vlm_model = vlm_with_expert.get_vlm_model()
    vision_model = getattr(vlm_model, "vision_model", None)
    connector = getattr(vlm_model, "connector", None)
    if not isinstance(vision_model, nn.Module) or not isinstance(connector, nn.Module):
        raise TypeError("SmolVLM 缺少 vision_model 或 connector。")

    handle = SmolVLASigLIPHandle(
        source=smolvla,
        vlm_with_expert=vlm_with_expert,
        vision_model=vision_model,
        connector=connector,
    )
    if freeze:
        handle.freeze()
    return handle


def load_smolvla_siglip(
    pretrained_name_or_path: str | Path = "lerobot/smolvla_base",
    *,
    device: torch.device | str | None = None,
    freeze: bool = True,
    **from_pretrained_kwargs: Any,
) -> SmolVLASigLIPHandle:
    """主体尚未建立时临时加载 SmolVLA checkpoint，并返回视觉句柄。"""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path, **from_pretrained_kwargs)
    if device is not None:
        policy.to(device)
    policy.eval()
    return reuse_smolvla_siglip(policy, freeze=freeze)


@dataclass
class DemoEmbeddingCache:
    """冻结 ``E_match`` 的 DTW 特征及 Local State/时间索引缓存。"""

    config: DemoAlignmentConfig
    state_normalizer: DemoStateNormalizer
    timestamps: Tensor
    valid_mask: Tensor
    anchor_indices: Tensor
    state_frame_features: Tensor
    visual_chunk_embeddings: Tensor
    state_chunk_embeddings: Tensor
    chunk_valid_mask: Tensor
    chunk_valid_fraction: Tensor

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        siglip: SmolVLASigLIPHandle,
        preprocessed_images: Tensor,
        raw_states: Tensor,
        timestamps: Tensor,
        *,
        state_normalizer: DemoStateNormalizer,
        config: DemoAlignmentConfig | None = None,
        valid_mask: Tensor | None = None,
    ) -> Self:
        """用冻结 Matcher SigLIP 编码完整 Demo 的对齐特征。"""
        cfg = config or DemoAlignmentConfig()
        if not siglip.frozen:
            raise ValueError("Demo Matcher 只能使用冻结的 SigLIP snapshot。")
        siglip._validate_images(preprocessed_images)
        if len(raw_states) != len(preprocessed_images) or len(timestamps) != len(preprocessed_images):
            raise ValueError("Demo image/state/timestamp 的时间长度必须一致。")

        frame_valid = (
            torch.ones(len(preprocessed_images), dtype=torch.bool, device=preprocessed_images.device)
            if valid_mask is None
            else valid_mask.to(device=preprocessed_images.device, dtype=torch.bool)
        )
        if frame_valid.shape != preprocessed_images.shape[:1]:
            raise ValueError("valid_mask 必须与 Demo 图像帧数一致。")
        safe_images = torch.where(
            frame_valid[:, None, None, None],
            preprocessed_images,
            torch.zeros_like(preprocessed_images),
        )

        cache_device = torch.device(cfg.cache_device)
        frame_batches: list[Tensor] = []
        for start in range(0, len(safe_images), cfg.demo_encode_batch_size):
            tokens = siglip.encode_visual_tokens(safe_images[start : start + cfg.demo_encode_batch_size])
            # Matcher 只保存 pooled feature；模型使用的 spatial tokens
            # 由 Policy 通过独立的 E_vision 路径维护。
            frame_batches.append(
                pool_visual_tokens(
                    tokens,
                    normalize=False,
                    eps=cfg.eps,
                ).to(cache_device)
            )
        return cls.from_embeddings(
            torch.cat(frame_batches),
            raw_states,
            timestamps,
            state_normalizer=state_normalizer,
            config=cfg,
            valid_mask=frame_valid,
        )

    @classmethod
    def from_embeddings(
        cls,
        visual_frame_embeddings: Tensor,
        raw_states: Tensor,
        timestamps: Tensor,
        *,
        state_normalizer: DemoStateNormalizer,
        config: DemoAlignmentConfig | None = None,
        valid_mask: Tensor | None = None,
    ) -> Self:
        """从已经分批编码好的帧特征构建缓存。

        该入口适合长视频和测试工具：视觉编码可以流式进行，同时正式代码
        不再需要调用私有的窗口 helper。``raw_states`` 必须保持数据集真实
        State 宽度且尚未归一化；本函数使用显式传入的 normalizer 统一处理。
        """
        cfg = config or DemoAlignmentConfig()
        if visual_frame_embeddings.ndim != 2 or raw_states.ndim != 2 or timestamps.ndim != 1:
            raise ValueError("Demo visual/state/timestamp 形状必须分别为 (T,Dv)/(T,Ds)/(T,)。")
        num_frames = len(timestamps)
        if num_frames == 0 or len(visual_frame_embeddings) != num_frames or len(raw_states) != num_frames:
            raise ValueError("Demo 三种输入必须非空且时间长度一致。")
        device = torch.device(cfg.cache_device)
        times = timestamps.detach().to(device=device, dtype=torch.float64)
        visual_values = visual_frame_embeddings.detach().to(device=device, dtype=torch.float32)
        frame_valid = (
            torch.ones(num_frames, dtype=torch.bool, device=device)
            if valid_mask is None
            else valid_mask.detach().to(device=device, dtype=torch.bool)
        )
        if frame_valid.shape != times.shape:
            raise ValueError("valid_mask 必须与 Demo 帧数一致。")
        if torch.any(~torch.isfinite(visual_values[frame_valid])):
            raise ValueError("有效 Demo 帧的视觉特征必须只包含有限值。")

        # Matcher 的公开入口只接收真实维度的 raw State，并在内部统一归一化。
        # 这样已 padding 到 32 维的 State 会在这里直接失败，而 Demo/Query
        # 也不再依赖调用者分别执行同一套预处理。
        raw_state_values = raw_states.detach().to(device=device, dtype=torch.float32)
        state_values = state_normalizer.normalize(raw_state_values, valid_mask=frame_valid)

        visual_values = torch.where(
            frame_valid.unsqueeze(-1),
            visual_values,
            torch.zeros_like(visual_values),
        )
        state_features = extract_state_features(
            state_values,
            times,
            valid_mask=frame_valid,
            eps=cfg.eps,
        )

        # State/时间保留原始时间分辨率供 Local 窗口读取；DTW 视觉特征只在
        # 与 Query 重规划频率一致的锚点上建立滚动窗口，逐帧 pooled feature
        # 构建完成后即丢弃，避免它被误当成模型 E_vision 的 Local 输入。
        alignment_indices = _select_alignment_indices(times, cfg.alignment_hz, cfg.eps)
        alignment_times = times[alignment_indices]
        alignment_valid = frame_valid[alignment_indices]
        alignment_states = state_values[alignment_indices]
        alignment_visual = visual_values[alignment_indices]
        if cfg.normalize_visual_features:
            # 归一化只服务于 Matcher 距离；Local Encoder 的视觉输入由
            # Policy 持有的独立 E_vision cache 提供。
            alignment_visual = F.normalize(alignment_visual, dim=-1, eps=cfg.eps)
        matching_state_features = extract_matching_state_features(
            alignment_states,
            alignment_times,
            excluded_indices=cfg.matching_state_excluded_indices,
            velocity_scale=cfg.matching_state_velocity_scale,
            valid_mask=alignment_valid,
            eps=cfg.eps,
        )
        visual_chunks, visual_valid, visual_fraction = _build_causal_chunk_embeddings(
            alignment_visual,
            alignment_valid,
            cfg,
            normalize=cfg.normalize_visual_features,
        )
        state_chunks, state_valid, state_fraction = _build_causal_chunk_embeddings(
            matching_state_features,
            alignment_valid,
            cfg,
            normalize=False,
        )

        return cls(
            config=cfg,
            state_normalizer=state_normalizer,
            timestamps=times,
            valid_mask=frame_valid,
            anchor_indices=alignment_indices,
            state_frame_features=state_features,
            visual_chunk_embeddings=visual_chunks,
            state_chunk_embeddings=state_chunks,
            chunk_valid_mask=visual_valid & state_valid,
            chunk_valid_fraction=torch.minimum(visual_fraction, state_fraction),
        )

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)

    @property
    def num_chunks(self) -> int:
        return len(self.visual_chunk_embeddings)

    def to_serializable(self) -> dict[str, Any]:
        """导出不含模型对象的 Matcher CPU payload。"""
        tensor_names = (
            "timestamps",
            "valid_mask",
            "anchor_indices",
            "state_frame_features",
            "visual_chunk_embeddings",
            "state_chunk_embeddings",
            "chunk_valid_mask",
            "chunk_valid_fraction",
        )
        tensors = {name: getattr(self, name).detach().cpu() for name in tensor_names}
        return {
            "version": 2,
            "config": asdict(self.config),
            "state_normalizer": {
                "mean": self.state_normalizer.mean.detach().cpu(),
                "std": self.state_normalizer.std.detach().cpu(),
                "eps": self.state_normalizer.eps,
            },
            "tensors": tensors,
        }

    @classmethod
    def from_serializable(cls, payload: dict[str, Any]) -> Self:
        """从 :meth:`to_serializable` 的纯数据 payload 恢复缓存。"""
        if payload.get("version") != 2:
            raise ValueError("不支持的 DemoEmbeddingCache 磁盘版本。")
        normalizer_payload = payload["state_normalizer"]
        tensors = payload["tensors"]
        return cls(
            config=DemoAlignmentConfig(**payload["config"]),
            state_normalizer=DemoStateNormalizer(
                mean=normalizer_payload["mean"],
                std=normalizer_payload["std"],
                eps=float(normalizer_payload["eps"]),
            ),
            timestamps=tensors["timestamps"],
            valid_mask=tensors["valid_mask"],
            anchor_indices=tensors["anchor_indices"],
            state_frame_features=tensors["state_frame_features"],
            visual_chunk_embeddings=tensors["visual_chunk_embeddings"],
            state_chunk_embeddings=tensors["state_chunk_embeddings"],
            chunk_valid_mask=tensors["chunk_valid_mask"],
            chunk_valid_fraction=tensors["chunk_valid_fraction"],
        )

    def timestamps_to_phase(self, timestamps: Tensor) -> Tensor:
        """按完整 Demo 的有效时间范围把 timestamp 映射到 ``[0,1]``。"""
        valid_times = self.timestamps[self.valid_mask]
        start_time = valid_times.min()
        duration = (valid_times.max() - start_time).clamp_min(self.config.eps)
        return ((timestamps - start_time) / duration).clamp(0.0, 1.0)

    def get_chunk(self, index: int) -> AlignmentChunkEmbedding:
        """返回指定 Demo 锚点的检索特征。"""
        if not 0 <= index < self.num_chunks:
            raise IndexError("Demo Chunk 索引越界。")
        anchor = int(self.anchor_indices[index])
        return AlignmentChunkEmbedding(
            visual=self.visual_chunk_embeddings[index],
            state=self.state_chunk_embeddings[index],
            anchor_index=anchor,
            timestamp=float(self.timestamps[anchor]),
            valid=bool(self.chunk_valid_mask[index]),
            valid_fraction=float(self.chunk_valid_fraction[index]),
            state_normalization_signature=self.state_normalizer.signature,
        )

    def extract_local_window(self, alignment: AlignmentResult | int) -> LocalDemoWindow:
        """截取 anchor 周围的 State/时间窗口；模型视觉特征由 Policy 单独读取。"""
        if isinstance(alignment, AlignmentResult):
            anchor = alignment.demo_observation_index
        else:
            anchor = int(alignment)
        if not 0 <= anchor < self.num_frames:
            raise IndexError("Local Demo 锚点越界。")

        anchor_position = self.config.local_anchor_position
        relative_indices = torch.arange(
            -anchor_position,
            self.config.local_chunk_size - anchor_position,
            device=self.timestamps.device,
        )
        requested = anchor + relative_indices
        in_bounds = (requested >= 0) & (requested < self.num_frames)
        source_indices = torch.where(in_bounds, requested, -torch.ones_like(requested))
        clamped = requested.clamp(0, self.num_frames - 1)
        local_valid = in_bounds & self.valid_mask[clamped]

        def gather_with_padding(source: Tensor) -> Tensor:
            output = source.new_zeros((len(requested), *source.shape[1:]))
            output[in_bounds] = source[requested[in_bounds]]
            # Demo 内部的无效帧也统一清零，消费方只需使用同一个 mask。
            expanded_mask = local_valid.reshape(len(local_valid), *([1] * (output.ndim - 1)))
            return torch.where(expanded_mask, output, torch.zeros_like(output))

        state_features = gather_with_padding(self.state_frame_features)

        # Local Demo 使用完整 Demo 的原始时间轴，而非较稀疏的 Matcher 频率。
        # 越界 padding 没有真实 timestamp，才用相邻帧的中位时间间隔外推。
        if self.num_frames > 1:
            demo_period_s = float(torch.median(self.timestamps[1:] - self.timestamps[:-1]))
        else:
            demo_period_s = 1.0 / self.config.alignment_hz
        relative_time = relative_indices.float() * demo_period_s
        relative_time[in_bounds] = (self.timestamps[requested[in_bounds]] - self.timestamps[anchor]).to(
            relative_time.dtype
        )
        local_timestamps = self.timestamps[anchor] + relative_time.to(self.timestamps.dtype)
        local_timestamps[in_bounds] = self.timestamps[requested[in_bounds]]

        return LocalDemoWindow(
            state_features=state_features,
            relative_time_s=relative_time,
            relative_position=relative_indices.float() / self.config.local_chunk_size,
            phase=self.timestamps_to_phase(local_timestamps),
            valid_mask=local_valid,
            source_indices=source_indices,
            anchor_position=anchor_position,
        )

    def to(self, device: torch.device | str) -> Self:
        """像 ``nn.Module.to`` 一样原地移动缓存并返回自身。"""
        target = torch.device(device)
        for name in (
            "timestamps",
            "valid_mask",
            "anchor_indices",
            "state_frame_features",
            "visual_chunk_embeddings",
            "state_chunk_embeddings",
            "chunk_valid_mask",
            "chunk_valid_fraction",
        ):
            setattr(self, name, getattr(self, name).to(target))
        return self


class ObservationHistoryBuffer:
    """单环境的严格因果 Query 缓存。

    对外只有逐帧 ``append``，内部最多保留 ``window_size + 1`` 帧；多出的
    一帧仅用于计算窗口首帧速度。因此最新 Query Chunk 不可能包含未来帧。

    配合 SmolVLA action chunking 时，应在 action 队列为空、调用
    ``_get_action_chunk`` 之前追加本次真实 Observation 并执行匹配。队列内的
    预测 action 及其推演 State 都不能追加；后续控制步必须等环境返回真实
    Observation，直到下一个重规划边界再形成新的 Query。
    """

    def __init__(
        self,
        config: DemoAlignmentConfig | None = None,
        *,
        state_normalizer: DemoStateNormalizer,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or DemoAlignmentConfig()
        self.state_normalizer = state_normalizer
        self.device = torch.device(device or self.config.cache_device)
        self._max_storage = self.config.window_size + 1
        self.reset()

    def reset(self) -> None:
        """开始新 episode 时清空所有历史。"""
        self._visual: deque[Tensor] = deque(maxlen=self._max_storage)
        self._states: deque[Tensor] = deque(maxlen=self._max_storage)
        self._timestamps: deque[float] = deque(maxlen=self._max_storage)
        self._valid: deque[bool] = deque(maxlen=self._max_storage)
        self._ids: deque[int | str | None] = deque(maxlen=self._max_storage)
        self._indices: deque[int] = deque(maxlen=self._max_storage)
        self._visual_dim: int | None = None
        self._state_dim: int | None = None
        self._seen = 0

    def __len__(self) -> int:
        return min(len(self._timestamps), self.config.window_size)

    @property
    def is_ready(self) -> bool:
        """当前帧有效且当前/历史窗口已积累足够有效帧。"""
        recent = list(self._valid)[-self.config.window_size :]
        required = max(1, math.ceil(self.config.window_size * self.config.min_valid_fraction))
        return bool(recent) and recent[-1] and len(recent) >= required and sum(recent) >= required

    def append(
        self,
        *,
        raw_state: Tensor,
        timestamp: float | Tensor,
        visual_tokens: Tensor | None = None,
        visual_embedding: Tensor | None = None,
        observation_id: int | str | None = None,
        valid: bool = True,
    ) -> None:
        """追加一帧 raw State 和视觉特征，并在内部执行共享归一化。"""
        frame_valid = bool(valid)
        if (visual_tokens is None) == (visual_embedding is None):
            raise ValueError("visual_tokens 和 visual_embedding 必须且只能提供一个。")

        if visual_tokens is not None:
            if visual_tokens.ndim == 3:
                if visual_tokens.shape[0] != 1:
                    raise ValueError("History Buffer 每次只能追加一个环境的一帧。")
                visual_tokens = visual_tokens[0]
            visual = pool_visual_tokens(
                visual_tokens,
                normalize=self.config.normalize_visual_features,
                eps=self.config.eps,
            )
        else:
            assert visual_embedding is not None
            if visual_embedding.ndim == 2:
                if visual_embedding.shape[0] != 1:
                    raise ValueError("History Buffer 每次只能追加一个环境的一帧。")
                visual_embedding = visual_embedding[0]
            visual = visual_embedding
            if visual.ndim != 1 or not visual.is_floating_point():
                raise ValueError("visual_embedding 必须是单帧浮点特征。")
            if visual.numel() == 0:
                raise ValueError("visual_embedding 特征维度必须大于 0。")
            visual = visual.float()
            if self.config.normalize_visual_features:
                visual = F.normalize(visual, dim=-1, eps=self.config.eps)

        if raw_state.ndim == 2:
            if raw_state.shape[0] != 1:
                raise ValueError("History Buffer 每次只能追加一个环境的一帧。")
            raw_state = raw_state[0]
        if raw_state.ndim != 1 or not raw_state.is_floating_point():
            raise ValueError("raw_state 必须是单帧浮点 State。")
        state = self.state_normalizer.normalize(
            raw_state,
            valid_mask=torch.tensor(frame_valid, device=raw_state.device),
        )

        if self._visual_dim is None:
            self._visual_dim = int(visual.shape[0])
            self._state_dim = int(state.shape[0])
        elif visual.shape[0] != self._visual_dim or state.shape[0] != self._state_dim:
            raise ValueError("Observation 的视觉和 State 特征维度必须在同一 episode 内保持一致。")

        if frame_valid and (torch.any(~torch.isfinite(visual)) or torch.any(~torch.isfinite(state))):
            raise ValueError("有效 Observation 必须只包含有限的视觉和 State 特征。")
        if not frame_valid:
            # 显式清零避免无效帧中的 NaN 通过 ``NaN * 0`` 污染滚动窗口。
            visual = torch.zeros_like(visual)
            state = torch.zeros_like(state)

        current_time = self._scalar_timestamp(timestamp)
        if self._timestamps and current_time <= self._timestamps[-1]:
            raise ValueError("Observation timestamp 必须严格递增。")

        self._visual.append(visual.detach().to(self.device, torch.float32))
        self._states.append(state.detach().to(self.device, torch.float32))
        self._timestamps.append(current_time)
        self._valid.append(frame_valid)
        self._ids.append(observation_id)
        self._indices.append(self._seen)
        self._seen += 1

    def get_latest_chunk(self) -> AlignmentChunkEmbedding:
        """构建右边界为最新帧的 Query Chunk，不读取任何未来帧。"""
        if not self._timestamps:
            raise RuntimeError("History Buffer 为空。")

        visual_all = list(self._visual)
        states_all = list(self._states)
        times_all = list(self._timestamps)
        valid_all = list(self._valid)
        num_frames = min(len(times_all), self.config.window_size)
        start = len(times_all) - num_frames

        visual = torch.stack(visual_all[start:])
        states = torch.stack(states_all[start:])
        times = torch.tensor(times_all[start:], dtype=torch.float64, device=self.device)
        valid = torch.tensor(valid_all[start:], dtype=torch.bool, device=self.device)

        matching_state_features = extract_matching_state_features(
            states,
            times,
            excluded_indices=self.config.matching_state_excluded_indices,
            velocity_scale=self.config.matching_state_velocity_scale,
            valid_mask=valid,
            previous_state=states_all[start - 1] if start > 0 else None,
            previous_timestamp=times_all[start - 1] if start > 0 else None,
            previous_valid=valid_all[start - 1] if start > 0 else True,
            eps=self.config.eps,
        )
        visual_chunks, visual_valid, visual_fraction = _build_causal_chunk_embeddings(
            visual,
            valid,
            self.config,
            normalize=self.config.normalize_visual_features,
        )
        state_chunks, state_valid, state_fraction = _build_causal_chunk_embeddings(
            matching_state_features,
            valid,
            self.config,
            normalize=False,
        )
        return AlignmentChunkEmbedding(
            visual=visual_chunks[-1],
            state=state_chunks[-1],
            anchor_index=self._indices[-1],
            timestamp=times_all[-1],
            valid=bool(visual_valid[-1] & state_valid[-1]),
            valid_fraction=float(torch.minimum(visual_fraction[-1], state_fraction[-1])),
            state_normalization_signature=self.state_normalizer.signature,
            observation_id=self._ids[-1],
        )

    @staticmethod
    def _scalar_timestamp(timestamp: float | Tensor) -> float:
        if isinstance(timestamp, Tensor):
            if timestamp.numel() != 1:
                raise ValueError("timestamp 必须是标量。")
            timestamp = float(timestamp.detach().cpu())
        value = float(timestamp)
        if not math.isfinite(value):
            raise ValueError("timestamp 必须是有限值。")
        return value


class OnlineDTWMatcher:
    """只沿 Demo 向前搜索的轻量在线 DTW。

    Query 较慢时可重复使用当前 Demo 锚点；已提交的输出不回退。
    ``dtw_max_advance`` 约束单条 DP path 的一次转移，而不是相邻两次
    argmin 输出的硬差值；后续接入控制时可在本类之外增加提交平滑。
    """

    def __init__(
        self,
        demo_cache: DemoEmbeddingCache,
        *,
        start_index: int | None = None,
    ) -> None:
        self.demo_cache = demo_cache
        # Demo/Query 的窗口尺寸、State 尺度和视觉归一化必须一致，
        # 因此 Matcher 不再接受第二份可能冲突的配置。
        self.config = demo_cache.config
        self._previous_costs: Tensor
        self._previous_start = 0
        self._active_index = 0
        self._cost_offset = 0.0
        self._num_updates = 0
        self._last_result: AlignmentResult | None = None
        self.reset(start_index=start_index)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def last_result(self) -> AlignmentResult | None:
        return self._last_result

    def create_observation_history(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> ObservationHistoryBuffer:
        """创建与 Demo 强制共享 State 统计量的 Query 历史缓存。"""
        return ObservationHistoryBuffer(
            self.config,
            state_normalizer=self.demo_cache.state_normalizer,
            device=device,
        )

    def reset(self, *, start_index: int | None = None) -> None:
        """从第一个有效 Demo Chunk（或显式锚点）开始新 Query episode。"""
        valid_indices = torch.nonzero(self.demo_cache.chunk_valid_mask).flatten()
        if len(valid_indices) == 0:
            raise ValueError("Demo 中没有有效 Chunk。")
        start = int(valid_indices[0]) if start_index is None else int(start_index)
        if not 0 <= start < self.demo_cache.num_chunks or not bool(self.demo_cache.chunk_valid_mask[start]):
            raise ValueError("DTW start_index 无效。")

        device = self.demo_cache.visual_chunk_embeddings.device
        self._previous_start = start
        self._previous_costs = torch.zeros(1, device=device)
        self._active_index = start
        self._cost_offset = 0.0
        self._num_updates = 0
        self._last_result = None

    @torch.no_grad()
    def update(self, query: AlignmentChunkEmbedding) -> AlignmentResult:
        """用最新因果 Query Chunk 推进一次 DTW。"""
        if not query.valid:
            raise ValueError("Query Chunk 尚未积累足够有效帧。")
        if (
            not self.config.rgb_only
            and query.state_normalization_signature != self.demo_cache.state_normalizer.signature
        ):
            raise ValueError("Query 与 Demo 必须使用同一套 State 归一化统计量。")

        search_start = self._active_index
        search_end = min(
            self.demo_cache.num_chunks,
            search_start + self.config.dtw_forward_window + 1,
        )
        local_costs = self._local_costs(query, search_start, search_end)

        row: list[Tensor] = []
        for offset, demo_index in enumerate(range(search_start, search_end)):
            stay_penalty = 0.0 if self._num_updates == 0 else self.config.dtw_stay_penalty
            transitions = [self._previous_cost(demo_index) + stay_penalty]
            for advance in range(1, self.config.dtw_max_advance + 1):
                previous_index = demo_index - advance
                if previous_index < 0:
                    break
                skip_penalty = self.config.dtw_skip_penalty * max(0, advance - 1)
                transitions.append(self._previous_cost(previous_index) + skip_penalty)
            row.append(local_costs[offset] + torch.stack(transitions).min())

        band_costs = torch.stack(row)
        finite = torch.isfinite(band_costs)
        if not torch.any(finite):
            raise RuntimeError("DTW 搜索带内没有可达的有效 Demo Chunk。")
        best_offset = int(torch.argmin(band_costs))
        best_index = search_start + best_offset
        row_minimum = band_costs[best_offset]
        accumulated_cost = self._cost_offset + float(row_minimum)

        normalized = torch.where(finite, band_costs - row_minimum, band_costs)
        confidence = float(torch.softmax(-normalized[finite] / self.config.dtw_temperature, dim=0).max())
        self._previous_start = search_start
        self._previous_costs = normalized
        self._active_index = best_index
        self._cost_offset = accumulated_cost
        self._num_updates += 1

        demo_index = int(self.demo_cache.anchor_indices[best_index])
        result = AlignmentResult(
            demo_chunk_index=best_index,
            demo_observation_index=demo_index,
            demo_timestamp=float(self.demo_cache.timestamps[demo_index]),
            phase=float(self.demo_cache.timestamps_to_phase(self.demo_cache.timestamps[demo_index])),
            confidence=confidence,
            local_cost=float(local_costs[best_offset]),
            accumulated_cost=accumulated_cost,
            observation_id=query.observation_id,
            observation_timestamp=query.timestamp,
        )
        self._last_result = result
        return result

    @torch.no_grad()
    def update_and_extract(
        self,
        query: AlignmentChunkEmbedding,
    ) -> tuple[AlignmentResult, LocalDemoWindow]:
        result = self.update(query)
        return result, self.demo_cache.extract_local_window(result)

    def _previous_cost(self, demo_index: int) -> Tensor:
        offset = demo_index - self._previous_start
        if 0 <= offset < len(self._previous_costs):
            return self._previous_costs[offset]
        return self._previous_costs.new_tensor(torch.inf)

    def _local_costs(
        self,
        query: AlignmentChunkEmbedding,
        search_start: int,
        search_end: int,
    ) -> Tensor:
        """计算当前 Demo 前向带的匹配代价。"""
        demo_visual = self.demo_cache.visual_chunk_embeddings[search_start:search_end].float()
        query_visual = query.visual.detach().to(demo_visual).unsqueeze(0)
        if query_visual.shape[1:] != demo_visual.shape[1:]:
            raise ValueError("Query 与 Demo 的视觉 Chunk 特征维度不一致。")

        visual_distance = 1 - F.cosine_similarity(
            demo_visual,
            query_visual,
            dim=-1,
            eps=self.config.eps,
        )

        if self.config.rgb_only:
            # RGB-only 消融必须在这里直接返回视觉距离，不能先计算 State
            # 再乘以 0；这样可以保证 State 的取值完全不影响匹配结果。
            cost = self.config.vision_distance_weight * visual_distance
        else:
            demo_state = self.demo_cache.state_chunk_embeddings[search_start:search_end].float()
            query_state = query.state.detach().to(demo_state).unsqueeze(0)
            if query_state.shape[1:] != demo_state.shape[1:]:
                raise ValueError("Query 与 Demo 的 State Chunk 特征维度不一致。")
            state_distance = F.smooth_l1_loss(
                demo_state,
                query_state.expand_as(demo_state),
                reduction="none",
            ).mean(dim=-1)
            cost = (
                self.config.vision_distance_weight * visual_distance
                + self.config.state_distance_weight * state_distance
            )

        valid = self.demo_cache.chunk_valid_mask[search_start:search_end].to(cost.device)
        return cost.masked_fill(~valid, torch.inf)
