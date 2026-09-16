"""SmolVLA-ICL 的纯配置定义。

本文件只放置可序列化的 dataclass，不导入视频骨干、Transformer
或 Stage Matcher 实现。这样配置可以被 LeRobot/draccus 安全保存，
也不会因为可选模型依赖导致导入失败。
"""

import math
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class GlobalEncoderConfig:
    """Global Demo Encoder 的结构、输入与训练配置。

    Global Encoder 把一条完整 Demo 编码为固定数量的 Global Task
    Tokens。输出宽度直接对齐 Demo Expert，因此默认为 SmolVLA
    VLM hidden size 960 乘以 0.75，即 720。

    ``backbone_name`` 暂时只支持 ``"s3d"``。后续的 MoViNet、
    VideoMAE 和 Swin3D 会通过相同的 backbone adapter 接口接入，
    不改变上层 Temporal Aggregator 和 Demo Expert 的数据契约。
    """

    # 视频骨干。首版只实现 TorchVision S3D。
    backbone_name: str = "s3d"
    pretrained_backbone: bool = True
    freeze_video_backbone: bool = True

    # S3D 接收的单帧空间尺寸。预训练权重会使用自带的
    # TorchVision transforms；无预训练权重时使用这里的尺寸。
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
    image_std: tuple[float, float, float] = (0.22803, 0.22145, 0.216989)

    # Processor 将完整 Demo 切分为固定帧数的 clip。首版使用
    # 16 帧无重叠分段；末尾不足一个 clip 的部分用 valid mask 补齐。
    clip_length: int = 16
    clip_stride: int = 16

    # Global State 路径的输入维度对齐 SmolVLA 的 max_state_dim。
    state_dim: int = 32
    state_feature_dim: int = 128

    # RGB/State 融合后交给时序 Transformer 的宽度。
    temporal_hidden_size: int = 512
    temporal_num_layers: int = 2
    temporal_num_heads: int = 8
    temporal_mlp_ratio: float = 4.0
    temporal_dropout: float = 0.0

    # 固定输出的 Task Query 数量和 Demo Expert hidden size。
    num_global_tokens: int = 8
    output_dim: int = 720

    # 一个 clip 至少需要达到该有效帧比例才会交给 Temporal
    # Aggregator。无效帧会在视频骨干和 State Encoder 前清零。
    min_valid_frame_fraction: float = 0.5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """验证所有会影响张量形状或数值稳定性的配置。"""
        if self.backbone_name != "s3d":
            raise ValueError("当前 Global Encoder 只支持 backbone_name='s3d'。")
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size 必须是两个正整数。")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean 和 image_std 必须各包含 3 个通道值。")
        if any(not math.isfinite(value) for value in (*self.image_mean, *self.image_std)):
            raise ValueError("图像均值和标准差必须是有限值。")
        if any(value <= 0 for value in self.image_std):
            raise ValueError("image_std 的每一维都必须为正数。")
        if self.clip_length < 14:
            raise ValueError("S3D 的 clip_length 至少为 14。")
        if not 1 <= self.clip_stride <= self.clip_length:
            raise ValueError("clip_stride 必须位于 [1, clip_length]，以保证完整覆盖 Demo。")

        positive_ints = {
            "state_dim": self.state_dim,
            "state_feature_dim": self.state_feature_dim,
            "temporal_hidden_size": self.temporal_hidden_size,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_num_heads": self.temporal_num_heads,
            "num_global_tokens": self.num_global_tokens,
            "output_dim": self.output_dim,
        }
        if any(value < 1 for value in positive_ints.values()):
            raise ValueError(f"{', '.join(positive_ints)} 都必须大于 0。")
        if self.temporal_hidden_size % self.temporal_num_heads != 0:
            raise ValueError("temporal_hidden_size 必须能被 temporal_num_heads 整除。")
        if not math.isfinite(self.temporal_mlp_ratio) or self.temporal_mlp_ratio <= 0:
            raise ValueError("temporal_mlp_ratio 必须是有限正数。")
        if not math.isfinite(self.temporal_dropout) or not 0 <= self.temporal_dropout < 1:
            raise ValueError("temporal_dropout 必须位于 [0, 1)。")
        if not 0 < self.min_valid_frame_fraction <= 1:
            raise ValueError("min_valid_frame_fraction 必须位于 (0, 1]。")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps 必须是有限正数。")


@dataclass
class DemoAlignmentConfig:
    """Demo 缓存、因果 Query 窗口和在线 DTW 的配置。"""

    # 对于 SmolVLA 的 action chunk 推理，这里应填写“重新生成 action
    # chunk”的实际频率，而不是机器人底层控制频率。Demo 可以保留完整
    # 控制频率；DemoEmbeddingCache 会按 timestamp 自动选择匹配锚点。
    alignment_hz: float = 10.0
    window_duration_s: float = 1.0

    # Demo 在任务开始前一次性编码。视觉 token 只在后续 Demo Expert
    # 需要空间信息时保存；Stage Match 本身只使用池化特征。
    demo_encode_batch_size: int = 16
    min_valid_fraction: float = 0.5
    normalize_visual_features: bool = True
    cache_device: str = "cpu"
    cache_visual_tokens: bool = True

    # ``rgb_only`` 用于消融实验。启用后 Matcher 完全跳过 State distance，
    # State 仍保留在缓存中，供匹配后的 Local Demo 使用。
    rgb_only: bool = False

    # Matcher 不比较绝对关节位置，只比较由已观测 State 差分得到的运动特征。
    # 默认排除最后一维，因为 LeRobot 机械臂通常把夹爪 State 放在最后；这可
    # 防止模型依赖夹爪开合事件。不同 State 排列可在构建配置时显式修改。
    # 支持 Python 风格负索引，空 tuple 表示不排除任何维度。
    matching_state_excluded_indices: tuple[int, ...] = (-1,)

    # 可选的非夹爪关节速度尺度，顺序与排除维度后的 State 一致。已知数据集
    # 可使用跨 Demo 统计；新数据可从当前可用 Demo 即时估计。None 表示输入
    # State 已经使用统一尺度，或者明确希望直接使用原始速度。
    matching_state_velocity_scale: tuple[float, ...] | None = None

    # RGB 是主匹配依据；无夹爪的运动强度只作为弱时间正则，避免不同 layout
    # 导致的机械臂轨迹差异反过来压过视觉语义。
    vision_distance_weight: float = 1.0
    state_distance_weight: float = 0.01

    # 在线 DTW 只在当前匹配点之后的有限区间搜索，不回看已经通过的阶段。
    dtw_forward_window: int = 20
    dtw_max_advance: int = 2
    dtw_stay_penalty: float = 0.01
    dtw_skip_penalty: float = 0.05
    dtw_temperature: float = 0.1

    # 匹配完成后，从完整 Demo 中读取固定长度的 Local Chunk。
    # anchor 在 Local Chunk 中的位置由归一化比例决定：默认 0.4
    # 表示 anchor 之前保留约 40% 的历史，anchor 及其后内容占约 60%。
    local_chunk_size: int = 100
    local_anchor_position_ratio: float = 0.4
    eps: float = 1e-6

    @classmethod
    def for_action_chunking(
        cls,
        *,
        control_hz: float,
        n_action_steps: int,
        query_window_replans: int = 4,
        **kwargs: Any,
    ) -> Self:
        """按 SmolVLA action queue 的重规划周期创建匹配配置。

        SmolVLA 每生成一个 action chunk，最多连续执行 ``n_action_steps``
        个动作；因此不额外运行视觉模型时，Matcher 的自然更新频率是
        ``control_hz / n_action_steps``。窗口用重规划次数表达，避免默认
        1 秒窗口在长 action chunk 下不足两帧。
        """
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("控制频率必须是有限正数。")
        if n_action_steps < 1 or query_window_replans < 2:
            raise ValueError(
                "action steps 必须大于 0，窗口至少包含 2 次重规划。"
            )
        if "alignment_hz" in kwargs or "window_duration_s" in kwargs:
            raise ValueError("for_action_chunking 会自动设置 alignment_hz 和 window_duration_s。")

        replan_hz = control_hz / n_action_steps
        return cls(
            alignment_hz=replan_hz,
            window_duration_s=query_window_replans / replan_hz,
            **kwargs,
        )

    def __post_init__(self) -> None:
        """只保留会直接影响算法正确性的配置检查。"""
        positive_floats = {
            "alignment_hz": self.alignment_hz,
            "window_duration_s": self.window_duration_s,
            "dtw_temperature": self.dtw_temperature,
            "eps": self.eps,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive_floats.values()):
            raise ValueError(f"{', '.join(positive_floats)} 都必须是有限正数。")
        if self.window_size < 2:
            raise ValueError("Stage Match 的滚动窗口至少需要 2 帧。")
        if self.demo_encode_batch_size < 1:
            raise ValueError("demo_encode_batch_size 必须大于 0。")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction 必须位于 (0, 1]。")
        distance_weights = (self.vision_distance_weight, self.state_distance_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in distance_weights):
            raise ValueError("视觉和 State 距离权重必须是有限非负数。")
        if self.vision_distance_weight + self.state_distance_weight == 0:
            raise ValueError("视觉和 State 距离权重不能同时为 0。")
        if self.rgb_only and self.vision_distance_weight == 0:
            raise ValueError("rgb_only=True 时 vision_distance_weight 必须大于 0。")
        if len(set(self.matching_state_excluded_indices)) != len(
            self.matching_state_excluded_indices
        ):
            raise ValueError("matching_state_excluded_indices 不能包含重复索引。")
        if self.matching_state_velocity_scale is not None and any(
            not math.isfinite(scale) or scale <= 0
            for scale in self.matching_state_velocity_scale
        ):
            raise ValueError("matching_state_velocity_scale 的每一维都必须是有限正数。")
        if self.dtw_forward_window < 0 or self.dtw_max_advance < 1:
            raise ValueError("DTW 搜索窗口必须非负，最大前进步数必须大于 0。")
        penalties = (self.dtw_stay_penalty, self.dtw_skip_penalty)
        if any(not math.isfinite(penalty) or penalty < 0 for penalty in penalties):
            raise ValueError("DTW penalty 必须是有限非负数。")
        if self.local_chunk_size < 1:
            raise ValueError("local_chunk_size 必须大于 0。")
        if not math.isfinite(self.local_anchor_position_ratio) or not (
            0 <= self.local_anchor_position_ratio <= 1
        ):
            raise ValueError("local_anchor_position_ratio 必须位于 [0, 1]。")

    @property
    def window_size(self) -> int:
        """Query 可见的最大历史长度（包含当前帧）。"""
        return int(round(self.alignment_hz * self.window_duration_s))

    @property
    def local_anchor_position(self) -> int:
        """anchor 在 Local Chunk 内的整数位置。

        使用 ``floor(chunk_size * ratio)`` 将比例映射为序列下标。
        ratio=1 时将 anchor 放在最后一个位置，确保它始终在
        Local Chunk 内。
        """
        return min(
            self.local_chunk_size - 1,
            math.floor(self.local_chunk_size * self.local_anchor_position_ratio),
        )


__all__ = ["DemoAlignmentConfig", "GlobalEncoderConfig"]
