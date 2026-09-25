"""SmolVLA-ICL 的纯配置定义。

本文件只放置可序列化的 dataclass，不导入视频骨干、Transformer
或 Stage Matcher 实现。这样配置可以被 LeRobot/draccus 安全保存，
也不会因为可选模型依赖导致导入失败。
"""

import math
from dataclasses import dataclass, field, fields
from typing import Any, Self

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@dataclass
class GlobalEncoderConfig:
    """Global Demo Encoder 的结构、输入与训练配置。

    Global Encoder 把一条完整 Demo 编码为固定数量的 Global Task
    Tokens。输出宽度直接对齐 Demo Expert，因此默认为 SmolVLA
    VLM hidden size 960 乘以 0.75，即 720。

    当前实现固定使用 S3D；如果后续接入其他视频骨干，应增加独立的
    backbone adapter，而不是保留尚未生效的名称参数。
    """

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
    # 注册或离线缓存 Demo 时，每次送入 S3D 的 clip 数。完整视频保留在
    # CPU，只逐批上传，避免长 Demo 在 ``set_demo`` 阶段产生显存峰值。
    clip_encode_batch_size: int = 2

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
    # Aggregator。视频侧用有效帧补满后进入 S3D，State 侧按 mask 清零。
    min_valid_frame_fraction: float = 0.5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """验证所有会影响张量形状或数值稳定性的配置。"""
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
            "clip_encode_batch_size": self.clip_encode_batch_size,
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
class LocalEncoderConfig:
    """Local Demo Encoder 的逐帧多模态嵌入配置。

    Local Encoder 使用一个可学习 query 压缩每帧的 spatial visual
    tokens，再把已对齐的每个 Demo observation 转换为 Demo Expert
    宽度的一个 token。跨帧时序交互由后续 Demo Expert
    Self-Attention 负责，本配置因此不包含额外的 GRU 或 Transformer 参数。
    """

    # SmolVLA connector 输出宽度默认与 VLM hidden size 一致。
    visual_feature_dim: int = 960
    state_dim: int = 32

    # RGB 和 [State, dState/dt] 分别投影后再融合。
    visual_projection_dim: int = 512
    state_projection_dim: int = 128
    output_dim: int = 720

    # relative time、relative position 和 global phase 各使用一份
    # SmolVLA 风格的连续正弦/余弦编码。
    temporal_embedding_dim: int = 128
    min_period: float = 4e-3
    max_period: float = 4.0

    def __post_init__(self) -> None:
        """只验证会直接决定线性层和正弦编码形状的参数。"""
        dimensions = (
            self.visual_feature_dim,
            self.state_dim,
            self.visual_projection_dim,
            self.state_projection_dim,
            self.output_dim,
            self.temporal_embedding_dim,
        )
        if any(dimension < 1 for dimension in dimensions):
            raise ValueError("Local Encoder 的所有特征维度都必须大于 0。")
        if self.temporal_embedding_dim % 2 != 0:
            raise ValueError("temporal_embedding_dim 必须是偶数。")
        if (
            not math.isfinite(self.min_period)
            or not math.isfinite(self.max_period)
            or not 0 < self.min_period <= self.max_period
        ):
            raise ValueError("Local Encoder 的时间编码周期必须是有限正数且从小到大。")


@dataclass
class DemoAlignmentConfig:
    """Demo 缓存、因果 Query 窗口和在线 DTW 的配置。"""

    # 对于 SmolVLA 的 action chunk 推理，这里应填写“重新生成 action
    # chunk”的实际频率，而不是机器人底层控制频率。Demo 可以保留完整
    # 控制频率；DemoEmbeddingCache 会按 timestamp 自动选择匹配锚点。
    alignment_hz: float = 10.0
    window_duration_s: float = 1.0

    # Demo Matcher 在任务开始前用独立冻结 snapshot 一次性编码。
    # Matcher cache 不保存供模型训练使用的 Local spatial tokens。
    demo_encode_batch_size: int = 16
    min_valid_fraction: float = 0.5
    normalize_visual_features: bool = True
    cache_device: str = "cpu"

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
    local_chunk_size: int = 48
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
            raise ValueError("action steps 必须大于 0，窗口至少包含 2 次重规划。")
        if "alignment_hz" in kwargs or "window_duration_s" in kwargs:
            raise ValueError("for_action_chunking 会自动设置 alignment_hz 和 window_duration_s。")

        replan_hz = control_hz / n_action_steps
        return cls(
            alignment_hz=replan_hz,
            window_duration_s=query_window_replans / replan_hz,
            **kwargs,
        )

    def bind_action_chunking(
        self,
        *,
        control_hz: float,
        n_action_steps: int,
        query_window_replans: int = 4,
    ) -> Self:
        """保留 Matcher 超参数，只按 action chunking 重建时间语义。

        Builder 和训练数据工厂共同使用此入口，避免一侧手写
        ``alignment_hz``、另一侧从 ``n_action_steps`` 推导而产生漂移。
        """
        kwargs = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"alignment_hz", "window_duration_s"}
        }
        return type(self).for_action_chunking(
            control_hz=control_hz,
            n_action_steps=n_action_steps,
            query_window_replans=query_window_replans,
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
        if len(set(self.matching_state_excluded_indices)) != len(self.matching_state_excluded_indices):
            raise ValueError("matching_state_excluded_indices 不能包含重复索引。")
        if self.matching_state_velocity_scale is not None and any(
            not math.isfinite(scale) or scale <= 0 for scale in self.matching_state_velocity_scale
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


@PreTrainedConfig.register_subclass("smolvla_icl")
@dataclass
class SmolVLAICLConfig(SmolVLAConfig):
    """SmolVLA-ICL 顶层配置。

    VLM、Action Expert、flow matching 和图像/语言预处理参数全部继承
    SmolVLA 默认值；这里只增加 Demo 路径配置与两条新增 Cross-Attention
    的残差门控。首版固定使用设计文档中的 16 层、偶数层 Union、奇数层
    Cross 结构，避免缩减 Expert 层数后破坏层类型对应关系。
    """

    global_encoder: GlobalEncoderConfig = field(default_factory=GlobalEncoderConfig)
    local_encoder: LocalEncoderConfig = field(default_factory=LocalEncoderConfig)
    demo_alignment: DemoAlignmentConfig = field(default_factory=DemoAlignmentConfig)

    # V100 使用 FP32 参数存储，并由训练入口通过 FP16 autocast/GradScaler
    # 执行混合精度计算。这样既使用 Volta Tensor Core，也避免直接更新
    # FP16 参数带来的数值精度损失。A100/H100 可显式改回 ``bfloat16``。
    vlm_load_dtype: str = "float32"

    # Query RGB 和 Local Demo RGB 始终共享同一套 SigLIP+connector。
    # False：两条视觉路径都参与 Action Loss 反传；True：两条路径都只做
    # 冻结前向，但 Local Encoder 及后续模块仍正常训练。Matcher 始终使用
    # 另一份冻结 snapshot，因此不受该开关影响，也无需重建 pairing sidecar。
    freeze_vision_encoder: bool = False

    # Local Demo 一次最多送入视觉编码器的帧数。Local Chunk 的语义窗口
    # 保持不变，只在 GPU 上按小批次编码，避免 B*T 张图同时展开。
    local_vision_encode_batch_size: int = 2
    # 训练可学习视觉编码器时重算前向以降低激活显存；视觉冻结、rollout
    # 或外层 no_grad 时自动跳过，避免没有反向收益的重复计算。
    local_vision_gradient_checkpointing: bool = True

    # 训练数据只保存 demo_id/query_anchor/local_anchor；该目录只保存
    # 冻结 S3D 的 Global clip feature。Local RGB 由 Dataset 按 anchor 读取。
    training_demo_cache_dir: str | None = None
    # episode 级数据协议：唯一确定 train/val/test 以及 Demo/Query 划分。
    # 训练数据工厂不再使用 LeRobot 通用 eval_split 重新划分。
    data_manifest_path: str | None = None
    # 只由 Manifest train/demo + train/query 计算的 State/Action 统计。
    # Query、Demo、Matcher 与离线 cache 必须共用同一 fingerprint。
    training_stats_path: str | None = None
    # 离线 DTW 配对表。它只记录 Query episode/frame 到
    # ``demo_id + local_anchor`` 的映射，不保存任何 RGB 或模型 token。
    pairing_sidecar_path: str | None = None
    # 该 LRU 位于每个 DataLoader worker 内，只缓存最近使用的
    # Global S3D feature，不缓存 Local RGB 或模型视觉 tokens。
    training_demo_cache_memory_entries: int = 1
    # 训练期 Local Demo 只从离线解码的 CPU uint8 episode frame cache 读取；
    # 可训练 SigLIP/connector 的 token 仍在 forward 中在线计算。
    training_local_rgb_cache_dir: str | None = None

    # 使用很小的非零 gate：基本保持预训练 SmolVLA 的初始行为，同时让
    # Global/Local Encoder、Demo Expert 和 Cross-Attention 从第一次反传
    # 就能收到任务损失梯度。设为 0 会使首步只有 gate 自身有梯度。
    global_cross_gate_init: float = 1e-3
    local_cross_gate_init: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_demo_cache_memory_entries < 0:
            raise ValueError("training_demo_cache_memory_entries 不能为负数。")
        if self.local_vision_encode_batch_size < 1:
            raise ValueError("local_vision_encode_batch_size 必须大于 0。")
        if self.attention_mode != "cross_attn" or self.self_attn_every_n_layers != 2:
            raise ValueError(
                "SmolVLA-ICL 首版要求 attention_mode='cross_attn' 且 self_attn_every_n_layers=2。"
            )
        if self.num_vlm_layers != 16 or not math.isclose(
            self.expert_width_multiplier,
            0.75,
        ):
            raise ValueError("SmolVLA-ICL 首版固定使用 16 层 VLM 和 0.75 Expert 宽度。")
        if self.num_expert_layers > 0:
            raise ValueError(
                "SmolVLA-ICL 首版要求 num_expert_layers<=0，使 Action/Demo Expert 与 VLM 保持相同层数。"
            )
        if not self.use_cache:
            raise NotImplementedError("SmolVLA-ICL 推理固定使用 P/G/L condition cache。")
        if self.compile_model:
            raise NotImplementedError("SmolVLA-ICL 尚未接入 torch.compile。")
        if self.rtc_config is not None and self.rtc_config.enabled:
            raise NotImplementedError("SmolVLA-ICL 尚未接入 RTC。")
        if self.adapt_to_pi_aloha:
            raise NotImplementedError("SmolVLA-ICL 尚未统一 ALOHA Prefix 与 Demo Matcher 的 State 坐标系。")
        if self.global_encoder.output_dim != self.local_encoder.output_dim:
            raise ValueError("Global/Local Encoder 的 output_dim 必须一致。")
        if (
            self.global_encoder.state_dim != self.max_state_dim
            or self.local_encoder.state_dim != self.max_state_dim
        ):
            raise ValueError("Global/Local Encoder state_dim 必须等于 max_state_dim。")
        if not math.isfinite(self.global_cross_gate_init) or not math.isfinite(self.local_cross_gate_init):
            raise ValueError("Cross-Attention gate 初始值必须是有限数。")

    def validate_features(self) -> None:
        """验证真实 State/Action 维度能被 SmolVLA 的固定投影接收。"""
        super().validate_features()
        state_feature = self.robot_state_feature
        action_feature = self.action_feature
        if state_feature is None or len(state_feature.shape) != 1:
            raise ValueError("SmolVLA-ICL 需要一维 observation.state 特征。")
        if action_feature is None or len(action_feature.shape) != 1:
            raise ValueError("SmolVLA-ICL 需要一维 action 特征。")
        if state_feature.shape[0] > self.max_state_dim:
            raise ValueError("真实 State 维度不能超过 max_state_dim。")
        if action_feature.shape[0] > self.max_action_dim:
            raise ValueError("真实 Action 维度不能超过 max_action_dim。")


__all__ = [
    "DemoAlignmentConfig",
    "GlobalEncoderConfig",
    "LocalEncoderConfig",
    "SmolVLAICLConfig",
]
