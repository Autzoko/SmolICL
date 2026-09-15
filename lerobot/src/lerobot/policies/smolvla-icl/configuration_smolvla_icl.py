"""SmolVLA-ICL 的轻量配置。

当前主体模型仍在搭建，因此本文件只定义已经可独立运行的 Stage Match
配置。把纯配置与对齐实现分开后，后续 ``SmolVLAICLConfig`` 可以直接持有
``DemoAlignmentConfig``，而无需从特征处理模块导入类型。
"""

import math
from dataclasses import dataclass
from typing import Any, Self


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

    # 匹配完成后，允许从完整 Demo 中读取锚点之前和之后的内容。
    local_history_steps: int = 40
    local_future_steps: int = 60
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
        if self.local_history_steps < 0 or self.local_future_steps < 1:
            raise ValueError("Local Demo 必须包含锚点，历史长度不能为负数。")

    @property
    def window_size(self) -> int:
        """Query 可见的最大历史长度（包含当前帧）。"""
        return int(round(self.alignment_hz * self.window_duration_s))

    @property
    def local_chunk_size(self) -> int:
        """匹配后送给 Demo Expert 的固定 Demo 长度。"""
        return self.local_history_steps + self.local_future_steps


__all__ = ["DemoAlignmentConfig"]
