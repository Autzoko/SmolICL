"""LeRobot Dataset 的 Local Demo 窗口读取器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from .contracts import DemoSampleRef
from .local_rgb_cache import LocalRGBFrameCacheStore
from .types import RawLocalDemoSample

__all__ = ["LeRobotLocalDemoReader", "LeRobotMatcherEpisodeReader", "MatcherEpisodeData"]


def _stack_column(values: list[Tensor], *, name: str) -> Tensor:
    """将 Hugging Face Dataset 批量索引返回的 Tensor list 堆叠。"""
    if not values or not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"Demo 字段 {name!r} 必须是非空 Tensor 序列。")
    return torch.stack(values)


@dataclass(frozen=True, slots=True)
class MatcherEpisodeData:
    """离线 Matcher 编码一条 episode 所需的原始时序数据。"""

    images: Tensor
    states: Tensor
    timestamps: Tensor


@dataclass
class LeRobotMatcherEpisodeReader:
    """一次读取一条完整 episode，仅供离线冻结 Matcher 编码。

    State 和 timestamp 只从 Parquet 投影必需列；如果 RGB 保存为视频，
    整条 episode 通过一次 decoder 调用批量解码，不会按帧重复打开视频。
    """

    dataset: LeRobotDataset
    image_key: str
    state_key: str

    def __post_init__(self) -> None:
        if self.image_key not in self.dataset.meta.camera_keys:
            raise KeyError(f"Matcher Dataset 中不存在 RGB key: {self.image_key!r}。")
        if self.image_key in self.dataset.meta.depth_keys:
            raise ValueError("Matcher 只支持 RGB camera，不能使用 depth key。")
        if self.state_key not in self.dataset.meta.features:
            raise KeyError(f"Matcher Dataset 中不存在 State key: {self.state_key!r}。")

    def __call__(self, episode_index: int) -> MatcherEpisodeData:
        episode = self.dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        relative_indices = self._absolute_to_relative(range(episode_start, episode_end))

        # select_columns 是 zero-copy schema projection，避免为读取 State/时间
        # 解码其他 camera、action 或与 Matcher 无关的列。
        rows = self.dataset.hf_dataset.select_columns([self.state_key, "timestamp"])[relative_indices]
        states = _stack_column(rows[self.state_key], name=self.state_key).float()
        timestamps = _stack_column(rows["timestamp"], name="timestamp").flatten().to(dtype=torch.float64)
        images = self._read_images(episode_index, relative_indices, timestamps)
        if len(images) != len(states) or len(states) != len(timestamps):
            raise ValueError("Matcher episode 的 RGB/State/timestamp 长度不一致。")
        return MatcherEpisodeData(images=images, states=states, timestamps=timestamps)

    def _absolute_to_relative(self, absolute_indices: range) -> list[int]:
        mapping = self.dataset.absolute_to_relative_idx
        if mapping is None:
            return list(absolute_indices)
        try:
            return [mapping[index] for index in absolute_indices]
        except KeyError as error:
            raise KeyError("Matcher Dataset 未加载请求的 episode。") from error

    def _read_images(
        self,
        episode_index: int,
        relative_indices: list[int],
        timestamps: Tensor,
    ) -> Tensor:
        if self.image_key not in self.dataset.meta.video_keys:
            rows = self.dataset.hf_dataset.select_columns(self.image_key)[relative_indices]
            return _stack_column(rows[self.image_key], name=self.image_key)

        episode = self.dataset.meta.episodes[episode_index]
        video_start = float(episode[f"videos/{self.image_key}/from_timestamp"])
        video_path = self.dataset.root / self.dataset.meta.get_video_file_path(
            episode_index,
            self.image_key,
        )
        return decode_video_frames(
            video_path,
            [video_start + float(timestamp) for timestamp in timestamps],
            self.dataset.tolerance_s,
            self.dataset._video_backend,
            return_uint8=True,
        )


@dataclass
class LeRobotLocalDemoReader:
    """按 ``demo_id + local_anchor`` 读取一个 Local RGB+State 窗口。

    该 reader 使用一个独立的 Demo-only :class:`LeRobotDataset`：
    它没有 Query delta timestamps 和图像增广。RGB 来自离线 episode mmap，
    State/时间来自 Parquet；训练进程不会重复调用视频 decoder。
    """

    dataset: LeRobotDataset
    demo_id_to_episode: Mapping[str, int]
    image_key: str
    state_key: str
    chunk_size: int
    anchor_position: int
    rgb_cache: LocalRGBFrameCacheStore
    _timestamp_bounds: dict[int, tuple[float, float]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.demo_id_to_episode = dict(self.demo_id_to_episode)
        if self.image_key not in self.dataset.meta.camera_keys:
            raise KeyError(f"Demo Dataset 中不存在 RGB key: {self.image_key!r}。")
        if self.image_key in self.dataset.meta.depth_keys:
            raise ValueError("Local Demo 必须使用 RGB camera，不能使用 depth key。")
        if self.state_key not in self.dataset.meta.features:
            raise KeyError(f"Demo Dataset 中不存在 State key: {self.state_key!r}。")
        if self.chunk_size < 1 or not 0 <= self.anchor_position < self.chunk_size:
            raise ValueError("Local Demo chunk_size/anchor_position 配置无效。")
        for demo_id, episode_index in self.demo_id_to_episode.items():
            entry = self.rgb_cache.entries.get(demo_id)
            if entry is None or entry.episode_index != episode_index:
                raise ValueError(f"Local RGB cache 与 Demo episode 映射不一致：{demo_id}")

    def __call__(self, reference: DemoSampleRef) -> RawLocalDemoSample:
        try:
            episode_index = self.demo_id_to_episode[reference.demo_id]
        except KeyError as error:
            raise KeyError(f"未知的 Demo ID: {reference.demo_id!r}。") from error

        episode = self.dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        episode_length = episode_end - episode_start
        if not 0 <= reference.local_anchor < episode_length:
            raise IndexError(
                f"Demo {reference.demo_id!r} local_anchor={reference.local_anchor} "
                f"超出 episode 长度 {episode_length}。"
            )

        relative_slots = torch.arange(self.chunk_size) - self.anchor_position
        requested_local = reference.local_anchor + relative_slots
        valid_mask = (requested_local >= 0) & (requested_local < episode_length)
        valid_slots = torch.nonzero(valid_mask, as_tuple=False).flatten()
        absolute_indices = episode_start + requested_local[valid_mask]
        relative_indices = self._absolute_to_relative(absolute_indices.tolist())

        # 当窗口左边界不是 episode 起点时，额外读取紧邻的前一帧
        # State/时间。它只用于首个 token 的后向差分，不读 RGB，也不
        # 改变 Local chunk 的 48-token 长度。
        window_start = int(requested_local[0])
        previous_relative_index: int | None = None
        if window_start > 0:
            previous_relative_index = self._absolute_to_relative(
                [episode_start + window_start - 1]
            )[0]
        state_row_indices = (
            [previous_relative_index, *relative_indices]
            if previous_relative_index is not None
            else relative_indices
        )

        # State/时间从 Parquet 投影读取；RGB 从 episode 级 mmap 切片，
        # 训练 worker 不再打开或解码 MP4。
        rows = self.dataset.hf_dataset.select_columns([self.state_key, "timestamp"])[state_row_indices]
        all_states = _stack_column(rows[self.state_key], name=self.state_key).float()
        all_timestamps = _stack_column(rows["timestamp"], name="timestamp").flatten().to(
            dtype=torch.float64
        )
        if previous_relative_index is None:
            previous_state = None
            previous_timestamp = None
            valid_states = all_states
            valid_timestamps = all_timestamps
        else:
            previous_state = all_states[0]
            previous_timestamp = float(all_timestamps[0])
            valid_states = all_states[1:]
            valid_timestamps = all_timestamps[1:]
        valid_images = self.rgb_cache.read(
            reference.demo_id,
            requested_local[valid_mask].to(dtype=torch.long),
        )

        images = valid_images.new_zeros(
            self.chunk_size,
            *valid_images.shape[1:],
        )
        states = valid_states.new_zeros(self.chunk_size, valid_states.shape[-1])
        images[valid_slots] = valid_images
        states[valid_slots] = valid_states

        # 有效帧保留 Dataset 的真实 timestamp，与 rollout 完整 Demo 路径
        # 一致；只对 episode 外的 padding 按 fps 向左/右外推。
        timestamps = valid_timestamps.new_empty(self.chunk_size)
        timestamps[valid_slots] = valid_timestamps
        period_s = 1.0 / self.dataset.meta.fps
        first_valid_slot = int(valid_slots[0])
        if first_valid_slot > 0:
            left_steps = torch.arange(
                first_valid_slot,
                0,
                -1,
                dtype=torch.float64,
                device=timestamps.device,
            )
            timestamps[:first_valid_slot] = valid_timestamps[0] - left_steps * period_s
        last_valid_slot = int(valid_slots[-1])
        if last_valid_slot + 1 < self.chunk_size:
            right_steps = torch.arange(
                1,
                self.chunk_size - last_valid_slot,
                dtype=torch.float64,
                device=timestamps.device,
            )
            timestamps[last_valid_slot + 1 :] = valid_timestamps[-1] + right_steps * period_s

        demo_start_timestamp, demo_end_timestamp = self._episode_timestamp_bounds(
            episode_index,
            episode_start,
            episode_end,
        )
        return RawLocalDemoSample(
            images=images,
            states=states,
            timestamps=timestamps,
            valid_mask=valid_mask,
            previous_state=previous_state,
            previous_timestamp=previous_timestamp,
            anchor_position=self.anchor_position,
            demo_start_timestamp=demo_start_timestamp,
            demo_end_timestamp=demo_end_timestamp,
        )

    def _episode_timestamp_bounds(
        self,
        episode_index: int,
        episode_start: int,
        episode_end: int,
    ) -> tuple[float, float]:
        """每个 worker 只读取一次 Demo 的首尾时间，并且只投影时间列。"""
        cached = self._timestamp_bounds.get(episode_index)
        if cached is not None:
            return cached
        indices = self._absolute_to_relative([episode_start, episode_end - 1])
        rows = self.dataset.hf_dataset.select_columns("timestamp")[indices]
        timestamps = rows["timestamp"]
        bounds = (float(timestamps[0]), float(timestamps[1]))
        self._timestamp_bounds[episode_index] = bounds
        return bounds

    def _absolute_to_relative(self, absolute_indices: list[int]) -> list[int]:
        mapping = self.dataset.absolute_to_relative_idx
        if mapping is None:
            return absolute_indices
        try:
            return [mapping[index] for index in absolute_indices]
        except KeyError as error:
            raise KeyError("当前 Demo Dataset 未加载 sidecar 引用的 episode。") from error
