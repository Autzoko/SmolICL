"""LeRobot Dataset 的 Local Demo 窗口读取器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from .contracts import DemoSampleRef
from .types import RawLocalDemoSample


__all__ = ["LeRobotLocalDemoReader"]


def _stack_column(values: list[Tensor], *, name: str) -> Tensor:
    """将 Hugging Face Dataset 批量索引返回的 Tensor list 堆叠。"""
    if not values or not all(isinstance(value, Tensor) for value in values):
        raise TypeError(f"Demo 字段 {name!r} 必须是非空 Tensor 序列。")
    return torch.stack(values)


@dataclass
class LeRobotLocalDemoReader:
    """按 ``demo_id + local_anchor`` 解码一个 Local RGB+State 窗口。

    该 reader 使用一个独立的 Demo-only :class:`LeRobotDataset`：
    它没有 Query delta timestamps 和图像增广，因此不会解码整条 Demo，
    也不会把 Query 训练数据处理错用到 Demo 窗口上。
    """

    dataset: LeRobotDataset
    demo_id_to_episode: Mapping[str, int]
    image_key: str
    state_key: str
    chunk_size: int
    anchor_position: int

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

        # Parquet 一次读取连续窗口的 State/时间列；视频帧则在
        # 下方用同一次 decoder 调用批量解码，避免每帧重新打开 mp4。
        rows = self.dataset.hf_dataset[relative_indices]
        valid_states = _stack_column(rows[self.state_key], name=self.state_key).float()
        valid_timestamps = _stack_column(rows["timestamp"], name="timestamp").flatten().to(
            dtype=torch.float64
        )
        valid_images = self._read_images(
            episode_index,
            relative_indices,
            valid_timestamps,
        )

        images = valid_images.new_zeros(
            self.chunk_size,
            *valid_images.shape[1:],
        )
        states = valid_states.new_zeros(self.chunk_size, valid_states.shape[-1])
        images[valid_slots] = valid_images
        states[valid_slots] = valid_states

        # LeRobot episode 以固定 fps 采样。对越界 padding 使用相同
        # 周期外推 timestamp，保证 State 速度特征所需的严格单调时间轴。
        anchor_slot = int((requested_local[valid_mask] == reference.local_anchor).nonzero()[0])
        anchor_time = valid_timestamps[anchor_slot]
        timestamps = anchor_time + relative_slots.to(torch.float64) / self.dataset.meta.fps

        first_row, last_row = self.dataset.hf_dataset[
            self._absolute_to_relative([episode_start, episode_end - 1])
        ]["timestamp"]
        return RawLocalDemoSample(
            images=images,
            states=states,
            timestamps=timestamps,
            valid_mask=valid_mask,
            anchor_position=self.anchor_position,
            demo_start_timestamp=float(first_row),
            demo_end_timestamp=float(last_row),
        )

    def _absolute_to_relative(self, absolute_indices: list[int]) -> list[int]:
        mapping = self.dataset.absolute_to_relative_idx
        if mapping is None:
            return absolute_indices
        try:
            return [mapping[index] for index in absolute_indices]
        except KeyError as error:
            raise KeyError("当前 Demo Dataset 未加载 sidecar 引用的 episode。") from error

    def _read_images(
        self,
        episode_index: int,
        relative_indices: list[int],
        timestamps: Tensor,
    ) -> Tensor:
        if self.image_key not in self.dataset.meta.video_keys:
            rows = self.dataset.hf_dataset[relative_indices]
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
