"""SmolVLA-ICL 的 epoch-aware LeRobot sampler。"""

from __future__ import annotations

from collections.abc import Iterator

from lerobot.datasets.sampler import EpisodeAwareSampler

from .contracts import SmolVLAICLSampleIndex

__all__ = ["SmolVLAICLEpisodeAwareSampler"]


class SmolVLAICLEpisodeAwareSampler(EpisodeAwareSampler):
    """LeRobot episode sampler，但将当前 epoch 与每个索引一起传给 Dataset。"""

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[SmolVLAICLSampleIndex]:
        for dataset_index in super()._iter_epoch(epoch, start):
            yield SmolVLAICLSampleIndex(dataset_index=dataset_index, epoch=epoch)
