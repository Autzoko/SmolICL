"""与具体机器人数据集无关的 SmolVLA-ICL Query Dataset Adapter。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .contracts import DemoReferenceResolver, DemoSampleRef, SmolVLAICLSampleIndex
from .types import RawLocalDemoSample, SMOLVLA_ICL_DEMO_REF


__all__ = ["SmolVLAICLQueryDataset"]


def _scalar_int(value: Any, *, field_name: str) -> int:
    """将 Dataset 中的标量 Tensor/Python 数值安全转换为整数。"""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} 必须是标量，实际形状为 {tuple(value.shape)}。")
        value = value.detach().cpu().item()
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} 必须能够转换为整数。") from error
    if result < 0:
        raise ValueError(f"{field_name} 不能为负数。")
    return result


def _scalar_float(value: Any, *, field_name: str) -> float:
    """将 Dataset 中的标量 Tensor/Python 数值安全转换为浮点数。"""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field_name} 必须是标量，实际形状为 {tuple(value.shape)}。")
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} 必须能够转换为浮点数。") from error


class SmolVLAICLQueryDataset(torch.utils.data.Dataset):
    """为任意 map-style Query Dataset 附加轻量 Demo 引用。

    Adapter 只读取 Query 样本的 episode/frame/timestamp 元数据，并委托
    :class:`DemoReferenceResolver` 生成 ``DemoSampleRef``。它不会读取 Demo
    cache、解码 Demo RGB 或运行 DTW，因此可以独立于最终的数据选择策略实现。

    普通整数索引使用 ``default_epoch``，适合固定配对的验证集；训练 sampler
    可传入 :class:`SmolVLAICLSampleIndex`，显式携带当前 epoch。
    """

    def __init__(
        self,
        query_dataset: torch.utils.data.Dataset,
        demo_ref_resolver: DemoReferenceResolver,
        local_demo_reader: Callable[[DemoSampleRef], RawLocalDemoSample],
        *,
        default_epoch: int = 0,
    ) -> None:
        if default_epoch < 0:
            raise ValueError("default_epoch 不能为负数。")
        self.query_dataset = query_dataset
        self.demo_ref_resolver = demo_ref_resolver
        # Reader 由具体数据集实现：它只按 ref 解码一个 Local 窗口，
        # 不解码整条 Demo，也不执行 SigLIP/S3D。
        self.local_demo_reader = local_demo_reader
        self.default_epoch = default_epoch

    def __len__(self) -> int:
        return len(self.query_dataset)

    @property
    def meta(self) -> Any:
        """透传 LeRobot metadata，供 Policy factory、processor 和 sampler 使用。"""
        return self.query_dataset.meta

    @property
    def episodes(self) -> Any:
        """透传被选择的 Query episode 列表。"""
        return self.query_dataset.episodes

    @property
    def num_frames(self) -> int:
        """返回 Query Dataset 帧数；不把 Demo 帧计入训练样本数。"""
        return int(getattr(self.query_dataset, "num_frames", len(self.query_dataset)))

    @property
    def num_episodes(self) -> int:
        """透传 Query episode 数量。"""
        return int(self.query_dataset.num_episodes)

    @property
    def absolute_to_relative_idx(self) -> Any:
        """透传 episode-filtered Dataset 的绝对到相对索引映射。"""
        return self.query_dataset.absolute_to_relative_idx

    @property
    def hf_dataset(self) -> Any:
        """兼容训练器当前基于 ``hf_dataset`` 的验证集子采样逻辑。"""
        return self.query_dataset.hf_dataset

    def __getitem__(
        self,
        index: int | slice | SmolVLAICLSampleIndex,
    ) -> dict[str, Any] | None | list[dict[str, Any] | None]:
        if isinstance(index, slice):
            return [self[item_index] for item_index in range(*index.indices(len(self)))]

        dataset_index, epoch = self._resolve_index(index)
        sample = self.query_dataset[dataset_index]
        return self._attach_demo_reference(sample, dataset_index=dataset_index, epoch=epoch)

    def __getitems__(
        self,
        indices: Sequence[int | SmolVLAICLSampleIndex],
    ) -> list[dict[str, Any] | None]:
        """批量读取 Query 数据后逐样本附加引用，保留底层批量读取优化。"""
        resolved = [self._resolve_index(index) for index in indices]
        dataset_indices = [dataset_index for dataset_index, _ in resolved]

        getitems = getattr(self.query_dataset, "__getitems__", None)
        if callable(getitems):
            samples = getitems(dataset_indices)
        else:
            samples = [self.query_dataset[index] for index in dataset_indices]
        if len(samples) != len(resolved):
            raise ValueError("底层 Query Dataset 批量读取返回的样本数与索引数不一致。")

        return [
            self._attach_demo_reference(sample, dataset_index=dataset_index, epoch=epoch)
            for sample, (dataset_index, epoch) in zip(samples, resolved, strict=True)
        ]

    def _resolve_index(self, index: int | SmolVLAICLSampleIndex) -> tuple[int, int]:
        if isinstance(index, SmolVLAICLSampleIndex):
            dataset_index, epoch = index.dataset_index, index.epoch
            if dataset_index >= len(self):
                raise IndexError("Query Dataset 索引越界。")
            return dataset_index, epoch
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("SmolVLAICLQueryDataset 索引必须是 int 或 SmolVLAICLSampleIndex。")
        if index < 0:
            # 与多数 PyTorch Dataset 一致地支持负索引，但在交给 Resolver 前转换为
            # 非负相对索引，避免同一 Query frame 出现两种 dataset_index 表示。
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Query Dataset 索引越界。")
        return index, self.default_epoch

    def _attach_demo_reference(
        self,
        sample: Any,
        *,
        dataset_index: int,
        epoch: int,
    ) -> dict[str, Any] | None:
        # 部分 LeRobot recipe 可以过滤样本并返回 None；保持该行为，让现有
        # collator 继续负责丢弃 None，而不是为无效样本解析 Demo。
        if sample is None:
            return None
        if not isinstance(sample, dict):
            raise TypeError("底层 Query Dataset 的单个样本必须是 dict 或 None。")
        if SMOLVLA_ICL_DEMO_REF in sample:
            raise KeyError(f"Query 样本已经包含保留字段 {SMOLVLA_ICL_DEMO_REF!r}。")

        required = ("episode_index", "frame_index", "timestamp")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"Query 样本缺少 Demo 引用解析所需字段：{missing}。")

        episode_index = _scalar_int(sample["episode_index"], field_name="episode_index")
        frame_index = _scalar_int(sample["frame_index"], field_name="frame_index")
        timestamp = _scalar_float(sample["timestamp"], field_name="timestamp")
        task_value = sample.get("task")
        task = task_value if isinstance(task_value, str) else None
        reference = self.demo_ref_resolver.resolve(
            epoch=epoch,
            dataset_index=dataset_index,
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
            task=task,
        )
        if not isinstance(reference, DemoSampleRef):
            raise TypeError("DemoReferenceResolver.resolve() 必须返回 DemoSampleRef。")

        # 返回浅拷贝，避免 Adapter 修改底层 Dataset 可能复用的 row cache。
        output = dict(sample)
        output[SMOLVLA_ICL_DEMO_REF] = reference
        return output
