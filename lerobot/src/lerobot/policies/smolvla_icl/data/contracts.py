"""SmolVLA-ICL 数据层与模型侧之间的轻量端口契约。

本模块只定义不依赖具体数据集格式、磁盘缓存格式或匹配算法的数据对象。
Dataset 侧负责产生 :class:`DemoSampleRef`，Collator 侧负责消费它并构建
``GlobalDemoBatch``/``LocalDemoBatch``。这样后续接入 LIBERO、其他 LeRobot
数据集或人工配对表时，都不需要修改模型输入接口。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


__all__ = [
    "DemoReferenceLookupKey",
    "DemoReferenceResolver",
    "DemoSampleRef",
    "FixedDemoReferenceResolver",
    "MappingDemoReferenceResolver",
    "SmolVLAICLSampleIndex",
]


@dataclass(frozen=True, slots=True)
class DemoSampleRef:
    """一个 Query 样本对已缓存 Demo 的轻量引用。

    ``query_anchor`` 与 ``local_anchor`` 都使用各自 episode 内的 observation
    索引，而不是整个数据集的绝对行号。前者用于追踪当前 Query 样本，后者
    用于从完整 Demo cache 中截取已经匹配好的 Local Chunk。
    """

    demo_id: str
    query_anchor: int
    local_anchor: int

    def __post_init__(self) -> None:
        if not self.demo_id:
            raise ValueError("demo_id 不能为空。")
        if self.query_anchor < 0 or self.local_anchor < 0:
            raise ValueError("query_anchor 和 local_anchor 不能为负数。")


@dataclass(frozen=True, slots=True)
class SmolVLAICLSampleIndex:
    """Sampler 传给 Query Dataset Adapter 的 epoch-aware 索引。

    普通 LeRobot sampler 只传递 ``dataset_index``。SmolVLA-ICL 还需要
    ``epoch``，才能保证同一 Query episode 在一个 epoch 内使用同一条 Demo，
    并允许在下一个 epoch 确定性地重新选择 Demo。
    """

    dataset_index: int
    epoch: int

    def __post_init__(self) -> None:
        if self.dataset_index < 0:
            raise ValueError("dataset_index 不能为负数。")
        if self.epoch < 0:
            raise ValueError("epoch 不能为负数。")


class DemoReferenceResolver(Protocol):
    """把 Query 样本元数据解析为 :class:`DemoSampleRef` 的可替换端口。

    后续的 pairing sidecar 只需要实现该接口。Resolver 不应返回完整 Demo
    Tensor，也不应修改 Query 样本；磁盘读取和 Local Chunk 提取由 Collator
    统一完成。
    """

    def resolve(
        self,
        *,
        epoch: int,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        task: str | None,
    ) -> DemoSampleRef:
        """返回当前 Query 样本在指定 epoch 使用的 Demo 引用。"""
        ...


@dataclass(frozen=True, slots=True)
class FixedDemoReferenceResolver:
    """始终返回同一 Demo/Local anchor 的最小 Resolver。

    该实现主要用于端口测试和人工 smoke test。正式训练应替换为后续的
    ``PairingSidecarResolver``。
    """

    demo_id: str
    local_anchor: int = 0

    def __post_init__(self) -> None:
        # 复用正式引用的校验规则，避免测试 Resolver 接受生产路径会拒绝的值。
        DemoSampleRef(self.demo_id, query_anchor=0, local_anchor=self.local_anchor)

    def resolve(
        self,
        *,
        epoch: int,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        task: str | None,
    ) -> DemoSampleRef:
        del epoch, dataset_index, episode_index, timestamp, task
        return DemoSampleRef(
            demo_id=self.demo_id,
            query_anchor=frame_index,
            local_anchor=self.local_anchor,
        )


# Mapping Resolver 使用稳定、与具体 Dataset 类无关的三元组作为 key。
# dataset_index 不进入 key，因为 episode 过滤后 Dataset 相对索引可能变化；
# episode 内 frame_index 才是 pairing sidecar 应保存的稳定 Query anchor。
type DemoReferenceLookupKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MappingDemoReferenceResolver:
    """从 ``(epoch, episode_index, frame_index)`` 映射中读取 Demo 引用。

    这是 sidecar reader 接入前的通用过渡实现，也便于测试 epoch 变化行为。
    映射会在初始化时复制，避免调用方在 DataLoader worker 启动后修改原字典，
    导致不同 worker 观察到不一致的数据。
    """

    mapping: Mapping[DemoReferenceLookupKey, DemoSampleRef]

    def __post_init__(self) -> None:
        copied = dict(self.mapping)
        for key, reference in copied.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
            ):
                raise TypeError(
                    "Mapping key 必须是 (epoch, episode_index, frame_index) 整数三元组。"
                )
            if any(value < 0 for value in key):
                raise ValueError("Mapping key 中的 epoch/episode/frame 不能为负数。")
            if not isinstance(reference, DemoSampleRef):
                raise TypeError("Mapping value 必须是 DemoSampleRef。")
        object.__setattr__(self, "mapping", copied)

    def resolve(
        self,
        *,
        epoch: int,
        dataset_index: int,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        task: str | None,
    ) -> DemoSampleRef:
        del dataset_index, timestamp, task
        key = (epoch, episode_index, frame_index)
        try:
            return self.mapping[key]
        except KeyError as error:
            raise KeyError(
                "没有为 Query 样本配置 Demo 引用："
                f"epoch={epoch}, episode_index={episode_index}, frame_index={frame_index}。"
            ) from error
