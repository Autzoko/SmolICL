"""SmolVLA-ICL 数据层与模型侧之间的轻量端口契约。

本模块只定义不依赖具体数据集格式、磁盘缓存格式或匹配算法的数据对象。
Dataset 侧负责产生 :class:`DemoSampleRef`，Collator 侧负责消费它并构建
``GlobalDemoBatch``/``LocalDemoBatch``。这样后续接入 LIBERO、其他 LeRobot
数据集或人工配对表时，都不需要修改模型输入接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "DemoReferenceResolver",
    "DemoSampleRef",
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
