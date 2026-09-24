"""SmolVLA-ICL 训练用的 Demo 数据边界。

缓存边界刻意停在冻结模块之后：

* Global 保存 S3D 的逐 clip feature；
* Matcher 使用独立冻结 snapshot，其 embedding/anchor sidecar 不进入模型；
* Local 根据 sidecar 只读取 raw RGB+State，不缓存 connector tokens；
* Global/Local Encoder 的可训练输出均不保存。

Dataset 单样本只需携带 :class:`DemoSampleRef`。Collator 会按 ``demo_id``
去重磁盘读取，并用 inverse index 将唯一 Demo 的 Global 编码结果映射回每个
query 样本。
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.utils.collate import lerobot_collate_fn

from ..components.global_encoder import GlobalDemoEncoder
from .collate import (
    collate_global_demo_samples,
    collate_raw_local_demo_samples,
    validate_smolvla_icl_training_batch,
)
from .contracts import DemoSampleRef
from .preprocessing import build_global_demo_clips
from .state import DemoStateNormalizer
from .types import (
    SMOLVLA_ICL_DEMO_REF,
    SMOLVLA_ICL_GLOBAL_DEMO,
    SMOLVLA_ICL_LOCAL_DEMO,
    GlobalDemoClips,
    GlobalDemoSample,
    RawLocalDemoSample,
)

__all__ = [
    "CachedTrainingDemo",
    "DemoFeatureStore",
    "SmolVLAICLCollator",
    "cache_training_demo",
    "precompute_training_demo",
]


@dataclass(frozen=True, slots=True)
class CachedTrainingDemo:
    """一条 Demo 的 Global 冻结 S3D 特征。"""

    demo_id: str
    global_demo: GlobalDemoSample
    cache_identity: str

    def __post_init__(self) -> None:
        if not self.demo_id:
            raise ValueError("demo_id 不能为空。")
        if len(self.cache_identity) != 64 or any(
            char not in "0123456789abcdef" for char in self.cache_identity.lower()
        ):
            raise ValueError("cache_identity 必须是 SHA-256 字符串。")

    def to_serializable(self) -> dict[str, Any]:
        """导出仅含基础类型和 CPU Tensor 的版本化 payload。"""
        global_demo = self.global_demo
        return {
            "version": 3,
            "demo_id": self.demo_id,
            "cache_identity": self.cache_identity,
            "global": {
                "video_features": global_demo.video_features.detach().cpu(),
                "states": global_demo.states.detach().cpu(),
                "timestamps": global_demo.timestamps.detach().cpu(),
                "valid_mask": global_demo.valid_mask.detach().cpu(),
            },
        }

    @classmethod
    def from_serializable(cls, payload: dict[str, Any]) -> CachedTrainingDemo:
        """从 :meth:`to_serializable` 的 payload 恢复对象。"""
        if payload.get("version") != 3:
            raise ValueError("不支持的 SmolVLA-ICL 训练缓存版本。")
        demo_id = str(payload["demo_id"])
        global_payload = payload["global"]
        return cls(
            demo_id=demo_id,
            global_demo=GlobalDemoSample(
                video_features=global_payload["video_features"],
                states=global_payload["states"],
                timestamps=global_payload["timestamps"],
                valid_mask=global_payload["valid_mask"],
            ),
            cache_identity=str(payload["cache_identity"]),
        )


class DemoFeatureStore:
    """按 ``demo_id`` 管理一条一文件的离线特征缓存。

    每个 DataLoader worker 拥有自己的小型 CPU LRU。它避免同一 Demo 在相邻
    batch 反复读盘；单个 batch 内的去重由 :class:`SmolVLAICLCollator` 保证。
    """

    def __init__(self, root: str | Path, *, memory_entries: int = 1) -> None:
        self.root = Path(root).expanduser()
        if memory_entries < 0:
            raise ValueError("memory_entries 不能为负数。")
        self.memory_entries = memory_entries
        self._memory: OrderedDict[str, CachedTrainingDemo] = OrderedDict()

    @staticmethod
    def _filename(demo_id: str) -> str:
        digest = hashlib.sha256(demo_id.encode("utf-8")).hexdigest()
        return f"{digest}.pt"

    def path_for(self, demo_id: str) -> Path:
        """返回稳定且不受 demo_id 路径字符影响的 cache 路径。"""
        if not demo_id:
            raise ValueError("demo_id 不能为空。")
        return self.root / self._filename(demo_id)

    def save(self, demo: CachedTrainingDemo) -> Path:
        """原子写入一条 Demo cache，并更新当前进程内存副本。"""
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(demo.demo_id)
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        try:
            torch.save(demo.to_serializable(), temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self._remember(demo)
        return target

    def load(self, demo_id: str) -> CachedTrainingDemo:
        """从进程内 LRU 或磁盘读取一条 Demo；不会触发任何视觉模型。"""
        cached = self._memory.pop(demo_id, None)
        if cached is not None:
            self._memory[demo_id] = cached
            return cached

        path = self.path_for(demo_id)
        if not path.is_file():
            raise FileNotFoundError(f"Demo feature cache 不存在：{path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        demo = CachedTrainingDemo.from_serializable(payload)
        if demo.demo_id != demo_id:
            raise ValueError(f"Demo cache ID 不一致：请求 {demo_id!r}，文件保存 {demo.demo_id!r}。")
        self._remember(demo)
        return demo

    def _remember(self, demo: CachedTrainingDemo) -> None:
        if self.memory_entries == 0:
            return
        self._memory.pop(demo.demo_id, None)
        self._memory[demo.demo_id] = demo
        while len(self._memory) > self.memory_entries:
            self._memory.popitem(last=False)


@torch.no_grad()
def cache_training_demo(
    demo_id: str,
    global_demo: GlobalDemoClips,
    global_encoder: GlobalDemoEncoder,
    *,
    cache_identity: str,
) -> CachedTrainingDemo:
    """运行一次冻结 S3D，不触碰 Matcher 或 Local 视觉路径。"""
    if not global_encoder.config.freeze_video_backbone:
        raise ValueError("S3D 未冻结时不能生成离线 Global feature cache。")
    features = (
        global_encoder.encode_video_clips_batched(
            global_demo.video.unsqueeze(0),
            global_demo.valid_mask.unsqueeze(0),
        )[0]
        .detach()
        .cpu()
    )
    cached_global = GlobalDemoSample(
        video_features=features,
        states=global_demo.states.detach().cpu(),
        timestamps=global_demo.timestamps.detach().cpu(),
        valid_mask=global_demo.valid_mask.detach().cpu(),
    )
    return CachedTrainingDemo(
        demo_id=demo_id,
        global_demo=cached_global,
        cache_identity=cache_identity,
    )


@torch.no_grad()
def precompute_training_demo(
    demo_id: str,
    video: torch.Tensor,
    raw_states: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    state_normalizer: DemoStateNormalizer,
    global_encoder: GlobalDemoEncoder,
    valid_mask: torch.Tensor | None = None,
    cache_identity: str,
) -> CachedTrainingDemo:
    """从一次 Demo 解码结果生成冻结 Global S3D 缓存。"""
    global_clips = build_global_demo_clips(
        video,
        raw_states,
        timestamps,
        state_normalizer=state_normalizer,
        config=global_encoder.config,
        valid_mask=valid_mask,
    )
    return cache_training_demo(
        demo_id,
        global_clips,
        global_encoder,
        cache_identity=cache_identity,
    )


@dataclass
class SmolVLAICLCollator:
    """组装 LeRobot Query、冻结 Global feature 和 raw Local 窗口。"""

    cache_root: str | Path
    local_demo_reader: Callable[[DemoSampleRef], RawLocalDemoSample]
    state_normalizer: DemoStateNormalizer
    expected_state_dim: int = 32
    memory_entries: int = 1
    _store: DemoFeatureStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = DemoFeatureStore(
            self.cache_root,
            memory_entries=self.memory_entries,
        )

    def __call__(self, samples: list[dict[str, Any] | None]) -> dict[str, Any] | None:
        valid_samples = [sample for sample in samples if sample is not None]
        if not valid_samples:
            return None
        if any(SMOLVLA_ICL_DEMO_REF not in sample for sample in valid_samples):
            raise KeyError(f"磁盘缓存训练样本必须包含 {SMOLVLA_ICL_DEMO_REF!r}。")

        refs = [sample[SMOLVLA_ICL_DEMO_REF] for sample in valid_samples]
        if not all(isinstance(ref, DemoSampleRef) for ref in refs):
            raise TypeError(f"{SMOLVLA_ICL_DEMO_REF} 必须保存 DemoSampleRef。")

        unique_ids: list[str] = []
        id_to_index: dict[str, int] = {}
        inverse: list[int] = []
        for ref in refs:
            assert isinstance(ref, DemoSampleRef)
            if ref.demo_id not in id_to_index:
                id_to_index[ref.demo_id] = len(unique_ids)
                unique_ids.append(ref.demo_id)
            inverse.append(id_to_index[ref.demo_id])

        # 每个 demo_id 在一个 batch 内只调用一次 load；store 的 worker-local
        # LRU 还会消除相邻 batch 的重复磁盘读取。
        cached_demos = [self._store.load(demo_id) for demo_id in unique_ids]
        # DTW 常让相邻 Query frame 停留在同一 anchor。按
        # ``(demo_id, local_anchor)`` 去重可避免一个 batch 内重复切片同一窗口；
        # query_anchor 只用于追踪 Query，不影响 Local Demo 内容。
        local_by_anchor: dict[tuple[str, int], RawLocalDemoSample] = {}
        local_samples: list[RawLocalDemoSample] = []
        for ref in refs:
            key = (ref.demo_id, ref.local_anchor)
            sample = local_by_anchor.get(key)
            if sample is None:
                sample = self.local_demo_reader(ref)
                local_by_anchor[key] = sample
            local_samples.append(sample)

        policy_samples = [
            {
                key: value
                for key, value in sample.items()
                if key
                not in (
                    SMOLVLA_ICL_DEMO_REF,
                    SMOLVLA_ICL_GLOBAL_DEMO,
                    SMOLVLA_ICL_LOCAL_DEMO,
                )
            }
            for sample in valid_samples
        ]
        batch = lerobot_collate_fn(policy_samples)
        if batch is None:
            return None
        batch[SMOLVLA_ICL_GLOBAL_DEMO] = collate_global_demo_samples(
            [cached.global_demo for cached in cached_demos],
            sample_to_demo=inverse,
        )
        batch[SMOLVLA_ICL_LOCAL_DEMO] = collate_raw_local_demo_samples(
            local_samples,
            state_normalizer=self.state_normalizer,
            expected_state_dim=self.expected_state_dim,
        )
        validate_smolvla_icl_training_batch(
            batch,
            expected_state_dim=self.expected_state_dim,
        )
        return batch
