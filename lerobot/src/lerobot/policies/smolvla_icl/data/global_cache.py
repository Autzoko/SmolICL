"""Global Demo 冻结 S3D cache 的身份、清单与训练前校验。

每条 ``.pt`` 文件只保存冻结 S3D 的 clip feature，以及后续可训练 State
路径仍需使用的归一化 State、timestamp 和 valid mask。目录级 JSON manifest
负责把这些 Tensor 绑定到明确的数据 revision、camera、S3D snapshot、clip
配置和 State 统计量。训练启动时一次性检查全部文件，并确认 entry 集合覆盖
当前 sidecar 引用的 Demo；DataLoader worker 不再承担发现陈旧缓存的职责。
"""

from __future__ import annotations

import hashlib
import json
import os
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import torch

from ..configuration_smolvla_icl import GlobalEncoderConfig
from .libero_manifest import LiberoDataManifest
from .sidecar import PairingSidecar
from .state import DemoStateNormalizer

if TYPE_CHECKING:
    from .cache import CachedTrainingDemo

GLOBAL_CACHE_MANIFEST_NAME = "cache_manifest.json"

__all__ = [
    "GLOBAL_CACHE_MANIFEST_NAME",
    "GlobalDemoCacheEntry",
    "GlobalDemoCacheManifest",
    "global_cache_feature_config",
    "global_cache_identity",
    "preflight_global_demo_cache",
    "s3d_snapshot_identity",
    "validate_global_demo_cache_entry",
]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def global_cache_feature_config(config: GlobalEncoderConfig) -> dict[str, Any]:
    """只返回真正决定磁盘 cache 内容的 Global 配置。

    Temporal Aggregator、Task Queries 和 fusion 都在训练中更新，不能成为
    离线缓存的一部分。``clip_encode_batch_size`` 只影响显存峰值，也不应导致
    相同 feature 被重复生成。
    """
    payload = asdict(config)
    keys = ["pretrained_backbone", "clip_length", "clip_stride", "state_dim"]
    # 使用预训练权重时，S3D 自带 transform 完全决定 resize/normalize；
    # 自定义 image 参数只在无预训练权重路径中生效。
    if not config.pretrained_backbone:
        keys.extend(("image_size", "image_mean", "image_std"))
    # 立即规范为 JSON 基础类型，避免 tuple 写盘后变为 list，导致同一配置
    # 在重新加载 manifest 时出现伪不一致。
    return json.loads(json.dumps({key: payload[key] for key in keys}))


def s3d_snapshot_identity(config: GlobalEncoderConfig) -> dict[str, Any]:
    """返回无需加载权重即可计算的 TorchVision S3D 身份。"""
    import torchvision
    from torchvision.models.video import S3D_Weights

    weights = S3D_Weights.DEFAULT if config.pretrained_backbone else None
    return {
        "implementation": "torchvision.models.video.s3d",
        "torchvision_version": torchvision.__version__,
        "weights": None if weights is None else weights.name,
        "weights_url": None if weights is None else weights.url,
    }


def _normalization_payload(normalizer: DemoStateNormalizer) -> dict[str, Any]:
    mean, std, eps = normalizer.signature
    return {"mean": list(mean), "std": list(std), "eps": eps}


def _cache_identity_payload(
    manifest: LiberoDataManifest,
    *,
    config: GlobalEncoderConfig,
    state_normalizer: DemoStateNormalizer,
    state_key: str,
    stats_fingerprint: str,
) -> dict[str, Any]:
    return {
        "dataset": {
            "repo_id": manifest.repo_id,
            "revision": manifest.revision,
            "image_key": manifest.image_key,
            "state_key": state_key,
        },
        "feature_config": global_cache_feature_config(config),
        "s3d_snapshot": s3d_snapshot_identity(config),
        "state_normalization": _normalization_payload(state_normalizer),
        "stats_fingerprint": stats_fingerprint,
    }


def global_cache_identity(
    manifest: LiberoDataManifest,
    *,
    config: GlobalEncoderConfig,
    state_normalizer: DemoStateNormalizer,
    state_key: str,
    stats_fingerprint: str,
) -> str:
    """计算单条 cache 文件携带的稳定特征身份。"""
    return _canonical_sha256(
        _cache_identity_payload(
            manifest,
            config=config,
            state_normalizer=state_normalizer,
            state_key=state_key,
            stats_fingerprint=stats_fingerprint,
        )
    )


@dataclass(frozen=True, slots=True)
class GlobalDemoCacheEntry:
    """cache manifest 中一条 Demo 文件的轻量索引。"""

    demo_id: str
    episode_index: int
    episode_length: int
    num_clips: int
    feature_dim: int

    def __post_init__(self) -> None:
        if not self.demo_id:
            raise ValueError("Global cache entry 的 demo_id 不能为空。")
        if self.episode_index < 0 or self.episode_length < 1:
            raise ValueError("Global cache entry 的 episode index/length 无效。")
        if self.num_clips < 1 or self.feature_dim < 1:
            raise ValueError("Global cache entry 的 clip 数和 feature 维度必须大于 0。")


@dataclass(frozen=True, slots=True)
class GlobalDemoCacheManifest:
    """一组可直接供 SmolVLA-ICL 训练使用的 Global Demo cache。"""

    manifest_fingerprint: str
    repo_id: str
    revision: str
    image_key: str
    state_key: str
    feature_config: dict[str, Any]
    s3d_snapshot: dict[str, Any]
    state_normalization: dict[str, Any]
    stats_fingerprint: str
    cache_identity: str
    entries: tuple[GlobalDemoCacheEntry, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_fingerprint", self.manifest_fingerprint),
            ("cache_identity", self.cache_identity),
            ("stats_fingerprint", self.stats_fingerprint),
        ):
            if len(value) != 64 or any(char not in string.hexdigits for char in value):
                raise ValueError(f"{name} 必须是 SHA-256 字符串。")
        if not all((self.repo_id, self.revision, self.image_key, self.state_key)):
            raise ValueError("Global cache manifest 的 Dataset 身份字段不能为空。")
        if not self.entries:
            raise ValueError("Global cache manifest 至少需要一条 Demo entry。")
        demo_ids = [entry.demo_id for entry in self.entries]
        episode_indices = [entry.episode_index for entry in self.entries]
        if len(set(demo_ids)) != len(demo_ids) or len(set(episode_indices)) != len(episode_indices):
            raise ValueError("Global cache manifest 的 demo_id/episode_index 必须唯一。")
        if tuple(sorted(self.entries, key=lambda entry: entry.demo_id)) != self.entries:
            raise ValueError("Global cache manifest entries 必须按 demo_id 排序。")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self._payload_without_fingerprint())

    def _payload_without_fingerprint(self) -> dict[str, Any]:
        return {
            "version": 2,
            "manifest_fingerprint": self.manifest_fingerprint,
            "dataset": {
                "repo_id": self.repo_id,
                "revision": self.revision,
                "image_key": self.image_key,
                "state_key": self.state_key,
            },
            "feature_config": self.feature_config,
            "s3d_snapshot": self.s3d_snapshot,
            "state_normalization": self.state_normalization,
            "stats_fingerprint": self.stats_fingerprint,
            "cache_identity": self.cache_identity,
            "entries": [asdict(entry) for entry in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_fingerprint()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def create(
        cls,
        manifest: LiberoDataManifest,
        *,
        config: GlobalEncoderConfig,
        state_normalizer: DemoStateNormalizer,
        state_key: str,
        stats_fingerprint: str,
        entries: tuple[GlobalDemoCacheEntry, ...],
    ) -> Self:
        """由当前数据与特征配置构造一份排序后的 cache manifest。"""
        identity_payload = _cache_identity_payload(
            manifest,
            config=config,
            state_normalizer=state_normalizer,
            state_key=state_key,
            stats_fingerprint=stats_fingerprint,
        )
        return cls(
            manifest_fingerprint=manifest.fingerprint,
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            image_key=manifest.image_key,
            state_key=state_key,
            feature_config=identity_payload["feature_config"],
            s3d_snapshot=identity_payload["s3d_snapshot"],
            state_normalization=identity_payload["state_normalization"],
            stats_fingerprint=identity_payload["stats_fingerprint"],
            cache_identity=_canonical_sha256(identity_payload),
            entries=tuple(sorted(entries, key=lambda entry: entry.demo_id)),
        )

    def save(self, root: str | Path) -> Path:
        """原子写入目录级 manifest；它的出现代表整组 cache 已经完成。"""
        directory = Path(root).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / GLOBAL_CACHE_MANIFEST_NAME
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        if payload.get("version") != 2:
            raise ValueError("不支持的 Global Demo cache manifest 版本。")
        dataset = payload["dataset"]
        result = cls(
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            repo_id=str(dataset["repo_id"]),
            revision=str(dataset["revision"]),
            image_key=str(dataset["image_key"]),
            state_key=str(dataset["state_key"]),
            feature_config=dict(payload["feature_config"]),
            s3d_snapshot=dict(payload["s3d_snapshot"]),
            state_normalization=dict(payload["state_normalization"]),
            stats_fingerprint=str(payload["stats_fingerprint"]),
            cache_identity=str(payload["cache_identity"]),
            entries=tuple(GlobalDemoCacheEntry(**entry) for entry in payload["entries"]),
        )
        if payload.get("fingerprint") != result.fingerprint:
            raise ValueError("Global Demo cache manifest fingerprint 不匹配。")
        return result

    @classmethod
    def load(cls, root: str | Path) -> Self:
        source = Path(root).expanduser() / GLOBAL_CACHE_MANIFEST_NAME
        if not source.is_file():
            raise FileNotFoundError(f"Global Demo cache manifest 不存在：{source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Global Demo cache manifest 顶层必须是 JSON object。")
        return cls.from_dict(payload)


def _expected_num_clips(num_frames: int, config: GlobalEncoderConfig) -> int:
    if num_frames <= config.clip_length:
        return 1
    last_start = num_frames - config.clip_length
    regular_starts = last_start // config.clip_stride + 1
    return regular_starts + int(last_start % config.clip_stride != 0)


def validate_global_demo_cache_entry(
    cached: CachedTrainingDemo,
    entry: GlobalDemoCacheEntry,
    *,
    config: GlobalEncoderConfig,
    expected_identity: str,
) -> None:
    if cached.cache_identity != expected_identity:
        raise ValueError(f"Demo cache 特征身份不匹配：{entry.demo_id}")
    sample = cached.global_demo
    if sample.video_features.ndim != 2 or not sample.video_features.is_floating_point():
        raise ValueError(f"Demo cache video_features 形状或 dtype 无效：{entry.demo_id}")
    if sample.states.shape != (entry.num_clips, config.clip_length, config.state_dim):
        raise ValueError(f"Demo cache states 形状无效：{entry.demo_id}")
    if sample.timestamps.shape != (entry.num_clips, config.clip_length):
        raise ValueError(f"Demo cache timestamps 形状无效：{entry.demo_id}")
    if sample.valid_mask.shape != (entry.num_clips, config.clip_length):
        raise ValueError(f"Demo cache valid_mask 形状无效：{entry.demo_id}")
    if sample.video_features.shape != (entry.num_clips, entry.feature_dim):
        raise ValueError(f"Demo cache feature shape 与 manifest 不一致：{entry.demo_id}")
    if not sample.states.is_floating_point() or not sample.timestamps.is_floating_point():
        raise ValueError(f"Demo cache State/timestamp 必须是浮点 Tensor：{entry.demo_id}")
    if sample.valid_mask.dtype != torch.bool or not torch.any(sample.valid_mask):
        raise ValueError(f"Demo cache valid_mask 无效：{entry.demo_id}")
    if int(sample.valid_mask.sum()) < entry.episode_length:
        raise ValueError(f"Demo cache 有效帧数少于原 episode 长度：{entry.demo_id}")
    if not torch.all(torch.isfinite(sample.video_features)):
        raise ValueError(f"Demo cache video_features 包含非有限值：{entry.demo_id}")
    if not torch.all(torch.isfinite(sample.states[sample.valid_mask])):
        raise ValueError(f"Demo cache 有效 State 包含非有限值：{entry.demo_id}")
    if not torch.all(torch.isfinite(sample.timestamps[sample.valid_mask])):
        raise ValueError(f"Demo cache 有效 timestamp 包含非有限值：{entry.demo_id}")


def preflight_global_demo_cache(
    root: str | Path,
    *,
    manifest: LiberoDataManifest,
    sidecar: PairingSidecar,
    config: GlobalEncoderConfig,
    state_normalizer: DemoStateNormalizer,
    state_key: str,
    stats_fingerprint: str,
    validate_tensors: bool = True,
) -> GlobalDemoCacheManifest:
    """在 DataLoader 启动前校验训练会访问的全部 cache。

    分布式训练只需主进程完整反序列化 Tensor；其他 rank 在主进程完成并经过
    barrier 后，仅复核 manifest 和文件存在性，避免同时扫描共享存储。
    """
    # 延迟导入避免配置注册期间形成
    # demo_alignment -> data -> factory -> global_cache -> collate 的环。
    from .cache import DemoFeatureStore

    if sidecar.stats_fingerprint != stats_fingerprint:
        raise ValueError("pairing sidecar 与 Global cache 使用的 train-only stats 不一致。")
    cache_manifest = GlobalDemoCacheManifest.load(root)
    expected_identity_payload = _cache_identity_payload(
        manifest,
        config=config,
        state_normalizer=state_normalizer,
        state_key=state_key,
        stats_fingerprint=stats_fingerprint,
    )
    expected_identity = _canonical_sha256(expected_identity_payload)
    expected_fields = {
        "manifest_fingerprint": manifest.fingerprint,
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "image_key": manifest.image_key,
        "state_key": state_key,
        "feature_config": expected_identity_payload["feature_config"],
        "s3d_snapshot": expected_identity_payload["s3d_snapshot"],
        "state_normalization": expected_identity_payload["state_normalization"],
        "stats_fingerprint": stats_fingerprint,
        "cache_identity": expected_identity,
    }
    for field_name, expected in expected_fields.items():
        actual = getattr(cache_manifest, field_name)
        if actual != expected:
            raise ValueError(f"Global cache manifest 的 {field_name} 与当前训练配置不一致。")

    expected_demos = sidecar.demo_id_to_episode
    entries = {entry.demo_id: entry for entry in cache_manifest.entries}
    if set(entries) != set(expected_demos):
        raise ValueError(
            "Global cache manifest 未精确覆盖 pairing sidecar 的 Demo；"
            f"missing={sorted(set(expected_demos) - set(entries))}, "
            f"extra={sorted(set(entries) - set(expected_demos))}。"
        )

    episode_by_index = {episode.episode_index: episode for episode in manifest.episodes}
    store = DemoFeatureStore(root, memory_entries=0)
    feature_dim: int | None = None
    for demo_id in sorted(expected_demos):
        entry = entries[demo_id]
        episode_index = expected_demos[demo_id]
        episode = episode_by_index[episode_index]
        if entry.episode_index != episode_index or entry.episode_length != episode.length:
            raise ValueError(f"Global cache entry 的 episode 身份不一致：{demo_id}")
        expected_num_clips = _expected_num_clips(episode.length, config)
        if entry.num_clips != expected_num_clips:
            raise ValueError(f"Global cache entry 的 clip 数不正确：{demo_id}")
        cache_path = store.path_for(demo_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Global Demo cache 不存在：{cache_path}")
        if validate_tensors:
            cached = store.load(demo_id)
            validate_global_demo_cache_entry(
                cached,
                entry,
                config=config,
                expected_identity=expected_identity,
            )
        if feature_dim is None:
            feature_dim = entry.feature_dim
        elif entry.feature_dim != feature_dim:
            raise ValueError("Global cache 中不同 Demo 的 S3D feature_dim 不一致。")
    return cache_manifest
