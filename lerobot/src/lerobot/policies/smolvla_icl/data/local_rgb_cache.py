"""训练期 Local Demo 的 episode 级 CPU uint8 RGB cache。"""

from __future__ import annotations

import hashlib
import json
import os
import string
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from .libero_manifest import LiberoDataManifest
from .sidecar import PairingSidecar

__all__ = [
    "LocalRGBFrameCacheManifest",
    "LocalRGBFrameCacheStore",
    "LocalRGBFrameEntry",
    "local_rgb_cache_identity",
    "preflight_local_rgb_cache",
]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _identity_payload(manifest: LiberoDataManifest) -> dict[str, Any]:
    return {
        "version": 1,
        "manifest_fingerprint": manifest.fingerprint,
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "image_key": manifest.image_key,
        "dtype": "uint8",
        "layout": "TCHW",
    }


def local_rgb_cache_identity(manifest: LiberoDataManifest) -> str:
    """返回由 Dataset revision、camera 和 Manifest 共同确定的 cache 身份。"""
    return _canonical_sha256(_identity_payload(manifest))


@dataclass(frozen=True, slots=True)
class LocalRGBFrameEntry:
    demo_id: str
    episode_index: int
    episode_length: int
    shape: tuple[int, int, int, int]
    file: str

    def __post_init__(self) -> None:
        if not self.demo_id or self.episode_index < 0 or self.episode_length < 1:
            raise ValueError("Local RGB cache entry 的 Demo/episode 身份无效。")
        if len(self.shape) != 4 or self.shape[0] != self.episode_length or self.shape[1] != 3:
            raise ValueError("Local RGB cache shape 必须是与 episode 等长的 (T,3,H,W)。")
        if any(size < 1 for size in self.shape) or not self.file:
            raise ValueError("Local RGB cache entry 的 shape/file 无效。")


@dataclass(frozen=True, slots=True)
class LocalRGBFrameCacheManifest:
    manifest_fingerprint: str
    repo_id: str
    revision: str
    image_key: str
    cache_identity: str
    entries: tuple[LocalRGBFrameEntry, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_fingerprint", self.manifest_fingerprint),
            ("cache_identity", self.cache_identity),
        ):
            if len(value) != 64 or any(char not in string.hexdigits for char in value):
                raise ValueError(f"{name} 必须是 SHA-256 字符串。")
        if not all((self.repo_id, self.revision, self.image_key)) or not self.entries:
            raise ValueError("Local RGB cache manifest 的 Dataset 身份和 entries 不能为空。")
        if tuple(sorted(self.entries, key=lambda item: item.demo_id)) != self.entries:
            raise ValueError("Local RGB cache entries 必须按 demo_id 排序。")
        if len({item.demo_id for item in self.entries}) != len(self.entries):
            raise ValueError("Local RGB cache demo_id 必须唯一。")

    @classmethod
    def create(
        cls,
        manifest: LiberoDataManifest,
        entries: tuple[LocalRGBFrameEntry, ...],
    ) -> Self:
        return cls(
            manifest_fingerprint=manifest.fingerprint,
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            image_key=manifest.image_key,
            cache_identity=local_rgb_cache_identity(manifest),
            entries=tuple(sorted(entries, key=lambda item: item.demo_id)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": 1,
            "manifest_fingerprint": self.manifest_fingerprint,
            "dataset": {
                "repo_id": self.repo_id,
                "revision": self.revision,
                "image_key": self.image_key,
            },
            "cache_identity": self.cache_identity,
            "entries": [asdict(entry) for entry in self.entries],
        }
        payload["fingerprint"] = _canonical_sha256(payload)
        return payload

    def save(self, root: str | Path) -> Path:
        target = Path(root).expanduser() / "cache_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
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
    def load(cls, root: str | Path) -> Self:
        path = Path(root).expanduser() / "cache_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Local RGB cache manifest 不存在：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = payload.pop("fingerprint", None)
        if fingerprint != _canonical_sha256(payload):
            raise ValueError("Local RGB cache manifest fingerprint 不匹配。")
        if payload.get("version") != 1:
            raise ValueError("不支持的 Local RGB cache manifest 版本。")
        dataset = payload["dataset"]
        return cls(
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            repo_id=str(dataset["repo_id"]),
            revision=str(dataset["revision"]),
            image_key=str(dataset["image_key"]),
            cache_identity=str(payload["cache_identity"]),
            entries=tuple(
                LocalRGBFrameEntry(
                    demo_id=str(entry["demo_id"]),
                    episode_index=int(entry["episode_index"]),
                    episode_length=int(entry["episode_length"]),
                    shape=tuple(int(value) for value in entry["shape"]),
                    file=str(entry["file"]),
                )
                for entry in payload["entries"]
            ),
        )


def _entry_path(root: Path, entry: LocalRGBFrameEntry) -> Path:
    path = (root / entry.file).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Local RGB cache file 越出 cache root：{entry.file}")
    return path


def preflight_local_rgb_cache(
    root: str | Path,
    *,
    manifest: LiberoDataManifest,
    sidecar: PairingSidecar,
) -> LocalRGBFrameCacheManifest:
    """训练前校验身份、Demo 覆盖以及所有 NPY header。"""
    cache_root = Path(root).expanduser()
    cache_manifest = LocalRGBFrameCacheManifest.load(cache_root)
    expected_identity = local_rgb_cache_identity(manifest)
    expected_fields = {
        "manifest_fingerprint": manifest.fingerprint,
        "repo_id": manifest.repo_id,
        "revision": manifest.revision,
        "image_key": manifest.image_key,
        "cache_identity": expected_identity,
    }
    for name, expected in expected_fields.items():
        if getattr(cache_manifest, name) != expected:
            raise ValueError(f"Local RGB cache manifest 的 {name} 与当前 Dataset 不一致。")

    entries = {entry.demo_id: entry for entry in cache_manifest.entries}
    expected_demos = sidecar.demo_id_to_episode
    if set(entries) != set(expected_demos):
        raise ValueError("Local RGB cache 必须精确覆盖 pairing sidecar 的 Demo 集合。")
    episodes = {episode.episode_index: episode for episode in manifest.episodes}
    for demo_id, episode_index in expected_demos.items():
        entry = entries[demo_id]
        episode = episodes[episode_index]
        if entry.episode_index != episode_index or entry.episode_length != episode.length:
            raise ValueError(f"Local RGB cache episode 身份不一致：{demo_id}")
        array = np.load(_entry_path(cache_root, entry), mmap_mode="r", allow_pickle=False)
        if array.dtype != np.uint8 or tuple(array.shape) != entry.shape:
            raise ValueError(f"Local RGB cache dtype/shape 不一致：{demo_id}")
    return cache_manifest


class LocalRGBFrameCacheStore:
    """按 episode mmap CPU uint8 帧；LRU 只管理文件句柄。"""

    def __init__(
        self,
        root: str | Path,
        cache_manifest: LocalRGBFrameCacheManifest,
        *,
        open_entries: int = 2,
    ) -> None:
        if open_entries < 1:
            raise ValueError("Local RGB mmap open_entries 必须大于 0。")
        self.root = Path(root).expanduser()
        self.entries = {entry.demo_id: entry for entry in cache_manifest.entries}
        self.open_entries = open_entries
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def read(self, demo_id: str, local_indices: Tensor) -> Tensor:
        """读取 episode 内下标并复制为可安全传给 PyTorch 的 CPU Tensor。"""
        if local_indices.ndim != 1 or local_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Local RGB cache indices 必须是一维整数 Tensor。")
        entry = self.entries.get(demo_id)
        if entry is None:
            raise KeyError(f"Local RGB cache 中不存在 Demo: {demo_id!r}。")
        indices = local_indices.detach().cpu().numpy()
        if len(indices) == 0 or indices.min() < 0 or indices.max() >= entry.episode_length:
            raise IndexError(f"Local RGB cache indices 超出 Demo {demo_id!r} 边界。")
        array = self._open(demo_id, entry)
        # 高级索引产生独立的紧凑 CPU buffer；模型仍在 forward 内在线编码。
        return torch.from_numpy(np.asarray(array[indices], dtype=np.uint8))

    def _open(self, demo_id: str, entry: LocalRGBFrameEntry) -> np.ndarray:
        cached = self._arrays.pop(demo_id, None)
        if cached is None:
            cached = np.load(_entry_path(self.root, entry), mmap_mode="r", allow_pickle=False)
        self._arrays[demo_id] = cached
        while len(self._arrays) > self.open_entries:
            self._arrays.popitem(last=False)
        return cached
