"""SmolVLA-ICL 离线 DTW 配对 sidecar。

Sidecar 只保存轻量索引，不包含 RGB、State 或模型特征。每个 epoch
中，一条 Query episode 只对应一条 Demo；``local_anchors`` 则给出
该 Query episode 每个 frame 的离线 DTW 匹配位置。
"""

from __future__ import annotations

import json
import math
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from ..configuration_smolvla_icl import DemoAlignmentConfig
from .contracts import DemoReferenceResolver, DemoSampleRef

__all__ = [
    "EpisodeDemoPairing",
    "PairingSidecar",
    "PairingSidecarResolver",
]


@dataclass(frozen=True, slots=True)
class EpisodeDemoPairing:
    """一条 Query episode 在某个 epoch 内的固定 Demo 配对。"""

    demo_id: str
    demo_episode_index: int
    local_anchors: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.demo_id:
            raise ValueError("sidecar demo_id 不能为空。")
        if self.demo_episode_index < 0:
            raise ValueError("sidecar demo_episode_index 不能为负数。")
        if not self.local_anchors or any(anchor < 0 for anchor in self.local_anchors):
            raise ValueError("sidecar local_anchors 必须是非空非负整数序列。")


@dataclass(frozen=True, slots=True)
class PairingSidecar:
    """可序列化的 epoch 级 Demo 配对表。

    ``epochs[e][query_episode_index]`` 存放该 Query episode 在 epoch
    ``e`` 使用的唯一 Demo 以及逐帧 Local anchor。``image_key`` 使用
    原始 LeRobot Dataset 字段名，便于 Local reader 在 processor 前解码。
    """

    manifest_fingerprint: str
    matcher_snapshot: str
    image_key: str
    stats_fingerprint: str
    alignment_config: DemoAlignmentConfig
    control_hz: float
    n_action_steps: int
    query_window_replans: int
    epochs: tuple[dict[int, EpisodeDemoPairing], ...]

    def __post_init__(self) -> None:
        if len(self.manifest_fingerprint) != 64 or any(
            char not in string.hexdigits for char in self.manifest_fingerprint
        ):
            raise ValueError("sidecar manifest_fingerprint 必须是 SHA-256 字符串。")
        if not self.matcher_snapshot:
            raise ValueError("sidecar matcher_snapshot 不能为空。")
        if not self.image_key:
            raise ValueError("sidecar image_key 不能为空。")
        if len(self.stats_fingerprint) != 64 or any(
            char not in string.hexdigits for char in self.stats_fingerprint
        ):
            raise ValueError("sidecar stats_fingerprint 必须是 SHA-256 字符串。")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("sidecar control_hz 必须是有限正数。")
        if self.n_action_steps < 1:
            raise ValueError("sidecar n_action_steps 必须大于 0。")
        if self.query_window_replans < 2:
            raise ValueError("sidecar query_window_replans 至少为 2。")
        expected_alignment = self.alignment_config.bind_action_chunking(
            control_hz=self.control_hz,
            n_action_steps=self.n_action_steps,
            query_window_replans=self.query_window_replans,
        )
        if self.alignment_config != expected_alignment:
            raise ValueError(
                "sidecar alignment_config 与 control_hz/n_action_steps 不一致："
                f"期望 alignment_hz={expected_alignment.alignment_hz:g}, "
                f"window_duration_s={expected_alignment.window_duration_s:g}。"
            )
        if not self.epochs:
            raise ValueError("sidecar 至少需要一个 epoch。")

        demo_to_episode: dict[str, int] = {}
        normalized_epochs: list[dict[int, EpisodeDemoPairing]] = []
        query_episodes: set[int] | None = None
        for epoch in self.epochs:
            if not epoch:
                raise ValueError("sidecar 的每个 epoch 至少需要一条 Query episode。")
            normalized: dict[int, EpisodeDemoPairing] = {}
            for query_episode, pairing in epoch.items():
                if isinstance(query_episode, bool) or query_episode < 0:
                    raise ValueError("sidecar Query episode index 必须是非负整数。")
                if not isinstance(pairing, EpisodeDemoPairing):
                    raise TypeError("sidecar epoch 的 value 必须是 EpisodeDemoPairing。")
                previous = demo_to_episode.setdefault(pairing.demo_id, pairing.demo_episode_index)
                if previous != pairing.demo_episode_index:
                    raise ValueError(f"demo_id={pairing.demo_id!r} 在 sidecar 中对应了多个 episode。")
                if query_episode == pairing.demo_episode_index:
                    raise ValueError(f"Query episode={query_episode} 不能使用自身作为 Demo。")
                normalized[int(query_episode)] = pairing
            current_query_episodes = set(normalized)
            if query_episodes is None:
                query_episodes = current_query_episodes
            elif current_query_episodes != query_episodes:
                raise ValueError("sidecar 的所有 epoch 必须覆盖同一组 Query episodes。")
            normalized_epochs.append(normalized)
        assert query_episodes is not None
        overlap = sorted(query_episodes & set(demo_to_episode.values()))
        if overlap:
            raise ValueError(
                f"sidecar 的 Query/Demo episode 子集必须不相交，当前重叠：{overlap}。"
            )
        object.__setattr__(self, "epochs", tuple(normalized_epochs))

    @property
    def query_episode_indices(self) -> list[int]:
        """返回训练 Dataset 应保留的 Query episode 子集。"""
        return sorted(self.epochs[0])

    @property
    def demo_id_to_episode(self) -> dict[str, int]:
        """返回 Local reader 所需的唯一 Demo ID 映射。"""
        result: dict[str, int] = {}
        for epoch in self.epochs:
            for pairing in epoch.values():
                result[pairing.demo_id] = pairing.demo_episode_index
        return result

    @property
    def demo_episode_indices(self) -> list[int]:
        """返回需要打开的 Demo episode，避免 Dataset 加载其他 episode。"""
        return sorted(set(self.demo_id_to_episode.values()))

    def to_dict(self) -> dict[str, Any]:
        """转换为紧凑且可人工检查的 JSON 结构。"""
        return {
            "version": 4,
            "manifest_fingerprint": self.manifest_fingerprint,
            "matcher_snapshot": self.matcher_snapshot,
            "image_key": self.image_key,
            "stats_fingerprint": self.stats_fingerprint,
            "alignment": {
                "control_hz": self.control_hz,
                "n_action_steps": self.n_action_steps,
                "query_window_replans": self.query_window_replans,
                "config": asdict(self.alignment_config),
            },
            "epochs": [
                {
                    str(query_episode): {
                        "demo_id": pairing.demo_id,
                        "demo_episode_index": pairing.demo_episode_index,
                        "local_anchors": list(pairing.local_anchors),
                    }
                    for query_episode, pairing in sorted(epoch.items())
                }
                for epoch in self.epochs
            ],
        }

    def save(self, path: str | Path) -> Path:
        """保存 sidecar；目录由调用方明确指定。"""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """从 JSON payload 构建 sidecar。"""
        if payload.get("version") != 4:
            raise ValueError(
                "不支持的 SmolVLA-ICL pairing sidecar 版本；"
                "旧 sidecar 未绑定 train-only stats/action chunking，请重新生成。"
            )
        raw_alignment = payload.get("alignment")
        if not isinstance(raw_alignment, dict) or not isinstance(raw_alignment.get("config"), dict):
            raise TypeError("sidecar alignment 必须包含 control/action chunking 和 config。")
        alignment_payload = dict(raw_alignment["config"])
        for key in ("matching_state_excluded_indices", "matching_state_velocity_scale"):
            if alignment_payload.get(key) is not None:
                alignment_payload[key] = tuple(alignment_payload[key])
        raw_epochs = payload.get("epochs")
        if not isinstance(raw_epochs, list):
            raise TypeError("sidecar epochs 必须是 list。")

        epochs: list[dict[int, EpisodeDemoPairing]] = []
        for raw_epoch in raw_epochs:
            if not isinstance(raw_epoch, dict):
                raise TypeError("sidecar 的每个 epoch 必须是 object。")
            epoch: dict[int, EpisodeDemoPairing] = {}
            for raw_query_episode, raw_pairing in raw_epoch.items():
                if not isinstance(raw_pairing, dict):
                    raise TypeError("sidecar Query episode 配对必须是 object。")
                query_episode = int(raw_query_episode)
                epoch[query_episode] = EpisodeDemoPairing(
                    demo_id=str(raw_pairing["demo_id"]),
                    demo_episode_index=int(raw_pairing["demo_episode_index"]),
                    local_anchors=tuple(int(value) for value in raw_pairing["local_anchors"]),
                )
            epochs.append(epoch)
        return cls(
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
            matcher_snapshot=str(payload["matcher_snapshot"]),
            image_key=str(payload["image_key"]),
            stats_fingerprint=str(payload["stats_fingerprint"]),
            alignment_config=DemoAlignmentConfig(**alignment_payload),
            control_hz=float(raw_alignment["control_hz"]),
            n_action_steps=int(raw_alignment["n_action_steps"]),
            query_window_replans=int(raw_alignment["query_window_replans"]),
            epochs=tuple(epochs),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """从磁盘读取 sidecar。"""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"pairing sidecar 不存在：{source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("pairing sidecar 顶层必须是 JSON object。")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PairingSidecarResolver(DemoReferenceResolver):
    """将 Query frame 解析为 sidecar 中预计算的 Demo anchor。"""

    sidecar: PairingSidecar

    def validate_query_episodes(self, episode_indices: list[int]) -> None:
        """在 DataLoader 启动前确认每个 epoch 都覆盖 Query 集。"""
        required = set(episode_indices)
        for epoch_index, epoch in enumerate(self.sidecar.epochs):
            missing = sorted(required - epoch.keys())
            if missing:
                raise ValueError(f"pairing sidecar epoch={epoch_index} 缺少 Query episodes: {missing}。")

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
        # 一份 sidecar 可以预生成若干种 epoch 配对并循环复用；
        # 这保证 resume 后的配对仍只由训练 epoch 决定。
        sidecar_epoch = self.sidecar.epochs[epoch % len(self.sidecar.epochs)]
        try:
            pairing = sidecar_epoch[episode_index]
        except KeyError as error:
            raise KeyError(
                f"pairing sidecar 未配置 epoch={epoch}, query episode={episode_index}。"
            ) from error
        if not 0 <= frame_index < len(pairing.local_anchors):
            raise IndexError(
                f"Query frame={frame_index} 超出 sidecar 中 episode={episode_index} "
                f"的 {len(pairing.local_anchors)} 个 anchors。"
            )
        return DemoSampleRef(
            demo_id=pairing.demo_id,
            query_anchor=frame_index,
            local_anchor=pairing.local_anchors[frame_index],
        )
