"""LIBERO 的 episode 级数据清单与确定性划分工具.

Manifest 是原始 LeRobot Dataset 与 SmolVLA-ICL 离线预处理之间的唯一数据选择
边界。它只读取 metadata，以及数据 Parquet 中很小的 ``episode_index`` 和
``task_index`` 两列；不会读取 RGB、State、Action，也不会解码视频。

构建顺序固定为：

1. 排除未纳入实验的 suite（当前默认排除 LIBERO-Long）；
2. 在每个 suite 内按 task 划分互斥的 train/val/test，形成 unseen-task split；
3. 在每个 task 内把 episode 划分为互斥的 Demo/Query；
4. 保存数据 revision、随机种子和内容指纹，供 Matcher cache、DTW sidecar
   与 Global cache 验证自己是否由同一份数据协议生成。

后续模块不得自行重新抽取 Demo/Query，否则会破坏可复现性，并可能让验证或
测试 Query 使用训练 split 的 Demo。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow.parquet as pq

LiberoSuite = Literal[
    "libero_long",
    "libero_goal",
    "libero_object",
    "libero_spatial",
]
DatasetSplit = Literal["train", "val", "test"]
EpisodeRole = Literal["demo", "query"]

LIBERO_LONG = "libero_long"
LIBERO_GOAL = "libero_goal"
LIBERO_OBJECT = "libero_object"
LIBERO_SPATIAL = "libero_spatial"

DEFAULT_LIBERO_SUITES: tuple[LiberoSuite, ...] = (
    LIBERO_GOAL,
    LIBERO_OBJECT,
    LIBERO_SPATIAL,
)
_ALL_LIBERO_SUITES = frozenset((LIBERO_LONG, LIBERO_GOAL, LIBERO_OBJECT, LIBERO_SPATIAL))
_SPLITS: tuple[DatasetSplit, ...] = ("train", "val", "test")
_ROLES: tuple[EpisodeRole, ...] = ("demo", "query")


def libero_suite_from_task_index(task_index: int) -> LiberoSuite:
    """返回 ``lerobot/libero`` 固定 task 排列对应的 suite.

    该数据集由 Long、Goal、Object、Spatial 四个 10-task suite 顺序合并而成。
    Builder 对范围做严格检查，避免把其他 LIBERO 仓库的 task index 静默套用到
    这份映射。
    """
    if 0 <= task_index < 10:
        return LIBERO_LONG
    if 10 <= task_index < 20:
        return LIBERO_GOAL
    if 20 <= task_index < 30:
        return LIBERO_OBJECT
    if 30 <= task_index < 40:
        return LIBERO_SPATIAL
    raise ValueError(f"LIBERO task_index 必须位于 [0, 39]，实际为 {task_index}。")


def libero_task_index(suite: str, suite_task_id: int) -> int:
    """把 LIBERO simulator 的 suite 内 task id 映射到 Dataset task_index。"""
    offsets = {
        LIBERO_LONG: 0,
        LIBERO_GOAL: 10,
        LIBERO_OBJECT: 20,
        LIBERO_SPATIAL: 30,
    }
    if suite not in offsets:
        raise ValueError(f"未知 LIBERO suite：{suite!r}。")
    if not 0 <= suite_task_id < 10:
        raise ValueError(f"LIBERO suite task id 必须位于 [0, 9]，实际为 {suite_task_id}。")
    return offsets[suite] + suite_task_id


@dataclass(frozen=True, slots=True)
class LiberoManifestConfig:
    """只描述数据划分；三个 split ratio 作用于每个 suite 的 task 数。"""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    demo_ratio: float = 0.2
    seed: int = 42
    included_suites: tuple[LiberoSuite, ...] = DEFAULT_LIBERO_SUITES

    def __post_init__(self) -> None:
        """验证比例、随机种子和 suite 选择."""
        split_ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        if any(not math.isfinite(ratio) or ratio <= 0 for ratio in split_ratios):
            raise ValueError("train/val/test ratio 必须都是有限正数。")
        if not math.isclose(sum(split_ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train/val/test ratio 之和必须等于 1。")
        if not math.isfinite(self.demo_ratio) or not 0 < self.demo_ratio < 1:
            raise ValueError("demo_ratio 必须位于 (0, 1)。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须是非负整数。")
        if not self.included_suites:
            raise ValueError("included_suites 不能为空。")
        if len(set(self.included_suites)) != len(self.included_suites):
            raise ValueError("included_suites 不能包含重复 suite。")
        unknown = set(self.included_suites) - _ALL_LIBERO_SUITES
        if unknown:
            raise ValueError(f"未知的 LIBERO suite：{sorted(unknown)}。")


@dataclass(frozen=True, slots=True)
class LiberoSourceEpisode:
    """从 LeRobot metadata 提取、尚未划分的数据集 episode."""

    episode_index: int
    task_index: int
    task: str
    length: int

    def __post_init__(self) -> None:
        """验证源 episode 的必要索引和长度."""
        if self.episode_index < 0 or self.task_index < 0:
            raise ValueError("episode_index/task_index 不能为负数。")
        if not self.task:
            raise ValueError("LIBERO task 文本不能为空。")
        if self.length < 1:
            raise ValueError("LIBERO episode length 必须大于 0。")


@dataclass(frozen=True, slots=True)
class LiberoManifestEpisode:
    """Manifest 中已经固定 split 与 Demo/Query role 的 episode."""

    episode_index: int
    suite: LiberoSuite
    task_index: int
    task: str
    length: int
    split: DatasetSplit
    role: EpisodeRole


@dataclass(frozen=True, slots=True)
class LiberoDataManifest:
    """一份可序列化、可验证、可复现的 LIBERO 数据实验协议."""

    repo_id: str
    revision: str
    fps: float
    image_key: str
    config: LiberoManifestConfig
    episodes: tuple[LiberoManifestEpisode, ...]

    def __post_init__(self) -> None:
        """验证数据身份、episode 唯一性和 split/role 完整性."""
        if not self.repo_id or not self.revision:
            raise ValueError("Manifest 必须固定非空 repo_id 和 revision。")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Manifest fps 必须是有限正数。")
        if not self.image_key:
            raise ValueError("Manifest image_key 不能为空。")
        if not self.episodes:
            raise ValueError("Manifest 至少需要一个 episode。")

        episode_indices = [episode.episode_index for episode in self.episodes]
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("同一个 episode 不能在 Manifest 中出现多次。")
        if tuple(sorted(self.episodes, key=lambda episode: episode.episode_index)) != self.episodes:
            raise ValueError("Manifest episodes 必须按 episode_index 升序保存。")

        # 同一 task 的 suite 和语言描述必须一致，否则无法保证“同任务配对”。
        task_identity: dict[int, tuple[str, str]] = {}
        task_splits: dict[int, DatasetSplit] = {}
        grouped_roles: dict[int, set[str]] = defaultdict(set)
        for episode in self.episodes:
            if episode.suite not in self.config.included_suites:
                raise ValueError(f"Manifest 包含未启用 suite：{episode.suite}。")
            if episode.split not in _SPLITS or episode.role not in _ROLES:
                raise ValueError("Manifest episode 的 split/role 无效。")
            expected_suite = libero_suite_from_task_index(episode.task_index)
            if episode.suite != expected_suite:
                raise ValueError(
                    f"task_index={episode.task_index} 应属于 {expected_suite}，实际记录为 {episode.suite}。"
                )
            identity = (episode.suite, episode.task)
            previous = task_identity.setdefault(episode.task_index, identity)
            if previous != identity:
                raise ValueError(f"task_index={episode.task_index} 对应了多个任务身份。")
            previous_split = task_splits.setdefault(episode.task_index, episode.split)
            if previous_split != episode.split:
                raise ValueError(
                    f"unseen-task Manifest 中 task_index={episode.task_index} "
                    f"不能同时属于 {previous_split} 和 {episode.split}。"
                )
            grouped_roles[episode.task_index].add(episode.role)

        # 每个 task 只属于一个 split，并且其 episode 同时包含 Demo/Query。
        # Pairing Builder 因此总能在同 split、同 task 内找到 Demo，不需要跨
        # split 兜底，也不会让 seen task 泄漏到 unseen-task validation/test。
        for task_index in task_identity:
            if grouped_roles[task_index] != set(_ROLES):
                raise ValueError(f"task_index={task_index} 必须同时包含 Demo 和 Query episode。")
        if set(task_splits.values()) != set(_SPLITS):
            raise ValueError("unseen-task Manifest 必须同时包含 train、val 和 test task。")

    def episode_indices(
        self,
        *,
        split: DatasetSplit | None = None,
        role: EpisodeRole | None = None,
    ) -> list[int]:
        """返回指定 split/role 的 episode 索引，供 Dataset factory 直接使用."""
        return [
            episode.episode_index
            for episode in self.episodes
            if (split is None or episode.split == split) and (role is None or episode.role == role)
        ]

    def task_indices(self, *, split: DatasetSplit | None = None) -> list[int]:
        """返回指定 split 的唯一 task indices，供训练和在线评测选择 task。"""
        return sorted(
            {
                episode.task_index
                for episode in self.episodes
                if split is None or episode.split == split
            }
        )

    def demo_id(self, episode_index: int) -> str:
        """构造跨 cache 目录仍稳定且不会与其他数据 revision 冲突的 Demo ID."""
        episode = next(
            (item for item in self.episodes if item.episode_index == episode_index),
            None,
        )
        if episode is None:
            raise KeyError(f"Manifest 不包含 episode={episode_index}。")
        if episode.role != "demo":
            raise ValueError(f"episode={episode_index} 不是 Demo episode。")
        return f"{self.repo_id}@{self.revision}:episode_{episode_index:06d}"

    def _payload_without_fingerprint(self) -> dict[str, Any]:
        return {
            "version": 2,
            "split_strategy": "unseen_task",
            "dataset": {
                "repo_id": self.repo_id,
                "revision": self.revision,
                "fps": self.fps,
                "image_key": self.image_key,
            },
            "split_config": asdict(self.config),
            "episodes": [asdict(episode) for episode in self.episodes],
        }

    @property
    def fingerprint(self) -> str:
        """返回 canonical JSON 的 SHA-256，用于拒绝错配的 sidecar/cache."""
        canonical = json.dumps(
            self._payload_without_fingerprint(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """转换为带内容指纹的版本化 JSON payload."""
        payload = self._payload_without_fingerprint()
        payload["fingerprint"] = self.fingerprint
        return payload

    def save(self, path: str | Path) -> Path:
        """保存可人工检查的 JSON Manifest."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """从 JSON payload 恢复并校验 Manifest 指纹."""
        if payload.get("version") != 2 or payload.get("split_strategy") != "unseen_task":
            raise ValueError("Manifest 必须使用 version=2 的 unseen-task split，请重新生成。")
        dataset = payload["dataset"]
        raw_config = dict(payload["split_config"])
        raw_config["included_suites"] = tuple(raw_config["included_suites"])
        manifest = cls(
            repo_id=str(dataset["repo_id"]),
            revision=str(dataset["revision"]),
            fps=float(dataset["fps"]),
            image_key=str(dataset["image_key"]),
            config=LiberoManifestConfig(**raw_config),
            episodes=tuple(LiberoManifestEpisode(**episode) for episode in payload["episodes"]),
        )
        if payload.get("fingerprint") != manifest.fingerprint:
            raise ValueError("LIBERO Manifest fingerprint 不匹配，文件可能被修改或损坏。")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """从磁盘读取并校验 Manifest."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"LIBERO Manifest 不存在：{source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("LIBERO Manifest 顶层必须是 JSON object。")
        return cls.from_dict(payload)


def _derived_seed(seed: int, namespace: str, task_index: int) -> int:
    """生成与 Python hash 随机化无关的 task 局部随机种子."""
    digest = hashlib.sha256(f"{seed}:{namespace}:{task_index}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _allocate_task_split_counts(
    total: int,
    config: LiberoManifestConfig,
    suite: LiberoSuite,
) -> dict[DatasetSplit, int]:
    """按最大余数法分配 suite 内 task，并保证每个 split 至少一个 task。"""
    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]

    tie_breakers = [
        _derived_seed(config.seed, f"task-split-remainder:{suite}:{split}", 0)
        for split in _SPLITS
    ]
    for index in sorted(
        range(len(_SPLITS)),
        key=lambda item: (raw_counts[item] - counts[item], tie_breakers[item]),
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1

    minimum_per_split = 1
    for receiver in range(len(counts)):
        while counts[receiver] < minimum_per_split:
            donors = [index for index, count in enumerate(counts) if count > minimum_per_split]
            if not donors:
                raise ValueError(
                    f"suite={suite} 只有 {total} 个 task，无法构建 unseen-task train/val/test。"
                )
            donor = max(donors, key=lambda index: (counts[index], -index))
            counts[donor] -= 1
            counts[receiver] += 1
    return dict(zip(_SPLITS, counts, strict=True))


def build_libero_manifest(
    source_episodes: Sequence[LiberoSourceEpisode],
    *,
    repo_id: str,
    revision: str,
    fps: float,
    image_key: str = "observation.images.image",
    config: LiberoManifestConfig | None = None,
) -> LiberoDataManifest:
    """从 episode 级 metadata 构建确定性的 LIBERO Manifest."""
    cfg = config or LiberoManifestConfig()
    grouped: dict[int, list[LiberoSourceEpisode]] = defaultdict(list)
    seen_episode_indices: set[int] = set()
    for episode in source_episodes:
        if episode.episode_index in seen_episode_indices:
            raise ValueError(f"重复的 episode_index={episode.episode_index}。")
        seen_episode_indices.add(episode.episode_index)
        suite = libero_suite_from_task_index(episode.task_index)
        if suite in cfg.included_suites:
            grouped[episode.task_index].append(episode)

    expected_tasks = {
        task_index
        for task_index in range(40)
        if libero_suite_from_task_index(task_index) in cfg.included_suites
    }
    missing_tasks = sorted(expected_tasks - grouped.keys())
    if missing_tasks:
        raise ValueError(f"LIBERO Dataset 缺少启用 suite 的 task indices：{missing_tasks}。")

    task_splits: dict[int, DatasetSplit] = {}
    for suite in cfg.included_suites:
        suite_tasks = sorted(
            task_index
            for task_index in grouped
            if libero_suite_from_task_index(task_index) == suite
        )
        split_rng = random.Random(_derived_seed(cfg.seed, f"task-split:{suite}", 0))
        split_rng.shuffle(suite_tasks)
        split_counts = _allocate_task_split_counts(len(suite_tasks), cfg, suite)
        cursor = 0
        for split in _SPLITS:
            for task_index in suite_tasks[cursor : cursor + split_counts[split]]:
                task_splits[task_index] = split
            cursor += split_counts[split]

    output: list[LiberoManifestEpisode] = []
    for task_index in sorted(grouped):
        task_episodes = grouped[task_index]
        task_names = {episode.task for episode in task_episodes}
        if len(task_names) != 1:
            raise ValueError(f"task_index={task_index} 对应了多个 task 文本。")

        if len(task_episodes) < 2:
            raise ValueError(f"task_index={task_index} 至少需要两条 episode 才能划分 Demo/Query。")
        split = task_splits[task_index]
        shuffled = sorted(task_episodes, key=lambda episode: episode.episode_index)
        random.Random(_derived_seed(cfg.seed, f"role:{split}", task_index)).shuffle(shuffled)
        demo_count = math.floor(len(shuffled) * cfg.demo_ratio + 0.5)
        demo_count = min(max(demo_count, 1), len(shuffled) - 1)
        demo_indices = {episode.episode_index for episode in shuffled[:demo_count]}

        suite = libero_suite_from_task_index(task_index)
        output.extend(
            LiberoManifestEpisode(
                episode_index=episode.episode_index,
                suite=suite,
                task_index=episode.task_index,
                task=episode.task,
                length=episode.length,
                split=split,
                role="demo" if episode.episode_index in demo_indices else "query",
            )
            for episode in shuffled
        )

    return LiberoDataManifest(
        repo_id=repo_id,
        revision=revision,
        fps=fps,
        image_key=image_key,
        config=cfg,
        episodes=tuple(sorted(output, key=lambda episode: episode.episode_index)),
    )


def _read_episode_task_indices(
    dataset_root: Path,
    data_path_template: str,
    episode_rows: Sequence[dict[str, Any]],
) -> dict[int, int]:
    """只扫描数据 Parquet 的两个索引列，恢复 episode 到 task 的映射.

    LeRobot v3 的 ``meta/episodes`` 保存长度和数据文件位置，但当前 LIBERO
    转换数据没有在其中重复保存 task。直接读取完整 Dataset 会连 State/Action
    一起物化；这里按文件投影两个 int64 列，开销很小且不会访问视频。
    """
    data_files = {
        dataset_root
        / data_path_template.format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        for episode in episode_rows
    }
    episode_tasks: dict[int, int] = {}
    for path in sorted(data_files):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episode_indices = table.column("episode_index").to_pylist()
        task_indices = table.column("task_index").to_pylist()
        for raw_episode_index, raw_task_index in zip(
            episode_indices,
            task_indices,
            strict=True,
        ):
            episode_index = int(raw_episode_index)
            task_index = int(raw_task_index)
            previous = episode_tasks.setdefault(episode_index, task_index)
            if previous != task_index:
                raise ValueError(f"episode={episode_index} 同时包含 task_index={previous} 和 {task_index}。")
    return episode_tasks


def _read_libero_v3_metadata(
    dataset_root: Path,
    *,
    image_key: str,
) -> tuple[float, list[LiberoSourceEpisode]]:
    """直接读取 LeRobot v3 metadata，不实例化完整 Dataset.

    数据准备机器只需要 PyArrow，不需要安装训练模型、Transformers 或视频解码
    依赖。这里显式处理当前 v3 tasks parquet 把 task 文本保存在索引列中的格式。
    """
    info_path = dataset_root / "meta/info.json"
    tasks_path = dataset_root / "meta/tasks.parquet"
    episode_paths = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    if not info_path.is_file() or not tasks_path.is_file() or not episode_paths:
        raise FileNotFoundError(
            f"{dataset_root} 不是完整的 LeRobot v3 Dataset：缺少 info/tasks/episodes metadata。"
        )

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("LIBERO Manifest Builder 当前只支持 LeRobot v3.0。")
    if int(info.get("total_tasks", -1)) != 40:
        raise ValueError(
            f"当前 suite 映射要求标准 40-task LIBERO Dataset，实际为 {info.get('total_tasks')}。"
        )
    features = info.get("features", {})
    image_feature = features.get(image_key)
    if not isinstance(image_feature, dict) or image_feature.get("dtype") not in ("image", "video"):
        raise ValueError(f"image_key={image_key!r} 必须是 Dataset 中的 RGB image/video feature。")

    task_table = pq.read_table(tasks_path)
    text_columns = [name for name in task_table.column_names if name != "task_index"]
    if len(text_columns) != 1:
        raise ValueError(
            f"tasks.parquet 必须包含 task_index 和唯一 task 文本列，实际为 {task_table.column_names}。"
        )
    task_names = {
        int(task_index): str(task)
        for task_index, task in zip(
            task_table.column("task_index").to_pylist(),
            task_table.column(text_columns[0]).to_pylist(),
            strict=True,
        )
    }

    # episode metadata 很小，可以一次合并；这里只选择构建 Manifest 所需的四列。
    episode_table = pq.read_table(
        episode_paths,
        columns=[
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
        ],
    )
    episode_rows = episode_table.to_pylist()
    if len(episode_rows) != int(info.get("total_episodes", -1)):
        raise ValueError("meta/episodes 的行数与 info.json total_episodes 不一致。")

    data_path_template = str(info["data_path"])
    episode_tasks = _read_episode_task_indices(
        dataset_root,
        data_path_template,
        episode_rows,
    )
    source_episodes: list[LiberoSourceEpisode] = []
    for episode in episode_rows:
        episode_index = int(episode["episode_index"])
        try:
            task_index = episode_tasks[episode_index]
            task = task_names[task_index]
        except KeyError as error:
            raise ValueError(f"无法为 episode={episode_index} 解析唯一 task。") from error
        source_episodes.append(
            LiberoSourceEpisode(
                episode_index=episode_index,
                task_index=task_index,
                task=task,
                length=int(episode["length"]),
            )
        )
    return float(info["fps"]), source_episodes


def build_libero_manifest_from_dataset(
    dataset_root: str | Path,
    *,
    repo_id: str,
    revision: str,
    output_path: str | Path | None = None,
    image_key: str = "observation.images.image",
    config: LiberoManifestConfig | None = None,
) -> LiberoDataManifest:
    """读取本地 LeRobot v3 LIBERO metadata 并构建 Manifest."""
    root = Path(dataset_root).expanduser()
    fps, source_episodes = _read_libero_v3_metadata(root, image_key=image_key)

    manifest = build_libero_manifest(
        source_episodes,
        repo_id=repo_id,
        revision=revision,
        fps=fps,
        image_key=image_key,
        config=config,
    )
    if output_path is not None:
        manifest.save(output_path)
    return manifest


def _count_by_split_and_role(
    episodes: Iterable[LiberoManifestEpisode],
) -> dict[str, dict[str, int]]:
    counts = {split: dict.fromkeys(_ROLES, 0) for split in _SPLITS}
    for episode in episodes:
        counts[episode.split][episode.role] += 1
    return counts


def main() -> None:
    """命令行入口；适合在保存 LIBERO 数据的训练服务器上直接运行."""
    parser = argparse.ArgumentParser(description="构建 SmolVLA-ICL 的 LIBERO 数据 Manifest。")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image-key", default="observation.images.image")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="每个 suite 的 train task 比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="每个 suite 的 val task 比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="每个 suite 的 test task 比例")
    parser.add_argument("--demo-ratio", type=float, default=0.2, help="每个 task 内的 Demo episode 比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = LiberoManifestConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        demo_ratio=args.demo_ratio,
        seed=args.seed,
    )
    manifest = build_libero_manifest_from_dataset(
        args.dataset_root,
        repo_id=args.repo_id,
        revision=args.revision,
        output_path=args.output,
        image_key=args.image_key,
        config=config,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser()),
                "fingerprint": manifest.fingerprint,
                "episodes": len(manifest.episodes),
                "task_counts": {
                    split: len(manifest.task_indices(split=split))
                    for split in _SPLITS
                },
                "counts": _count_by_split_and_role(manifest.episodes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_LIBERO_SUITES",
    "LiberoDataManifest",
    "LiberoManifestConfig",
    "LiberoManifestEpisode",
    "LiberoSourceEpisode",
    "build_libero_manifest",
    "build_libero_manifest_from_dataset",
    "libero_suite_from_task_index",
    "libero_task_index",
]
