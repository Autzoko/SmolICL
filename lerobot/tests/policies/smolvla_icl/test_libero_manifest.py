"""LIBERO Manifest 的划分、序列化和防泄漏契约测试."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lerobot.policies.smolvla_icl.data.factory import _validate_pairings
from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
    LiberoSourceEpisode,
    build_libero_manifest,
)
from lerobot.policies.smolvla_icl.data.sidecar import EpisodeDemoPairing, PairingSidecar


def _source_episodes(episodes_per_task: int = 12) -> list[LiberoSourceEpisode]:
    """构造完整 40-task metadata；每个 task 的 episode index 均不重叠."""
    return [
        LiberoSourceEpisode(
            episode_index=task_index * episodes_per_task + offset,
            task_index=task_index,
            task=f"task {task_index}",
            length=100 + offset,
        )
        for task_index in range(40)
        for offset in range(episodes_per_task)
    ]


def _build(*, seed: int = 42) -> LiberoDataManifest:
    return build_libero_manifest(
        _source_episodes(),
        repo_id="lerobot/libero",
        revision="a1aaacb7",
        fps=10.0,
        config=LiberoManifestConfig(seed=seed),
    )


def test_manifest_excludes_long_and_stratifies_every_task() -> None:
    """默认排除 Long，并在每个保留 task 内完成两级分层划分."""
    manifest = _build()

    # 只保留 Goal/Object/Spatial 的 30 个 task；Long 的 task_index=0..9
    # 完全不会进入后续 cache 或 sidecar 构建。
    assert len(manifest.episodes) == 30 * 12
    assert {episode.task_index for episode in manifest.episodes} == set(range(10, 40))
    assert {episode.suite for episode in manifest.episodes} == {
        "libero_goal",
        "libero_object",
        "libero_spatial",
    }

    for task_index in range(10, 40):
        task_episodes = [episode for episode in manifest.episodes if episode.task_index == task_index]
        # 12 条 episode 在 80/10/10 且每个 split 至少两条的约束下得到 8/2/2。
        assert sum(episode.split == "train" for episode in task_episodes) == 8
        assert sum(episode.split == "val" for episode in task_episodes) == 2
        assert sum(episode.split == "test" for episode in task_episodes) == 2
        for split in ("train", "val", "test"):
            roles = {episode.role for episode in task_episodes if episode.split == split}
            assert roles == {"demo", "query"}


def test_manifest_is_reproducible_and_seed_changes_assignment() -> None:
    """相同 seed 必须逐 episode 复现，不同 seed 应改变划分."""
    first = _build(seed=7)
    second = _build(seed=7)
    different = _build(seed=8)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.episodes != different.episodes
    assert first.fingerprint != different.fingerprint


def test_equal_val_test_ratios_do_not_systematically_break_ties_to_one_split() -> None:
    """相同 val/test 比例的逐 task 余数平票不会总偏向同一 split."""
    manifest = build_libero_manifest(
        _source_episodes(episodes_per_task=34),
        repo_id="lerobot/libero",
        revision="a1aaacb7",
        fps=10.0,
    )

    val_count = len(manifest.episode_indices(split="val"))
    test_count = len(manifest.episode_indices(split="test"))
    assert abs(val_count - test_count) <= 10


def test_manifest_round_trip_and_fingerprint_validation(tmp_path) -> None:
    """JSON 往返保持内容一致，篡改内容会触发指纹错误."""
    manifest = _build()
    path = manifest.save(tmp_path / "libero_manifest.json")

    assert LiberoDataManifest.load(path) == manifest

    payload = json.loads(path.read_text())
    payload["episodes"][0]["length"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        LiberoDataManifest.load(path)


def test_manifest_rejects_task_with_too_few_episodes() -> None:
    """Episode 不足时拒绝通过跨 split 借用 Demo 来兜底."""
    sources = _source_episodes()
    # task 10 只留下五条，无法让三个 split 都同时具有 Demo 和 Query。
    sources = [episode for episode in sources if episode.task_index != 10 or episode.episode_index % 12 < 5]
    with pytest.raises(ValueError, match="无法让 train/val/test"):
        build_libero_manifest(
            sources,
            repo_id="lerobot/libero",
            revision="a1aaacb7",
            fps=10.0,
        )


def _make_pairing_manifest() -> LiberoDataManifest:
    """构造每个 split 各含一条 Demo/Query 的最小合法 Manifest."""
    assignments = (
        (0, "train", "demo"),
        (1, "train", "query"),
        (2, "val", "demo"),
        (3, "val", "query"),
        (4, "test", "demo"),
        (5, "test", "query"),
    )
    return LiberoDataManifest(
        repo_id="lerobot/libero",
        revision="revision",
        fps=10.0,
        image_key="observation.images.image",
        config=LiberoManifestConfig(),
        episodes=tuple(
            LiberoManifestEpisode(
                episode_index=index,
                suite="libero_goal",
                task_index=10,
                task="pick",
                length=2,
                split=split,
                role=role,
            )
            for index, split, role in assignments
        ),
    )


def _pairing_metadata() -> SimpleNamespace:
    """返回 factory pairing 边界检查所需的最小 metadata."""
    return SimpleNamespace(
        episodes=[{"dataset_from_index": index * 2, "dataset_to_index": index * 2 + 2} for index in range(6)]
    )


def test_pairing_validation_accepts_same_split_and_task() -> None:
    manifest = _make_pairing_manifest()
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="lerobot/smolvla_base@test",
        image_key=manifest.image_key,
        epochs=(
            {
                1: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
                3: EpisodeDemoPairing(manifest.demo_id(2), 2, (0, 1)),
            },
        ),
    )

    _validate_pairings(sidecar, manifest, _pairing_metadata())


def test_pairing_validation_rejects_cross_split_demo() -> None:
    manifest = _make_pairing_manifest()
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="lerobot/smolvla_base@test",
        image_key=manifest.image_key,
        epochs=(
            {
                1: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
                3: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
            },
        ),
    )

    with pytest.raises(ValueError, match="同一 split"):
        _validate_pairings(sidecar, manifest, _pairing_metadata())
