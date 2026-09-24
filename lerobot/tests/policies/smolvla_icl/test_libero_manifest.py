"""LIBERO Manifest 的划分、序列化和防泄漏契约测试."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lerobot.policies.smolvla_icl.configuration_smolvla_icl import DemoAlignmentConfig
from lerobot.policies.smolvla_icl.data.factory import _validate_data_contract, _validate_pairings
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


def test_manifest_excludes_long_and_builds_suite_stratified_unseen_task_split() -> None:
    """默认排除 Long，并在每个 suite 内把 task 按 8/1/1 互斥划分。"""
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

    split_tasks = {
        split: set(manifest.task_indices(split=split))
        for split in ("train", "val", "test")
    }
    assert split_tasks["train"].isdisjoint(split_tasks["val"])
    assert split_tasks["train"].isdisjoint(split_tasks["test"])
    assert split_tasks["val"].isdisjoint(split_tasks["test"])
    for suite_start in (10, 20, 30):
        suite_tasks = set(range(suite_start, suite_start + 10))
        assert len(split_tasks["train"] & suite_tasks) == 8
        assert len(split_tasks["val"] & suite_tasks) == 1
        assert len(split_tasks["test"] & suite_tasks) == 1

    for task_index in range(10, 40):
        task_episodes = [episode for episode in manifest.episodes if episode.task_index == task_index]
        assert len({episode.split for episode in task_episodes}) == 1
        assert {episode.role for episode in task_episodes} == {"demo", "query"}


def test_manifest_is_reproducible_and_seed_changes_assignment() -> None:
    """相同 seed 必须逐 episode 复现，不同 seed 应改变划分."""
    first = _build(seed=7)
    second = _build(seed=7)
    different = _build(seed=8)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.episodes != different.episodes
    assert first.fingerprint != different.fingerprint


def test_equal_val_test_task_ratios_are_balanced_per_suite() -> None:
    """相同 val/test 比例在每个 suite 中各分到一个 unseen task。"""
    manifest = build_libero_manifest(
        _source_episodes(episodes_per_task=34),
        repo_id="lerobot/libero",
        revision="a1aaacb7",
        fps=10.0,
    )

    assert len(manifest.task_indices(split="train")) == 24
    assert len(manifest.task_indices(split="val")) == 3
    assert len(manifest.task_indices(split="test")) == 3


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


def test_manifest_rejects_legacy_seen_task_split(tmp_path) -> None:
    """旧 version=1 Manifest 可能让同一 task 跨 split，必须强制重建。"""
    payload = _build().to_dict()
    payload["version"] = 1
    payload.pop("split_strategy")
    path = tmp_path / "legacy_manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unseen-task"):
        LiberoDataManifest.load(path)


def test_manifest_rejects_task_with_too_few_episodes() -> None:
    """Episode 不足时拒绝通过跨 split 借用 Demo 来兜底."""
    sources = _source_episodes()
    # task 10 只留下一条，无法在其所属 split 内同时产生 Demo 和 Query。
    sources = [episode for episode in sources if episode.task_index != 10 or episode.episode_index % 12 < 1]
    with pytest.raises(ValueError, match="至少需要两条"):
        build_libero_manifest(
            sources,
            repo_id="lerobot/libero",
            revision="a1aaacb7",
            fps=10.0,
        )


def _make_pairing_manifest() -> LiberoDataManifest:
    """构造每个 split 各含一条 Demo/Query 的最小合法 Manifest."""
    assignments = (
        (0, 10, "train", "demo"),
        (1, 10, "train", "query"),
        (2, 11, "val", "demo"),
        (3, 11, "val", "query"),
        (4, 12, "test", "demo"),
        (5, 12, "test", "query"),
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
                task_index=task_index,
                task=f"task {task_index}",
                length=2,
                split=split,
                role=role,
            )
            for index, task_index, split, role in assignments
        ),
    )


def _pairing_metadata() -> SimpleNamespace:
    """返回 factory pairing 边界检查所需的最小 metadata."""
    return SimpleNamespace(
        repo_id="lerobot/libero",
        fps=10.0,
        camera_keys=["observation.images.image"],
        depth_keys=[],
        episodes=[{"dataset_from_index": index * 2, "dataset_to_index": index * 2 + 2} for index in range(6)]
    )


def _training_config(manifest: LiberoDataManifest, *, n_action_steps: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id=manifest.repo_id,
            revision=manifest.revision,
            episodes=None,
            exclude_episodes=None,
            eval_split=0.0,
        ),
        trainable_config=SimpleNamespace(
            n_action_steps=n_action_steps,
            demo_alignment=DemoAlignmentConfig.for_action_chunking(
                control_hz=manifest.fps,
                n_action_steps=n_action_steps,
            ),
        ),
    )


def test_pairing_validation_accepts_same_split_and_task() -> None:
    manifest = _make_pairing_manifest()
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="lerobot/smolvla_base@test",
        image_key=manifest.image_key,
        stats_fingerprint="b" * 64,
        alignment_config=DemoAlignmentConfig.for_action_chunking(
            control_hz=manifest.fps, n_action_steps=1
        ),
        control_hz=manifest.fps,
        n_action_steps=1,
        query_window_replans=4,
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
        stats_fingerprint="b" * 64,
        alignment_config=DemoAlignmentConfig.for_action_chunking(
            control_hz=manifest.fps, n_action_steps=1
        ),
        control_hz=manifest.fps,
        n_action_steps=1,
        query_window_replans=4,
        epochs=(
            {
                1: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
                3: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
            },
        ),
    )

    with pytest.raises(ValueError, match="同一 split"):
        _validate_pairings(sidecar, manifest, _pairing_metadata())


def test_data_contract_rejects_sidecar_from_different_action_chunking() -> None:
    manifest = _make_pairing_manifest()
    sidecar = PairingSidecar(
        manifest_fingerprint=manifest.fingerprint,
        matcher_snapshot="lerobot/smolvla_base@test",
        image_key=manifest.image_key,
        stats_fingerprint="b" * 64,
        alignment_config=DemoAlignmentConfig.for_action_chunking(
            control_hz=manifest.fps, n_action_steps=1
        ),
        control_hz=manifest.fps,
        n_action_steps=1,
        query_window_replans=4,
        epochs=(
            {
                1: EpisodeDemoPairing(manifest.demo_id(0), 0, (0, 1)),
                3: EpisodeDemoPairing(manifest.demo_id(2), 2, (0, 1)),
            },
        ),
    )

    with pytest.raises(ValueError, match="n_action_steps 不一致"):
        _validate_data_contract(
            _training_config(manifest, n_action_steps=5),
            manifest,
            sidecar,
            SimpleNamespace(fingerprint="b" * 64),
            _pairing_metadata(),
        )

    wrong_timing_cfg = _training_config(manifest, n_action_steps=1)
    wrong_timing_cfg.trainable_config.demo_alignment = DemoAlignmentConfig()
    with pytest.raises(ValueError, match="Policy demo_alignment"):
        _validate_data_contract(
            wrong_timing_cfg,
            manifest,
            sidecar,
            SimpleNamespace(fingerprint="b" * 64),
            _pairing_metadata(),
        )
