"""SmolVLA-ICL 专用在线 test 的轻量协议测试。"""

from types import SimpleNamespace

import pytest

from lerobot.policies.smolvla_icl.data.libero_manifest import (
    LiberoDataManifest,
    LiberoManifestConfig,
    LiberoManifestEpisode,
    libero_task_index,
)
from lerobot.policies.smolvla_icl.evaluation import (
    AlignmentTraceCollector,
    select_test_demos,
    summarize_demo_swap,
)


def _manifest() -> LiberoDataManifest:
    assignments = (
        (0, 10, "train", "demo"),
        (1, 10, "train", "query"),
        (2, 11, "val", "demo"),
        (3, 11, "val", "query"),
        (4, 12, "test", "demo"),
        (5, 12, "test", "demo"),
        (6, 12, "test", "query"),
        (7, 12, "test", "query"),
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
                length=10,
                split=split,
                role=role,
            )
            for index, task_index, split, role in assignments
        ),
    )


def test_libero_online_task_id_maps_to_dataset_task_index() -> None:
    assert libero_task_index("libero_goal", 0) == 10
    assert libero_task_index("libero_object", 9) == 29
    assert libero_task_index("libero_spatial", 4) == 34
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        libero_task_index("libero_goal", 10)


def test_test_demo_selection_never_uses_query_or_other_split() -> None:
    manifest = _manifest()
    first = select_test_demos(manifest, task_index=12, seed=3, swap_demo_count=1)
    second = select_test_demos(manifest, task_index=12, seed=3, swap_demo_count=1)

    assert first == second
    assert {episode.episode_index for episode in first} == {4, 5}
    assert all(episode.split == "test" and episode.role == "demo" for episode in first)


def test_alignment_trace_only_records_new_replans() -> None:
    states = iter(
        (
            {"replan_index": 1, "demo_observation_index": 2},
            {"replan_index": 1, "demo_observation_index": 2},
            {"replan_index": 2, "demo_observation_index": 4},
        )
    )
    policy = SimpleNamespace(alignment_diagnostics=lambda: next(states))
    collector = AlignmentTraceCollector()

    collector(policy)
    collector(policy)
    collector(policy)

    assert [item["control_step"] for item in collector.trace] == [0, 2]
    assert [item["demo_observation_index"] for item in collector.trace] == [2, 4]


def test_demo_swap_sensitivity_uses_paired_seeds() -> None:
    results = [
        {
            "demo_episode_index": 4,
            "episodes": [
                {"seed": 10, "success": True},
                {"seed": 11, "success": False},
            ],
        },
        {
            "demo_episode_index": 5,
            "episodes": [
                {"seed": 10, "success": False},
                {"seed": 11, "success": False},
            ],
        },
    ]

    summary = summarize_demo_swap(results)

    assert summary["available"] is True
    assert summary["primary_success_rate"] == 0.5
    assert summary["alternatives"][0]["success_rate_delta"] == -0.5
    assert summary["alternatives"][0]["paired_success_flip_rate"] == 0.5
