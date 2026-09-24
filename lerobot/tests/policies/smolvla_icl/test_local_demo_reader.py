"""Local Demo reader 的窗口边界与速度上下文测试。"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.policies.smolvla_icl.data.collate import collate_raw_local_demo_samples
from lerobot.policies.smolvla_icl.data.contracts import DemoSampleRef
from lerobot.policies.smolvla_icl.data.reader import LeRobotLocalDemoReader
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer


class _FakeTable:
    def __init__(self, columns: dict[str, list[torch.Tensor]]) -> None:
        self.columns = columns

    def select_columns(self, keys: str | list[str]) -> _FakeTable:
        selected = [keys] if isinstance(keys, str) else keys
        return type(self)({key: self.columns[key] for key in selected})

    def __getitem__(self, indices: list[int]) -> dict[str, list[torch.Tensor]]:
        return {key: [values[index] for index in indices] for key, values in self.columns.items()}


class _FakeRGBCache:
    def __init__(self) -> None:
        self.entries = {"demo": SimpleNamespace(episode_index=0)}
        self.frames = torch.arange(8 * 3 * 2 * 2, dtype=torch.uint8).reshape(8, 3, 2, 2)

    def read(self, demo_id: str, indices: torch.Tensor) -> torch.Tensor:
        assert demo_id == "demo"
        return self.frames[indices]


def _reader() -> LeRobotLocalDemoReader:
    timestamps = [0.0, 0.1, 0.21, 0.31, 0.42, 0.52, 0.63, 0.73]
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            camera_keys=["rgb"],
            depth_keys=[],
            features={"state": {}},
            episodes=[{"dataset_from_index": 0, "dataset_to_index": 8}],
            fps=10.0,
        ),
        hf_dataset=_FakeTable(
            {
                "state": [torch.tensor([float(index)]) for index in range(8)],
                "timestamp": [torch.tensor(value, dtype=torch.float64) for value in timestamps],
            }
        ),
        absolute_to_relative_idx=None,
    )
    return LeRobotLocalDemoReader(
        dataset=dataset,
        demo_id_to_episode={"demo": 0},
        image_key="rgb",
        state_key="state",
        chunk_size=4,
        anchor_position=1,
        rgb_cache=_FakeRGBCache(),
    )


def test_local_reader_reads_preceding_state_without_adding_token() -> None:
    sample = _reader()(DemoSampleRef("demo", query_anchor=0, local_anchor=4))

    assert sample.images.shape[0] == 4
    assert sample.states[:, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert sample.previous_state is not None
    assert sample.previous_state.tolist() == [2.0]
    assert sample.previous_timestamp == 0.21
    torch.testing.assert_close(
        sample.timestamps,
        torch.tensor([0.31, 0.42, 0.52, 0.63], dtype=torch.float64),
    )

    local = collate_raw_local_demo_samples(
        [sample],
        state_normalizer=DemoStateNormalizer(torch.zeros(1), torch.ones(1)),
        expected_state_dim=1,
    )
    expected_first_velocity = torch.tensor((3.0 - 2.0) / (0.31 - 0.21))
    torch.testing.assert_close(local.state_features[0, 0, 1], expected_first_velocity)


def test_episode_start_has_no_predecessor_and_zero_first_valid_velocity() -> None:
    sample = _reader()(DemoSampleRef("demo", query_anchor=0, local_anchor=0))

    assert sample.previous_state is None
    assert sample.previous_timestamp is None
    assert sample.valid_mask.tolist() == [False, True, True, True]
    local = collate_raw_local_demo_samples(
        [sample],
        state_normalizer=DemoStateNormalizer(torch.zeros(1), torch.ones(1)),
        expected_state_dim=1,
    )
    assert float(local.state_features[0, 1, 1]) == 0.0
