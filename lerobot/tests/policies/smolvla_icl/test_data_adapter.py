"""SmolVLA-ICL Query Dataset 与轻量 Demo 引用端口测试。"""

from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.policies.smolvla_icl.data.contracts import (
    DemoSampleRef,
    SmolVLAICLSampleIndex,
)
from lerobot.policies.smolvla_icl.data.dataset import SmolVLAICLQueryDataset
from lerobot.policies.smolvla_icl.data.sidecar import (
    EpisodeDemoPairing,
    PairingSidecar,
    PairingSidecarResolver,
)
from lerobot.policies.smolvla_icl.data.types import SMOLVLA_ICL_DEMO_REF


class FakeQueryDataset(torch.utils.data.Dataset):
    """提供 LeRobot wrapper 依赖的最小属性，并记录是否走批量读取。"""

    def __init__(self) -> None:
        self.meta = SimpleNamespace(name="fake-meta")
        self.episodes = [3]
        self.num_frames = 3
        self.num_episodes = 1
        self.absolute_to_relative_idx = {100: 0, 101: 1, 102: 2}
        self.hf_dataset = "fake-hf-dataset"
        self.batch_calls = 0
        self.samples = [
            {
                "episode_index": torch.tensor(3),
                "frame_index": torch.tensor(10 + index),
                "timestamp": torch.tensor(index * 0.1),
                "task": "pick",
                "observation.state": torch.tensor([float(index), 0.0]),
            }
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        self.batch_calls += 1
        return [self.samples[index] for index in indices]


def make_dataset(base: FakeQueryDataset, resolver) -> SmolVLAICLQueryDataset:
    """Local reader 不在 Dataset ``__getitem__`` 阶段执行，测试使用哑元函数。"""
    return SmolVLAICLQueryDataset(base, resolver, lambda reference: reference)


def make_resolver(*epochs: dict[int, EpisodeDemoPairing]) -> PairingSidecarResolver:
    return PairingSidecarResolver(
        PairingSidecar(
            manifest_fingerprint="a" * 64,
            matcher_snapshot="matcher@test",
            image_key="observation.images.top",
            epochs=epochs,
        )
    )


def test_sidecar_resolver_adds_reference_without_mutating_query_sample() -> None:
    base = FakeQueryDataset()
    dataset = make_dataset(
        base,
        make_resolver({3: EpisodeDemoPairing("demo-A", 8, (7,) * 13)}),
    )

    sample = dataset[1]
    assert sample is not None
    assert sample[SMOLVLA_ICL_DEMO_REF] == DemoSampleRef("demo-A", 11, 7)
    assert SMOLVLA_ICL_DEMO_REF not in base.samples[1]
    assert dataset.meta is base.meta
    assert dataset.episodes == [3]
    assert dataset.num_frames == 3
    assert dataset.num_episodes == 1
    assert dataset.absolute_to_relative_idx == base.absolute_to_relative_idx
    assert dataset.hf_dataset == "fake-hf-dataset"


def test_epoch_aware_index_selects_different_precomputed_reference() -> None:
    base = FakeQueryDataset()
    resolver = make_resolver(
        {3: EpisodeDemoPairing("demo-A", 8, (20,) * 13)},
        {3: EpisodeDemoPairing("demo-B", 9, (30,) * 13)},
    )
    dataset = make_dataset(base, resolver)

    epoch_zero = dataset[SmolVLAICLSampleIndex(dataset_index=0, epoch=0)]
    epoch_one = dataset[SmolVLAICLSampleIndex(dataset_index=0, epoch=1)]
    assert epoch_zero is not None and epoch_one is not None
    assert epoch_zero[SMOLVLA_ICL_DEMO_REF].demo_id == "demo-A"
    assert epoch_one[SMOLVLA_ICL_DEMO_REF].demo_id == "demo-B"


def test_batch_read_keeps_each_sample_epoch_and_uses_underlying_getitems() -> None:
    base = FakeQueryDataset()
    resolver = make_resolver(
        {3: EpisodeDemoPairing("demo-A", 8, (20,) * 13)},
        {3: EpisodeDemoPairing("demo-B", 9, (30,) * 13)},
    )
    dataset = make_dataset(base, resolver)

    samples = dataset.__getitems__(
        [
            SmolVLAICLSampleIndex(0, 0),
            SmolVLAICLSampleIndex(1, 1),
        ]
    )

    assert base.batch_calls == 1
    assert samples[0] is not None and samples[1] is not None
    assert samples[0][SMOLVLA_ICL_DEMO_REF].demo_id == "demo-A"
    assert samples[1][SMOLVLA_ICL_DEMO_REF].demo_id == "demo-B"


def test_sidecar_resolver_reports_missing_pair_at_query_boundary() -> None:
    dataset = make_dataset(
        FakeQueryDataset(),
        make_resolver({4: EpisodeDemoPairing("demo-A", 8, (20,) * 13)}),
    )

    try:
        dataset[SmolVLAICLSampleIndex(0, 2)]
    except KeyError as error:
        assert "epoch=2" in str(error)
        assert "query episode=3" in str(error)
    else:
        raise AssertionError("缺失 Demo 引用时必须立即报错。")


def test_pairing_sidecar_roundtrip_and_epoch_cycle(tmp_path: Path) -> None:
    sidecar = PairingSidecar(
        manifest_fingerprint="a" * 64,
        matcher_snapshot="lerobot/smolvla_base@test",
        image_key="observation.images.top",
        epochs=(
            {3: EpisodeDemoPairing("demo-A", 8, (1, 2, 3))},
            {3: EpisodeDemoPairing("demo-B", 9, (4, 5, 6))},
        ),
    )
    restored = PairingSidecar.load(sidecar.save(tmp_path / "pairing.json"))
    resolver = PairingSidecarResolver(restored)

    reference = resolver.resolve(
        epoch=3,
        dataset_index=1,
        episode_index=3,
        frame_index=1,
        timestamp=0.1,
        task="pick",
    )
    assert reference == DemoSampleRef("demo-B", query_anchor=1, local_anchor=5)
