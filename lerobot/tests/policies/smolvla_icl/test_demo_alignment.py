"""Stage Matcher 选定锚点后的 Local Demo 窗口契约测试。"""

import torch

from lerobot.policies.smolvla_icl.components.demo_alignment import (
    DemoEmbeddingCache,
    ObservationHistoryBuffer,
    OnlineDTWMatcher,
)
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import DemoAlignmentConfig
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer


def make_state_normalizer(dim: int = 2) -> DemoStateNormalizer:
    """创建真实 State 维度明确的 identity normalizer。"""
    return DemoStateNormalizer(mean=torch.zeros(dim), std=torch.ones(dim))


def test_default_local_window_uses_48_frame_40_to_60_split() -> None:
    """默认窗口包含 48 帧，并保持约 4:6 的历史/未来比例。"""
    config = DemoAlignmentConfig()

    assert config.local_chunk_size == 48
    assert config.local_anchor_position == 19


def test_matcher_cache_does_not_store_model_visual_features() -> None:
    """Matcher cache 只保存 DTW pooled feature，Local 窗口不携带视觉 token。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        normalize_visual_features=True,
        local_chunk_size=3,
        local_anchor_position_ratio=1 / 3,
    )
    visual_embeddings = torch.tensor([[3.0, 0.0], [0.0, 4.0], [5.0, 0.0], [0.0, 6.0]])
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=visual_embeddings,
        raw_states=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        timestamps=torch.arange(4, dtype=torch.float64) * 0.1,
        state_normalizer=make_state_normalizer(),
        config=config,
    )

    chunk = cache.extract_local_window(1)

    assert not hasattr(cache, "visual_frame_embeddings")
    assert not hasattr(chunk, "visual_embeddings")
    torch.testing.assert_close(
        cache.visual_chunk_embeddings.norm(dim=-1),
        torch.ones(cache.num_chunks),
    )


def make_cache() -> DemoEmbeddingCache:
    """构建六帧小型 Demo，Local 窗口为两帧历史和三帧当前/未来。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=5,
        local_anchor_position_ratio=0.4,
    )
    num_frames = 6
    return DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.eye(num_frames),
        raw_states=torch.arange(num_frames * 2, dtype=torch.float32).reshape(num_frames, 2),
        timestamps=torch.arange(num_frames, dtype=torch.float64) * 0.1,
        state_normalizer=make_state_normalizer(),
        config=config,
    )


def test_local_chunk_keeps_anchor_at_fixed_position_and_left_pads() -> None:
    """靠近 Demo 开头时保持锚点位置，不把窗口整体向右移。"""
    chunk = make_cache().extract_local_window(1)

    assert len(chunk.valid_mask) == 5
    assert chunk.anchor_position == 2
    assert chunk.source_indices.tolist() == [-1, 0, 1, 2, 3]
    assert chunk.valid_mask.tolist() == [False, True, True, True, True]
    assert torch.count_nonzero(chunk.state_features[0]) == 0


def test_local_chunk_right_pads_at_demo_end() -> None:
    """靠近 Demo 末尾时只在右侧补齐，保留固定的历史/未来语义。"""
    chunk = make_cache().extract_local_window(5)

    assert chunk.anchor_position == 2
    assert chunk.source_indices.tolist() == [3, 4, 5, -1, -1]
    assert chunk.valid_mask.tolist() == [True, True, True, False, False]
    assert torch.count_nonzero(chunk.state_features[-2:]) == 0


def test_local_phase_uses_demo_timestamps_instead_of_frame_indices() -> None:
    """不规则采样时 Local phase 应与 Global phase 使用同一时间定义。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=3,
        local_anchor_position_ratio=1 / 3,
    )
    timestamps = torch.tensor([0.0, 0.1, 0.8, 1.0], dtype=torch.float64)
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.eye(4),
        raw_states=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        timestamps=timestamps,
        state_normalizer=make_state_normalizer(),
        config=config,
    )

    chunk = cache.extract_local_window(1)

    torch.testing.assert_close(
        chunk.phase,
        torch.tensor([0.0, 0.1, 0.8], dtype=torch.float64),
    )


def test_local_anchor_position_ratio_changes_history_future_split() -> None:
    """改变 anchor 比例时只改变窗口分割，不改变 Local Chunk 总长度。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=5,
        local_anchor_position_ratio=0.6,
    )
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.eye(8),
        raw_states=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        timestamps=torch.arange(8, dtype=torch.float64) * 0.1,
        state_normalizer=make_state_normalizer(),
        config=config,
    )

    chunk = cache.extract_local_window(4)

    assert len(chunk.valid_mask) == 5
    assert chunk.anchor_position == 3
    assert chunk.source_indices.tolist() == [1, 2, 3, 4, 5]


def test_matcher_rejects_padded_state_before_gripper_exclusion() -> None:
    """真实 State 为 2 维时，已补到 4 维的输入必须显式失败。"""
    try:
        DemoEmbeddingCache.from_embeddings(
            visual_frame_embeddings=torch.eye(4),
            raw_states=torch.zeros(4, 4),
            timestamps=torch.arange(4, dtype=torch.float64) * 0.1,
            state_normalizer=make_state_normalizer(dim=2),
            config=DemoAlignmentConfig(alignment_hz=10.0, window_duration_s=0.2),
        )
    except ValueError as error:
        assert "raw State" in str(error)
    else:
        raise AssertionError("Matcher 不应接受已 padding 的 State。")


def test_matcher_rejects_different_demo_query_normalization_stats() -> None:
    """Demo 和 Query 的统计量不一致时必须在计算 DTW 距离前拒绝。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=3,
    )
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.eye(4),
        raw_states=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        timestamps=torch.arange(4, dtype=torch.float64) * 0.1,
        state_normalizer=make_state_normalizer(),
        config=config,
    )
    query_history = ObservationHistoryBuffer(
        config,
        state_normalizer=DemoStateNormalizer(
            mean=torch.ones(2),
            std=torch.ones(2),
        ),
    )
    for index in range(2):
        query_history.append(
            raw_state=torch.tensor([float(index), float(index + 1)]),
            visual_embedding=torch.eye(4)[index],
            timestamp=index * 0.1,
        )

    try:
        OnlineDTWMatcher(cache).update(query_history.get_latest_chunk())
    except ValueError as error:
        assert "同一套 State 归一化统计量" in str(error)
    else:
        raise AssertionError("Matcher 不应接受使用不同统计量的 Query。")


def test_matcher_created_history_reuses_demo_state_normalizer() -> None:
    """Matcher 工厂创建的 Query Buffer 应自动复用 Demo 的统计量并正常匹配。"""
    config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=3,
    )
    normalizer = DemoStateNormalizer(
        mean=torch.tensor([1.0, 2.0]),
        std=torch.tensor([2.0, 4.0]),
    )
    raw_states = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]])
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.eye(4),
        raw_states=raw_states,
        timestamps=torch.arange(4, dtype=torch.float64) * 0.1,
        state_normalizer=normalizer,
        config=config,
    )
    matcher = OnlineDTWMatcher(cache)
    query_history = matcher.create_observation_history()
    for index in range(2):
        query_history.append(
            raw_state=raw_states[index],
            visual_embedding=torch.eye(4)[index],
            timestamp=index * 0.1,
        )

    result = matcher.update(query_history.get_latest_chunk())

    torch.testing.assert_close(
        cache.state_frame_features[:2, :2],
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert result.demo_observation_index >= 0
