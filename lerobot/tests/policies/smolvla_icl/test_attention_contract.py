"""SmolVLA-ICL 四区域 Attention 与位置编号的纯张量测试。"""

import torch

from lerobot.policies.smolvla_icl.smolvla_with_demo_expert import (
    FourRegionInputs,
    build_cross_attention_masks,
    build_demo_attention_mask,
    build_region_position_ids,
    build_union_attention_mask,
)


def make_inputs() -> FourRegionInputs:
    """构造含 Prefix/Local padding 的最小四区域输入。"""
    return FourRegionInputs(
        prefix_hidden=torch.zeros(1, 4, 6),
        global_hidden=torch.zeros(1, 2, 4),
        local_hidden=torch.zeros(1, 5, 4),
        action_hidden=torch.zeros(1, 3, 4),
        prefix_valid_mask=torch.tensor([[True, True, False, True]]),
        global_valid_mask=torch.tensor([[True, True]]),
        local_valid_mask=torch.tensor([[False, True, True, True, False]]),
        action_valid_mask=torch.tensor([[True, True, True]]),
        prefix_block_mask=torch.tensor([[False, False, False, True]]),
        local_anchor_positions=torch.tensor([2]),
    )


def test_union_mask_contains_only_designed_visibility_edges() -> None:
    """Union 层不得意外打开 P/G/L/A 之间未设计的边。"""
    inputs = make_inputs()
    mask = build_union_attention_mask(inputs)[0]
    layout = inputs.layout

    assert mask[layout.global_demo, layout.global_demo].all()
    assert not mask[layout.global_demo, layout.prefix].any()
    assert not mask[layout.local_demo, layout.global_demo].any()
    assert not mask[layout.action, layout.global_demo].any()
    assert not mask[layout.action, layout.local_demo].any()

    # Action 可读取有效 Prefix，但 Prefix padding 列必须不可见。
    assert mask[layout.action, layout.prefix][:, [0, 1, 3]].all()
    assert not mask[layout.action, layout.prefix][:, 2].any()

    # Action 自注意力保持因果关系。
    expected_causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    torch.testing.assert_close(mask[layout.action, layout.action], expected_causal)


def test_demo_and_cross_masks_respect_padding_and_direction() -> None:
    """Demo block diagonal 与三条定向 Cross 边必须使用各自有效 mask。"""
    inputs = make_inputs()
    demo_mask = build_demo_attention_mask(inputs)[0]
    cross = build_cross_attention_masks(inputs)

    assert demo_mask[:2, :2].all()
    assert not demo_mask[:2, 2:].any()
    assert not demo_mask[2:, :2].any()
    assert not demo_mask[2].any()
    assert not demo_mask[:, 2].any()

    assert cross.prefix_from_global.shape == (1, 4, 2)
    assert not cross.prefix_from_global[0, 2].any()
    assert cross.action_from_prefix[0, :, [0, 1, 3]].all()
    assert not cross.action_from_prefix[0, :, 2].any()
    assert not cross.action_from_local[0, :, [0, 4]].any()
    assert cross.action_from_local[0, :, 1:4].all()


def test_position_ids_keep_local_anchor_at_zero_without_compressing_padding() -> None:
    """Local RoPE 以锚点为零，padding 不得压缩真实 slot 的相对位置。"""
    positions = build_region_position_ids(make_inputs())

    assert positions.prefix.tolist() == [[0, 1, 1, 2]]
    assert positions.global_demo.tolist() == [[0, 1]]
    assert positions.local_demo.tolist() == [[-2, -1, 0, 1, 2]]
    assert positions.action.tolist() == [[3, 4, 5]]
