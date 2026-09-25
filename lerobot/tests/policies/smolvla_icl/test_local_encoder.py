"""Local Demo Encoder 的独立形状、Mask 和梯度测试。"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolvla_icl.components.demo_alignment import DemoEmbeddingCache
from lerobot.policies.smolvla_icl.components.local_encoder import LocalDemoEncoder
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import (
    DemoAlignmentConfig,
    LocalEncoderConfig,
)
from lerobot.policies.smolvla_icl.data.collate import build_encoded_local_demo_batch
from lerobot.policies.smolvla_icl.data.state import DemoStateNormalizer
from lerobot.policies.smolvla_icl.data.types import LocalDemoBatch
from lerobot.policies.smolvla_icl.modeling_smolvla_icl import SmolVLAICLPolicy, VLAFlowMatchingICL
from lerobot.policies.smolvla_icl.smolvla_with_demo_expert import SmolVLMWithDemoExpertModel


def make_config() -> LocalEncoderConfig:
    """使用小维度配置验证结构，不改变正式模型数据契约。"""
    return LocalEncoderConfig(
        visual_feature_dim=12,
        state_dim=4,
        visual_projection_dim=8,
        state_projection_dim=6,
        output_dim=10,
        temporal_embedding_dim=8,
    )


def make_inputs() -> tuple[torch.Tensor, ...]:
    """构建包含一个 padding 位置的五帧 Local Demo。"""
    batch_size, length = 2, 5
    return (
        torch.randn(batch_size, length, 8),
        torch.randn(batch_size, length, 8),
        torch.linspace(-0.2, 0.2, length).expand(batch_size, -1).clone(),
        torch.linspace(-0.4, 0.4, length).expand(batch_size, -1).clone(),
        torch.linspace(0.2, 0.6, length).expand(batch_size, -1).clone(),
        torch.tensor([[False, True, True, True, True], [True, True, True, True, True]]),
    )


def test_local_encoder_returns_one_masked_token_per_demo_frame() -> None:
    """每帧产生一个 token，padding 位置必须严格为零。"""
    encoder = LocalDemoEncoder(make_config())
    inputs = make_inputs()

    output = encoder(*inputs)

    assert output.local_tokens.shape == (2, 5, 10)
    assert output.local_mask.shape == (2, 5)
    assert torch.count_nonzero(output.local_tokens[0, 0]) == 0
    assert torch.count_nonzero(output.local_tokens[output.local_mask]) > 0


def test_padding_values_do_not_change_valid_local_tokens() -> None:
    """padding 内容可以任意变化，但不得影响有效 Local token。"""
    encoder = LocalDemoEncoder(make_config()).eval()
    inputs = list(make_inputs())
    reference = encoder(*inputs).local_tokens

    changed = [value.clone() for value in inputs]
    changed[0][0, 0] = 10_000
    changed[1][0, 0] = -10_000
    changed[2][0, 0] = 10_000
    changed[3][0, 0] = -10_000
    changed[4][0, 0] = 10_000
    actual = encoder(*changed).local_tokens

    torch.testing.assert_close(actual, reference)


def test_local_encoder_trains_state_and_temporal_paths() -> None:
    """动作损失的梯度必须到达视觉输入、State 和时间路径。"""
    encoder = LocalDemoEncoder(make_config())
    inputs = list(make_inputs())
    inputs[0].requires_grad_()
    inputs[1].requires_grad_()

    output = encoder(*inputs)
    output.local_tokens.square().mean().backward()

    assert encoder.state_projection[0].weight.grad is not None
    assert encoder.temporal_projection[0].weight.grad is not None
    assert inputs[0].grad is not None
    assert torch.count_nonzero(inputs[0].grad) > 0
    assert torch.count_nonzero(encoder.state_projection[0].weight.grad) > 0
    assert torch.count_nonzero(encoder.temporal_projection[0].weight.grad) > 0


def test_spatial_pooling_preserves_visual_gradient_path() -> None:
    """Action 侧损失应穿过 learned spatial pooling 到达视觉 tokens。"""
    encoder = LocalDemoEncoder(make_config())
    inputs = list(make_inputs())
    visual_tokens = torch.randn(2, 5, 4, 12, requires_grad=True)

    output = encoder(encoder.pool_spatial_tokens(visual_tokens), *inputs[1:])
    output.local_tokens[..., 0].sum().backward()

    assert visual_tokens.grad is not None
    assert encoder.spatial_query.grad is not None
    assert torch.count_nonzero(visual_tokens.grad) > 0
    assert torch.count_nonzero(encoder.spatial_query.grad) > 0


def test_matcher_processor_encoder_contract_is_end_to_end_compatible() -> None:
    """Matcher 截取、Processor 补齐和 Local Encoder 的张量排列必须一致。"""
    alignment_config = DemoAlignmentConfig(
        alignment_hz=10.0,
        window_duration_s=0.2,
        local_chunk_size=5,
        local_anchor_position_ratio=0.4,
    )
    num_frames = 8
    cache = DemoEmbeddingCache.from_embeddings(
        visual_frame_embeddings=torch.randn(num_frames, 12),
        raw_states=torch.randn(num_frames, 4),
        timestamps=torch.arange(num_frames, dtype=torch.float64) * 0.1,
        state_normalizer=DemoStateNormalizer(
            mean=torch.zeros(4),
            std=torch.ones(4),
        ),
        config=alignment_config,
    )
    visual_hidden = torch.randn(num_frames, make_config().visual_projection_dim)
    batch = build_encoded_local_demo_batch(
        cache.extract_local_window(4),
        visual_hidden,
        expected_state_dim=4,
    )
    encoder = LocalDemoEncoder(make_config())

    output = encoder(
        batch.visual_hidden,
        batch.state_features,
        batch.relative_time_s,
        batch.relative_position,
        batch.phase,
        batch.valid_mask,
    )

    assert output.local_tokens.shape == (1, 5, 10)
    assert output.local_mask.all()
    assert batch.visual_hidden.shape == (1, 5, 8)


class TinyVisionModel(nn.Module):
    """记录视觉 batch 大小的极小共享编码器，用于验证设备与梯度边界。"""

    def __init__(self) -> None:
        super().__init__()
        self.connector = nn.Linear(3, 4)
        self.batch_sizes: list[int] = []

    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(images))
        pixels = images.mean(dim=(-2, -1))
        return self.connector(torch.stack((pixels, pixels.square()), dim=1))


class TinyVLM(nn.Module):
    """只提供视觉前端，用于验证 ICL 对官方冻结逻辑的最终覆盖。"""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.vision_model = nn.Linear(4, 4)
        self.model.connector = nn.Linear(4, 4)


def test_local_rgb_microbatches_preserve_shared_vision_gradients() -> None:
    """CPU uint8 RGB 应分批编码，且 Action 侧损失仍能更新共享视觉参数。"""
    encoder_config = LocalEncoderConfig(
        visual_feature_dim=4,
        state_dim=2,
        visual_projection_dim=4,
        state_projection_dim=4,
        output_dim=6,
        temporal_embedding_dim=4,
    )
    model = VLAFlowMatchingICL.__new__(VLAFlowMatchingICL)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        resize_imgs_with_padding=None,
        freeze_vision_encoder=False,
        local_vision_encode_batch_size=2,
        local_vision_gradient_checkpointing=True,
        local_encoder=encoder_config,
    )
    model.vlm_with_expert = TinyVisionModel()
    model.local_encoder = LocalDemoEncoder(encoder_config)
    model.train()

    valid_mask = torch.tensor([[True, True, False, True, True]])
    demo = LocalDemoBatch(
        images=torch.randint(0, 256, (1, 5, 3, 6, 6), dtype=torch.uint8),
        state_features=torch.randn(1, 5, 4),
        relative_time_s=torch.linspace(-0.2, 0.2, 5).unsqueeze(0),
        relative_position=torch.linspace(-0.4, 0.4, 5).unsqueeze(0),
        phase=torch.linspace(0.1, 0.5, 5).unsqueeze(0),
        valid_mask=valid_mask,
        anchor_positions=torch.tensor([2]),
    )

    output = model.encode_local_demo(demo)
    output.local_tokens.square().mean().backward()

    assert demo.images.device.type == "cpu"
    assert demo.images.dtype == torch.uint8
    assert max(model.vlm_with_expert.batch_sizes) <= 2
    assert model.vlm_with_expert.connector.weight.grad is not None
    assert torch.count_nonzero(model.vlm_with_expert.connector.weight.grad) > 0


def test_frozen_local_vision_skips_graph_but_trains_local_encoder() -> None:
    """冻结 SigLIP 时不保存视觉图，但 Local Encoder 仍接收 Action 侧梯度。"""
    encoder_config = LocalEncoderConfig(
        visual_feature_dim=4,
        state_dim=2,
        visual_projection_dim=4,
        state_projection_dim=4,
        output_dim=6,
        temporal_embedding_dim=4,
    )
    model = VLAFlowMatchingICL.__new__(VLAFlowMatchingICL)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        resize_imgs_with_padding=None,
        freeze_vision_encoder=True,
        local_vision_encode_batch_size=2,
        # 即使配置仍为 True，冻结模式也必须自动跳过视觉 checkpoint。
        local_vision_gradient_checkpointing=True,
        local_encoder=encoder_config,
    )
    model.vlm_with_expert = TinyVisionModel()
    model.vlm_with_expert.requires_grad_(False)
    model.local_encoder = LocalDemoEncoder(encoder_config)
    model.train()

    demo = LocalDemoBatch(
        images=torch.randint(0, 256, (1, 5, 3, 6, 6), dtype=torch.uint8),
        state_features=torch.randn(1, 5, 4),
        relative_time_s=torch.linspace(-0.2, 0.2, 5).unsqueeze(0),
        relative_position=torch.linspace(-0.4, 0.4, 5).unsqueeze(0),
        phase=torch.linspace(0.1, 0.5, 5).unsqueeze(0),
        valid_mask=torch.ones(1, 5, dtype=torch.bool),
        anchor_positions=torch.tensor([2]),
    )

    output = model.encode_local_demo(demo)
    output.local_tokens.square().mean().backward()

    assert model.vlm_with_expert.connector.weight.grad is None
    projection_weight = model.local_encoder.visual_projection[0].weight
    assert projection_weight.grad is not None
    assert torch.count_nonzero(projection_weight.grad) > 0


def test_freeze_shared_siglip_is_respected_by_default_peft_targets() -> None:
    """PEFT 不得用 LoRA 或 modules_to_save 绕过视觉冻结开关。"""
    policy = SmolVLAICLPolicy.__new__(SmolVLAICLPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(freeze_vision_encoder=True, num_vlm_layers=16)

    frozen_targets = policy._get_default_peft_targets()

    assert "vision_model" not in frozen_targets["target_modules"]
    assert all("connector" not in module for module in frozen_targets["modules_to_save"])

    policy.config.freeze_vision_encoder = False
    trainable_targets = policy._get_default_peft_targets()

    assert "vision_model" in trainable_targets["target_modules"]
    assert any("connector" in module for module in trainable_targets["modules_to_save"])


def test_freeze_shared_vision_also_freezes_connector_outside_expert_only_mode() -> None:
    """即使官方全 VLM 训练逻辑保留 connector，ICL 也必须将其重新冻结。"""
    model = SmolVLMWithDemoExpertModel.__new__(SmolVLMWithDemoExpertModel)
    nn.Module.__init__(model)
    model.vlm = TinyVLM()
    model.freeze_vision_encoder = True
    model.train_expert_only = False

    # 模拟官方 ``train_expert_only=False`` 执行完毕后的状态：connector 以及
    # 其他未命中的 VLM 参数仍然可训练。ICL override 必须建立最终后置条件。
    with patch.object(SmolVLMWithExpertModel, "set_requires_grad", return_value=None):
        model.set_requires_grad()

    vision_parameters = model.get_vlm_model().vision_model.parameters()
    connector_parameters = model.get_vlm_model().connector.parameters()
    assert all(not parameter.requires_grad for parameter in vision_parameters)
    assert all(not parameter.requires_grad for parameter in connector_parameters)

    model.train()
    assert not model.get_vlm_model().vision_model.training
    assert not model.get_vlm_model().connector.training
