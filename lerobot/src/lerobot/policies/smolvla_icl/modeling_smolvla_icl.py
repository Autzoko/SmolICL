"""SmolVLA-ICL 的顶层模型外壳。

本文件负责创建 VLM、Demo Expert、Action Expert、Global/Local Encoder
以及 flow-matching 所需的输入/输出投影，并提供训练 forward 与基于条件
KV cache 的多步 Euler 去噪。
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any

import torch
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.import_utils import require_package

from ..common.flow_matching import euler_integrate, sample_noise, sample_time_beta
from ..common.vla_utils import create_sinusoidal_pos_embedding, resize_with_pad
from ..pretrained import PreTrainedPolicy
from ..smolvla.modeling_smolvla import SmolVLAPolicy, pad_tensor
from ..utils import log_model_loading_keys
from .components.demo_alignment import (
    DemoEmbeddingCache,
    ObservationHistoryBuffer,
    OnlineDTWMatcher,
    SmolVLASigLIPHandle,
    pool_visual_tokens,
    reuse_smolvla_siglip,
)
from .components.global_encoder import GlobalDemoEncoder, GlobalEncoderOutput
from .components.local_encoder import LocalDemoEncoder, LocalEncoderOutput
from .configuration_smolvla_icl import SmolVLAICLConfig
from .data.collate import (
    build_encoded_local_demo_batch,
    get_smolvla_icl_demo_batches,
)
from .data.preprocessing import build_global_demo_clips
from .data.state import DemoStateNormalizer
from .data.types import EncodedLocalDemoBatch, GlobalDemoBatch, LocalDemoBatch
from .smolvla_with_demo_expert import (
    FourRegionInputs,
    SmolVLAICLConditionCache,
    SmolVLMWithDemoExpertModel,
)

__all__ = ["SmolVLAICLPolicy", "VLAFlowMatchingICL"]


class VLAFlowMatchingICL(nn.Module):
    """SmolVLA-ICL 神经网络主体，不包含 Policy 的 episode 状态管理。

    参数命名尽量保持与官方 :class:`VLAFlowMatching` 一致，使预训练
    SmolVLA 的 VLM、Action Expert、State/Action 投影能够按原路径迁移；
    ``global_encoder``、``local_encoder``、``demo_expert`` 和两条 Demo
    Cross-Attention adapter 是 ICL 新增参数。
    """

    def __init__(self, config: SmolVLAICLConfig) -> None:
        super().__init__()
        self.config = config

        # 直接复用官方 SmolVLA 的 VLM 和 Action Expert 构造逻辑，并在其上
        # 增加独立 Demo Expert。这里显式传入所有结构参数，不能依赖父类中
        # 面向通用模型的默认值，否则会丢失 16 层 Union/Cross 交替结构。
        self.vlm_with_expert = SmolVLMWithDemoExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
            vlm_load_dtype=config.vlm_load_dtype,
            global_cross_gate_init=config.global_cross_gate_init,
            local_cross_gate_init=config.local_cross_gate_init,
        )

        vlm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size
        if config.global_encoder.output_dim != expert_hidden_size:
            raise ValueError(
                "Global/Local Encoder output_dim 必须等于 Demo Expert hidden size："
                f"{config.global_encoder.output_dim} != {expert_hidden_size}。"
            )
        if config.local_encoder.visual_feature_dim != vlm_hidden_size:
            raise ValueError(
                "Local visual_feature_dim 必须等于 SmolVLM connector 输出宽度："
                f"{config.local_encoder.visual_feature_dim} != {vlm_hidden_size}。"
            )

        self.global_encoder = GlobalDemoEncoder(config.global_encoder)
        self.local_encoder = LocalDemoEncoder(config.local_encoder)

        # 以下投影沿用 SmolVLA 的名称和形状。Action Expert 继续接收每个
        # noisy action 对应的一个 token，并从最终 Action hidden 预测速度场。
        self.state_proj = nn.Linear(config.max_state_dim, vlm_hidden_size)
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.set_requires_grad()

        tokenizer = self.vlm_with_expert.processor.tokenizer
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        self.register_buffer(
            "global_image_start_token",
            torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "image_end_token",
            torch.tensor([self.fake_image_token], dtype=torch.long),
            persistent=False,
        )
        self.add_image_special_tokens = config.add_image_special_tokens
        self.prefix_length = config.prefix_length

    def set_requires_grad(self) -> None:
        """保持 SmolVLA 的 State projection 训练开关。"""
        for parameter in self.state_proj.parameters():
            parameter.requires_grad_(self.config.train_state_proj)

    def sample_noise(self, shape: tuple[int, ...], device: torch.device | str) -> Tensor:
        """沿用 SmolVLA 的标准高斯 flow 起点。"""
        return sample_noise(shape, device)

    def sample_time(self, batch_size: int, device: torch.device | str) -> Tensor:
        """沿用 SmolVLA 的 Beta flow-time 采样分布。"""
        return sample_time_beta(
            batch_size,
            device,
            alpha=1.5,
            beta=1.0,
            scale=0.999,
            offset=0.001,
        )

    def embed_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        image_embeddings: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """复用 SmolVLA 语义构造当前 Observation 的 V/L/S Prefix。

        返回值依次是 Prefix hidden、valid mask 和一维 Prefix block mask。
        图像/语言属于 block 0；State 属于 block 1，因此 State 可以读取
        前面的视觉语言，而视觉语言不能反向读取 State。
        """
        embeddings: list[Tensor] = []
        valid_masks: list[Tensor] = []
        block_values: list[int] = []

        if image_embeddings is None:
            image_embeddings = self.encode_images(images)

        for image, image_mask, image_embedding in zip(
            images,
            image_masks,
            image_embeddings,
            strict=True,
        ):
            if self.add_image_special_tokens:
                start_embedding = (
                    self.vlm_with_expert.embed_language_tokens(self.global_image_start_token.to(image.device))
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                embeddings.append(start_embedding)
                valid_masks.append(
                    torch.ones(
                        start_embedding.shape[:2],
                        dtype=torch.bool,
                        device=start_embedding.device,
                    )
                )
                block_values.extend([0] * start_embedding.shape[1])

            image_embedding = image_embedding * math.sqrt(image_embedding.shape[-1])
            batch_size, image_length = image_embedding.shape[:2]
            embeddings.append(image_embedding)
            valid_masks.append(
                image_mask.to(device=image_embedding.device, dtype=torch.bool)[:, None].expand(
                    batch_size,
                    image_length,
                )
            )
            block_values.extend([0] * image_length)

            if self.add_image_special_tokens:
                end_embedding = (
                    self.vlm_with_expert.embed_language_tokens(self.image_end_token.to(image.device))
                    .unsqueeze(0)
                    .expand(image.shape[0], -1, -1)
                )
                embeddings.append(end_embedding)
                valid_masks.append(
                    torch.ones(
                        end_embedding.shape[:2],
                        dtype=torch.bool,
                        device=end_embedding.device,
                    )
                )
                block_values.extend([0] * end_embedding.shape[1])

        language_embedding = self.vlm_with_expert.embed_language_tokens(language_tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embeddings.append(language_embedding)
        valid_masks.append(language_masks.to(device=language_embedding.device, dtype=torch.bool))
        block_values.extend([0] * language_embedding.shape[1])

        state_embedding = self.state_proj(state)
        if state_embedding.ndim == 2:
            state_embedding = state_embedding[:, None, :]
        embeddings.append(state_embedding)
        valid_masks.append(
            torch.ones(
                state_embedding.shape[:2],
                dtype=torch.bool,
                device=state_embedding.device,
            )
        )
        block_values.extend([1] * state_embedding.shape[1])

        prefix_hidden = torch.cat(embeddings, dim=1)
        prefix_valid_mask = torch.cat(valid_masks, dim=1)
        prefix_block_mask = torch.tensor(
            block_values,
            dtype=torch.bool,
            device=prefix_hidden.device,
        )[None, :].expand(prefix_hidden.shape[0], -1)

        if prefix_hidden.shape[1] < self.prefix_length:
            prefix_hidden = pad_tensor(prefix_hidden, self.prefix_length, pad_value=0)
            prefix_valid_mask = pad_tensor(
                prefix_valid_mask,
                self.prefix_length,
                pad_value=0,
            )
            prefix_block_mask = pad_tensor(
                prefix_block_mask,
                self.prefix_length,
                pad_value=0,
            )
        return prefix_hidden, prefix_valid_mask, prefix_block_mask

    def encode_images(self, images: list[Tensor]) -> list[Tensor]:
        """编码当前 Observation 图像，供 Prefix 和 Stage Matcher 共享。"""
        return [self.vlm_with_expert.embed_image(image) for image in images]

    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """将 noisy action 与 flow timestep 融合为 Action Expert tokens。"""
        action_hidden = self.action_in_proj(noisy_actions)
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=action_hidden.device,
        ).to(dtype=action_hidden.dtype)
        time_embedding = time_embedding[:, None, :].expand_as(action_hidden)
        action_hidden = torch.cat([action_hidden, time_embedding], dim=-1)
        action_hidden = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_hidden)))
        action_valid_mask = torch.ones(
            action_hidden.shape[:2],
            dtype=torch.bool,
            device=action_hidden.device,
        )
        return action_hidden, action_valid_mask

    def encode_global_demo(self, demo: GlobalDemoBatch) -> GlobalEncoderOutput:
        """把缓存的 S3D feature 与可训练 State 路径编码为 Global Tokens。"""
        output = self.global_encoder(
            demo.video_features,
            demo.states,
            demo.timestamps,
            demo.valid_mask,
        )
        inverse = demo.sample_to_demo
        return GlobalEncoderOutput(
            global_tokens=output.global_tokens.index_select(0, inverse),
            global_mask=output.global_mask.index_select(0, inverse),
        )

    def _prepare_local_vision_images(self, images: Tensor) -> Tensor:
        """只预处理当前视觉 micro-batch，避免展开整段 float32 RGB。"""
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        elif images.is_floating_point():
            images = images.float()
        else:
            raise TypeError("Local RGB 必须是 uint8 或浮点 Tensor。")
        if self.config.resize_imgs_with_padding is not None:
            target_width, target_height = self.config.resize_imgs_with_padding
            images = resize_with_pad(
                images,
                target_height,
                target_width,
                pad_value=0,
            )
        return images.mul(2.0).sub(1.0)

    def _encode_and_pool_local_images(self, images: Tensor) -> Tensor:
        """共享 SigLIP 编码后立即压缩 spatial tokens。"""
        visual_tokens = self.vlm_with_expert.embed_image(images)
        return self.local_encoder.pool_spatial_tokens(visual_tokens)

    def _prepare_and_encode_local_images(self, raw_images: Tensor) -> Tensor:
        """预处理图像并执行共享 SigLIP+connector，不进入 Local Encoder。"""
        images = self._prepare_local_vision_images(raw_images)
        return self.vlm_with_expert.embed_image(images)

    def _prepare_encode_and_pool_local_images(self, raw_images: Tensor) -> Tensor:
        """在同一 checkpoint 单元内完成图像展开、编码与池化。"""
        return self._encode_and_pool_local_images(self._prepare_local_vision_images(raw_images))

    def encode_local_demo(self, demo: LocalDemoBatch) -> LocalEncoderOutput:
        """从 CPU 逐批上传 RGB，并按配置训练或冻结共享 ``E_vision``。"""
        batch_size, chunk_length = demo.images.shape[:2]
        flat_images = demo.images.flatten(0, 1)
        if flat_images.device.type != "cpu" or flat_images.dtype != torch.uint8:
            raise ValueError("训练期 Local RGB 必须是留在 CPU 的 uint8 Tensor。")

        # 使用连续 view 保留 DataLoader 的 pinned storage，确保 non_blocking
        # H2D 不会因离散 CPU index_select 产生的新内存而退化为同步拷贝。
        # 边界 padding 帧也随所在 micro-batch 编码，随后在视觉输出处清零；
        # 可训练模式下，checkpoint 单元一直覆盖到 spatial pooling，因此
        # backward 前无需保留每帧的 SigLIP 中间激活或完整 spatial token。
        # 冻结模式不为 SigLIP+connector 建立计算图；spatial pooling 仍属于
        # 可训练 Local Encoder，所以不会改变其输入、输出或后续梯度链路。
        model_device = demo.state_features.device
        visual_batches: list[Tensor] = []
        train_vision_encoder = not self.config.freeze_vision_encoder
        use_checkpoint = (
            train_vision_encoder
            and self.training
            and torch.is_grad_enabled()
            and self.config.local_vision_gradient_checkpointing
        )
        encode_batch_size = self.config.local_vision_encode_batch_size
        for start in range(0, len(flat_images), encode_batch_size):
            end = min(start + encode_batch_size, len(flat_images))
            # DataLoader pin_memory=True 时这是异步 H2D；任何时刻 GPU 上只
            # 存在当前 micro-batch 的 raw RGB，而不是完整 B*T Local 视频。
            image_batch = flat_images[start:end].to(
                model_device,
                non_blocking=True,
            )
            if use_checkpoint:
                visual_hidden = checkpoint(
                    self._prepare_encode_and_pool_local_images,
                    image_batch,
                    use_reentrant=False,
                )
            elif train_vision_encoder:
                visual_hidden = self._prepare_encode_and_pool_local_images(image_batch)
            else:
                # 视觉冻结时显式关闭 autograd，避免为 48 帧 Local RGB 保存
                # 无用激活。作用域只覆盖 SigLIP+connector；spatial pooling
                # 和下方 Local Encoder 均在普通梯度上下文中执行并正常更新。
                with torch.no_grad():
                    visual_tokens = self._prepare_and_encode_local_images(image_batch)
                visual_hidden = self.local_encoder.pool_spatial_tokens(visual_tokens)
            visual_batches.append(visual_hidden)

        flat_hidden = torch.cat(visual_batches, dim=0)
        flat_hidden = torch.where(
            demo.valid_mask.flatten().unsqueeze(-1),
            flat_hidden,
            torch.zeros_like(flat_hidden),
        )
        return self.local_encoder(
            flat_hidden.unflatten(0, (batch_size, chunk_length)),
            demo.state_features,
            demo.relative_time_s,
            demo.relative_position,
            demo.phase,
            demo.valid_mask,
        )

    def encode_encoded_local_demo(
        self,
        demo: EncodedLocalDemoBatch,
    ) -> LocalEncoderOutput:
        """rollout 路径：消费 ``set_demo`` 时缓存的逐帧视觉 hidden。"""
        return self.local_encoder(
            demo.visual_hidden,
            demo.state_features,
            demo.relative_time_s,
            demo.relative_position,
            demo.phase,
            demo.valid_mask,
        )

    def build_four_region_inputs(
        self,
        *,
        prefix_hidden: Tensor,
        prefix_valid_mask: Tensor,
        prefix_block_mask: Tensor,
        global_output: GlobalEncoderOutput,
        local_output: LocalEncoderOutput,
        local_anchor_positions: Tensor,
        action_hidden: Tensor,
        action_valid_mask: Tensor,
    ) -> FourRegionInputs:
        """按固定 ``[P;G;L;A]`` 顺序建立三分支 Transformer 输入契约。"""
        return FourRegionInputs(
            prefix_hidden=prefix_hidden,
            global_hidden=global_output.global_tokens,
            local_hidden=local_output.local_tokens,
            action_hidden=action_hidden,
            prefix_valid_mask=prefix_valid_mask,
            global_valid_mask=global_output.global_mask,
            local_valid_mask=local_output.local_mask,
            action_valid_mask=action_valid_mask,
            prefix_block_mask=prefix_block_mask,
            local_anchor_positions=local_anchor_positions,
        )

    def forward(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        global_demo: GlobalDemoBatch,
        local_demo: LocalDemoBatch,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """执行一次完整训练前向并返回逐元素 flow-matching MSE。

        ``actions`` 已由上层 Policy 右侧补齐到 ``max_action_dim``；Global 和
        Local Demo 也已经由各自 Processor 整理成结构化 batch。本函数不运行
        Stage Matcher，因为随机训练 batch 不具有 Matcher 所需的在线历史状态。
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 与 SmolVLA 相同的 conditional flow matching 路径：t=0 对应真实
        # action，t=1 对应高斯噪声，监督目标是在整条路径上恒定的速度场。
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        prefix_hidden, prefix_valid_mask, prefix_block_mask = self.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
        )
        action_hidden, action_valid_mask = self.embed_suffix(noisy_actions, time)
        global_output = self.encode_global_demo(global_demo)
        local_output = self.encode_local_demo(local_demo)
        transformer_inputs = self.build_four_region_inputs(
            prefix_hidden=prefix_hidden,
            prefix_valid_mask=prefix_valid_mask,
            prefix_block_mask=prefix_block_mask,
            global_output=global_output,
            local_output=local_output,
            local_anchor_positions=local_demo.anchor_positions,
            action_hidden=action_hidden,
            action_valid_mask=action_valid_mask,
        )

        # 三分支主干最终仍保持 P/G/L/A 分区；训练 head 只消费 Action 区域。
        transformer_output = self.vlm_with_expert(transformer_inputs)
        action_output = transformer_output.action_hidden.to(dtype=torch.float32)
        predicted_velocity = self.action_out_proj(action_output)
        return F.mse_loss(
            target_velocity,
            predicted_velocity,
            reduction="none",
        )

    def denoise_step(
        self,
        x_t: Tensor,
        timestep: Tensor,
        condition_cache: SmolVLAICLConditionCache,
    ) -> Tensor:
        """使用固定 P/G/L Condition KV 执行一个 flow-matching 去噪步。

        与训练路径相同，当前 ``x_t`` 和 timestep 先融合为 Action tokens；
        随后只运行 Action Expert，并通过输出投影得到 Euler integration 所需
        的速度场。条件 cache 必须由本次重规划的同一 P/G/L 输入构建。
        """
        action_hidden, _ = self.embed_suffix(x_t, timestep)
        action_output = self.run_cached_action_layers(
            action_hidden,
            condition_cache,
        )
        return self.action_out_proj(action_output.to(dtype=torch.float32))

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        global_output: GlobalEncoderOutput,
        local_demo: EncodedLocalDemoBatch,
        noise: Tensor | None = None,
        image_embeddings: list[Tensor] | None = None,
    ) -> Tensor:
        """从高斯噪声采样一个完整 action chunk。

        ``global_output`` 已在注册 Demo 时编码并跨重规划复用；当前 Observation
        和匹配得到的 Local Chunk 在整个 Euler 轨迹中保持不变，因此这里只
        编码 Local 并建立 P/G/L condition cache。每个去噪步只重新计算依赖
        ``x_t`` 和 timestep 的 Action 分支。

        返回张量形状为 ``[B, chunk_size, max_action_dim]``；真实机器人动作维度
        的裁剪与反归一化仍由后续 Policy wrapper 负责。
        """
        batch_size = state.shape[0]
        device = state.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        prefix_hidden, prefix_valid_mask, prefix_block_mask = self.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            image_embeddings=image_embeddings,
        )
        local_output = self.encode_encoded_local_demo(local_demo)

        # condition cache 只需要 Action 区域的长度、mask 和位置，不读取其
        # hidden 数值，因此不必额外执行一次 action/time embedding。
        action_hidden = global_output.global_tokens.new_zeros(
            batch_size,
            noise.shape[1],
            self.vlm_with_expert.expert_hidden_size,
        )
        action_valid_mask = torch.ones(
            noise.shape[:2],
            dtype=torch.bool,
            device=device,
        )
        transformer_inputs = self.build_four_region_inputs(
            prefix_hidden=prefix_hidden,
            prefix_valid_mask=prefix_valid_mask,
            prefix_block_mask=prefix_block_mask,
            global_output=global_output,
            local_output=local_output,
            local_anchor_positions=local_demo.anchor_positions,
            action_hidden=action_hidden,
            action_valid_mask=action_valid_mask,
        )
        condition_cache = self.build_condition_cache(transformer_inputs)

        # 与训练路径相反地从 t=1 积分至 t=0：
        # x_t <- x_t - (1 / num_steps) * v_theta(x_t, t)。
        return euler_integrate(
            lambda x_t, timestep: self.denoise_step(
                x_t=x_t,
                timestep=timestep,
                condition_cache=condition_cache,
            ),
            noise,
            self.config.num_steps,
        )

    def build_condition_cache(
        self,
        inputs: FourRegionInputs,
    ) -> SmolVLAICLConditionCache:
        """预填充一次 P/G/L 条件分支，供多步 flow 去噪重复使用。"""
        return self.vlm_with_expert.build_condition_cache(inputs)

    def run_cached_action_layers(
        self,
        action_hidden: Tensor,
        cache: SmolVLAICLConditionCache,
    ) -> Tensor:
        """复用条件 K/V，只运行依赖当前 noisy action 的 Action 分支。"""
        return self.vlm_with_expert.run_action_with_condition_cache(
            action_hidden,
            cache,
        )


class SmolVLAICLPolicy(SmolVLAPolicy):
    """将 Demo 对齐和 SmolVLA-ICL 模型接入 LeRobot Policy 生命周期。

    ``set_demo`` 在 rollout 前调用一次，用完整 RGB+State Demo 建立 Global
    输入和 Stage Matcher cache。之后每次动作队列耗尽时，Policy 只用最新
    真实 Observation 推进 Matcher、提取 Local Chunk，并生成新的动作块。

    输入 State 和模型输出 Action 遵循 LeRobot 标准预/后处理契约：进入本类
    时已经归一化；真实动作维度的裁剪在本类完成，数值反归一化由配套的
    ``make_smolvla_icl_pre_post_processors`` postprocessor 完成。
    """

    config_class = SmolVLAICLConfig
    name = "smolvla_icl"

    _BASELINE_MISSING_PREFIXES = (
        "model.global_encoder.",
        "model.local_encoder.",
        "model.vlm_with_expert.demo_expert.",
        "model.vlm_with_expert.prefix_from_global.",
        "model.vlm_with_expert.action_from_local.",
    )

    @classmethod
    def _load_as_safetensor(
        cls,
        model: SmolVLAICLPolicy,
        model_file: str,
        map_location: str,
        strict: bool,
    ) -> SmolVLAICLPolicy:
        """加载 ICL 或 SmolVLA baseline checkpoint，并审计参数键。

        从 baseline 初始化时，只允许 ICL 新模块缺失。任何其他 missing key 或
        unexpected key 都说明预训练参数路径、层数或 checkpoint 类型没有对齐，
        不应在 ``strict=False`` 下静默继续训练。
        """
        missing_keys, unexpected_keys = load_model_as_safetensor(
            model,
            model_file,
            strict=strict,
            device=resolve_safetensors_device(map_location),
        )
        invalid_missing = [key for key in missing_keys if not key.startswith(cls._BASELINE_MISSING_PREFIXES)]
        if invalid_missing or unexpected_keys:
            raise RuntimeError(
                "SmolVLA checkpoint 与 SmolVLA-ICL 共享参数路径不一致："
                f"非预期 missing keys={invalid_missing}，"
                f"unexpected keys={unexpected_keys}。"
            )
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    def __init__(
        self,
        config: SmolVLAICLConfig,
        *,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        require_package("transformers", extra="smolvla")
        # 不调用 SmolVLAPolicy.__init__，避免先创建一份无用的 baseline 模型；
        # 这里只复用其图像、State、Action preparation 与动作队列方法。
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = VLAFlowMatchingICL(config)
        self.state_normalizer = (
            DemoStateNormalizer.from_dataset_stats(dataset_stats) if dataset_stats is not None else None
        )
        if self.state_normalizer is not None:
            self._validate_state_normalizer(self.state_normalizer)

        self._global_output: GlobalEncoderOutput | None = None
        self._demo_cache: DemoEmbeddingCache | None = None
        self._demo_visual_hidden: Tensor | None = None
        self._matcher: OnlineDTWMatcher | None = None
        self._matcher_siglip: SmolVLASigLIPHandle | None = None
        self._observation_history: ObservationHistoryBuffer | None = None
        self._matcher_image_key: str | None = None
        self._replan_index = 0
        self.reset()

    def _validate_state_normalizer(self, normalizer: DemoStateNormalizer) -> None:
        """保证统计量覆盖真实 State，而不是已经补齐到 32 维的向量。"""
        state_feature = self.config.robot_state_feature
        assert state_feature is not None
        if normalizer.state_dim != state_feature.shape[0]:
            raise ValueError(
                "State 统计量维度必须等于真实 observation.state 维度："
                f"{normalizer.state_dim} != {state_feature.shape[0]}。"
            )

    def supports_rtc(self) -> bool:
        """首版 ICL Policy 尚未把 RTC guidance 接入三分支 cache。"""
        return False

    def _icl_modules_to_save(self) -> list[str]:
        """返回 PEFT 中需要完整训练和保存的 ICL 新增模块。

        这些模块没有对应的 SmolVLA 预训练参数，不能只依靠主干上的 LoRA。
        两个 Cross-Attention 使用 ``ModuleDict`` 保存逐层 adapter；这里登记
        每个具体 adapter，而不是登记容器本身，保证字符串索引语义不变，
        并让其中的标量 gate 一同训练和保存。
        """
        modules = [
            "model.global_encoder.state_encoder",
            "model.global_encoder.rgb_state_fusion",
            "model.global_encoder.temporal_aggregator",
            "model.global_encoder.task_query_bank",
            "model.global_encoder.output_projection",
            "model.local_encoder",
        ]
        # PEFT 的 modules_to_save 会把模块设为可训练；只有显式选择训练
        # SigLIP 时才加入 connector，防止 freeze_vision_encoder=True 被
        # PEFT 静默绕过。冻结的 connector 由基础 checkpoint 直接提供。
        if not self.config.freeze_vision_encoder:
            modules.insert(0, "model.vlm_with_expert.vlm.model.connector")
        modules.extend(
            f"model.vlm_with_expert.demo_expert.layers.{layer_idx}"
            for layer_idx in range(self.config.num_vlm_layers)
        )
        for layer_idx in range(1, self.config.num_vlm_layers, 2):
            modules.extend(
                [
                    f"model.vlm_with_expert.prefix_from_global.{layer_idx}",
                    f"model.vlm_with_expert.action_from_local.{layer_idx}",
                ]
            )
        return modules

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """在 SmolVLA LoRA 目标上补充需要完整训练的 ICL 模块。"""
        targets = super()._get_default_peft_targets()
        # PEFT 会先冻结全部 base parameters。仅在视觉可训练模式下为
        # SigLIP attention 注入 LoRA；connector 则由 modules_to_save 完整训练。
        if not self.config.freeze_vision_encoder:
            targets["target_modules"] = (
                rf"({targets['target_modules']}|"
                r"model\.vlm_with_expert\.vlm\.model\.vision_model\..*\.(q|v)_proj)"
            )
        targets["modules_to_save"] = self._icl_modules_to_save()
        return targets

    def _validate_peft_config(self, peft_config: Any) -> None:
        """验证 ICL 必训模块，并维护共享视觉冻结的强契约。"""
        super()._validate_peft_config(peft_config)
        configured = set(peft_config.modules_to_save or [])
        missing = set(self._icl_modules_to_save()) - configured
        if missing:
            raise ValueError(
                "SmolVLA-ICL 的 PEFT 配置必须通过 modules_to_save 完整训练并保存 "
                f"ICL 新增模块，当前缺少：{sorted(missing)}。"
            )

        if not self.config.freeze_vision_encoder:
            return

        # PEFT 的 target_modules 支持正则字符串、名称后缀列表以及
        # ``all-linear``。不能只搜索配置文本中的 ``vision_model``：例如
        # ["q_proj"] 同样会命中 SigLIP。这里用当前完整 Policy 的真实模块名
        # 复现这些匹配规则，确保自定义 CLI 无法绕过视觉冻结选项。
        visual_prefixes = (
            "model.vlm_with_expert.vlm.model.vision_model",
            "model.vlm_with_expert.vlm.model.connector",
        )
        visual_modules = {
            name: module
            for name, module in self.named_modules()
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in visual_prefixes)
        }
        target_matches = self._match_peft_target_modules(
            peft_config.target_modules,
            visual_modules,
        )
        saved_matches = self._match_peft_module_suffixes(
            peft_config.modules_to_save,
            visual_modules,
        )
        if target_matches or saved_matches:
            raise ValueError(
                "freeze_vision_encoder=True 时 PEFT 不得训练 SigLIP 或 connector："
                f"target_modules 命中={target_matches}，"
                f"modules_to_save 命中={saved_matches}。"
            )

    @staticmethod
    def _match_peft_module_suffixes(
        configured_modules: Any,
        candidate_modules: dict[str, nn.Module],
    ) -> list[str]:
        """返回被 PEFT 名称/后缀列表命中的候选模块名。"""
        if not configured_modules:
            return []
        suffixes = (
            (configured_modules,)
            if isinstance(configured_modules, str)
            else tuple(configured_modules)
        )
        return sorted(
            name
            for name in candidate_modules
            if any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes)
        )

    @classmethod
    def _match_peft_target_modules(
        cls,
        target_modules: Any,
        candidate_modules: dict[str, nn.Module],
    ) -> list[str]:
        """按 PEFT 的正则、后缀和 all-linear 语义解析 target_modules。"""
        if not target_modules:
            return []
        if not isinstance(target_modules, str):
            return cls._match_peft_module_suffixes(target_modules, candidate_modules)
        if target_modules == "all-linear":
            return sorted(
                name for name, module in candidate_modules.items() if isinstance(module, nn.Linear)
            )
        try:
            pattern = re.compile(target_modules)
        except re.error as error:
            raise ValueError(f"PEFT target_modules 正则表达式无效：{target_modules!r}。") from error
        return sorted(name for name in candidate_modules if pattern.fullmatch(name))

    def reset(self) -> None:
        """开始新 episode：清空动作队列，并让 Matcher 从 Demo 起点重启。"""
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._replan_index = 0
        if self._matcher is not None:
            self._matcher.reset()
            self._observation_history = self._matcher.create_observation_history()

    def alignment_diagnostics(self) -> dict[str, int | float | None] | None:
        """返回当前 rollout 的只读对齐状态，供专用 evaluator 记录轨迹。

        Action queue 中间的控制步不会重新运行 Matcher，因此调用方应按
        ``replan_index`` 去重。历史窗口尚未填满时 ``matcher_updates`` 为 0，
        但 ``demo_observation_index`` 仍会报告当前实际使用的初始锚点。
        """
        if self._matcher is None or self._demo_cache is None:
            return None

        result = self._matcher.last_result
        active_index = self._matcher.active_index
        anchor_index = int(self._demo_cache.anchor_indices[active_index])
        anchor_timestamp = self._demo_cache.timestamps[anchor_index]
        diagnostics: dict[str, int | float | None] = {
            "replan_index": self._replan_index,
            "matcher_updates": self._matcher.num_updates,
            "demo_chunk_index": active_index,
            "demo_observation_index": anchor_index,
            "demo_timestamp": float(anchor_timestamp),
            "phase": float(self._demo_cache.timestamps_to_phase(anchor_timestamp)),
            "confidence": None,
            "local_cost": None,
            "accumulated_cost": None,
            "observation_id": None,
            "observation_timestamp": None,
        }
        if result is not None:
            diagnostics.update(
                demo_chunk_index=result.demo_chunk_index,
                demo_observation_index=result.demo_observation_index,
                demo_timestamp=result.demo_timestamp,
                phase=result.phase,
                confidence=result.confidence,
                local_cost=result.local_cost,
                accumulated_cost=result.accumulated_cost,
                observation_id=result.observation_id,
                observation_timestamp=result.observation_timestamp,
            )
        return diagnostics

    @torch.no_grad()
    def set_demo(
        self,
        video: Tensor,
        states: Tensor,
        timestamps: Tensor,
        *,
        valid_mask: Tensor | None = None,
        state_normalizer: DemoStateNormalizer | None = None,
        matcher_image_key: str | None = None,
        matcher_siglip: SmolVLASigLIPHandle | None = None,
    ) -> None:
        """注册一条完整 Demo，并一次性建立 Global 与 Matcher 输入。

        ``video`` 使用 ``[T,3,H,W]``、值域 ``[0,1]``；``states`` 必须保留
        机器人的真实维度且尚未归一化；timestamps 的单位为秒。Demo cache
        默认保存在 ``demo_alignment.cache_device``；Global Demo 在这里编码
        一次，得到的 Task Tokens 会一直复用到下一次 ``set_demo``。
        """
        normalizer = state_normalizer or self.state_normalizer
        if normalizer is None:
            raise ValueError("set_demo 需要 dataset_stats 或显式 state_normalizer。")
        self._validate_state_normalizer(normalizer)

        image_key = matcher_image_key or next(iter(self.config.image_features), None)
        if image_key is None:
            raise ValueError("SmolVLA-ICL 至少需要一个 Observation 图像字段。")

        frame_mask = (
            torch.ones(video.shape[0], dtype=torch.bool, device=video.device)
            if valid_mask is None
            else valid_mask.to(device=video.device, dtype=torch.bool)
        )
        global_clips = build_global_demo_clips(
            video,
            states,
            timestamps,
            state_normalizer=normalizer,
            config=self.config.global_encoder,
            valid_mask=frame_mask,
        )
        model_device = next(self.model.parameters()).device
        global_mask = global_clips.valid_mask.unsqueeze(0).to(model_device)
        # 完整 S3D 输入留在 CPU，每次只上传少量 clip。S3D feature 很小，
        # 可以在 GPU 上拼接后继续进入可训练的 State/Fusion/Temporal 路径。
        global_features = self.model.global_encoder.encode_video_clips_batched(
            global_clips.video.unsqueeze(0),
            global_clips.valid_mask.unsqueeze(0),
        )
        global_output = self.model.global_encoder(
            global_features,
            global_clips.states.unsqueeze(0).to(model_device),
            global_clips.timestamps.unsqueeze(0).to(model_device),
            global_mask,
        )

        # Matcher snapshot 和模型视觉编码器是两条独立路径。只有当模型的
        # vision model 与 connector 都已冻结时，才允许直接复用它做 Matcher。
        model_siglip = reuse_smolvla_siglip(self.model, freeze=False)
        vision_trainable = any(
            parameter.requires_grad for parameter in model_siglip.vision_model.parameters()
        )
        connector_trainable = any(
            parameter.requires_grad for parameter in model_siglip.connector.parameters()
        )
        model_visual_trainable = vision_trainable or connector_trainable
        if matcher_siglip is None:
            if model_visual_trainable:
                raise ValueError(
                    "当 SmolVLA vision model 或 connector 可训练时，"
                    "set_demo 必须传入独立冻结的 matcher_siglip snapshot。"
                )
            matcher_siglip = reuse_smolvla_siglip(self.model, freeze=True)
        if not matcher_siglip.frozen:
            raise ValueError("matcher_siglip 必须是全程冻结的 snapshot。")
        shares_trainable_module = (
            matcher_siglip.vision_model is model_siglip.vision_model and vision_trainable
        ) or (matcher_siglip.connector is model_siglip.connector and connector_trainable)
        if shares_trainable_module:
            raise ValueError("Matcher 不能引用正在训练的模型 vision model 或 connector。")

        # 完整 Demo 始终留在来源设备（通常为 CPU）。每轮只上传一个视觉
        # batch，完成 resize/编码后立刻把缓存结果移回 cache_device，避免
        # ``set_demo`` 因整段 512x512 float 视频产生显存峰值。
        matcher_device = next(matcher_siglip.vision_model.parameters()).device
        cache_device = torch.device(self.config.demo_alignment.cache_device)
        matcher_feature_batches: list[Tensor] = []
        local_hidden_batches: list[Tensor] = []
        encode_batch_size = self.config.demo_alignment.demo_encode_batch_size
        same_visual_modules = (
            matcher_siglip.vision_model is model_siglip.vision_model
            and matcher_siglip.connector is model_siglip.connector
        )
        for start in range(0, len(video), encode_batch_size):
            end = min(start + encode_batch_size, len(video))
            batch_valid = frame_mask[start:end]

            matcher_batch = self.model._prepare_local_vision_images(video[start:end].to(matcher_device))
            matcher_batch = torch.where(
                batch_valid.to(matcher_device)[:, None, None, None],
                matcher_batch,
                torch.zeros_like(matcher_batch),
            )
            matcher_tokens = matcher_siglip.encode_visual_tokens(matcher_batch)
            matcher_feature_batches.append(
                pool_visual_tokens(matcher_tokens, normalize=False).detach().to(cache_device)
            )

            if same_visual_modules:
                local_tokens = matcher_tokens
            else:
                local_batch = self.model._prepare_local_vision_images(video[start:end].to(model_device))
                local_batch = torch.where(
                    batch_valid.to(model_device)[:, None, None, None],
                    local_batch,
                    torch.zeros_like(local_batch),
                )
                local_tokens = model_siglip.encode_visual_tokens(local_batch)
            local_hidden_batches.append(
                self.model.local_encoder.pool_spatial_tokens(local_tokens).detach().to(cache_device)
            )

        demo_cache = DemoEmbeddingCache.from_embeddings(
            torch.cat(matcher_feature_batches),
            states,
            timestamps,
            state_normalizer=normalizer,
            config=self.config.demo_alignment,
            valid_mask=frame_mask,
        )

        # rollout 时模型权重固定，因此每帧只缓存完成 spatial pooling 后的
        # visual hidden。它不属于 Matcher cache，也不会写入训练磁盘缓存。
        demo_visual_hidden = torch.cat(local_hidden_batches)
        local_valid = frame_mask.to(device=cache_device)
        demo_visual_hidden = torch.where(
            local_valid[:, None],
            demo_visual_hidden,
            torch.zeros_like(demo_visual_hidden),
        )

        # 两条 Demo 路径都成功后再一起替换，避免构建中途失败时留下
        # “新 Global + 旧 Matcher”的不一致 Policy 状态。
        self._global_output = global_output
        self._demo_cache = demo_cache
        # 这是 rollout 专用的模型视觉 cache。它与只保存 E_match pooled
        # feature 的 DemoEmbeddingCache 分开，Matcher 无法把自己的视觉空间
        # 混入 Local Demo Expert 输入。
        self._demo_visual_hidden = demo_visual_hidden
        self.state_normalizer = normalizer
        self._matcher_image_key = image_key
        self._matcher_siglip = matcher_siglip
        self._matcher = OnlineDTWMatcher(demo_cache)
        self.reset()

    def _normalized_state_to_raw(self, state: Tensor) -> Tensor:
        """恢复 Matcher 所需的未归一化真实维度 State。"""
        assert self.state_normalizer is not None
        normalizer = self.state_normalizer
        state = state[..., : normalizer.state_dim]
        mean = normalizer.mean.to(device=state.device, dtype=state.dtype)
        std = normalizer.std.to(device=state.device, dtype=state.dtype)
        return state * (std + normalizer.eps) + mean

    def _observation_timestamp(self, batch: dict[str, Tensor]) -> float:
        """优先读取真实 timestamp；缺失时使用配置的 Matcher 周期。"""
        timestamp = batch.get("timestamp")
        if timestamp is not None:
            return float(torch.as_tensor(timestamp).reshape(-1)[0].detach().cpu())
        return self._replan_index / self.config.demo_alignment.alignment_hz

    def _observation_id(self, batch: dict[str, Tensor]) -> int | None:
        """读取可选帧编号，只用于保存对齐结果的可追踪元数据。"""
        value = batch.get("frame_index", batch.get("index"))
        if value is None:
            return None
        return int(torch.as_tensor(value).reshape(-1)[0].detach().cpu())

    def _match_local_demo(
        self,
        batch: dict[str, Tensor],
        images: list[Tensor],
        image_embeddings: list[Tensor],
        image_masks: list[Tensor],
    ) -> EncodedLocalDemoBatch:
        """追加当前真实 Observation，推进 Matcher 并组装 Local batch。"""
        if (
            self._demo_cache is None
            or self._demo_visual_hidden is None
            or self._matcher is None
            or self._observation_history is None
            or self._matcher_image_key is None
            or self._matcher_siglip is None
        ):
            raise RuntimeError("推理前必须先调用 set_demo(...)。")

        present_image_keys = [key for key in self.config.image_features if key in batch]
        if self._matcher_image_key not in present_image_keys:
            raise KeyError(f"当前 Observation 缺少 Matcher 图像 {self._matcher_image_key!r}。")
        image_index = present_image_keys.index(self._matcher_image_key)

        current_state = batch[OBS_STATE]
        if current_state.ndim > 2:
            current_state = current_state[:, -1]
        raw_state = self._normalized_state_to_raw(current_state)
        model_siglip = reuse_smolvla_siglip(self.model, freeze=False)
        if (
            self._matcher_siglip.vision_model is model_siglip.vision_model
            and self._matcher_siglip.connector is model_siglip.connector
        ):
            matcher_tokens = image_embeddings[image_index]
        else:
            matcher_tokens = self._matcher_siglip.encode_visual_tokens(images[image_index])
        self._observation_history.append(
            raw_state=raw_state,
            timestamp=self._observation_timestamp(batch),
            visual_tokens=matcher_tokens,
            observation_id=self._observation_id(batch),
            valid=bool(image_masks[image_index].reshape(-1)[0]),
        )

        if self._observation_history.is_ready:
            _, local_window = self._matcher.update_and_extract(self._observation_history.get_latest_chunk())
        else:
            # episode 起始阶段历史窗口尚未填满时，从 Matcher 当前的第一个
            # 有效锚点读取 Local Chunk；积累充分后自然切换到 DTW 输出。
            anchor = int(self._demo_cache.anchor_indices[self._matcher.active_index])
            local_window = self._demo_cache.extract_local_window(anchor)

        self._replan_index += 1
        model_device = next(self.model.parameters()).device
        return build_encoded_local_demo_batch(
            local_window,
            self._demo_visual_hidden,
            expected_state_dim=self.config.max_state_dim,
        ).to(model_device)

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """匹配当前阶段并生成一个裁剪到真实维度的动作块。"""
        del kwargs
        if self._global_output is None:
            raise RuntimeError("推理前必须先调用 set_demo(...)。")
        if batch[OBS_STATE].shape[0] != 1:
            raise ValueError("当前有状态 Stage Matcher 的在线推理仅支持 batch_size=1。")

        images, image_masks = self.prepare_images(batch)
        # Query Prefix 使用训练后的 E_vision；Matcher 在下方使用独立
        # E_match snapshot。当两者确实是同一个冻结对象时才会复用 tokens。
        image_embeddings = self.model.encode_images(images)
        local_demo = self._match_local_demo(batch, images, image_embeddings, image_masks)
        state = self.prepare_state(batch)
        actions = self.model.sample_actions(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            self._global_output,
            local_demo,
            noise=noise,
            image_embeddings=image_embeddings,
        )

        action_dim = self.config.action_feature.shape[0]
        return actions[:, :, :action_dim]

    def forward(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """执行标准 ``policy(batch)`` 训练入口。

        Global/Local Demo 由 SmolVLA-ICL collate 放在 batch 的专用字段中；
        Trainer 无需了解结构化 Demo，也不需要额外的位置参数。
        """
        global_demo, local_demo = get_smolvla_icl_demo_batches(batch)
        model_device = batch[OBS_STATE].device
        global_demo = global_demo.to(model_device, non_blocking=True)
        # Local RGB 保留为 CPU uint8；仅 State/时间/mask 等轻量字段进入 GPU。
        # encode_local_demo 逐个上传视觉 micro-batch：视觉可训练时保留梯度，
        # 冻结时只关闭 SigLIP 路径的 autograd，不影响后续 Local Encoder。
        local_demo = local_demo.to(model_device, non_blocking=True)
        images, image_masks = self.prepare_images(batch)
        losses = self.model(
            images,
            image_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            self.prepare_state(batch),
            self.prepare_action(batch),
            global_demo,
            local_demo,
            noise,
            time,
        )
        losses = losses[:, :, : self.config.action_feature.shape[0]]
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = (~action_is_pad).unsqueeze(-1)
            losses = losses * valid

        if reduction == "none":
            if action_is_pad is None:
                loss = losses.mean(dim=(1, 2))
            else:
                per_sample_count = ((~action_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                loss = losses.sum(dim=(1, 2)) / per_sample_count
            scalar_loss = loss.mean()
        else:
            if action_is_pad is None:
                loss = losses.mean()
            else:
                valid_count = ((~action_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / valid_count
            scalar_loss = loss
        metrics = {
            "loss": float(scalar_loss.detach()),
            **{
                name: float(value.cpu())
                for name, value in self.model.vlm_with_expert.demo_gate_statistics().items()
            },
        }
        return loss, metrics
