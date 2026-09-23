"""SmolVLA-ICL 的 ``[P;G;L;A]`` 输入契约、Attention Mask 与位置编号。

P 是当前 Vision/Language/State Prefix，G/L 是 Global/Local Demo Tokens，
A 是 noisy Action Tokens。四个分支先分别计算 Q/K/V，再按此顺序联合；本模块
不直接拼接宽度不同的 raw hidden。所有 bool Attention Mask 均以 ``True`` 表示可见。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..common.vla_utils import make_att_2d_masks
from ..smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope

__all__ = [
    "CrossAttentionMasks",
    "FourRegionInputs",
    "FourRegionOutput",
    "LayerConditionKVCache",
    "RegionPositionIds",
    "SmolVLAICLConditionCache",
    "SmolVLMWithDemoExpertModel",
    "TokenRegionLayout",
    "build_cross_attention_masks",
    "build_demo_attention_mask",
    "build_region_position_ids",
    "build_union_attention_mask",
]


@dataclass(frozen=True, slots=True)
class TokenRegionLayout:
    """记录 ``[P;G;L;A]`` 在逻辑联合序列中的边界。"""

    prefix_length: int
    global_length: int
    local_length: int
    action_length: int

    @property
    def prefix(self) -> slice:
        return slice(0, self.prefix_length)

    @property
    def global_demo(self) -> slice:
        start = self.prefix_length
        return slice(start, start + self.global_length)

    @property
    def local_demo(self) -> slice:
        start = self.prefix_length + self.global_length
        return slice(start, start + self.local_length)

    @property
    def action(self) -> slice:
        start = self.prefix_length + self.global_length + self.local_length
        return slice(start, start + self.action_length)

    @property
    def total_length(self) -> int:
        return self.prefix_length + self.global_length + self.local_length + self.action_length


@dataclass(frozen=True, slots=True)
class FourRegionInputs:
    """四分区 hidden、valid mask 与 Prefix 内部分块信息。

    ``prefix_block_mask`` 沿用 SmolVLA 的一维 Prefix-LM 标记：相同累计值
    属于同一个双向可见块，值发生递增后，后面的 token 可以读取此前的块。
    它会与 ``prefix_valid_mask`` 一起交给 ``make_att_2d_masks``，从而完整
    保留 Vision/Language/State 的原始可见关系。
    """

    prefix_hidden: Tensor
    global_hidden: Tensor
    local_hidden: Tensor
    action_hidden: Tensor
    prefix_valid_mask: Tensor
    global_valid_mask: Tensor
    local_valid_mask: Tensor
    action_valid_mask: Tensor
    prefix_block_mask: Tensor
    local_anchor_positions: Tensor

    def __post_init__(self) -> None:
        """只检查后续区域切分和 QKV 拼接所必需的核心契约。"""
        hiddens = (self.prefix_hidden, self.global_hidden, self.local_hidden, self.action_hidden)
        masks = (
            self.prefix_valid_mask,
            self.global_valid_mask,
            self.local_valid_mask,
            self.action_valid_mask,
        )
        if any(hidden.ndim != 3 for hidden in hiddens):
            raise ValueError("P/G/L/A hidden 必须全部为 (B,N,D) Tensor。")
        if len({hidden.shape[0] for hidden in hiddens}) != 1:
            raise ValueError("P/G/L/A hidden 的 batch size 必须一致。")
        if any(mask.shape != hidden.shape[:2] for mask, hidden in zip(masks, hiddens, strict=True)):
            raise ValueError("P/G/L/A valid mask 必须与对应 hidden 的 (B,N) 对齐。")
        if self.prefix_block_mask.shape != self.prefix_valid_mask.shape:
            raise ValueError("prefix_block_mask 必须与 prefix_valid_mask 形状一致。")
        if self.local_anchor_positions.shape != (self.prefix_hidden.shape[0],):
            raise ValueError("local_anchor_positions 必须为 (B,) Tensor。")
        if len({hidden.shape[-1] for hidden in hiddens[1:]}) != 1:
            raise ValueError("Global、Local 和 Action hidden width 必须一致。")
        if (
            len(
                {
                    tensor.device
                    for tensor in (
                        *hiddens,
                        *masks,
                        self.prefix_block_mask,
                        self.local_anchor_positions,
                    )
                }
            )
            != 1
        ):
            raise ValueError("P/G/L/A hidden 和 mask 必须位于同一设备。")

    @property
    def batch_size(self) -> int:
        """返回 batch size。"""
        return self.prefix_hidden.shape[0]

    @property
    def layout(self) -> TokenRegionLayout:
        """根据运行时 token 数生成逻辑区域布局。"""
        return TokenRegionLayout(
            prefix_length=self.prefix_hidden.shape[1],
            global_length=self.global_hidden.shape[1],
            local_length=self.local_hidden.shape[1],
            action_length=self.action_hidden.shape[1],
        )


@dataclass(frozen=True, slots=True)
class FourRegionOutput:
    """保存三分支 Transformer 某一深度的四区域 hidden。"""

    prefix_hidden: Tensor
    global_hidden: Tensor
    local_hidden: Tensor
    action_hidden: Tensor


@dataclass(frozen=True, slots=True)
class LayerConditionKVCache:
    """一个 Transformer layer 中供 Action 分支重复读取的条件 K/V。"""

    union_prefix_key: Tensor | None = None
    union_prefix_value: Tensor | None = None
    cross_prefix_key: Tensor | None = None
    cross_prefix_value: Tensor | None = None
    cross_local_key: Tensor | None = None
    cross_local_value: Tensor | None = None


@dataclass(frozen=True, slots=True)
class SmolVLAICLConditionCache:
    """一次策略重规划期间固定的 P/G/L 条件缓存。

    Even/Union 层只缓存 Action 可读取的 Prefix K/V；Odd/Cross 层缓存
    ``A<-P'`` 与 ``A<-L`` 的两组 K/V。Action 自身的 K/V 依赖当前去噪
    状态，必须在每个 Euler step 重新计算，因此不放入本缓存。
    """

    layers: tuple[LayerConditionKVCache, ...]
    union_action_mask: Tensor
    action_from_prefix_mask: Tensor
    action_from_local_mask: Tensor
    action_position_ids: Tensor
    action_valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class CrossAttentionMasks:
    """三条允许的有向 Cross-Attention 边。"""

    prefix_from_global: Tensor
    action_from_prefix: Tensor
    action_from_local: Tensor


@dataclass(frozen=True, slots=True)
class RegionPositionIds:
    """四个区域各自的 RoPE position IDs。"""

    prefix: Tensor
    global_demo: Tensor
    local_demo: Tensor
    action: Tensor

    @property
    def concatenated(self) -> Tensor:
        """按照 ``[P;G;L;A]`` 顺序返回联合 position IDs。"""
        return torch.cat(
            [self.prefix, self.global_demo, self.local_demo, self.action],
            dim=1,
        )


class _DirectedCrossAttention(nn.Module):
    """保存一条新增有向 Cross-Attention 的独立参数。

    ``P<-G`` 和 ``A<-L`` 分别实例化本模块，不共享 Norm、Q/K/V、输出投影
    或 gate。实际 attention 计算继续调用 SmolVLA 原有实现，以保持 head
    展开和数值精度一致。预训练的 ``A<-P`` 使用 Action Expert 原参数，
    不经过本模块。
    """

    def __init__(
        self,
        *,
        prefix_hidden_size: int,
        demo_hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.RMSNorm(prefix_hidden_size, eps=rms_norm_eps)
        self.key_value_norm = nn.RMSNorm(demo_hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(
            prefix_hidden_size,
            num_attention_heads * head_dim,
            bias=attention_bias,
        )
        self.k_proj = nn.Linear(
            demo_hidden_size,
            num_key_value_heads * head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            demo_hidden_size,
            num_key_value_heads * head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            prefix_hidden_size,
            bias=attention_bias,
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))


class SmolVLMWithDemoExpertModel(SmolVLMWithExpertModel):
    """在官方 SmolVLM+Action Expert 上增加 ICL 所需的分支接口。

    当前阶段保持父类创建的 ``vlm``、``lm_expert``、视觉编码器、connector、
    tokenizer 和参数路径不变；Demo Expert 使用与 Action Expert 相同的缩放配置
    独立初始化，并增加独立的 ``P<-G``、``A<-L`` adapters。

    VLM 在两类层中的行为为：

    * Union 层：外部联合计算 ``[P;G;L;A]`` attention，本类只负责把 P 区域
      输出送入原 VLM output projection、残差和 MLP；
    * Cross 层：先执行原 VLM Prefix Self-Attention block，再通过独立参数
      执行 ``P<-G``；Action 随后复用预训练参数执行 ``A<-P'``，再通过
      独立参数执行 ``A<-L``；Demo 同时在 ``G/L`` block-diagonal Mask 下
      执行自己的 Self-Attention。两个 Expert 最后分别运行自己的 Norm/MLP。
    """

    def __init__(
        self,
        *args,
        global_cross_gate_init: float = 1e-3,
        local_cross_gate_init: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.num_expert_layers != self.num_vlm_layers
            or self.attention_mode != "cross_attn"
            or self.self_attn_every_n_layers != 2
        ):
            raise ValueError("SmolVLA-ICL 要求 VLM/Expert 等深，且固定采用偶数 Union、奇数 Cross。")
        text_config = self.config.text_config
        expert_config = self.lm_expert.config

        # Demo Expert 使用与 Action Expert 完全相同的 Transformer 配置，保证
        # Union Attention 中三条分支的 head 数和 head_dim 可以直接拼接。
        # 这里从 config 新建模型而不是 deepcopy Action Expert，因此两者既不
        # 共享参数，也不会错误继承已经训练过的 Action 权重。
        self.demo_expert = type(self.lm_expert)(copy.deepcopy(expert_config))
        self.demo_expert.embed_tokens = None
        for demo_layer, action_layer in zip(
            self.demo_expert.layers,
            self.lm_expert.layers,
            strict=True,
        ):
            reference_weight = action_layer.self_attn.q_proj.weight
            demo_layer.to(device=reference_weight.device, dtype=reference_weight.dtype)
        final_reference = self.lm_expert.norm.weight
        self.demo_expert.norm.to(device=final_reference.device, dtype=final_reference.dtype)

        cross_layer_indices = range(1, self.num_vlm_layers, 2)

        # 每个 Cross 层拥有独立的 Global->Prefix 参数。adapter 跟随对应 VLM
        # layer 的 device/dtype，兼容预训练模型的 bf16 和可能的模型切分。
        self.prefix_from_global = nn.ModuleDict()
        for layer_idx in cross_layer_indices:
            vlm_layer = self.get_vlm_model().text_model.layers[layer_idx]
            reference_weight = vlm_layer.self_attn.q_proj.weight
            adapter = _DirectedCrossAttention(
                prefix_hidden_size=text_config.hidden_size,
                demo_hidden_size=self.expert_hidden_size,
                num_attention_heads=text_config.num_attention_heads,
                num_key_value_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                rms_norm_eps=text_config.rms_norm_eps,
                attention_bias=text_config.attention_bias,
                gate_init=global_cross_gate_init,
            ).to(device=reference_weight.device, dtype=reference_weight.dtype)
            self.prefix_from_global[str(layer_idx)] = adapter

        # ``A<-L`` 的 Query/Key-Value 都是 Expert hidden width，但仍与
        # ``P<-G`` 使用完全独立的参数和 gate。
        self.action_from_local = nn.ModuleDict()
        for layer_idx in cross_layer_indices:
            action_layer = self._get_action_layer(layer_idx)
            reference_weight = action_layer.self_attn.q_proj.weight
            adapter = _DirectedCrossAttention(
                prefix_hidden_size=self.expert_hidden_size,
                demo_hidden_size=self.expert_hidden_size,
                num_attention_heads=expert_config.num_attention_heads,
                num_key_value_heads=expert_config.num_key_value_heads,
                head_dim=expert_config.head_dim,
                rms_norm_eps=expert_config.rms_norm_eps,
                attention_bias=expert_config.attention_bias,
                gate_init=local_cross_gate_init,
            ).to(device=reference_weight.device, dtype=reference_weight.dtype)
            self.action_from_local[str(layer_idx)] = adapter

    def set_requires_grad(self) -> None:
        """保留 expert-only 训练，但允许 Query/Local 共享的视觉路径更新。

        官方 SmolVLA 的 ``train_expert_only=True`` 会冻结整个 VLM。ICL 在
        ``freeze_vision_encoder=False`` 时只重新开启 vision model 和 connector，
        text model 仍按官方 expert-only 语义冻结。
        """
        super().set_requires_grad()
        if not self.freeze_vision_encoder:
            for module in (self.get_vlm_model().vision_model, self.get_vlm_model().connector):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        """让可训练视觉路径保持 train mode，其余冻结 VLM 保持 eval。"""
        super().train(mode)
        if mode and not self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.train()
            self.get_vlm_model().connector.train()
        return self

    def _get_action_layer(self, layer_idx: int) -> nn.Module:
        """取得与 VLM 同深度的 Action Expert layer。"""
        return self.lm_expert.layers[layer_idx]

    def _get_demo_layer(self, layer_idx: int) -> nn.Module:
        """取得与 VLM 同深度的 Demo Expert layer。"""
        return self.demo_expert.layers[layer_idx]

    def demo_gate_statistics(self) -> dict[str, Tensor]:
        """返回两类 Demo Cross-Attention gate 的平均绝对值。

        该指标只用于训练日志，便于及时发现 gate 始终接近初始化值、模型可能
        忽略 Demo 的情况；不参与 loss，也不改变 Attention 计算图。
        """
        global_gates = torch.stack(
            [adapter.gate.detach().float().abs() for adapter in self.prefix_from_global.values()]
        )
        local_gates = torch.stack(
            [adapter.gate.detach().float().abs() for adapter in self.action_from_local.values()]
        )
        return {
            "demo_global_gate_abs_mean": global_gates.mean(),
            "demo_local_gate_abs_mean": local_gates.mean(),
        }

    def _project_vlm_qkv(self, prefix_hidden: Tensor, layer_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """使用指定 VLM layer 的原始参数计算 Prefix Q/K/V。"""
        layer = self.get_vlm_model().text_model.layers[layer_idx]
        normalized = layer.input_layernorm(prefix_hidden)
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(normalized).view(hidden_shape)
        key = layer.self_attn.k_proj(normalized).view(hidden_shape)
        value = layer.self_attn.v_proj(normalized).view(hidden_shape)
        return query, key, value

    def apply_vlm_attention_output(
        self,
        prefix_hidden: Tensor,
        attention_output: Tensor,
        layer_idx: int,
        prefix_valid_mask: Tensor,
    ) -> Tensor:
        """执行官方 VLM layer 的 output projection、两个残差和 MLP。

        该函数同时服务两类层：Union 层传入联合 Attention 中切出的 P 输出；
        Prefix Self 层传入本类刚计算的 P Self-Attention 输出。最后重新清零
        padding hidden，避免全屏蔽 Query 行经 softmax 后产生的数值残留。
        """
        layer = self.get_vlm_model().text_model.layers[layer_idx]
        attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
        after_attention = prefix_hidden + layer.self_attn.o_proj(attention_output)
        output = after_attention + layer.mlp(layer.post_attention_layernorm(after_attention))
        return output * prefix_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def forward_vlm_self_layer(
        self,
        prefix_hidden: Tensor,
        layer_idx: int,
        attention_mask: Tensor,
        position_ids: Tensor,
        prefix_valid_mask: Tensor,
    ) -> Tensor:
        """按 SmolVLA 原顺序执行一个完整的 VLM Prefix Self-Attention block。"""
        query, key, value = self._project_vlm_qkv(prefix_hidden, layer_idx)
        query = apply_rope(query, position_ids)
        key = apply_rope(key, position_ids)
        attention_output = self.eager_attention_forward(
            attention_mask,
            prefix_hidden.shape[0],
            self.config.text_config.head_dim,
            query,
            key,
            value,
        )
        return self.apply_vlm_attention_output(
            prefix_hidden,
            attention_output,
            layer_idx,
            prefix_valid_mask,
        )

    def condition_vlm_on_global(
        self,
        prefix_hidden: Tensor,
        global_hidden: Tensor,
        layer_idx: int,
        attention_mask: Tensor,
        prefix_position_ids: Tensor,
        global_position_ids: Tensor,
        prefix_valid_mask: Tensor,
    ) -> Tensor:
        """执行设计中的 ``P<-G``，并以门控残差写回 Prefix。

        Global 只在该有向路径中改变 VLM；本函数不允许 Prefix 反向写入 Demo，
        也不读取 Local 或 Action。默认 gate 是很小的非零值，因此新增分支
        初始仅产生微弱修正，但 Demo 路径从第一次反传起即可联合训练。
        """
        adapter = self.prefix_from_global[str(layer_idx)]
        query_hidden = adapter.query_norm(prefix_hidden).to(dtype=adapter.q_proj.weight.dtype)
        key_value_hidden = adapter.key_value_norm(global_hidden).to(dtype=adapter.k_proj.weight.dtype)

        batch_size = prefix_hidden.shape[0]
        head_dim = self.config.text_config.head_dim
        query = adapter.q_proj(query_hidden).view(batch_size, prefix_hidden.shape[1], -1, head_dim)
        key = adapter.k_proj(key_value_hidden).view(batch_size, global_hidden.shape[1], -1, head_dim)
        value = adapter.v_proj(key_value_hidden).view(batch_size, global_hidden.shape[1], -1, head_dim)
        query = apply_rope(query, prefix_position_ids)
        key = apply_rope(key, global_position_ids)

        attention_output = self.eager_attention_forward(
            attention_mask,
            batch_size,
            head_dim,
            query,
            key,
            value,
        )
        attention_output = adapter.o_proj(attention_output.to(dtype=adapter.o_proj.weight.dtype))
        # 没有有效 Global Key 的 Query 不应接收 eager softmax 的均匀输出。
        attention_output = attention_output * attention_mask.any(dim=-1).unsqueeze(-1)
        gate = adapter.gate.to(dtype=attention_output.dtype)
        output = prefix_hidden + gate * attention_output
        return output * prefix_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def finalize_vlm_hidden(self, prefix_hidden: Tensor, prefix_valid_mask: Tensor) -> Tensor:
        """应用官方 VLM final norm，并保持 padding token 为零。"""
        output = self.get_vlm_model().text_model.norm(prefix_hidden)
        return output * prefix_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def _project_demo_qkv(
        self,
        demo_hidden: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """使用 Demo Expert 独立参数计算 ``[G;L]`` 的 Q/K/V。

        Demo Expert 和 Action Expert 采用相同的 hidden width、head 数与
        ``head_dim``，因此本函数产生的 Q/K/V 可以在 Union 层与 P、A 的
        Q/K/V 沿 token 维直接拼接；raw hidden 本身仍不跨分支拼接。
        """
        layer = self._get_demo_layer(layer_idx)
        normalized = layer.input_layernorm(demo_hidden)
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(normalized).view(hidden_shape)
        key = layer.self_attn.k_proj(normalized).view(hidden_shape)
        value = layer.self_attn.v_proj(normalized).view(hidden_shape)
        return query, key, value

    def apply_demo_attention_output(
        self,
        demo_hidden: Tensor,
        attention_output: Tensor,
        layer_idx: int,
        demo_valid_mask: Tensor,
    ) -> Tensor:
        """将 Attention 输出写回 Demo Expert，并完成该层的 MLP。

        Union 层传入联合 Attention 中切出的 ``[G;L]`` 输出；Demo Self 层
        传入 block-diagonal Self-Attention 输出。二者共用同一套 Demo layer
        的 output projection、两个残差和 MLP，不额外引入旁路参数。
        """
        layer = self._get_demo_layer(layer_idx)
        attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
        after_attention = demo_hidden + layer.self_attn.o_proj(attention_output)
        output = after_attention + layer.mlp(layer.post_attention_layernorm(after_attention))
        return output * demo_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def forward_demo_self_layer(
        self,
        demo_hidden: Tensor,
        layer_idx: int,
        attention_mask: Tensor,
        position_ids: Tensor,
        demo_valid_mask: Tensor,
    ) -> Tensor:
        """执行一个完整的 Demo Expert ``[G;L]`` Self-Attention block。

        调用方应传入 :func:`build_demo_attention_mask` 生成的 block-diagonal
        Mask。这样 G 和 L 共享 Demo Expert 的层参数，但在该注意力步骤中
        只能读取各自区域，不会发生 ``G<->L`` 信息泄漏。
        """
        layer = self._get_demo_layer(layer_idx)
        query, key, value = self._project_demo_qkv(demo_hidden, layer_idx)
        query = apply_rope(query, position_ids)
        key = apply_rope(key, position_ids)
        attention_output = self.eager_attention_forward(
            attention_mask,
            demo_hidden.shape[0],
            layer.self_attn.head_dim,
            query,
            key,
            value,
        )
        return self.apply_demo_attention_output(
            demo_hidden,
            attention_output,
            layer_idx,
            demo_valid_mask,
        )

    def finalize_demo_hidden(self, demo_hidden: Tensor, demo_valid_mask: Tensor) -> Tensor:
        """应用 Demo Expert 独立的 final norm，并保持 padding token 为零。"""
        output = self.demo_expert.norm(demo_hidden)
        return output * demo_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def _project_action_qkv(
        self,
        action_hidden: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """使用 Action Expert 原参数计算 Union 层中的 Action Q/K/V。"""
        layer = self._get_action_layer(layer_idx)
        normalized = layer.input_layernorm(action_hidden)
        normalized = normalized.to(dtype=layer.self_attn.q_proj.weight.dtype)
        hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(normalized).view(hidden_shape)
        key = layer.self_attn.k_proj(normalized).view(hidden_shape)
        value = layer.self_attn.v_proj(normalized).view(hidden_shape)
        return query, key, value

    def apply_action_attention_output(
        self,
        action_hidden: Tensor,
        attention_output: Tensor,
        layer_idx: int,
        action_valid_mask: Tensor,
    ) -> Tensor:
        """将 Union Attention 的 A 区域输出写回 Action Expert block。"""
        layer = self._get_action_layer(layer_idx)
        attention_output = attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
        after_attention = action_hidden + layer.self_attn.o_proj(attention_output)
        output = after_attention + layer.mlp(layer.post_attention_layernorm(after_attention))
        return output * action_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def _project_prefix_kv_for_action(
        self,
        prefix_hidden: Tensor,
        layer_idx: int,
        prefix_position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """生成 Odd/Cross 层 ``A<-P'`` 使用的已变换 Prefix K/V。"""
        action_layer = self._get_action_layer(layer_idx)
        vlm_layer = self.get_vlm_model().text_model.layers[layer_idx]
        batch_size = prefix_hidden.shape[0]
        prefix_normalized = vlm_layer.input_layernorm(prefix_hidden)
        prefix_normalized = prefix_normalized.to(dtype=vlm_layer.self_attn.k_proj.weight.dtype)
        prefix_key = vlm_layer.self_attn.k_proj(prefix_normalized).view(
            batch_size,
            prefix_hidden.shape[1],
            -1,
            vlm_layer.self_attn.head_dim,
        )
        prefix_value = vlm_layer.self_attn.v_proj(prefix_normalized).view(
            batch_size,
            prefix_hidden.shape[1],
            -1,
            vlm_layer.self_attn.head_dim,
        )
        prefix_key = apply_rope(prefix_key, prefix_position_ids)

        # 官方 Odd-layer Action k/v projection 将 VLM KV heads 映射到
        # Action Expert KV heads；缓存保存的是完成该映射后的结果。
        prefix_key = action_layer.self_attn.k_proj(prefix_key.flatten(-2)).view(
            batch_size,
            prefix_hidden.shape[1],
            -1,
            action_layer.self_attn.head_dim,
        )
        prefix_value = action_layer.self_attn.v_proj(prefix_value.flatten(-2)).view(
            batch_size,
            prefix_hidden.shape[1],
            -1,
            action_layer.self_attn.head_dim,
        )
        return prefix_key, prefix_value

    def _project_local_kv_for_action(
        self,
        local_hidden: Tensor,
        layer_idx: int,
        local_position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """生成 Odd/Cross 层 ``A<-L`` 使用的 Local K/V。"""
        adapter = self.action_from_local[str(layer_idx)]
        key_value_hidden = adapter.key_value_norm(local_hidden).to(dtype=adapter.k_proj.weight.dtype)
        batch_size = local_hidden.shape[0]
        head_dim = self.lm_expert.config.head_dim
        key = adapter.k_proj(key_value_hidden).view(
            batch_size,
            local_hidden.shape[1],
            -1,
            head_dim,
        )
        value = adapter.v_proj(key_value_hidden).view(
            batch_size,
            local_hidden.shape[1],
            -1,
            head_dim,
        )
        return apply_rope(key, local_position_ids), value

    def condition_action_on_prefix(
        self,
        action_hidden: Tensor,
        prefix_hidden: Tensor,
        layer_idx: int,
        attention_mask: Tensor,
        action_position_ids: Tensor,
        prefix_position_ids: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        """复用预训练 Action Expert 参数执行 ``A<-P'``。

        Action 是 Query，已经融合 Global 的 Prefix 是 Key/Value。这里沿用
        SmolVLA Cross 层的投影结构：VLM 先生成 Prefix K/V，Action Expert
        再用原有 K/V 投影把它们映射到自身空间。该函数只完成 attention
        output projection 和第一个残差；Action MLP 要等 ``A<-L`` 完成后执行。
        """
        action_layer = self._get_action_layer(layer_idx)
        batch_size = action_hidden.shape[0]
        head_dim = action_layer.self_attn.head_dim

        action_normalized = action_layer.input_layernorm(action_hidden)
        action_normalized = action_normalized.to(dtype=action_layer.self_attn.q_proj.weight.dtype)
        action_query = action_layer.self_attn.q_proj(action_normalized).view(
            batch_size,
            action_hidden.shape[1],
            -1,
            action_layer.self_attn.head_dim,
        )
        # 与官方 Cross-Attention 一致，Action Query 的 RoPE 从 0 开始。
        action_cross_position_ids = (
            action_position_ids
            - action_position_ids.min(
                dim=1,
                keepdim=True,
            ).values
        )
        action_query = apply_rope(action_query, action_cross_position_ids)
        prefix_key, prefix_value = self._project_prefix_kv_for_action(
            prefix_hidden,
            layer_idx,
            prefix_position_ids,
        )
        attention_output = self.eager_attention_forward(
            attention_mask,
            batch_size,
            head_dim,
            action_query,
            prefix_key,
            prefix_value,
        )
        attention_output = attention_output * attention_mask.any(dim=-1).unsqueeze(-1)
        attention_output = attention_output.to(dtype=action_layer.self_attn.o_proj.weight.dtype)
        output = action_hidden + action_layer.self_attn.o_proj(attention_output)
        return output * action_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def condition_action_on_local(
        self,
        action_hidden: Tensor,
        local_hidden: Tensor,
        layer_idx: int,
        attention_mask: Tensor,
        action_position_ids: Tensor,
        local_position_ids: Tensor,
        action_valid_mask: Tensor,
    ) -> Tensor:
        """使用独立参数执行 ``A<-L``，输入 Action 已经完成 ``A<-P'``。"""
        adapter = self.action_from_local[str(layer_idx)]
        query_hidden = adapter.query_norm(action_hidden).to(dtype=adapter.q_proj.weight.dtype)

        batch_size = action_hidden.shape[0]
        head_dim = self.lm_expert.config.head_dim
        query = adapter.q_proj(query_hidden).view(batch_size, action_hidden.shape[1], -1, head_dim)
        key, value = self._project_local_kv_for_action(
            local_hidden,
            layer_idx,
            local_position_ids,
        )
        action_cross_position_ids = (
            action_position_ids
            - action_position_ids.min(
                dim=1,
                keepdim=True,
            ).values
        )
        query = apply_rope(query, action_cross_position_ids)

        attention_output = self.eager_attention_forward(
            attention_mask,
            batch_size,
            head_dim,
            query,
            key,
            value,
        )
        attention_output = adapter.o_proj(attention_output.to(dtype=adapter.o_proj.weight.dtype))
        attention_output = attention_output * attention_mask.any(dim=-1).unsqueeze(-1)
        gate = adapter.gate.to(dtype=attention_output.dtype)
        output = action_hidden + gate * attention_output
        return output * action_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def finish_action_cross_layer(
        self,
        action_hidden: Tensor,
        layer_idx: int,
        action_valid_mask: Tensor,
    ) -> Tensor:
        """在 ``A<-P'``、``A<-L`` 之后执行 Action Expert 自己的 Norm/MLP。"""
        layer = self._get_action_layer(layer_idx)
        output = action_hidden + layer.mlp(layer.post_attention_layernorm(action_hidden))
        return output * action_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def finalize_action_hidden(self, action_hidden: Tensor, action_valid_mask: Tensor) -> Tensor:
        """应用官方 Action Expert final norm，并保持 padding token 为零。"""
        output = self.lm_expert.norm(action_hidden)
        return output * action_valid_mask.to(dtype=output.dtype).unsqueeze(-1)

    def _is_union_layer(self, layer_idx: int) -> bool:
        """首版固定偶数层执行 Union、奇数层执行 Cross。"""
        return layer_idx % 2 == 0

    def _forward_union_layer(
        self,
        hidden: FourRegionOutput,
        inputs: FourRegionInputs,
        layer_idx: int,
        attention_mask: Tensor,
        position_ids: RegionPositionIds,
    ) -> FourRegionOutput:
        """执行一个 ``[P;G;L;A]`` Union Self-Attention layer。

        四个区域先使用所属分支的独立 Norm 和 Q/K/V 投影，再沿 token 维
        组成一次 Attention。输出按区域切回后，分别经过各自的输出投影、
        残差和 MLP；不同分支从始至终不共享 Transformer 参数。
        """
        demo_hidden = torch.cat([hidden.global_hidden, hidden.local_hidden], dim=1)
        prefix_qkv = self._project_vlm_qkv(hidden.prefix_hidden, layer_idx)
        demo_qkv = self._project_demo_qkv(demo_hidden, layer_idx)
        action_qkv = self._project_action_qkv(hidden.action_hidden, layer_idx)

        # 三个分支的 head 数与 head_dim 相同；这里只拼接投影后的 Q/K/V，
        # 不直接拼接宽度分别为 960 和 720 的 raw hidden。
        query = torch.cat([prefix_qkv[0], demo_qkv[0], action_qkv[0]], dim=1)
        key = torch.cat([prefix_qkv[1], demo_qkv[1], action_qkv[1]], dim=1)
        value = torch.cat([prefix_qkv[2], demo_qkv[2], action_qkv[2]], dim=1)
        query = apply_rope(query, position_ids.concatenated)
        key = apply_rope(key, position_ids.concatenated)
        attention_output = self.eager_attention_forward(
            attention_mask,
            inputs.batch_size,
            self.config.text_config.head_dim,
            query,
            key,
            value,
        )

        layout = inputs.layout
        prefix_attention = attention_output[:, layout.prefix]
        demo_attention = attention_output[
            :,
            layout.global_demo.start : layout.local_demo.stop,
        ]
        action_attention = attention_output[:, layout.action]
        demo_valid_mask = torch.cat(
            [inputs.global_valid_mask, inputs.local_valid_mask],
            dim=1,
        )

        prefix_hidden = self.apply_vlm_attention_output(
            hidden.prefix_hidden,
            prefix_attention,
            layer_idx,
            inputs.prefix_valid_mask,
        )
        demo_hidden = self.apply_demo_attention_output(
            demo_hidden,
            demo_attention,
            layer_idx,
            demo_valid_mask,
        )
        action_hidden = self.apply_action_attention_output(
            hidden.action_hidden,
            action_attention,
            layer_idx,
            inputs.action_valid_mask,
        )
        global_length = inputs.layout.global_length
        return FourRegionOutput(
            prefix_hidden,
            demo_hidden[:, :global_length],
            demo_hidden[:, global_length:],
            action_hidden,
        )

    def _forward_cross_layer(
        self,
        hidden: FourRegionOutput,
        inputs: FourRegionInputs,
        layer_idx: int,
        union_mask: Tensor,
        demo_mask: Tensor,
        cross_masks: CrossAttentionMasks,
        position_ids: RegionPositionIds,
    ) -> FourRegionOutput:
        """按设计顺序执行一个 Cross-Conditioning layer。"""
        layout = inputs.layout

        # 1. VLM 先独立完成原有 Prefix Self-Attention、残差和 MLP。
        prefix_hidden = self.forward_vlm_self_layer(
            hidden.prefix_hidden,
            layer_idx,
            union_mask[:, layout.prefix, layout.prefix],
            position_ids.prefix,
            inputs.prefix_valid_mask,
        )

        # 2. Demo Expert 同时更新 G/L，但 block-diagonal Mask 禁止两区互读。
        demo_hidden = torch.cat([hidden.global_hidden, hidden.local_hidden], dim=1)
        demo_valid_mask = torch.cat(
            [inputs.global_valid_mask, inputs.local_valid_mask],
            dim=1,
        )
        demo_position_ids = torch.cat(
            [position_ids.global_demo, position_ids.local_demo],
            dim=1,
        )
        demo_hidden = self.forward_demo_self_layer(
            demo_hidden,
            layer_idx,
            demo_mask,
            demo_position_ids,
            demo_valid_mask,
        )
        global_length = layout.global_length
        global_hidden = demo_hidden[:, :global_length]
        local_hidden = demo_hidden[:, global_length:]

        # 3. 更新后的 Global 写入 Prefix。这样下一步 Action 读取到的是 P'。
        prefix_hidden = self.condition_vlm_on_global(
            prefix_hidden,
            global_hidden,
            layer_idx,
            cross_masks.prefix_from_global,
            position_ids.prefix,
            position_ids.global_demo,
            inputs.prefix_valid_mask,
        )

        # 4-5. Action 先读取融合 Global 后的 P'，再用更新后的 Local 修正。
        action_hidden = self.condition_action_on_prefix(
            hidden.action_hidden,
            prefix_hidden,
            layer_idx,
            cross_masks.action_from_prefix,
            position_ids.action,
            position_ids.prefix,
            inputs.action_valid_mask,
        )
        action_hidden = self.condition_action_on_local(
            action_hidden,
            local_hidden,
            layer_idx,
            cross_masks.action_from_local,
            position_ids.action,
            position_ids.local_demo,
            inputs.action_valid_mask,
        )
        action_hidden = self.finish_action_cross_layer(
            action_hidden,
            layer_idx,
            inputs.action_valid_mask,
        )
        return FourRegionOutput(
            prefix_hidden,
            global_hidden,
            local_hidden,
            action_hidden,
        )

    def forward(self, inputs: FourRegionInputs) -> FourRegionOutput:
        """运行 SmolVLA-ICL 三分支 Transformer。

        显式覆盖父类只支持 ``[VLM, Action]`` 的 ``forward``，避免调用本模型
        时静默绕过 Demo Expert。原始模态 embedding 和 flow-matching loss
        由外层 :class:`VLAFlowMatchingICL` 负责。
        """
        return self.run_transformer_layers(inputs)

    def run_transformer_layers(self, inputs: FourRegionInputs) -> FourRegionOutput:
        """运行完整的 Union/Cross 交替主干，但不承担顶层模型 ``forward``。

        本接口只消费已经构造好的 P/G/L/A hidden 与 Mask 契约。图像、语言、
        State、Demo 和 noisy action 的 embedding，以及最终 flow-matching head，
        将在顶层 ``forward`` 中统一接入。
        """
        union_mask = build_union_attention_mask(inputs)
        demo_mask = build_demo_attention_mask(inputs)
        cross_masks = build_cross_attention_masks(inputs)
        position_ids = build_region_position_ids(inputs)
        hidden = FourRegionOutput(
            inputs.prefix_hidden,
            inputs.global_hidden,
            inputs.local_hidden,
            inputs.action_hidden,
        )

        for layer_idx in range(self.num_vlm_layers):
            if self._is_union_layer(layer_idx):
                hidden = self._forward_union_layer(
                    hidden,
                    inputs,
                    layer_idx,
                    union_mask,
                    position_ids,
                )
            else:
                hidden = self._forward_cross_layer(
                    hidden,
                    inputs,
                    layer_idx,
                    union_mask,
                    demo_mask,
                    cross_masks,
                    position_ids,
                )

        demo_hidden = torch.cat([hidden.global_hidden, hidden.local_hidden], dim=1)
        demo_valid_mask = torch.cat(
            [inputs.global_valid_mask, inputs.local_valid_mask],
            dim=1,
        )
        demo_hidden = self.finalize_demo_hidden(demo_hidden, demo_valid_mask)
        global_length = inputs.layout.global_length
        return FourRegionOutput(
            prefix_hidden=self.finalize_vlm_hidden(
                hidden.prefix_hidden,
                inputs.prefix_valid_mask,
            ),
            global_hidden=demo_hidden[:, :global_length],
            local_hidden=demo_hidden[:, global_length:],
            action_hidden=self.finalize_action_hidden(
                hidden.action_hidden,
                inputs.action_valid_mask,
            ),
        )

    def build_condition_cache(self, inputs: FourRegionInputs) -> SmolVLAICLConditionCache:
        """预计算一次去噪过程中不随 Action 改变的 P/G/L hidden 与 K/V。

        Union Mask 禁止 P/G/L 读取 A，Cross 层也只允许条件分支写入 Action，
        因而 P/G/L 的逐层演化与 noisy action 无关。这里先完整推进三个条件
        区域，并只缓存 Action 在对应层实际能够读取的 K/V。
        """
        demo_mask = build_demo_attention_mask(inputs)
        cross_masks = build_cross_attention_masks(inputs)
        position_ids = build_region_position_ids(inputs)
        layout = inputs.layout
        prefix_valid_mask = _as_bool(inputs.prefix_valid_mask)
        action_valid_mask = _as_bool(inputs.action_valid_mask)
        prefix_self_mask = make_att_2d_masks(
            prefix_valid_mask,
            _as_bool(inputs.prefix_block_mask),
        )
        # 缓存路径只计算 Action 行，因此无需分配完整的 [P;G;L;A] 方阵。
        union_action_mask = torch.cat(
            [
                _full_cross_visibility(action_valid_mask, prefix_valid_mask),
                make_att_2d_masks(action_valid_mask, torch.ones_like(action_valid_mask)),
            ],
            dim=-1,
        )
        demo_valid_mask = torch.cat(
            [inputs.global_valid_mask, inputs.local_valid_mask],
            dim=1,
        )
        demo_position_ids = torch.cat(
            [position_ids.global_demo, position_ids.local_demo],
            dim=1,
        )

        prefix_hidden = inputs.prefix_hidden
        demo_hidden = torch.cat([inputs.global_hidden, inputs.local_hidden], dim=1)
        layer_caches: list[LayerConditionKVCache] = []

        for layer_idx in range(self.num_vlm_layers):
            if self._is_union_layer(layer_idx):
                # Prefix 在 Union 层只做自身 Self-Attention。保存的 K/V 是该层
                # Action 行读取的 Prefix K/V，与训练时完整 Union 计算完全一致。
                prefix_query, prefix_key, prefix_value = self._project_vlm_qkv(
                    prefix_hidden,
                    layer_idx,
                )
                prefix_query = apply_rope(prefix_query, position_ids.prefix)
                prefix_key = apply_rope(prefix_key, position_ids.prefix)
                prefix_attention = self.eager_attention_forward(
                    prefix_self_mask,
                    inputs.batch_size,
                    self.config.text_config.head_dim,
                    prefix_query,
                    prefix_key,
                    prefix_value,
                )
                prefix_hidden = self.apply_vlm_attention_output(
                    prefix_hidden,
                    prefix_attention,
                    layer_idx,
                    inputs.prefix_valid_mask,
                )
                demo_hidden = self.forward_demo_self_layer(
                    demo_hidden,
                    layer_idx,
                    demo_mask,
                    demo_position_ids,
                    demo_valid_mask,
                )

                layer_caches.append(
                    LayerConditionKVCache(
                        union_prefix_key=prefix_key,
                        union_prefix_value=prefix_value,
                    )
                )
                continue

            # Odd/Cross 层先按设计更新 Prefix 和 Demo，再执行 P<-G。
            prefix_hidden = self.forward_vlm_self_layer(
                prefix_hidden,
                layer_idx,
                prefix_self_mask,
                position_ids.prefix,
                inputs.prefix_valid_mask,
            )
            demo_hidden = self.forward_demo_self_layer(
                demo_hidden,
                layer_idx,
                demo_mask,
                demo_position_ids,
                demo_valid_mask,
            )
            global_length = layout.global_length
            global_hidden = demo_hidden[:, :global_length]
            local_hidden = demo_hidden[:, global_length:]
            prefix_hidden = self.condition_vlm_on_global(
                prefix_hidden,
                global_hidden,
                layer_idx,
                cross_masks.prefix_from_global,
                position_ids.prefix,
                position_ids.global_demo,
                inputs.prefix_valid_mask,
            )

            prefix_key, prefix_value = self._project_prefix_kv_for_action(
                prefix_hidden,
                layer_idx,
                position_ids.prefix,
            )
            local_key, local_value = self._project_local_kv_for_action(
                local_hidden,
                layer_idx,
                position_ids.local_demo,
            )
            layer_caches.append(
                LayerConditionKVCache(
                    cross_prefix_key=prefix_key,
                    cross_prefix_value=prefix_value,
                    cross_local_key=local_key,
                    cross_local_value=local_value,
                )
            )

        return SmolVLAICLConditionCache(
            layers=tuple(layer_caches),
            # Union 中 G/L 两块对 Action 始终不可见，Action-only 推理时可将
            # 它们从 KV 序列中删去，只保留 [P;A] 而不改变 Attention 结果。
            union_action_mask=union_action_mask,
            action_from_prefix_mask=cross_masks.action_from_prefix,
            action_from_local_mask=cross_masks.action_from_local,
            action_position_ids=position_ids.action,
            action_valid_mask=inputs.action_valid_mask,
        )

    def run_action_with_condition_cache(
        self,
        action_hidden: Tensor,
        cache: SmolVLAICLConditionCache,
    ) -> Tensor:
        """只重算 Action 分支，复用一次重规划内固定的条件 K/V。"""
        if action_hidden.shape[:2] != cache.action_valid_mask.shape:
            raise ValueError("Action hidden 必须与 condition cache 的 batch/序列长度一致。")
        batch_size = action_hidden.shape[0]
        action_cross_position_ids = (
            cache.action_position_ids
            - cache.action_position_ids.min(
                dim=1,
                keepdim=True,
            ).values
        )

        for layer_idx, layer_cache in enumerate(cache.layers):
            action_layer = self._get_action_layer(layer_idx)
            if self._is_union_layer(layer_idx):
                if layer_cache.union_prefix_key is None or layer_cache.union_prefix_value is None:
                    raise RuntimeError("Union layer 缺少 Prefix K/V cache。")
                query, action_key, action_value = self._project_action_qkv(
                    action_hidden,
                    layer_idx,
                )
                query = apply_rope(query, cache.action_position_ids)
                action_key = apply_rope(action_key, cache.action_position_ids)
                key = torch.cat([layer_cache.union_prefix_key, action_key], dim=1)
                value = torch.cat([layer_cache.union_prefix_value, action_value], dim=1)
                attention_output = self.eager_attention_forward(
                    cache.union_action_mask,
                    batch_size,
                    action_layer.self_attn.head_dim,
                    query,
                    key,
                    value,
                )
                action_hidden = self.apply_action_attention_output(
                    action_hidden,
                    attention_output,
                    layer_idx,
                    cache.action_valid_mask,
                )
                continue

            if (
                layer_cache.cross_prefix_key is None
                or layer_cache.cross_prefix_value is None
                or layer_cache.cross_local_key is None
                or layer_cache.cross_local_value is None
            ):
                raise RuntimeError("Cross layer 缺少 Prefix/Local K/V cache。")

            # A<-P'：Query 仍由当前去噪状态产生，K/V 直接读取预填充结果。
            action_normalized = action_layer.input_layernorm(action_hidden)
            action_normalized = action_normalized.to(dtype=action_layer.self_attn.q_proj.weight.dtype)
            prefix_query = action_layer.self_attn.q_proj(action_normalized).view(
                batch_size,
                action_hidden.shape[1],
                -1,
                action_layer.self_attn.head_dim,
            )
            prefix_query = apply_rope(prefix_query, action_cross_position_ids)
            prefix_attention = self.eager_attention_forward(
                cache.action_from_prefix_mask,
                batch_size,
                action_layer.self_attn.head_dim,
                prefix_query,
                layer_cache.cross_prefix_key,
                layer_cache.cross_prefix_value,
            )
            prefix_attention = prefix_attention * cache.action_from_prefix_mask.any(dim=-1).unsqueeze(-1)
            prefix_attention = prefix_attention.to(dtype=action_layer.self_attn.o_proj.weight.dtype)
            action_hidden = action_hidden + action_layer.self_attn.o_proj(prefix_attention)
            action_hidden = action_hidden * cache.action_valid_mask.to(dtype=action_hidden.dtype).unsqueeze(
                -1
            )

            # A<-L：使用已经读取 P' 的 Action hidden 重新生成 Query。
            adapter = self.action_from_local[str(layer_idx)]
            local_query_hidden = adapter.query_norm(action_hidden).to(dtype=adapter.q_proj.weight.dtype)
            local_query = adapter.q_proj(local_query_hidden).view(
                batch_size,
                action_hidden.shape[1],
                -1,
                self.lm_expert.config.head_dim,
            )
            local_query = apply_rope(local_query, action_cross_position_ids)
            local_attention = self.eager_attention_forward(
                cache.action_from_local_mask,
                batch_size,
                self.lm_expert.config.head_dim,
                local_query,
                layer_cache.cross_local_key,
                layer_cache.cross_local_value,
            )
            local_attention = adapter.o_proj(local_attention.to(dtype=adapter.o_proj.weight.dtype))
            local_attention = local_attention * cache.action_from_local_mask.any(dim=-1).unsqueeze(-1)
            action_hidden = action_hidden + adapter.gate.to(dtype=local_attention.dtype) * local_attention
            action_hidden = action_hidden * cache.action_valid_mask.to(dtype=action_hidden.dtype).unsqueeze(
                -1
            )
            action_hidden = self.finish_action_cross_layer(
                action_hidden,
                layer_idx,
                cache.action_valid_mask,
            )

        return self.finalize_action_hidden(action_hidden, cache.action_valid_mask)


def _as_bool(mask: Tensor) -> Tensor:
    return mask.to(dtype=torch.bool)


def _full_cross_visibility(query_mask: Tensor, key_mask: Tensor) -> Tensor:
    return _as_bool(query_mask)[:, :, None] & _as_bool(key_mask)[:, None, :]


def build_union_attention_mask(inputs: FourRegionInputs) -> Tensor:
    """构造 Union Self-Attention 的 ``[B,N_total,N_total]`` Mask。

    当前首版可见关系为：

    * ``P -> P``：保留 SmolVLA Prefix-LM 规则；
    * ``G -> G``、``L -> L``：各区域内部双向可见；
    * ``A -> P``：Action 读取当前 Prefix；
    * ``A -> A``：保持 SmolVLA 的 causal Action 关系；
    * 其余跨区域 block 全部不可见。
    """
    layout = inputs.layout
    mask = torch.zeros(
        inputs.batch_size,
        layout.total_length,
        layout.total_length,
        dtype=torch.bool,
        device=inputs.prefix_hidden.device,
    )

    prefix_valid = _as_bool(inputs.prefix_valid_mask)
    global_valid = _as_bool(inputs.global_valid_mask)
    local_valid = _as_bool(inputs.local_valid_mask)
    action_valid = _as_bool(inputs.action_valid_mask)

    # Prefix 内部直接复用 SmolVLA 的 Vision/Language/State 分块规则。
    mask[:, layout.prefix, layout.prefix] = make_att_2d_masks(
        prefix_valid,
        _as_bool(inputs.prefix_block_mask),
    )

    # Global 与 Local 在首版中各自双向 Self-Attend，但互相不可见。
    mask[:, layout.global_demo, layout.global_demo] = _full_cross_visibility(
        global_valid,
        global_valid,
    )
    mask[:, layout.local_demo, layout.local_demo] = _full_cross_visibility(
        local_valid,
        local_valid,
    )

    # Action 可以读取全部有效 Prefix，但不能在 Union 层直接读取 Demo。
    mask[:, layout.action, layout.prefix] = _full_cross_visibility(
        action_valid,
        prefix_valid,
    )

    # 与 SmolVLA embed_suffix 的全 1 block marker 保持一致，生成 causal A->A。
    action_block_mask = torch.ones_like(action_valid, dtype=torch.bool)
    mask[:, layout.action, layout.action] = make_att_2d_masks(
        action_valid,
        action_block_mask,
    )
    return mask


def build_demo_attention_mask(inputs: FourRegionInputs) -> Tensor:
    """构造 Demo Expert 的 Global/Local block-diagonal Self-Attention Mask。"""
    global_valid = _as_bool(inputs.global_valid_mask)
    local_valid = _as_bool(inputs.local_valid_mask)
    global_length = global_valid.shape[1]
    total_length = global_length + local_valid.shape[1]
    mask = torch.zeros(
        inputs.batch_size,
        total_length,
        total_length,
        dtype=torch.bool,
        device=inputs.prefix_hidden.device,
    )
    mask[:, :global_length, :global_length] = _full_cross_visibility(
        global_valid,
        global_valid,
    )
    mask[:, global_length:, global_length:] = _full_cross_visibility(
        local_valid,
        local_valid,
    )
    return mask


def build_cross_attention_masks(inputs: FourRegionInputs) -> CrossAttentionMasks:
    """构造 ``P<-G``、``A<-P`` 和 ``A<-L`` 三张矩形可见性 Mask。"""
    return CrossAttentionMasks(
        prefix_from_global=_full_cross_visibility(
            inputs.prefix_valid_mask,
            inputs.global_valid_mask,
        ),
        action_from_prefix=_full_cross_visibility(
            inputs.action_valid_mask,
            inputs.prefix_valid_mask,
        ),
        action_from_local=_full_cross_visibility(
            inputs.action_valid_mask,
            inputs.local_valid_mask,
        ),
    )


def _valid_token_position_ids(valid_mask: Tensor) -> Tensor:
    """复用 SmolVLA 的编号方式：只累计有效 token。"""
    return torch.cumsum(_as_bool(valid_mask).long(), dim=1) - 1


def _slot_position_ids(reference_mask: Tensor) -> Tensor:
    """按固定序列槽位编号；padding 不改变其他 token 的位置。"""
    return torch.arange(reference_mask.shape[1], device=reference_mask.device).expand(
        reference_mask.shape[0], -1
    )


def build_region_position_ids(inputs: FourRegionInputs) -> RegionPositionIds:
    """为 P/G/L/A 构造彼此兼容的位置编号。

    Prefix 和 Action 沿用 SmolVLA 的有效 token 累计编号。Global 使用固定 slot
    编号；Local 使用 ``slot-anchor``，使匹配锚点恒为 0，且开头/结尾 padding
    不会压缩位置。Action 仅以有效 Prefix 长度作为 offset，逻辑上插入的 G/L
    不改变其预训练位置分布；在 ``A<-L`` 中 Action Query 会再从 0 开始。
    """
    prefix = _valid_token_position_ids(inputs.prefix_valid_mask)
    global_demo = _slot_position_ids(inputs.global_valid_mask)
    local_demo = (
        _slot_position_ids(inputs.local_valid_mask)
        - inputs.local_anchor_positions.to(dtype=torch.long)[:, None]
    )
    prefix_length = _as_bool(inputs.prefix_valid_mask).long().sum(dim=1, keepdim=True)
    action = prefix_length + _valid_token_position_ids(inputs.action_valid_mask)
    return RegionPositionIds(
        prefix=prefix,
        global_demo=global_demo,
        local_demo=local_demo,
        action=action,
    )
