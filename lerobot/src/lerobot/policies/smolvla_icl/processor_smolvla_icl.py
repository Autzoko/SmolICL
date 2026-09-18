"""SmolVLA-ICL 的标准 Policy pre/post processor。

Demo 配对、缓存和 batch 组装属于数据层，统一放在 :mod:`.data` 下；
这里与官方 SmolVLA 一样，只负责构建 Policy 输入、输出处理管线。
"""

from typing import Any

import torch

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import PolicyProcessorPipeline

from ..smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .configuration_smolvla_icl import SmolVLAICLConfig


def make_smolvla_icl_pre_post_processors(
    config: SmolVLAICLConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """复用 SmolVLA 的输入归一化、语言处理和动作反归一化管线。

    完整 Demo 和 Local Demo 是 Policy 的补充输入，不进入标准 observation
    processor；训练 collator 会在 Query batch 完成后附加这些数据。
    """
    return make_smolvla_pre_post_processors(config, dataset_stats)


__all__ = ["make_smolvla_icl_pre_post_processors"]
