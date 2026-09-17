"""SmolVLA-ICL policy components。"""

from .configuration_smolvla_icl import (
    DemoAlignmentConfig,
    GlobalEncoderConfig,
    LocalEncoderConfig,
    SmolVLAICLConfig,
)
from .components import (
    DemoStateNormalizer,
    GlobalDemoEncoder,
    GlobalEncoderOutput,
    LocalDemoEncoder,
    LocalEncoderOutput,
    S3DVideoBackbone,
)
from .processor_smolvla_icl import (
    GlobalDemoBatch,
    GlobalDemoSample,
    LocalDemoBatch,
    build_global_demo_sample,
    collate_global_demo_samples,
    collate_local_demo_chunks,
    make_smolvla_icl_pre_post_processors,
)
from .modeling_smolvla_icl import SmolVLAICLPolicy, VLAFlowMatchingICL
from .smolvla_with_demo_expert import (
    CrossAttentionMasks,
    FourRegionInputs,
    FourRegionOutput,
    LayerConditionKVCache,
    RegionPositionIds,
    SmolVLAICLConditionCache,
    SmolVLMWithDemoExpertModel,
    TokenRegionLayout,
    build_cross_attention_masks,
    build_demo_attention_mask,
    build_region_position_ids,
    build_union_attention_mask,
)

__all__ = [
    "DemoAlignmentConfig",
    "DemoStateNormalizer",
    "CrossAttentionMasks",
    "FourRegionInputs",
    "FourRegionOutput",
    "LayerConditionKVCache",
    "GlobalDemoBatch",
    "GlobalDemoEncoder",
    "GlobalDemoSample",
    "GlobalEncoderConfig",
    "GlobalEncoderOutput",
    "LocalDemoBatch",
    "LocalDemoEncoder",
    "LocalEncoderConfig",
    "LocalEncoderOutput",
    "SmolVLAICLConfig",
    "SmolVLAICLConditionCache",
    "SmolVLAICLPolicy",
    "VLAFlowMatchingICL",
    "RegionPositionIds",
    "S3DVideoBackbone",
    "SmolVLMWithDemoExpertModel",
    "TokenRegionLayout",
    "build_global_demo_sample",
    "collate_global_demo_samples",
    "collate_local_demo_chunks",
    "make_smolvla_icl_pre_post_processors",
    "build_cross_attention_masks",
    "build_demo_attention_mask",
    "build_region_position_ids",
    "build_union_attention_mask",
]
