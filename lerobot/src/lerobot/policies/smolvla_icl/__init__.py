"""SmolVLA-ICL policy components。"""

from .components import (
    GlobalDemoEncoder,
    GlobalEncoderOutput,
    LocalDemoEncoder,
    LocalEncoderOutput,
    S3DVideoBackbone,
)
from .configuration_smolvla_icl import (
    DemoAlignmentConfig,
    GlobalEncoderConfig,
    LocalEncoderConfig,
    SmolVLAICLConfig,
)
from .data.cache import (
    CachedTrainingDemo,
    DemoFeatureStore,
    SmolVLAICLCollator,
    cache_training_demo,
    precompute_training_demo,
)
from .data.collate import (
    build_encoded_local_demo_batch,
    collate_global_demo_samples,
    collate_raw_local_demo_samples,
    get_smolvla_icl_demo_batches,
    validate_smolvla_icl_training_batch,
)
from .data.contracts import (
    DemoReferenceLookupKey,
    DemoReferenceResolver,
    DemoSampleRef,
    FixedDemoReferenceResolver,
    MappingDemoReferenceResolver,
    SmolVLAICLSampleIndex,
)
from .data.dataset import SmolVLAICLQueryDataset
from .data.factory import prepare_smolvla_icl_datasets
from .data.preprocessing import build_global_demo_clips
from .data.reader import LeRobotLocalDemoReader
from .data.sampler import SmolVLAICLEpisodeAwareSampler
from .data.sidecar import EpisodeDemoPairing, PairingSidecar, PairingSidecarResolver
from .data.state import DemoStateNormalizer
from .data.types import (
    EncodedLocalDemoBatch,
    GlobalDemoBatch,
    GlobalDemoClips,
    GlobalDemoSample,
    LocalDemoBatch,
    RawLocalDemoSample,
    SMOLVLA_ICL_DEMO_REF,
    SMOLVLA_ICL_GLOBAL_DEMO,
    SMOLVLA_ICL_LOCAL_DEMO,
)
from .modeling_smolvla_icl import SmolVLAICLPolicy, VLAFlowMatchingICL
from .processor_smolvla_icl import make_smolvla_icl_pre_post_processors
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
    "DemoFeatureStore",
    "EpisodeDemoPairing",
    "DemoReferenceLookupKey",
    "DemoReferenceResolver",
    "DemoSampleRef",
    "DemoStateNormalizer",
    "CrossAttentionMasks",
    "FourRegionInputs",
    "FourRegionOutput",
    "LayerConditionKVCache",
    "GlobalDemoBatch",
    "GlobalDemoClips",
    "GlobalDemoEncoder",
    "GlobalDemoSample",
    "GlobalEncoderConfig",
    "GlobalEncoderOutput",
    "LocalDemoBatch",
    "EncodedLocalDemoBatch",
    "RawLocalDemoSample",
    "LocalDemoEncoder",
    "LeRobotLocalDemoReader",
    "LocalEncoderConfig",
    "LocalEncoderOutput",
    "FixedDemoReferenceResolver",
    "MappingDemoReferenceResolver",
    "PairingSidecar",
    "PairingSidecarResolver",
    "CachedTrainingDemo",
    "SMOLVLA_ICL_DEMO_REF",
    "SMOLVLA_ICL_GLOBAL_DEMO",
    "SMOLVLA_ICL_LOCAL_DEMO",
    "SmolVLAICLConfig",
    "SmolVLAICLConditionCache",
    "SmolVLAICLPolicy",
    "SmolVLAICLQueryDataset",
    "SmolVLAICLEpisodeAwareSampler",
    "SmolVLAICLSampleIndex",
    "VLAFlowMatchingICL",
    "RegionPositionIds",
    "S3DVideoBackbone",
    "SmolVLMWithDemoExpertModel",
    "TokenRegionLayout",
    "build_global_demo_clips",
    "build_encoded_local_demo_batch",
    "collate_global_demo_samples",
    "collate_raw_local_demo_samples",
    "cache_training_demo",
    "precompute_training_demo",
    "prepare_smolvla_icl_datasets",
    "get_smolvla_icl_demo_batches",
    "make_smolvla_icl_pre_post_processors",
    "validate_smolvla_icl_training_batch",
    "SmolVLAICLCollator",
    "build_cross_attention_masks",
    "build_demo_attention_mask",
    "build_region_position_ids",
    "build_union_attention_mask",
]
