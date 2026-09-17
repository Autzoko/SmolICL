"""SmolVLA-ICL 中可独立测试和替换的模型组件。"""

from .demo_alignment import (
    AlignmentChunkEmbedding,
    AlignmentResult,
    DemoEmbeddingCache,
    LocalDemoChunk,
    ObservationHistoryBuffer,
    OnlineDTWMatcher,
    SmolVLASigLIPHandle,
    extract_matching_state_features,
    extract_state_features,
    load_smolvla_siglip,
    pool_visual_tokens,
    reuse_smolvla_siglip,
)
from .global_encoder import GlobalDemoEncoder, GlobalEncoderOutput, S3DVideoBackbone
from .local_encoder import LocalDemoEncoder, LocalEncoderOutput
from .state_normalizer import DemoStateNormalizer, StateNormalizationSignature

__all__ = [
    "AlignmentChunkEmbedding",
    "AlignmentResult",
    "DemoEmbeddingCache",
    "DemoStateNormalizer",
    "GlobalDemoEncoder",
    "GlobalEncoderOutput",
    "LocalDemoChunk",
    "LocalDemoEncoder",
    "LocalEncoderOutput",
    "ObservationHistoryBuffer",
    "OnlineDTWMatcher",
    "S3DVideoBackbone",
    "SmolVLASigLIPHandle",
    "StateNormalizationSignature",
    "extract_matching_state_features",
    "extract_state_features",
    "load_smolvla_siglip",
    "pool_visual_tokens",
    "reuse_smolvla_siglip",
]
