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

__all__ = [
    "AlignmentChunkEmbedding",
    "AlignmentResult",
    "DemoEmbeddingCache",
    "GlobalDemoEncoder",
    "GlobalEncoderOutput",
    "LocalDemoChunk",
    "ObservationHistoryBuffer",
    "OnlineDTWMatcher",
    "S3DVideoBackbone",
    "SmolVLASigLIPHandle",
    "extract_matching_state_features",
    "extract_state_features",
    "load_smolvla_siglip",
    "pool_visual_tokens",
    "reuse_smolvla_siglip",
]
