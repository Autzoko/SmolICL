"""SmolVLA-ICL 中可独立测试和替换的模型组件。"""

from .demo_alignment import (
    AlignmentChunkEmbedding,
    AlignmentResult,
    DemoEmbeddingCache,
    LocalDemoWindow,
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

__all__ = [
    "AlignmentChunkEmbedding",
    "AlignmentResult",
    "DemoEmbeddingCache",
    "GlobalDemoEncoder",
    "GlobalEncoderOutput",
    "LocalDemoWindow",
    "LocalDemoEncoder",
    "LocalEncoderOutput",
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
