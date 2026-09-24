"""SmolVLA-ICL 硬件相关配置测试。"""

import pytest

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla_icl.configuration_smolvla_icl import SmolVLAICLConfig


def test_smolvla_icl_defaults_target_v100_training() -> None:
    config = SmolVLAICLConfig()

    # V100 用 FP32 保存可训练参数，实际矩阵计算由训练入口的 FP16 AMP 完成。
    assert config.vlm_load_dtype == "float32"
    assert config.local_vision_encode_batch_size == 2
    assert config.local_vision_gradient_checkpointing


def test_vlm_load_dtype_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="vlm_load_dtype"):
        SmolVLAConfig(vlm_load_dtype="tf32")
