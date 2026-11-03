"""
Unit tests for Model Registry, pinned revisions, and loaders.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from model_registry import ModelRegistry, ModelDefinition, model_registry


def test_default_registry_contains_core_models():
    models = model_registry.list_models()
    assert "cogvideox" in models
    assert "dreamer" in models
    assert "animator" in models
    assert "studio_sdxl" in models
    assert "studio_epic" in models
    assert "studio_svd" in models
    assert "wan_t2v" in models
    assert "wan_14b" in models
    assert "ltx_video" in models
    
    wan_t2v = model_registry.get_definition("wan_t2v")
    assert wan_t2v.task == "text-to-video"
    assert wan_t2v.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    ltx_video = model_registry.get_definition("ltx_video")
    assert ltx_video.task == "text-to-video"
    assert ltx_video.model_id == "Lightricks/LTX-Video"
    
    studio_epic = model_registry.get_definition("studio_epic")
    assert studio_epic.task == "text-to-image"
    assert studio_epic.model_id == "emilianJR/epiCRealism"

    studio_sdxl = model_registry.get_definition("studio_sdxl")
    assert studio_sdxl.task == "text-to-image"
    assert studio_sdxl.model_id == "stabilityai/sdxl-turbo"

    studio_svd = model_registry.get_definition("studio_svd")
    assert studio_svd.task == "image-to-video"
    assert studio_svd.model_id == "stabilityai/stable-video-diffusion-img2vid-xt"

    dreamer = model_registry.get_definition("dreamer")
    assert dreamer.task == "text-to-video"
    assert dreamer.model_id in ("stablediffusionapi/toonyou", "emilianJR/epiCRealism")

    animator = model_registry.get_definition("animator")
    assert animator.task == "motion-adapter"
    assert animator.model_id == "ByteDance/AnimateDiff-Lightning"


def test_custom_model_registration():
    registry = ModelRegistry()
    custom_def = ModelDefinition(
        id="test_model",
        model_id="org/test-model",
        revision="1234567890abcdef1234567890abcdef12345678",
        task="text-to-image",
        dtype="float16",
        expected_vram_mb=2048,
    )
    
    mock_pipeline = object()
    registry.register(custom_def, loader=lambda defn, device, dtype, cpu_offload: mock_pipeline)
    
    retrieved = registry.get_definition("test_model")
    assert retrieved.id == "test_model"
    assert retrieved.revision == "1234567890abcdef1234567890abcdef12345678"
    
    pipe = registry.load("test_model", device="cpu", dtype="float32")
    assert pipe is mock_pipeline


def test_unknown_model_raises_keyerror():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.get_definition("non_existent_model")

    with pytest.raises(KeyError):
        registry.load("non_existent_model", device="cpu", dtype="float32")
