"""
Unit and integration tests for Studio Photorealism Engine (SDXL + SVD + Real-ESRGAN).
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import config
from security import SafeGenerationRequest
from model_registry import model_registry
from engine import HybridEngine
from job_queue import Job, JobStatus
from prompt_director import ideate_cinematic_scenes, SceneConcept


def test_studio_model_pins_registered():
    """Verifies that SDXL-Turbo and SVD-XT are pinned in config and registered in registry."""
    assert config.studio_sdxl.id == "studio_sdxl"
    assert config.studio_sdxl.model_id == "stabilityai/sdxl-turbo"
    assert config.studio_svd.id == "studio_svd"
    assert config.studio_svd.model_id == "stabilityai/stable-video-diffusion-img2vid-xt"

    models = model_registry.list_models()
    assert "studio_sdxl" in models
    assert "studio_svd" in models

    sdxl_def = model_registry.get_definition("studio_sdxl")
    assert sdxl_def.task == "text-to-image"
    svd_def = model_registry.get_definition("studio_svd")
    assert svd_def.task == "image-to-video"


def test_safe_generation_request_allows_studio_engine():
    """Verifies that SafeGenerationRequest schema accepts 'studio' and 'studio_photoreal'."""
    req_studio = SafeGenerationRequest(
        prompt="A magnificent golden eagle soaring over snow-capped mountains",
        engine="studio",
    )
    assert req_studio.engine == "studio"

    req_photoreal = SafeGenerationRequest(
        prompt="A magnificent golden eagle soaring over snow-capped mountains",
        engine="studio_photoreal",
    )
    assert req_photoreal.engine == "studio_photoreal"


def test_studio_prompt_enhancement():
    """Verifies that _build_studio_prompts injects National Geographic and 8k photography specs."""
    eng = HybridEngine(mock_mode=True)
    req = SafeGenerationRequest(prompt="a golden eagle", engine="studio")
    enhanced, neg = eng._build_studio_prompts(req)

    assert "National Geographic" in enhanced
    assert "85mm f/1.4 lens" in enhanced
    assert "a golden eagle" in enhanced
    assert "cartoon" in neg
    assert "anime" in neg
    assert "grain" in neg
    assert "pixelated" in neg


def test_studio_mock_execution_lifecycle(tmp_path):
    """Verifies mock execution lifecycle for studio engine requests."""
    eng = HybridEngine(mock_mode=True)
    req = SafeGenerationRequest(
        prompt="A wild snow leopard in Himalayas",
        engine="studio",
        fps=8,
    )
    job = Job(job_id="test_studio_mock", request=req)

    # Output to temp dir
    orig_output = config.outputs_dir
    config.outputs_dir = tmp_path
    try:
        eng.execute_job(job)
        assert job.status in (JobStatus.COMPLETED, JobStatus.QUEUED, JobStatus.RUNNING)
        assert job.video_path is not None
        assert Path(job.video_path).exists()
        assert job.metadata_path is not None
        assert Path(job.metadata_path).exists()
    finally:
        config.outputs_dir = orig_output


def test_eco_mode_and_aspect_ratios_validation():
    """Verifies that eco_mode and aspect_ratio validate properly in SafeGenerationRequest."""
    req_default = SafeGenerationRequest(prompt="A serene mountain lake")
    assert req_default.eco_mode is False
    assert req_default.aspect_ratio == "16:9"

    req_vertical = SafeGenerationRequest(
        prompt="A serene mountain lake",
        aspect_ratio="9:16",
        eco_mode=True,
    )
    assert req_vertical.aspect_ratio == "9:16"
    assert req_vertical.eco_mode is True

    req_square = SafeGenerationRequest(
        prompt="A serene mountain lake",
        aspect_ratio="1:1",
    )
    assert req_square.aspect_ratio == "1:1"

    # Verify aspect ratio dims in config
    for ar in ["16:9", "9:16", "1:1"]:
        assert ar in config.aspect_ratio_dims
        dims = config.aspect_ratio_dims[ar]
        assert "width" in dims and "height" in dims
        assert "target_w" in dims and "target_h" in dims


def test_ideation_concepts_generated():
    """Verifies that ideate_cinematic_scenes generates 3 high-quality scene concepts."""
    concepts = ideate_cinematic_scenes("cyberpunk street in rain")
    assert len(concepts) == 3
    for c in concepts:
        assert isinstance(c, SceneConcept)
        assert len(c.id) > 0
        assert len(c.title) > 0
        assert len(c.badge) > 0
        assert "cyberpunk street in rain" in c.enhanced_prompt
        assert len(c.negative_prompt) > 0
        assert len(c.camera_movement) > 0
        assert len(c.lighting) > 0

    # NatGeo / Portrait check
    assert "National Geographic" in concepts[0].enhanced_prompt or "photography" in concepts[0].enhanced_prompt.lower()
    assert concepts[0].badge == "NatGeo 8K"
    assert concepts[1].badge == "Dinamik"
    assert concepts[2].badge == "Altın Saat"


def test_studio_initial_image_execution(tmp_path):
    """Verifies that Image-to-Video mode execution runs cleanly in mock mode."""
    eng = HybridEngine(mock_mode=True)
    from PIL import Image

    dummy_img = Image.new("RGB", (1024, 576), color="blue")
    img_path = tmp_path / "dummy_input.png"
    dummy_img.save(str(img_path))

    req = SafeGenerationRequest(
        prompt="Gentle ocean waves rolling over the blue frame",
        engine="studio",
        aspect_ratio="16:9",
        eco_mode=True,
    )
    job = Job(job_id="test_studio_img2vid", request=req, initial_image_path=str(img_path))

    orig_output = config.outputs_dir
    config.outputs_dir = tmp_path
    try:
        eng.execute_job(job)
        assert job.status in (JobStatus.COMPLETED, JobStatus.QUEUED, JobStatus.RUNNING)
        assert job.video_path is not None
        assert Path(job.video_path).exists()
    finally:
        config.outputs_dir = orig_output

