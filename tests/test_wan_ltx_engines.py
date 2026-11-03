"""
Unit tests for Next-Gen Wan2.1 and LTX-Video DiT engines,
VRAM-to-RAM Group Offloading, and Cloud Studio Bridge.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import config
from security import SafeGenerationRequest
from model_registry import model_registry
from engine import HybridEngine
from job_queue import Job, JobStatus
from vram_manager import vram_manager
from cloud_provider import cloud_provider


def test_wan_and_ltx_config_pins():
    """Verifies that Wan2.1 and LTX-Video are configured with correct models and tasks."""
    assert config.wan_t2v.id == "wan_t2v"
    assert config.wan_t2v.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert config.wan_t2v.task == "text-to-video"

    assert config.ltx_video.id == "ltx_video"
    assert config.ltx_video.model_id == "Lightricks/LTX-Video"
    assert config.ltx_video.task == "text-to-video"

    assert config.enable_group_offload is True
    assert config.group_offload_num_blocks == 1


def test_model_registry_has_nextgen_dit_models():
    """Verifies that Wan2.1 and LTX-Video are registered in ModelRegistry."""
    models = model_registry.list_models()
    assert "wan_t2v" in models
    assert "wan_14b" in models
    assert "ltx_video" in models

    wan_def = model_registry.get_definition("wan_t2v")
    assert wan_def.task == "text-to-video"
    assert wan_def.expected_vram_mb == 6200

    ltx_def = model_registry.get_definition("ltx_video")
    assert ltx_def.task == "text-to-video"
    assert ltx_def.expected_vram_mb == 5600


def test_safe_generation_request_accepts_nextgen_engines():
    """Verifies SafeGenerationRequest validates 'wan', 'ltx', and 'cloud' engine identifiers."""
    req_wan = SafeGenerationRequest(prompt="Sunset over the fjords", engine="wan")
    assert req_wan.engine == "wan"

    req_ltx = SafeGenerationRequest(prompt="Futuristic city at night", engine="ltx")
    assert req_ltx.engine == "ltx"

    req_cloud = SafeGenerationRequest(prompt="A cinematic drone shot of Norway", engine="cloud")
    assert req_cloud.engine == "cloud"


def test_vram_manager_hybrid_group_offload():
    """Verifies VRAMManager.apply_hybrid_ram_vram_offload delegates to enable_group_offload."""
    mock_pipeline = MagicMock()
    mock_pipeline.enable_group_offload = MagicMock()

    # When on CUDA, applies group offload
    vram_manager.device = "cuda"
    success = vram_manager.apply_hybrid_ram_vram_offload(
        mock_pipeline,
        enable_group_offload=True,
        num_blocks_per_group=1,
        use_stream=True,
    )
    assert success is True
    assert mock_pipeline.enable_group_offload.called


def test_cloud_provider_mock_generation(tmp_path):
    """Verifies CloudVideoProvider produces mock video output in test mode."""
    target_video = tmp_path / "cloud_test.mp4"
    result = cloud_provider.generate_video(
        prompt="Golden hour over snowy mountains",
        output_path=target_video,
        model_name="google/veo-2",
        mock_mode=True,
    )
    assert result.exists()
    assert result.stat().st_size > 0


def test_mock_pipeline_execution_with_wan_engine(tmp_path):
    """Verifies HybridEngine completes mock execution cleanly for wan engine."""
    eng = HybridEngine(mock_mode=True)
    orig_output = config.outputs_dir
    config.outputs_dir = tmp_path
    try:
        req = SafeGenerationRequest(
            prompt="Dramatic snow-capped Norwegian mountain peaks at golden sunset",
            engine="wan",
            fps=16,
        )
        job = Job(request=req)
        eng.execute_job(job)

        assert job.video_path is not None
        assert Path(job.video_path).exists()
        assert job.metadata is not None
    finally:
        config.outputs_dir = orig_output
