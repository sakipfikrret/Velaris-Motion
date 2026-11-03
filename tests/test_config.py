"""
Unit tests for backend configuration and pinned models.
"""

import pytest
from pathlib import Path
import sys

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import config, VelarisConfig, ModelPin


def test_config_pinned_revisions():
    """Verify that models have valid configurations."""
    assert config.cogvideox.id == "cogvideox"
    assert config.cogvideox.model_id == "THUDM/CogVideoX-2b"
    assert config.cogvideox.expected_vram_mb > 0

    assert config.dreamer.id == "dreamer"
    assert config.dreamer.model_id in ("stablediffusionapi/toonyou", "emilianJR/epiCRealism")
    assert config.dreamer.expected_vram_mb > 0

    assert config.animator.id == "animator"
    assert config.animator.model_id == "ByteDance/AnimateDiff-Lightning"
    assert config.animator.expected_vram_mb > 0


def test_config_directories():
    """Verify that output and temp directories are created."""
    assert config.outputs_dir.exists()
    assert config.temp_dir.exists()


def test_config_constraints():
    """Verify numeric constraints are sensible."""
    assert config.min_prompt_length >= 1
    assert config.max_prompt_length == 1000
    assert config.min_inference_steps >= 1
    assert config.max_inference_steps >= 50
    assert config.max_queue_size > 0
