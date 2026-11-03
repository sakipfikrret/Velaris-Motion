"""
Unit tests for security, input validation, and path traversal protection.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from security import (
    SafeGenerationRequest,
    validate_safe_filename,
    resolve_safe_path,
)
from config import config


def test_valid_generation_request():
    req = SafeGenerationRequest(
        prompt="A tranquil landscape with mountains and lake",
        num_inference_steps=25,
        motion_bucket_id=127,
        fps=7,
        seed=12345,
    )
    assert req.prompt == "A tranquil landscape with mountains and lake"
    assert req.num_inference_steps == 25
    assert req.seed == 12345


def test_empty_or_whitespace_prompt_rejected():
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="")
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="    ")


def test_prompt_exceeding_max_length_rejected():
    long_prompt = "a" * (config.max_prompt_length + 1)
    with pytest.raises(ValueError, match="Prompt cannot exceed"):
        SafeGenerationRequest(prompt=long_prompt)


def test_null_bytes_in_prompt_rejected():
    with pytest.raises(ValueError, match="invalid null"):
        SafeGenerationRequest(prompt="Cyberpunk\0city")


def test_inference_steps_boundary_validation():
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", num_inference_steps=0)
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", num_inference_steps=51)


def test_fps_boundary_validation():
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", fps=0)
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", fps=31)


def test_motion_bucket_boundary_validation():
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", motion_bucket_id=0)
    with pytest.raises(ValueError):
        SafeGenerationRequest(prompt="Valid", motion_bucket_id=256)


def test_validate_safe_filename_success():
    assert validate_safe_filename("session_123.mp4") == "session_123.mp4"
    assert validate_safe_filename("uuid-abc-def.png") == "uuid-abc-def.png"
    assert validate_safe_filename("meta_456.json") == "meta_456.json"


def test_validate_safe_filename_rejection():
    # Path traversal attempts
    with pytest.raises(ValueError):
        validate_safe_filename("../secret.txt")
    with pytest.raises(ValueError):
        validate_safe_filename("..\\secret.mp4")
    with pytest.raises(ValueError):
        validate_safe_filename("/etc/passwd")
    with pytest.raises(ValueError):
        validate_safe_filename("C:\\Windows\\system32\\cmd.exe")
    # Disallowed extension
    with pytest.raises(ValueError):
        validate_safe_filename("malicious.exe")
    with pytest.raises(ValueError):
        validate_safe_filename("script.sh")
    with pytest.raises(ValueError):
        validate_safe_filename("image.jpg")


def test_resolve_safe_path_guarantees_containment(tmp_path):
    # Valid file resolution
    valid_file = "output_123.mp4"
    resolved = resolve_safe_path(tmp_path, valid_file)
    assert resolved == tmp_path / valid_file

    # Directory traversal attempts
    with pytest.raises(ValueError):
        resolve_safe_path(tmp_path, "../outside.mp4")
    with pytest.raises(ValueError):
        resolve_safe_path(tmp_path, "..\\outside.mp4")
