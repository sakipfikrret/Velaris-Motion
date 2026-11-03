"""
Velaris Motion - Sprint 4A & 5B Tests
Tests Direct Image-to-Video mode, image upload validation,
bypassing Dreamer stage, and benchmark API endpoints.
"""

import io
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from engine import HybridEngine, engine
engine.mock_mode = True

from api import app
from security import validate_uploaded_image, MAX_IMAGE_FILE_SIZE_BYTES
from job_queue import job_queue, Job, JobStatus, GenerationStage
from config import config

client = TestClient(app)


def _create_test_image_bytes(format: str = "PNG", size: tuple = (256, 256), color: tuple = (120, 180, 240)) -> bytes:
    """Helper creating raw image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Image Validation Tests
# ---------------------------------------------------------------------------

def test_validate_uploaded_image_valid_png():
    data = _create_test_image_bytes("PNG")
    # Should not raise
    validate_uploaded_image(data, "test_input.png")


def test_validate_uploaded_image_valid_jpeg():
    data = _create_test_image_bytes("JPEG")
    validate_uploaded_image(data, "photo.jpg")


def test_validate_uploaded_image_valid_webp():
    data = _create_test_image_bytes("WEBP")
    validate_uploaded_image(data, "frame.webp")


def test_validate_uploaded_image_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_uploaded_image(b"", "empty.png")


def test_validate_uploaded_image_rejects_oversized():
    fake_huge_data = b"x" * (MAX_IMAGE_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        validate_uploaded_image(fake_huge_data, "huge.png")


def test_validate_uploaded_image_rejects_invalid_extension():
    data = _create_test_image_bytes("PNG")
    with pytest.raises(ValueError, match="Unauthorized image extension"):
        validate_uploaded_image(data, "script.exe")


def test_validate_uploaded_image_rejects_corrupted_bytes():
    bad_bytes = b"\x89PNG\r\n\x1a\nThisIsCorruptDataNotAValidPngImageStream"
    with pytest.raises(ValueError, match="Corrupted or unsupported"):
        validate_uploaded_image(bad_bytes, "corrupt.png")


# ---------------------------------------------------------------------------
# POST /animate-image Endpoint Tests
# ---------------------------------------------------------------------------

def test_animate_image_endpoint_success():
    img_bytes = _create_test_image_bytes("PNG")
    response = client.post(
        "/animate-image",
        files={"image": ("scenery.png", img_bytes, "image/png")},
        data={
            "motion_bucket_id": "140",
            "num_inference_steps": "20",
            "fps": "7",
            "prompt": "Gently swaying pine trees in autumn breeze",
            "audio_mode": "none",
            "interpolate": "true",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "QUEUED"
    assert data["is_img2vid"] is True


def test_animate_image_endpoint_rejects_corrupt_file():
    response = client.post(
        "/animate-image",
        files={"image": ("broken.png", b"not-a-real-image", "image/png")},
    )
    assert response.status_code == 400
    assert "Corrupted or unsupported" in response.json()["error"]["message"]


def test_animate_image_endpoint_rejects_disallowed_extension():
    response = client.post(
        "/animate-image",
        files={"image": ("malicious.svg", b"<svg></svg>", "image/svg+xml")},
    )
    assert response.status_code == 400
    assert "Unauthorized image extension" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Direct Image Mode Engine Execution Tests
# ---------------------------------------------------------------------------

def test_engine_executes_direct_image_job_in_mock_mode(tmp_path):
    """Verifies that direct image-to-video jobs skip Dreamer and synthesize motion directly."""
    img_path = tmp_path / "source.png"
    Image.new("RGB", (512, 512), (100, 150, 200)).save(img_path)

    engine_instance = HybridEngine(mock_mode=True)
    from security import SafeGenerationRequest

    req = SafeGenerationRequest(
        prompt="Image animation test",
        motion_bucket_id=127,
        num_inference_steps=20,
        fps=7,
        seed=1001,
        interpolate=False,
        audio_mode="none",
    )
    job = Job(req, initial_image_path=str(img_path))
    engine_instance.execute_job(job)

    assert job.video_path is not None
    assert Path(job.video_path).exists()
    assert job.video_url is not None and job.video_url.startswith("/videos/")
    assert job.metadata is not None
    assert job.metadata["fps"] == 7


# ---------------------------------------------------------------------------
# Benchmark Endpoints Tests (Sprint 5B)
# ---------------------------------------------------------------------------

def test_benchmark_results_endpoint_returns_json():
    response = client.get("/benchmark-results")
    assert response.status_code == 200
    data = response.json()
    assert "velaris_version" in data
    assert "latency" in data
    assert "resolution" in data


def test_trigger_benchmark_mock_run():
    response = client.post("/benchmark/run?mock=true&prompts=2")
    assert response.status_code == 200
    report = response.json()
    assert report["mock_mode"] is True
    assert report["total_prompts"] == 2
    assert report["success_count"] == 2
    assert report["success_rate_percent"] == 100.0
