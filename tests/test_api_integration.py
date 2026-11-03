"""
Integration tests for FastAPI endpoints, lifecycle, job queue, and error schemas.
"""

import time
import pytest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from api import app
from engine import engine
from job_queue import job_queue
from config import config


@pytest.fixture(scope="module", autouse=True)
def setup_mock_engine():
    # Enable mock mode for fast, deterministic CI integration tests
    original_mock = engine.mock_mode
    engine.mock_mode = True
    yield
    engine.mock_mode = original_mock


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["version"] == config.version
    assert "vram" in data


def test_preflight_endpoint(client):
    response = client.get("/preflight")
    assert response.status_code == 200
    data = response.json()
    assert data["overall_status"] in ("PASS", "WARN", "FAIL")
    assert "checks" in data


def test_vram_status_endpoint(client):
    response = client.get("/vram")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "allocated_vram_mb" in data


def test_generate_and_poll_lifecycle(client):
    # Submit job
    payload = {
        "prompt": "Cinematic shot of a crystal waterfall in neon light",
        "num_inference_steps": 10,
        "dream_steps": 10,
        "seed": 42,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 202
    job_info = response.json()
    job_id = job_info["job_id"]
    assert job_info["status"] in ("QUEUED", "RUNNING")
    assert job_info["stage"] in ("queued", "loading_model")

    # Poll until completed
    timeout = 10.0
    start = time.time()
    completed = False
    final_data = None

    while time.time() - start < timeout:
        poll_resp = client.get(f"/jobs/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        if poll_data["status"] == "COMPLETED":
            completed = True
            final_data = poll_data
            break
        time.sleep(0.1)

    assert completed is True
    assert final_data is not None
    assert final_data["stage"] == "completed"
    assert final_data["progress"] == 1.0
    assert final_data["video_url"] == f"/videos/{job_id}.mp4"
    assert final_data["metadata"] is not None
    assert final_data["metadata"]["seed"] == 42


def test_concurrent_generation_requests(client):
    # Enqueue two generation jobs simultaneously
    resp1 = client.post("/generate", json={"prompt": "First concurrent prompt", "seed": 101})
    resp2 = client.post("/generate", json={"prompt": "Second concurrent prompt", "seed": 102})
    assert resp1.status_code == 202
    assert resp2.status_code == 202

    id1 = resp1.json()["job_id"]
    id2 = resp2.json()["job_id"]

    # Poll both until finished
    timeout = 15.0
    start = time.time()
    completed1 = False
    completed2 = False

    while time.time() - start < timeout:
        d1 = client.get(f"/jobs/{id1}").json()
        d2 = client.get(f"/jobs/{id2}").json()

        if d1["status"] == "COMPLETED":
            completed1 = True
        if d2["status"] == "COMPLETED":
            completed2 = True

        if completed1 and completed2:
            break
        time.sleep(0.1)

    assert completed1 is True
    assert completed2 is True


def test_cancel_endpoint(client):
    # Submit job and cancel it immediately
    resp = client.post("/generate", json={"prompt": "A prompt to cancel quickly"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    cancel_resp = client.post(f"/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    cancel_data = cancel_resp.json()
    assert cancel_data["job_id"] == job_id


def test_input_validation_errors_return_predictable_schema(client):
    # Empty prompt -> 422
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"

    # Out of range steps -> 422
    resp2 = client.post("/generate", json={"prompt": "Valid", "num_inference_steps": 999})
    assert resp2.status_code == 422
    data2 = resp2.json()
    assert data2["error"]["code"] == "VALIDATION_ERROR"


def test_safe_video_download_and_traversal_protection(client):
    # Missing video -> 404
    resp = client.get("/videos/non_existent_12345.mp4")
    assert resp.status_code == 404

    # Traversal attempt -> 400
    resp_traversal = client.get("/videos/..%2Fsecret.mp4")
    assert resp_traversal.status_code == 400
    err = resp_traversal.json()
    assert "error" in err
