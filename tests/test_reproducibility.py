"""
Unit tests for GenerationMetadata and deterministic reproducibility logging.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from reproducibility import GenerationMetadata


def test_generation_metadata_save_and_load(tmp_path):
    meta = GenerationMetadata(
        job_id="test-session-123",
        prompt="A cybernetic garden",
        seed=42,
        models={
            "dreamer": {
                "model_id": "Lykon/dreamshaper-8",
                "revision": "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
            },
            "animator": {
                "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
                "revision": "9e43909513c6714f1bc78bcb44d96e733cd242aa"
            }
        },
        resolution={"width": 1024, "height": 576},
        steps={"dream_steps": 30, "animator_steps": 25},
        motion_bucket_id=127,
        fps=7,
        scheduler="EulerAncestralDiscreteScheduler",
        dtype="float16",
        device="cuda",
        gpu_device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        vram_peak_mb=5432.1,
        generation_duration_seconds=38.5,
    )

    json_file = tmp_path / "test-session-123_metadata.json"
    saved_path = meta.save(json_file)
    assert saved_path.exists()

    loaded = GenerationMetadata.load(json_file)
    assert loaded.job_id == meta.job_id
    assert loaded.seed == 42
    assert loaded.models["dreamer"]["revision"] == "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
    assert loaded.vram_peak_mb == 5432.1
    assert loaded.velaris_version == "1.1.0"
