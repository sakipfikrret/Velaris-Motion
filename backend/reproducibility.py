"""
Velaris Motion - Reproducibility & Generation Metadata
Target Release: v1.1.0
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class GenerationMetadata(BaseModel):
    """
    Immutable metadata record saved with each generation.
    Ensures complete scientific reproducibility of results.
    """
    job_id: str
    prompt: str
    seed: int
    models: Dict[str, Dict[str, str]]
    resolution: Dict[str, int]
    steps: Dict[str, int]
    motion_bucket_id: int
    fps: int
    interpolated: bool = False
    interpolated_fps: Optional[int] = None
    audio_mode: str = "none"
    scheduler: str
    dtype: str
    device: str
    gpu_device_name: Optional[str] = None
    vram_peak_mb: float
    generation_duration_seconds: float
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    velaris_version: str = "1.1.0"

    def save(self, filepath: Path) -> Path:
        """Saves metadata to disk as indented JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "GenerationMetadata":
        """Loads and validates metadata from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
