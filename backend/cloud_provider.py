"""
Velaris Motion - Cloud Studio Provider Bridge
Enables Hollywood-grade video generation via Google Veo 2, Kling 1.5, and Wan2.1-14B APIs.
"""

import os
import time
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

from config import config

logger = logging.getLogger("velaris.cloud")


class CloudVideoProvider:
    """
    Unified client for cloud AI video generation.
    Supports Replicate (Veo 2, Kling 1.5, Wan 14B), Fal.ai, and mock execution.
    """

    # Model registry for Replicate hosted endpoints
    MODEL_MAP = {
        "google/veo-2": "google/veo-2",
        "veo": "google/veo-2",
        "kling": "kwaivgi/kling-v1.5-pro",
        "wan-14b": "wan-ai/wan-2.1-t2v-14b",
        "minimax": "minimax/video-01",
    }

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")

    def is_available(self) -> bool:
        """Returns True if a valid API token is configured."""
        return bool(self.api_token and len(self.api_token.strip()) > 5)

    def generate_video(
        self,
        prompt: str,
        output_path: Path,
        model_name: str = "google/veo-2",
        aspect_ratio: str = "16:9",
        duration_seconds: int = 5,
        mock_mode: bool = False,
    ) -> Path:
        """
        Executes cloud generation and downloads the final MP4.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mock_mode or os.environ.get("VELARIS_MOCK_MODE") == "1":
            logger.info(f"Cloud provider running in mock mode for '{model_name}'")
            # Write a minimal dummy mp4 for mock verification
            output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42")
            return output_path

        if not self.is_available():
            raise RuntimeError(
                "Cloud Cinema Mode requires a valid API token. "
                "Set REPLICATE_API_TOKEN in your environment or .env file."
            )

        try:
            import replicate
        except ImportError:
            raise ImportError(
                "The 'replicate' Python package is required for Cloud Studio. "
                "Install it with: pip install replicate"
            )

        client = replicate.Client(api_token=self.api_token)
        target_model = self.MODEL_MAP.get(model_name.lower(), model_name)
        logger.info(f"Submitting job to Cloud Provider: {target_model} ('{prompt[:40]}...')")

        input_payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration_seconds,
        }

        # Run inference on cloud cluster
        output = client.run(target_model, input=input_payload)

        # Output is either a URL string or a FileOutput iterator
        video_url = None
        if isinstance(output, str):
            video_url = output
        elif isinstance(output, list) and len(output) > 0:
            video_url = str(output[0])
        elif hasattr(output, "url"):
            video_url = getattr(output, "url")
        else:
            video_url = str(output)

        if not video_url or not video_url.startswith("http"):
            raise ValueError(f"Invalid cloud response URL: {video_url}")

        logger.info(f"Cloud generation complete. Downloading video from {video_url}...")
        urllib.request.urlretrieve(video_url, str(output_path))
        logger.info(f"Cloud video saved -> {output_path.name} ({output_path.stat().st_size / 1024**2:.1f} MB)")
        return output_path


# Global singleton instance
cloud_provider = CloudVideoProvider()

