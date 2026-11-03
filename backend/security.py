"""
Velaris Motion - Security, Input Validation & Path Traversal Protection
Target Release: v1.1.0
"""

import os
import re
from pathlib import Path
from typing import Optional, Any, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
from config import config


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StandardErrorResponse(BaseModel):
    error: ErrorDetail


class SafeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Prompt describing the anime scene to generate")
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=50, description="Inference steps: None=engine optimal (35 for CogVideoX, 8 for AnimateDiff)")
    motion_bucket_id: int = Field(default=40, description="Legacy field (not used in AnimateDiff-Lightning)")
    fps: int = Field(default=8, description="Output FPS (8fps native AnimateDiff, interpolated to 24fps)")
    seed: Optional[int] = Field(default=None, description="Deterministic seed for reproducibility")
    dream_steps: int = Field(default=4, description="AnimateDiff-Lightning inference steps")
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0, description="CFG scale: None=engine optimal (6.0 for CogVideoX, 1.5 for AnimateDiff)")
    interpolate: bool = Field(default=True, description="Apply FFmpeg frame interpolation to reach 24fps cinema quality")
    audio_mode: Literal["none", "music", "narration", "foley"] = Field(
        default="none",
        description="Audio synthesis mode: 'none' (silent), 'music' (MusicGen ambient), 'narration' (TTS), 'foley' (Auto-detected ambient sound effects)"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Negative prompt to exclude unwanted artifacts and distortions"
    )
    num_frames: int = Field(
        default=16,
        description="Number of video frames per segment (AnimateDiff-Lightning: 16 recommended)"
    )
    num_segments: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of chained animation segments (1=~3.5s, each +1 adds ~3.5s; max 20 for ~70s)"
    )
    quality: Literal["draft", "hd", "fhd", "4k"] = Field(
        default="fhd",
        description="Output quality preset: draft (576p fast), hd (720p), fhd (1080p), 4k (3840x2160 UHD)"
    )
    duration_mode: Literal["auto", "short", "medium", "long", "cinematic"] = Field(
        default="auto",
        description="Duration planning mode: auto (infer from prompt), short (~7s), medium (~21s), long (~60s), cinematic (~90s)"
    )
    target_duration: Optional[int] = Field(
        default=None,
        ge=4,
        le=180,
        description="Explicit target duration in seconds (4–180s). Overrides duration_mode."
    )
    use_director: bool = Field(
        default=False,
        description="Enable AI Director storyboard engine for intelligent multi-scene planning"
    )
    engine: Optional[Literal["auto", "wan", "ltx", "studio", "studio_photoreal", "cogvideox", "animatediff", "cloud"]] = Field(
        default="auto",
        description="Generation engine: auto (detect from prompt), wan (Wan2.1 SOTA 3D DiT), ltx (LTX-Video 24fps), studio (Legacy SDXL+SVD), cogvideox (DiT), animatediff (anime), cloud (Veo 2/Kling)"
    )
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Field(
        default="16:9",
        description="Target aspect ratio: 16:9 (Cinema/YouTube), 9:16 (Reels/Shorts/TikTok), 1:1 (Square)"
    )
    eco_mode: bool = Field(
        default=False,
        description="Eco / Silent mode: limits GPU thermal output and memory pressure to keep computer completely responsive"
    )
    camera_movement: Literal[
        "static", "dolly_in", "dolly_out", "pan_left", "pan_right",
        "drone_soar", "orbit_right", "tilt_up", "tilt_down", "handheld"
    ] = Field(
        default="static",
        description=(
            "Camera movement style to prefix into the prompt: "
            "'static' (no movement), 'dolly_in' (push-in tracking shot), "
            "'dolly_out' (pull-back reveal), 'pan_left'/'pan_right' (horizontal sweep), "
            "'drone_soar' (aerial upward sweep), 'orbit_right' (360 arc), "
            "'tilt_up'/'tilt_down' (vertical pivot), 'handheld' (documentary handheld shake)"
        )
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < config.min_prompt_length:
            raise ValueError(f"Prompt must be at least {config.min_prompt_length} character.")
        if len(stripped) > config.max_prompt_length:
            raise ValueError(f"Prompt cannot exceed {config.max_prompt_length} characters (got {len(stripped)}).")
        # Null bytes or control characters check
        if "\0" in stripped:
            raise ValueError("Prompt contains invalid null characters.")
        return stripped

    @field_validator("num_inference_steps")
    @classmethod
    def validate_animator_steps(cls, v: int) -> int:
        if v < config.min_inference_steps or v > config.max_inference_steps:
            raise ValueError(f"num_inference_steps must be between {config.min_inference_steps} and {config.max_inference_steps}.")
        return v

    @field_validator("dream_steps")
    @classmethod
    def validate_dream_steps(cls, v: int) -> int:
        if v < config.min_inference_steps or v > config.max_inference_steps:
            raise ValueError(f"dream_steps must be between {config.min_inference_steps} and {config.max_inference_steps}.")
        return v

    @field_validator("motion_bucket_id")
    @classmethod
    def validate_motion(cls, v: int) -> int:
        if v < config.min_motion_bucket_id or v > config.max_motion_bucket_id:
            raise ValueError(f"motion_bucket_id must be between {config.min_motion_bucket_id} and {config.max_motion_bucket_id}.")
        return v

    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        if v < config.min_fps or v > config.max_fps:
            raise ValueError(f"fps must be between {config.min_fps} and {config.max_fps}.")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 2**32 - 1):
            raise ValueError("Seed must be an unsigned 32-bit integer (0 to 4294967295).")
        return v


# Safe filename pattern: alphanumeric, hyphen, underscore, and safe extensions only
SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+\.(mp4|png|json)$")


def validate_safe_filename(filename: str) -> str:
    """
    Validates that a filename matches safe characters and allowed extensions.
    Prevents path traversal, shell injection, or illegal characters.
    """
    if not filename or len(filename) > 255:
        raise ValueError("Invalid filename length.")
    
    # Check for path traversal markers
    if ".." in filename or "/" in filename or "\\" in filename or "\0" in filename:
        raise ValueError("Path traversal sequences are strictly forbidden.")

    if not SAFE_FILENAME_REGEX.match(filename):
        raise ValueError(f"Filename '{filename}' is invalid or has an unauthorized extension. Allowed: .mp4, .png, .json")

    return filename


def resolve_safe_path(base_dir: Path, filename: str) -> Path:
    """
    Safely resolves a filename within base_dir.
    Guarantees the resolved path is strictly inside base_dir.
    """
    sanitized_name = validate_safe_filename(filename)
    resolved_base = base_dir.resolve()
    target_path = (resolved_base / sanitized_name).resolve()
    
    # Ensure target_path is within base_dir
    try:
        # os.path.commonpath verifies directory boundary containment
        common = Path(os.path.commonpath([str(resolved_base), str(target_path)]))
        if common != resolved_base:
            raise ValueError("Path resolution escaped allowed directory boundary.")
    except Exception as e:
        raise ValueError(f"Path traversal check failed: {e}")

    return target_path


MAX_IMAGE_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def validate_uploaded_image(file_bytes: bytes, filename: str) -> None:
    """
    Sprint 4A: Validates uploaded image for direct Image-to-Video mode.
    - Rejects empty uploads
    - Enforces maximum file size (15MB)
    - Validates extension (.png, .jpg, .jpeg, .webp)
    - Verifies image integrity and decodability with PIL
    """
    if not file_bytes:
        raise ValueError("Uploaded image file is empty.")

    if len(file_bytes) > MAX_IMAGE_FILE_SIZE_BYTES:
        raise ValueError(
            f"Uploaded image exceeds maximum size of 15MB (got {len(file_bytes) / (1024 * 1024):.2f}MB)."
        )

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unauthorized image extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    # Validate image bytes integrity using PIL
    import io
    from PIL import Image

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or unsupported image file: {e}")

