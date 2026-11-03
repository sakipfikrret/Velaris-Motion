"""
Velaris Motion - AI Director & Storyboard Planning Engine
Dynamically plans video duration, splits complex narratives into cohesive scenes,
and calculates precise hardware execution time estimates for RTX 4060 / CUDA GPUs.
Target Release: v1.1.0
"""

import re
import math
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

QualityPreset = Literal["draft", "hd", "fhd", "4k"]
DurationMode = Literal["auto", "short", "medium", "long", "cinematic"]

RESOLUTION_MAP: Dict[QualityPreset, Dict[str, Any]] = {
    "draft": {"width": 1024, "height": 576, "label": "576p SD (Fast Preview)"},
    "hd": {"width": 1280, "height": 720, "label": "720p HD (Balanced)"},
    "fhd": {"width": 1920, "height": 1080, "label": "1080p Full HD (Crisp Cinema)"},
    "4k": {"width": 3840, "height": 2160, "label": "4K Ultra HD (Master Quality)"},
}


class StoryboardScene(BaseModel):
    """A discrete cinematic shot within a multi-scene video."""
    scene_index: int
    prompt: str
    camera_movement: str
    duration_seconds: float
    motion_bucket_id: int = 65


class StoryboardPlan(BaseModel):
    """The master director plan for a video generation."""
    title: str
    target_duration_seconds: float
    duration_mode: DurationMode
    scenes: List[StoryboardScene]
    total_shots: int
    estimated_time_seconds: float
    formatted_estimate: str
    quality: QualityPreset
    target_resolution: Dict[str, int]
    is_long_form: bool  # True if >= 60 seconds


def detect_requested_duration(prompt: str, duration_mode: DurationMode, target_duration: Optional[int] = None) -> float:
    """
    Infers intended duration from prompt text, explicit seconds, or duration mode.
    Handles cues like '1 dakika', '60 seconds', '2 minutes', 'short clip', etc.
    """
    if target_duration is not None and target_duration > 0:
        return float(min(max(target_duration, 4), 180))

    prompt_lower = prompt.lower()

    # Regex search for explicit duration mentions in prompt
    # Turkish & English: "60 saniye", "1 dakika", "2 minutes", "90 seconds"
    minute_match = re.search(r"(\d+)\s*(?:dakika|minute|min|dk)", prompt_lower)
    if minute_match:
        minutes = int(minute_match.group(1))
        return float(min(minutes * 60, 180))

    second_match = re.search(r"(\d+)\s*(?:saniye|second|sec|sn)", prompt_lower)
    if second_match:
        secs = int(second_match.group(1))
        return float(min(max(secs, 4), 180))

    # Fallback to duration_mode preset
    if duration_mode == "short":
        return 7.0   # 1-2 shots, quick clip (~7s)
    elif duration_mode == "medium":
        return 21.0  # 3 shots (~21s)
    elif duration_mode == "long":
        return 60.0  # 1 full minute (7-8 shots)
    elif duration_mode == "cinematic":
        return 90.0  # 1.5 minutes extended film (10-12 shots)
    
    # "auto" mode: analyze prompt length and narrative complexity
    sentences = [s.strip() for s in re.split(r"[.,;\n]+", prompt) if len(s.strip()) > 5]
    if len(sentences) >= 4 or len(prompt) > 250:
        return 60.0  # Long narrative storyboard (>1 minute)
    elif len(sentences) >= 2 or len(prompt) > 100:
        return 21.0  # Medium scene (~21 seconds)
    else:
        return 7.0   # Standard scene (~7 seconds)


def plan_storyboard(
    prompt: str,
    duration_mode: DurationMode = "auto",
    target_duration: Optional[int] = None,
    quality: QualityPreset = "fhd",
) -> StoryboardPlan:
    """
    Director Planner: Splits narrative into sequential scenes, calculates timing,
    and returns a structured StoryboardPlan.
    """
    total_duration = detect_requested_duration(prompt, duration_mode, target_duration)
    is_long_form = total_duration >= 60.0

    raw_clauses = [c.strip() for c in re.split(r"[.,;\n]+", prompt) if len(c.strip()) > 3]

    shot_duration = 7.0
    num_shots = max(1, math.ceil(total_duration / shot_duration))

    scenes: List[StoryboardScene] = []

    camera_styles = [
        "cinematic wide establishing shot, smooth forward dolly tracking",
        "medium close-up, dynamic lateral panning",
        "low angle soaring perspective, sweeping motion",
        "epic wide aerial shot, slow majestic push-in",
        "side-profile tracking shot, continuous momentum",
        "dramatic hero angle, golden hour lighting transition",
    ]

    for i in range(num_shots):
        if i < len(raw_clauses):
            clause_text = raw_clauses[i]
        else:
            clause_text = f"{prompt} - phase {i + 1}, continuous cinematic progression"

        cam = camera_styles[i % len(camera_styles)]
        scene_prompt = f"{clause_text}, {cam}"

        scenes.append(
            StoryboardScene(
                scene_index=i + 1,
                prompt=scene_prompt,
                camera_movement=cam,
                duration_seconds=shot_duration,
                motion_bucket_id=65 if (i % 2 == 0) else 75,
            )
        )

    est_seconds = estimate_execution_time(
        num_scenes=num_shots,
        steps_per_scene=15,
        quality=quality,
        interpolate=True,
        has_audio=True,
    )

    formatted_time = format_seconds(est_seconds)
    res = RESOLUTION_MAP.get(quality, RESOLUTION_MAP["fhd"])

    return StoryboardPlan(
        title=prompt[:60] + ("..." if len(prompt) > 60 else ""),
        target_duration_seconds=round(num_shots * shot_duration, 1),
        duration_mode=duration_mode,
        scenes=scenes,
        total_shots=num_shots,
        estimated_time_seconds=est_seconds,
        formatted_estimate=formatted_time,
        quality=quality,
        target_resolution={"width": res["width"], "height": res["height"]},
        is_long_form=is_long_form,
    )


def estimate_execution_time(
    num_scenes: int,
    steps_per_scene: int = 15,
    quality: QualityPreset = "fhd",
    interpolate: bool = True,
    has_audio: bool = True,
) -> float:
    """
    Hardware-calibrated benchmark model for RTX 4060 GPU:
    - Dreamer base frame (SD1.5 768x432): ~3.0s per scene
    - SVD-XT diffusion with forward chunking: ~1.8s per step (15 steps = ~27s per shot)
    - 24fps Interpolation: ~5.0s per shot
    - Upscaling (FFmpeg Lanczos):
        - draft (576p): 0.5s
        - hd (720p): 1.5s
        - fhd (1080p): 3.0s
        - 4k (3840x2160): 8.5s per shot
    - Audio TTS & Mux: ~3.5s total
    """
    per_scene_dream = 3.0
    per_scene_diffusion = steps_per_scene * 1.8
    per_scene_interp = 5.0 if interpolate else 0.5

    upscale_costs = {
        "draft": 0.5,
        "hd": 1.5,
        "fhd": 3.0,
        "4k": 8.5,
    }
    per_scene_upscale = upscale_costs.get(quality, 3.0)

    total_scenes_time = num_scenes * (per_scene_dream + per_scene_diffusion + per_scene_interp + per_scene_upscale)
    audio_time = 4.0 if has_audio else 0.5

    return round(total_scenes_time + audio_time, 1)


def format_seconds(seconds: float) -> str:
    """Formats raw seconds into human-readable Turkish string, e.g. '1 dk 25 sn'."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins} dk {secs:02d} sn"
    return f"{secs} sn"
