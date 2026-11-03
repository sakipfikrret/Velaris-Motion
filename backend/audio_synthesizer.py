"""
Velaris Motion - Audio Synthesis Module
Sprint 2: AI-powered audio generation for video output.

Provides two independent audio modes — both run exclusively on CPU to avoid
competing with Dreamer/Animator for VRAM:

  1. Ambient Music (MusicGen-Small) — generates ~15s of atmospheric background
     music from the text prompt.
  2. Narration (Kokoro-82M TTS or edge-tts fallback) — synthesizes a voiced
     narration of the prompt text.

After audio generation, FFmpeg muxes the audio track into the final MP4.
"""

import logging
import subprocess
import shutil
import tempfile
import importlib
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger("velaris.audio")

AudioMode = Literal["none", "music", "narration"]

# ---------------------------------------------------------------------------
# Dependency availability guards
# ---------------------------------------------------------------------------

def _is_transformers_audio_available() -> bool:
    """Checks if transformers with audio extras is installed."""
    try:
        importlib.import_module("transformers")
        importlib.import_module("scipy")
        importlib.import_module("soundfile")
        return True
    except Exception:
        return False


def _is_kokoro_available() -> bool:
    """Checks if kokoro TTS is installed."""
    try:
        importlib.import_module("kokoro")
        return True
    except Exception:
        return False


def _is_edge_tts_available() -> bool:
    """Checks if edge-tts is installed (online TTS fallback)."""
    try:
        importlib.import_module("edge_tts")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ambient Music Generation (MusicGen-Small)
# ---------------------------------------------------------------------------

def generate_ambient_music(
    prompt: str,
    duration_seconds: float,
    output_path: Path,
) -> Optional[Path]:
    """
    Generates an ambient music track from a text prompt using MusicGen-Small.

    Always runs on CPU to avoid competing with Dreamer/Animator for VRAM.
    The musicgen-small model (~300MB) is fast enough on CPU for short clips.

    Args:
        prompt: Text description — used as the music generation prompt.
        duration_seconds: Desired music duration in seconds.
        output_path: Path to save the generated WAV file.

    Returns:
        output_path if successful, None if the package is unavailable or
        generation fails (non-fatal — caller decides whether to skip audio).
    """
    if not _is_transformers_audio_available():
        logger.warning(
            "MusicGen requires: pip install transformers soundfile scipy. "
            "Skipping ambient music generation."
        )
        return None

    try:
        import numpy as np
        import scipy.io.wavfile  # type: ignore
        from transformers import pipeline as hf_pipeline

        logger.info(f"Generating ambient music for: '{prompt[:60]}...' ({duration_seconds:.1f}s)")

        synthesizer = hf_pipeline(
            "text-to-audio",
            model="facebook/musicgen-small",
            device="cpu",   # Always CPU — VRAM reserved for Dreamer/Animator
        )

        # MusicGen encodes ~50 tokens/second at 32kHz
        max_tokens = int(duration_seconds * 51.2)

        result = synthesizer(
            prompt,
            forward_params={
                "do_sample": True,
                "max_new_tokens": max_tokens,
                "guidance_scale": 3.0,
            }
        )

        audio_array = np.array(result["audio"]).squeeze()
        sampling_rate: int = result["sampling_rate"]

        # Normalize to int16 range for WAV compatibility
        if audio_array.dtype != np.int16:
            audio_array = (audio_array / max(abs(audio_array.max()), 1e-8) * 32767).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(output_path), rate=sampling_rate, data=audio_array)

        logger.info(f"Ambient music saved: {output_path.name} ({sampling_rate}Hz)")
        return output_path

    except Exception as e:
        logger.warning(f"Ambient music generation failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# Narration / TTS Generation
# ---------------------------------------------------------------------------

def generate_narration(
    text: str,
    output_path: Path,
    voice: str = "af_heart",
) -> Optional[Path]:
    """
    Synthesizes a narration of the given text using Kokoro-82M TTS.
    Falls back to edge-tts (online) if Kokoro is not installed.

    Always runs on CPU — Kokoro-82M is real-time capable on CPU.

    Args:
        text: The text to narrate (typically the generation prompt).
        output_path: Path to save the generated WAV file.
        voice: Kokoro voice ID (default: 'af_heart' — natural English female).

    Returns:
        output_path if successful, None if all TTS options are unavailable.
    """
    # Primary: Kokoro-82M (offline, Apache 2.0)
    if _is_kokoro_available():
        return _generate_narration_kokoro(text, output_path, voice)

    # Fallback: edge-tts (online, no install required beyond pip)
    if _is_edge_tts_available():
        return _generate_narration_edge_tts(text, output_path)

    logger.warning(
        "No TTS engine available. Install: pip install kokoro>=0.9.4 "
        "(or pip install edge-tts for online fallback). Skipping narration."
    )
    return None


def _generate_narration_kokoro(text: str, output_path: Path, voice: str) -> Optional[Path]:
    """TTS via Kokoro-82M (offline, CPU-only, Apache 2.0)."""
    try:
        import numpy as np
        import scipy.io.wavfile  # type: ignore
        from kokoro import KPipeline  # pyright: ignore[reportMissingImports]  # type: ignore[import]

        logger.info(f"Generating Kokoro narration: '{text[:60]}...'")
        pipeline = KPipeline(lang_code="en-us")
        audio, sr = pipeline(text, voice=voice)

        audio_np = np.array(audio)
        if audio_np.dtype != np.int16:
            audio_np = (audio_np / max(abs(audio_np.max()), 1e-8) * 32767).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.wavfile.write(str(output_path), rate=sr, data=audio_np)
        logger.info(f"Kokoro narration saved: {output_path.name}")
        return output_path

    except Exception as e:
        logger.warning(f"Kokoro narration failed (non-fatal): {e}")
        return None


def _generate_narration_edge_tts(text: str, output_path: Path) -> Optional[Path]:
    """TTS via edge-tts (requires internet, no GPU)."""
    try:
        import asyncio
        import edge_tts  # pyright: ignore[reportMissingImports]  # type: ignore[import]

        logger.info(f"Generating edge-tts narration: '{text[:60]}...'")

        async def _synthesize():
            voice = "en-US-ChristopherNeural"
            communicate = edge_tts.Communicate(text, voice=voice)
            wav_path = output_path.with_suffix(".mp3")
            await communicate.save(str(wav_path))
            return wav_path

        wav_path = asyncio.run(_synthesize())
        logger.info(f"edge-tts narration saved: {wav_path.name}")

        # Convert MP3 to WAV so mux_audio_to_video works uniformly
        if shutil.which("ffmpeg"):
            converted = output_path.with_suffix(".wav")
            subprocess.run(
                ["ffmpeg", "-i", str(wav_path), str(converted), "-y"],
                check=True, capture_output=True
            )
            wav_path.unlink(missing_ok=True)
            return converted

        return wav_path  # return MP3 if ffmpeg not available

    except Exception as e:
        logger.warning(f"edge-tts narration failed (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# FFmpeg Audio Mux
# ---------------------------------------------------------------------------

def mux_audio_to_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_duration: Optional[float] = None,
) -> Path:
    """
    Muxes an audio track into a video using FFmpeg.

    Audio is trimmed or looped to match video duration.
    Video stream is copied without re-encoding for speed.

    Args:
        video_path: Source silent MP4.
        audio_path: Audio track (WAV or MP3).
        output_path: Output path for the combined MP4.
        video_duration: Exact duration of video in seconds (prevents premature cut).

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If FFmpeg is not available or muxing fails.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is required for audio muxing. "
            "Install FFmpeg from https://ffmpeg.org and add to PATH."
        )

    logger.info(f"Muxing audio into video: {video_path.name} + {audio_path.name}")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",          # Video stream: copy without re-encoding (fast)
        "-c:a", "aac",           # Audio: encode to AAC for MP4 compatibility
        "-b:a", "192k",          # Audio bitrate
    ]

    if video_duration is not None and video_duration > 0:
        cmd.extend(["-t", f"{video_duration:.2f}"])
    else:
        cmd.append("-shortest")

    cmd.extend([
        "-movflags", "+faststart",
        str(output_path),
        "-y",
    ])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Audio mux complete: {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio mux failed: {e.stderr}")
        raise RuntimeError(
            f"FFmpeg audio mux failed: {e.stderr[-400:]}"
        ) from e


# ---------------------------------------------------------------------------
# High-level entry point (called by engine.py)
# ---------------------------------------------------------------------------

def generate_audio_for_job(
    prompt: str,
    audio_mode: AudioMode,
    duration_seconds: float,
    temp_dir: Path,
    job_id: str,
) -> Optional[Path]:
    """
    Generates audio for a job based on the requested audio_mode.

    Args:
        prompt: The generation prompt (used for both music and TTS).
        audio_mode: 'none' | 'music' | 'narration'
        duration_seconds: Target audio duration (should match video duration).
        temp_dir: Temporary directory for intermediate audio files.
        job_id: Job identifier (used for unique temp file naming).

    Returns:
        Path to the generated audio file, or None if audio_mode is 'none'
        or generation fails.
    """
    if audio_mode == "none":
        return None

    audio_path = temp_dir / f"{job_id}_audio.wav"

    if audio_mode == "music":
        return generate_ambient_music(prompt, duration_seconds, audio_path)
    elif audio_mode == "narration":
        return generate_narration(prompt, audio_path)

    return None
