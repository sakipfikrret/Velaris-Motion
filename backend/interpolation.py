"""
Velaris Motion - Frame Interpolation Module
Sprint 1C: RIFE-style frame interpolation via FFmpeg minterpolate filter.
Converts low-framerate SVD-XT output (14 frames @ 7fps) to cinema-quality
smooth motion (56 frames @ 24fps) with zero additional Python dependencies.
"""

import subprocess
import shutil
import logging
import os
from pathlib import Path

logger = logging.getLogger("velaris.interpolation")


def _get_ffmpeg_exe() -> str:
    """
    Returns the FFmpeg executable path.
    Priority: system PATH → imageio_ffmpeg bundled binary.
    """
    # 1. System PATH (fastest)
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    # 2. imageio_ffmpeg bundled binary (installed with diffusers/moviepy)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    return ""


def is_ffmpeg_available() -> bool:
    """Checks whether FFmpeg is available (system PATH or imageio_ffmpeg bundle)."""
    return bool(_get_ffmpeg_exe())


def interpolate_video(
    input_path: Path,
    output_path: Path,
    target_fps: int = 24,
    quality_crf: int = 18,
    preset: str = "fast",
) -> Path:
    """
    Performs motion-compensated frame interpolation using FFmpeg's minterpolate filter.

    Converts a low-framerate video (e.g. 14 frames @ 7fps from SVD-XT) to a
    smooth, cinema-quality output (e.g. 56 frames @ 24fps) without requiring
    any additional Python dependencies beyond FFmpeg being on PATH.

    Args:
        input_path: Path to the source MP4 video.
        output_path: Path to write the interpolated MP4.
        target_fps: Target output framerate (default: 24 for cinema-quality).
        quality_crf: H.264 CRF quality (0=lossless, 51=worst; 18 is near-lossless).
        preset: FFmpeg encoding speed preset ('fast', 'medium', 'slow').

    Returns:
        output_path on success.

    Raises:
        RuntimeError: If FFmpeg is not available or interpolation fails.
        FileNotFoundError: If input_path does not exist.
    """
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError(
            "FFmpeg is not available. Install FFmpeg from https://ffmpeg.org/download.html "
            "or run: pip install imageio-ffmpeg"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # minterpolate filter parameters:
    # - fps: target framerate
    # - mi_mode=mci: Motion Compensated Interpolation (best quality)
    # - mc_mode=aobmc: Adaptive Overlapped Block Motion Compensation
    # - me_mode=bidir: Bidirectional motion estimation (uses past + future frames)
    # - vsbmc=1: Variable-size Block Motion Compensation (reduces blocking artifacts)
    vf_filter = (
        f"minterpolate=fps={target_fps}:"
        f"mi_mode=mci:"
        f"mc_mode=aobmc:"
        f"me_mode=bidir:"
        f"vsbmc=1"
    )

    cmd = [
        ffmpeg_exe,
        "-i", str(input_path),
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-crf", str(quality_crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",   # Ensures broad browser/player compatibility
        "-movflags", "+faststart",  # Web-optimized: metadata at file start for streaming
        str(output_path),
        "-y",                     # Overwrite output without asking
    ]

    logger.info(
        f"Interpolating video: {input_path.name} -> {output_path.name} "
        f"@ {target_fps}fps (CRF={quality_crf})"
    )

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Interpolation complete: {output_path.name}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg interpolation failed: {e.stderr}")
        raise RuntimeError(
            f"Frame interpolation failed for '{input_path.name}'. "
            f"FFmpeg stderr: {e.stderr[-500:]}"
        ) from e


def get_ffmpeg_version() -> str:
    """Returns FFmpeg version string, or 'Not installed' if unavailable."""
    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        return "Not installed"
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract just the first line: "ffmpeg version X.Y.Z ..."
        first_line = result.stdout.splitlines()[0] if result.stdout else "Unknown"
        return first_line.split("Copyright")[0].strip()
    except Exception:
        return "Unknown"
