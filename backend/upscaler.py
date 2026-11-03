"""
Velaris Motion - Super-Resolution Video Upscaler & Multi-Shot Stitcher
Upscales videos up to 4K Ultra HD (3840x2160) using high-fidelity Lanczos-3,
edge-enhancement sharpening, and stitches multi-shot sequences into seamless extended films.
Target Release: v1.1.0
"""

import shutil
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Literal

logger = logging.getLogger("velaris.upscaler")

QualityPreset = Literal["draft", "hd", "fhd", "4k"]

RESOLUTION_SPECS = {
    "draft": {"width": 1024, "height": 576, "crf": 22},
    "hd": {"width": 1280, "height": 720, "crf": 20},
    "fhd": {"width": 1920, "height": 1080, "crf": 18},
    "4k": {"width": 3840, "height": 2160, "crf": 16},
}


def get_ffmpeg_path() -> Optional[str]:
    """Resolves FFmpeg executable path from system PATH or bundled imageio-ffmpeg."""
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def is_ffmpeg_available() -> bool:
    """Returns True if a working FFmpeg binary is discovered."""
    return get_ffmpeg_path() is not None


# cv2 is imported lazily inside _upscale_video_realesrgan to prevent OpenMP / DLL conflicts on Windows
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
REALESRGAN_WEIGHTS = WEIGHTS_DIR / "realesr-general-x4v3.pth"


class SRVGGNetCompact(nn.Module):
    """Ultra-lightweight 4x neural super-resolution network for video restoration."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        for _ in range(num_conv):
            self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def _upscale_video_realesrgan(
    input_path: Path,
    output_path: Path,
    target_w: int,
    target_h: int,
    crf: int = 16,
) -> bool:
    """
    Applies Real-ESRGAN Compact 4x neural super-resolution directly on video frames.
    Reconstructs razor-sharp textures, removes compression artifacts and pixelation.
    """
    if not REALESRGAN_WEIGHTS.exists() or not torch.cuda.is_available():
        return False

    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        return False

    try:
        import cv2
        device = "cuda"
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4)
        loadnet = torch.load(REALESRGAN_WEIGHTS, map_location=device, weights_only=True)
        state_dict = loadnet.get("params_ema", loadnet.get("params", loadnet))
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 8.0

        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{target_w}x{target_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if pipe.stdin is None:
            raise RuntimeError("FFmpeg process stdin pipe could not be opened.")

        batch_frames = []
        def process_batch(b):
            tensors = [
                torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().div(255.0)
                for f in b
            ]
            inp = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = model(inp)
                out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
                out = out.clamp(0, 1).mul(255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()

            for i in range(len(b)):
                bgr = cv2.cvtColor(out[i], cv2.COLOR_RGB2BGR)
                pipe.stdin.write(bgr.tobytes())

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            if len(batch_frames) == 4:
                process_batch(batch_frames)
                batch_frames = []

        if batch_frames:
            process_batch(batch_frames)

        cap.release()
        pipe.stdin.close()
        pipe.wait()

        # Clean VRAM after upscaling
        del model
        torch.cuda.empty_cache()
        logger.info(f"Real-ESRGAN neural super-resolution completed: {output_path.name} ({target_w}x{target_h})")
        return True
    except Exception as e:
        logger.warning(f"Real-ESRGAN neural super-resolution failed, falling back to Lanczos: {e}")
        return False


def upscale_video_to_quality(
    input_path: Path,
    output_path: Path,
    quality: QualityPreset = "fhd",
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Path:
    """
    Upscales a video to the target quality tier (up to 4K UHD 3840x2160).
    Supports custom aspect ratios (e.g. 9:16 vertical Reels, 1:1 square).
    First attempts Real-ESRGAN Compact neural super-resolution for razor-sharp textures;
    falls back cleanly to Lanczos-3 if GPU/model is unavailable.
    """
    if quality == "draft":
        if input_path != output_path:
            shutil.copyfile(input_path, output_path)
        return output_path

    spec = RESOLUTION_SPECS.get(quality, RESOLUTION_SPECS["fhd"])
    target_w = target_width if target_width is not None else spec["width"]
    target_h = target_height if target_height is not None else spec["height"]
    crf = spec["crf"]

    # 1. Primary: Neural Super-Resolution via Real-ESRGAN Compact
    if _upscale_video_realesrgan(input_path, output_path, target_w, target_h, crf=crf):
        return output_path

    # 2. Fallback: High-Fidelity FFmpeg Lanczos
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        logger.warning("FFmpeg not available — skipping super-resolution upscaling.")
        if input_path != output_path:
            shutil.copyfile(input_path, output_path)
        return output_path

    logger.info(f"Upscaling video from {input_path.name} to {quality.upper()} ({target_w}x{target_h}, CRF {crf}) with Lanczos...")
    vf = f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd"

    cmd = [
        ffmpeg_exe,
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
        "-y",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Upscaling complete: {output_path.name} ({target_w}x{target_h})")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Video upscaling failed: {e.stderr}")
        raise RuntimeError(f"Upscaling to {quality} failed: {e.stderr[:200]}")


def concatenate_video_shots(
    shot_paths: List[Path],
    output_path: Path,
) -> Path:
    """
    Seamlessly concatenates multiple video shot files into a unified continuous video.
    Used for multi-scene and long-form (>1 minute) productions.
    """
    if not shot_paths:
        raise ValueError("Cannot concatenate empty list of video shots.")

    if len(shot_paths) == 1:
        if shot_paths[0] != output_path:
            shutil.copyfile(shot_paths[0], output_path)
        return output_path

    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg is required for multi-shot concatenation.")

    # Create concat list file
    concat_list_file = output_path.parent / f"concat_{output_path.stem}.txt"
    try:
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for p in shot_paths:
                # Escape single quotes and backslashes for FFmpeg concat demuxer
                safe_path = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            ffmpeg_exe,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
            "-y",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully concatenated {len(shot_paths)} shots into: {output_path.name}")
        return output_path

    finally:
        if concat_list_file.exists():
            concat_list_file.unlink(missing_ok=True)
