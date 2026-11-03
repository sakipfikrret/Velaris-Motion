"""
Norway Mountains Sunset - Full Production Pipeline
epiCRealism (master frame) -> SVD-XT (14 frames) -> FFmpeg (24fps)
"""
import sys, os, traceback, gc

# UTF-8 output - avoids Turkish cp1254 Unicode crash
import io
if isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if isinstance(sys.stderr, io.TextIOWrapper):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path

print("=== Norway Mountains Sunset - Production Run ===", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {free/1024**2:.0f} MB free / {total/1024**2:.0f} MB total", flush=True)

OUTPUTS = Path(__file__).parent / "outputs"
PROMPT = (
    "Dramatic snow-capped Norwegian mountain peaks at golden sunset, "
    "fjord valley below reflecting amber and rose light, atmospheric haze, "
    "long shadows across glacial terrain, serene cinematic 35mm photography, "
    "natural DSLR colors, f/5.6 aperture, Nikon D850, RAW converted"
)
NEG = (
    "cartoon, anime, painting, oversaturated, neon, HDR phone, dark, blurry, "
    "watermark, text, logo, people, buildings, ugly"
)

# =============================================================
# STAGE 1: epiCRealism Master Frame
# =============================================================
print("\n[Stage 1/3] epiCRealism master frame...", flush=True)
from diffusers import StableDiffusionPipeline

pipe_epic = StableDiffusionPipeline.from_pretrained(
    "emilianJR/epiCRealism",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True,
)
pipe_epic.enable_model_cpu_offload()

import time
t0 = time.time()
gen = torch.Generator(device="cuda").manual_seed(42)
out = pipe_epic(
    prompt=PROMPT,
    negative_prompt=NEG,
    num_inference_steps=20,
    guidance_scale=6.0,
    width=768, height=432,
    generator=gen,
)
master = out.images[0]
t1 = time.time()
print(f"  Master frame synthesized in {t1-t0:.1f}s", flush=True)

# Upscale to SVD-XT native resolution
from PIL import Image
master_1024 = master.resize((1024, 576), Image.Resampling.LANCZOS)
master_path = OUTPUTS / "norway_master.jpg"
master_1024.save(str(master_path), quality=97)
print(f"  Saved master frame -> {master_path.name}", flush=True)

del pipe_epic, out
gc.collect()
torch.cuda.empty_cache()
free, _ = torch.cuda.mem_get_info()
print(f"  VRAM after cleanup: {free/1024**2:.0f} MB", flush=True)

# =============================================================
# STAGE 2: SVD-XT Motion Synthesis (14 frames, 12 steps)
# =============================================================
print("\n[Stage 2/3] SVD-XT video motion synthesis (14 frames, 12 steps)...", flush=True)
from diffusers import StableVideoDiffusionPipeline

pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
)
pipe_svd.enable_model_cpu_offload()
print("  SVD-XT loaded.", flush=True)

t2 = time.time()
gen2 = torch.Generator(device="cuda").manual_seed(42)
svd_out = pipe_svd(
    master_1024,
    num_frames=14,
    num_inference_steps=12,
    motion_bucket_id=120,
    noise_aug_strength=0.00,
    fps=7,
    decode_chunk_size=1,
    generator=gen2,
)
frames = svd_out.frames[0]
t3 = time.time()
print(f"  SVD-XT complete: {len(frames)} frames in {t3-t2:.1f}s", flush=True)

# Export 7fps native
from diffusers.utils import export_to_video
raw_mp4 = OUTPUTS / "norway_7fps_raw.mp4"
export_to_video(frames, str(raw_mp4), fps=7)
print(f"  7fps video -> {raw_mp4.name}", flush=True)

del pipe_svd, svd_out, frames
gc.collect()
torch.cuda.empty_cache()

# =============================================================
# STAGE 3: FFmpeg Interpolation 7fps -> 24fps
# =============================================================
print("\n[Stage 3/3] FFmpeg interpolation 7fps -> 24fps...", flush=True)
from interpolation import is_ffmpeg_available, interpolate_video, _get_ffmpeg_exe

ffmpeg_exe = _get_ffmpeg_exe()
print(f"  FFmpeg: {ffmpeg_exe}", flush=True)

final_mp4 = OUTPUTS / "norway_sunset_24fps.mp4"
if ffmpeg_exe:
    t4 = time.time()
    interpolate_video(raw_mp4, final_mp4, target_fps=24, quality_crf=17, preset="medium")
    t5 = time.time()
    size_mb = final_mp4.stat().st_size / 1024**2
    print(f"  24fps video complete in {t5-t4:.1f}s -> {final_mp4.name} ({size_mb:.1f} MB)", flush=True)
    output_video = final_mp4
else:
    print("  FFmpeg not found - using 7fps video as final output", flush=True)
    output_video = raw_mp4

total_time = time.time() - t0
print(f"\n=== PRODUCTION COMPLETE ===", flush=True)
print(f"Total time: {total_time:.1f}s", flush=True)
print(f"Final video: {output_video}", flush=True)

