"""
Engine crash diagnostic v3 - ASCII safe print, safety_checker=None fix verified
Run: $env:HF_HUB_OFFLINE="1"; python -u backend/diagnose_engine.py 2>&1
"""
import sys, os, traceback, logging

# Force UTF-8 output so Unicode arrows don't crash on Turkish Windows locale
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ.setdefault("PYTHONUNBUFFERED", "1")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
print("=== Velaris Engine Diagnostics v3 ===", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {free/1024**2:.0f} MB free / {total/1024**2:.0f} MB total", flush=True)

# ---------------------------------------------------------------
# TEST A: epiCRealism with safety_checker=None (6 components, not 7)
# ---------------------------------------------------------------
print("\n[TEST A] epiCRealism with safety_checker=None...", flush=True)
try:
    from diffusers import StableDiffusionPipeline
    pipe_a = StableDiffusionPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe_a.enable_model_cpu_offload()
    print(f"  [A] Loaded: {type(pipe_a).__name__}", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(1337)
    out = pipe_a(
        prompt="Norwegian fjord at sunset, golden hour, photorealistic DSLR, 35mm film",
        negative_prompt="blurry, cartoon, fake, oversaturated, dark",
        num_inference_steps=20,
        guidance_scale=6.0,
        width=768, height=432,
        generator=gen,
    )
    img = out.images[0]
    out_p = os.path.join(os.path.dirname(__file__), "outputs", "diag_A_norway.jpg")
    img.save(out_p, quality=95)
    print(f"  [A] Inference OK -> {out_p}", flush=True)

    master_img = img
    del pipe_a, out
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        print(f"  [A] VRAM after cleanup: {free/1024**2:.0f} MB free", flush=True)

except Exception:
    print("  [A] FAILED:", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------
# TEST B: SVD-XT after epiCRealism cleanup
# ---------------------------------------------------------------
print("\n[TEST B] SVD-XT (14 frames, 12 steps)...", flush=True)
try:
    from diffusers import StableVideoDiffusionPipeline
    pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe_svd.enable_model_cpu_offload()
    print(f"  [B] SVD-XT loaded: {type(pipe_svd).__name__}", flush=True)

    from PIL import Image
    svd_in = master_img.resize((1024, 576), Image.Resampling.LANCZOS)
    gen2 = torch.Generator(device="cuda").manual_seed(1337)

    print("  [B] Running SVD-XT inference (12 steps, 14 frames)...", flush=True)
    svd_out = pipe_svd(
        svd_in,
        num_frames=14,
        num_inference_steps=12,
        motion_bucket_id=120,
        noise_aug_strength=0.00,
        fps=7,
        decode_chunk_size=1,
        generator=gen2,
    )
    frames = svd_out.frames[0]
    print(f"  [B] SVD-XT OK -- {len(frames)} frames", flush=True)

    from diffusers.utils import export_to_video
    vid_path = os.path.join(os.path.dirname(__file__), "outputs", "diag_B_norway_svd.mp4")
    export_to_video(frames, vid_path, fps=7)
    print(f"  [B] Video exported -> {vid_path}", flush=True)

    del pipe_svd, svd_out, frames
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception:
    print("  [B] FAILED:", flush=True)
    traceback.print_exc()
    sys.exit(2)

# ---------------------------------------------------------------
# TEST C: FFmpeg interpolation 7fps -> 24fps
# ---------------------------------------------------------------
print("\n[TEST C] FFmpeg interpolation 7fps -> 24fps...", flush=True)
try:
    from interpolation import interpolate_video, is_ffmpeg_available
    if is_ffmpeg_available():
        in_p = os.path.join(os.path.dirname(__file__), "outputs", "diag_B_norway_svd.mp4")
        out_p = os.path.join(os.path.dirname(__file__), "outputs", "diag_C_norway_24fps.mp4")
        interpolate_video(in_p, out_p, target_fps=24)
        print(f"  [C] Interpolated -> {out_p}", flush=True)
    else:
        print("  [C] FFmpeg not available -- skipped", flush=True)
except Exception:
    print("  [C] Interpolation FAILED (non-fatal):", flush=True)
    traceback.print_exc()

print("\n=== ALL TESTS PASSED ===", flush=True)
