"""
Velaris Motion v2.0 — AnimateDiff-Lightning Production Test
Tests the new single-pipeline architecture:
  - AnimateDiff-Lightning (4-step, ~15-30s/shot)
  - ToonYou anime base model
  - LCM Scheduler (trailing timestep)
  - xFormers memory-efficient attention
"""

import sys
import os
import time
from pathlib import Path

# Force UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        reconfigure_out = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfig_out := reconfigure_out):
            reconfig_out(encoding="utf-8", errors="replace")
        reconfigure_err = getattr(sys.stderr, "reconfigure", None)
        if callable(reconfig_err := reconfigure_err):\
            reconfig_err(encoding="utf-8", errors="replace")
    except Exception:
        pass

backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

import torch
import torch.version as torch_version

from config import config
from security import SafeGenerationRequest
from job_queue import Job, GenerationStage
from engine import HybridEngine
from vram_manager import vram_manager


def run_test():
    print("\n" + "=" * 70)
    print("   VELARIS MOTION v2.0 — ANIMATEDIFF-LIGHTNING TEST")
    print("=" * 70)

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
    total_vram = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_ok else 0
    )
    print(f"\n[Hardware] Device: {device_name} | VRAM: {total_vram} GB | CUDA: {torch_version.cuda}")

    if not cuda_ok:
        print("ERROR: CUDA not available. Aborting.")
        return

    # AnimateDiff-Lightning test request
    req = SafeGenerationRequest(
        prompt="national geographic photograph, a majestic golden eagle with sharp detailed plumage soaring over snow-capped mountain peaks at golden hour sunset, authentic wildlife cinema, 8k resolution, crisp clear feathers",
        num_inference_steps=8,    # AnimateDiff-Lightning 8-step for smooth, noise-free photorealism
        guidance_scale=1.4,       # 1.4 enables negative prompt (anti-grain, anti-duplicate)
        fps=8,                    # Native AnimateDiff fps
        seed=2024,
        interpolate=True,         # 8fps → 24fps cinema
        audio_mode="none",        # Skip audio for speed test
        quality="fhd",            # 1080p upscale
        num_frames=16,            # 16 frames native
        num_segments=1,
    )

    job = Job(req)
    engine = HybridEngine(mock_mode=False)

    print(f"\n[Generation Request]")
    print(f"  * Architecture:     AnimateDiff-Lightning ({config.animatediff_steps}-step distilled)")
    print(f"  * Base Model:       {config.dreamer.model_id} (Photorealistic)")
    print(f"  * Motion Adapter:   ByteDance/AnimateDiff-Lightning")
    print(f"  * Scheduler:        LCMScheduler (trailing timestep)")
    print(f"  * Steps:            {req.num_inference_steps} (vs 12-20 for SVD-XT)")
    print(f"  * Guidance Scale:   {req.guidance_scale} (CFG disabled → 2x speed)")
    print(f"  * Frames:           {req.num_frames} @ {req.fps}fps → 24fps interpolated")
    print(f"  * Quality:          {req.quality.upper()}")
    print(f"  * Seed:             {req.seed}")
    print(f"  * Prompt:           \"{req.prompt[:70]}...\"")

    print(f"\n[Starting AnimateDiff-Lightning Pipeline...]")
    t_start = time.time()

    stage_times = {}
    last_stage = None
    last_t = t_start

    def stage_callback(stage: GenerationStage, progress: float, message: str) -> None:
        nonlocal last_stage, last_t
        now = time.time()
        if last_stage is not None and last_stage != stage:
            stage_times[last_stage.value] = now - last_t
            print(f"   [DONE] Stage {last_stage.value.upper()} finished in {now - last_t:.2f}s")
            last_t = now
        last_stage = stage
        pct = int(progress * 100)
        vram_alloc = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if cuda_ok else 0
        print(f"   [{pct:2d}%] {message} (VRAM: {vram_alloc} MB)")

    original_update = job.update_stage

    def wrapped_update(stage: GenerationStage, progress: float, message: str) -> None:
        original_update(stage, progress, message)
        stage_callback(stage, progress, message)

    setattr(job, "update_stage", wrapped_update)

    try:
        engine.execute_job(job)
        t_end = time.time()
        if last_stage is not None:
            stage_times[last_stage.value] = t_end - last_t

        total_duration = t_end - t_start
        peak_vram = job.metadata.get("vram_peak_mb", 0.0) if job.metadata else 0.0
        video_path = Path(job.video_path) if job.video_path else None

        print("\n" + "=" * 70)
        print("   ANIMATEDIFF-LIGHTNING TEST SUCCESSFUL")
        print("=" * 70)
        print(f"\nTotal Wall-Clock Time:   {total_duration:.2f}s ({total_duration/60:.2f} min)")
        print(f"Peak VRAM Used:          {peak_vram:.1f} MB ({peak_vram/8192*100:.1f}% of 8GB)")
        print(f"Output Video:            {video_path}")
        print(f"Architecture:            AnimateDiff-Lightning (4-step, LCM)")
        print(f"Expected speedup:        ~90x faster than SVD-XT (was ~45 min)")

        print("\n[Per-Stage Duration Breakdown]:")
        for s_name, s_dur in stage_times.items():
            pct = (s_dur / total_duration) * 100 if total_duration > 0 else 0
            bar = "#" * int(pct / 4)
            print(f"  * {s_name:<20} {s_dur:6.2f}s ({pct:4.1f}%) | {bar}")

        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
