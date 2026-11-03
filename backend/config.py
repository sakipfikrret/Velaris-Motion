"""
Velaris Motion - Configuration & Environment Settings
v3.0 — Photorealistic Studio (SDXL-Turbo + SVD-XT + Real-ESRGAN) + AnimateDiff-Lightning
"""

import os
import shutil
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any

# Ensure bundled FFmpeg from imageio_ffmpeg is discovered on Windows PATH
if not shutil.which("ffmpeg"):
    try:
        import imageio_ffmpeg
        _img_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if _img_exe and os.path.exists(_img_exe):
            _bin_dir = str(Path(_img_exe).parent)
            _target = Path(_bin_dir) / "ffmpeg.exe"
            if not _target.exists():
                shutil.copyfile(_img_exe, _target)
            if _bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


class ModelPin(BaseModel):
    id: str
    model_id: str
    revision: str
    task: str
    dtype: str
    expected_vram_mb: int


class VelarisConfig(BaseModel):
    # App info
    app_name: str = "Velaris Motion AI Engine"
    version: str = "3.0.0"

    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000

    # Directories
    base_dir: Path = Path(__file__).resolve().parent
    workspace_root: Path = Path(__file__).resolve().parent.parent
    outputs_dir: Path = Path(__file__).resolve().parent / "outputs"
    temp_dir: Path = Path(__file__).resolve().parent / "temp"
    weights_dir: Path = Path(__file__).resolve().parent / "weights"

    # Concurrency and Job limits
    max_queue_size: int = 10
    request_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 600.0

    # Input validation constraints
    min_prompt_length: int = 1
    max_prompt_length: int = 1000
    min_inference_steps: int = 1
    max_inference_steps: int = 50
    min_fps: int = 1
    max_fps: int = 30
    min_motion_bucket_id: int = 1
    max_motion_bucket_id: int = 255
    max_dimension: int = 1920

    # Hardware & VRAM orchestration
    min_recommended_vram_mb: int = 6000
    force_cpu: bool = False
    enable_model_cpu_offload: bool = True
    vram_fraction_warning_threshold: float = 0.90

    # Inference acceleration & Hardware protection
    enable_torch_compile: bool = False
    enable_xformers: bool = False             # PyTorch 2.6+ SDPA used natively (FlashAttention)
    enable_interpolation: bool = True
    interpolation_target_fps: int = 24

    # Eco Mode (Silent Background Execution — protects laptop thermals & prevents UI lag)
    eco_mode: bool = False
    eco_mode_memory_fraction: float = 0.80       # Caps PyTorch allocation to 80% to keep system snappy
    eco_mode_step_cooldown_seconds: float = 0.05  # Micro-pause between steps to lower thermals & fan noise

    # Aspect Ratios: dimensions for generation and upscale
    default_aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    aspect_ratio_dims: Dict[str, Dict[str, int]] = {
        "16:9": {"width": 1024, "height": 576, "target_w": 1920, "target_h": 1080},
        "9:16": {"width": 576, "height": 1024, "target_w": 1080, "target_h": 1920},
        "1:1":  {"width": 768, "height": 768,  "target_w": 1080, "target_h": 1080},
    }

    # =========================================================================
    # Generation Architecture Selector
    # Options: "wan" (SOTA 3D DiT), "ltx" (Fast DiT), "studio" (Legacy SVD-XT),
    #          "cogvideox", "animatediff", "cloud" (Google Veo 2 / Kling)
    # =========================================================================
    active_engine: Literal["wan", "ltx", "studio", "cogvideox", "animatediff", "cloud"] = "wan"

    # =========================================================================
    # VRAM-to-RAM Block Swapping & Group Offloading Orchestration
    # Enables consumer 8GB GPUs to stream large DiT blocks via 32GB system RAM
    # =========================================================================
    enable_group_offload: bool = True
    group_offload_type: str = "block_level"     # "block_level" or "leaf_level"
    group_offload_num_blocks: int = 1           # 1 block per group minimizes peak VRAM on 8GB cards
    group_offload_use_stream: bool = True       # Asynchronous CUDA Stream for smooth PCIe pipelining

    # =========================================================================
    # Wan2.1 Stack (v4.0) — Flagship 3D Diffusion Transformer (Alibaba Wan-AI)
    # Native SOTA photorealism, flow matching, UMT5-XXL language comprehension
    # =========================================================================
    wan_t2v: ModelPin = ModelPin(
        id="wan_t2v",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="main",
        task="text-to-video",
        dtype="bfloat16",
        expected_vram_mb=6200,
    )
    wan_14b: ModelPin = ModelPin(
        id="wan_14b",
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        revision="main",
        task="text-to-video",
        dtype="bfloat16",
        expected_vram_mb=7800,
    )
    wan_steps: int = 30                     # 30 flow-matching steps: cinema convergence
    wan_guidance_scale: float = 5.0         # 5.0 CFG: rich contrast & precise adherence
    wan_num_frames: int = 81                # 81 frames = 5.0 seconds @ 16fps native
    wan_fps: int = 16                       # 16fps native Wan output
    wan_width: int = 832                    # 832x480 (16:9 cinematic widescreen)
    wan_height: int = 480

    # =========================================================================
    # LTX-Video Stack (v4.0) — Real-Time 24fps DiT (Lightricks)
    # Ultra-fast cinematic video generation with 1:192 3D VAE compression
    # =========================================================================
    ltx_video: ModelPin = ModelPin(
        id="ltx_video",
        model_id="Lightricks/LTX-Video",
        revision="main",
        task="text-to-video",
        dtype="bfloat16",
        expected_vram_mb=5600,
    )
    ltx_steps: int = 30                     # 30 steps: high-clarity output
    ltx_guidance_scale: float = 3.0         # 3.0 CFG
    ltx_num_frames: int = 97                # 97 frames = 4.0 seconds @ 24fps native
    ltx_fps: int = 24                       # 24fps native cinema motion
    ltx_width: int = 768                    # 768x512 (16:9 widescreen)
    ltx_height: int = 512

    # =========================================================================
    # Cloud Studio Bridge — Direct Google Veo 2 / Kling 1.5 API
    # =========================================================================
    cloud_provider: Literal["replicate", "fal"] = "replicate"
    cloud_model: str = "google/veo-2"       # Default cloud flagship

    # =========================================================================
    # Studio Photorealism Stack (v3.1) — Two-Stage Ultra High Fidelity Pipeline
    # Stage 1: epiCRealism DSLR (or SDXL-Turbo) Master Frame Synthesis
    # Stage 2: SVD-XT (Stable Video Diffusion XT, Fluid Motion Synthesis)
    # Stage 3: FFmpeg 24fps Motion Interpolation + Real-ESRGAN Upscale
    # =========================================================================
    studio_master_model: Literal["epicrealism", "sdxl_turbo"] = "epicrealism"
    studio_epic: ModelPin = ModelPin(
        id="studio_epic",
        model_id="emilianJR/epiCRealism",
        revision="main",
        task="text-to-image",
        dtype="float16",
        expected_vram_mb=3400,
    )
    studio_epic_steps: int = 20             # 20 steps: DSLR optical fidelity in ~3.7s
    studio_epic_guidance_scale: float = 6.0 # 6.0 CFG: natural filmic contrast & color roll-off

    studio_sdxl: ModelPin = ModelPin(
        id="studio_sdxl",
        model_id="stabilityai/sdxl-turbo",
        revision="main",
        task="text-to-image",
        dtype="float16",
        expected_vram_mb=4800,
    )
    studio_svd: ModelPin = ModelPin(
        id="studio_svd",
        model_id="stabilityai/stable-video-diffusion-img2vid-xt",
        revision="main",
        task="image-to-video",
        dtype="float16",
        expected_vram_mb=5600,
    )
    studio_sdxl_steps: int = 2              # 2-step SDXL-Turbo synthesis
    studio_sdxl_guidance_scale: float = 1.5 # 1.5 CFG
    studio_sdxl_width: int = 1024          # 1024x576 widescreen master frame
    studio_sdxl_height: int = 576
    studio_svd_num_frames: int = 14         # 14 frames = 8GB VRAM optimized
    studio_svd_steps: int = 12              # 12 steps: sharp temporal convergence without latent crawling
    studio_svd_motion_bucket_id: int = 120  # 120: fluid cinematic camera motion & parallax
    studio_svd_noise_aug_strength: float = 0.00 # 0.00: preserves exact optical fidelity of master frame
    studio_svd_fps: int = 7                 # 7fps native SVD (interpolated to 24fps cinema)

    # =========================================================================
    # CogVideoX-2B Stack (v2.5) — Flagship 3D Diffusion Transformer
    # Photorealistic cinematic generation on 8GB VRAM (DiT + T5-XXL)
    # =========================================================================
    cogvideox: ModelPin = ModelPin(
        id="cogvideox",
        model_id="THUDM/CogVideoX-2b",
        revision="main",
        task="text-to-video",
        dtype="bfloat16",
        expected_vram_mb=5500,
    )
    cogvideox_steps: int = 35               # 35 DPM steps: crisp convergence in ~1-2 min
    cogvideox_guidance_scale: float = 6.0   # 6.0 CFG for rich natural textures & prompt adherence
    cogvideox_num_frames: int = 49          # 49 frames = 6 seconds @ 8fps native
    cogvideox_width: int = 720              # 720x480 native 16:9 widescreen DiT resolution
    cogvideox_height: int = 480

    # =========================================================================
    # AnimateDiff-Lightning Stack (v2.0) — Fast Anime/Draft Mode
    # =========================================================================
    # Base model (ToonYou SD 1.5 fine-tune — fully cached offline)
    dreamer: ModelPin = ModelPin(
        id="dreamer",
        model_id="stablediffusionapi/toonyou",
        revision="main",
        task="text-to-image",
        dtype="float16",
        expected_vram_mb=3200,
    )

    # AnimateDiff-Lightning motion adapter (ByteDance — 8-step distilled high fidelity)
    animator: ModelPin = ModelPin(
        id="animator",
        model_id="ByteDance/AnimateDiff-Lightning",
        revision="main",
        task="text-to-video",
        dtype="float16",
        expected_vram_mb=4500,
    )

    # AnimateDiff Lightning step config: 8-step = smooth, sharp, zero graininess
    animatediff_steps: int = 8              # 8-step: ~15-20s/shot on RTX 4060, noise-free
    animatediff_guidance_scale: float = 1.4  # 1.4 enables negative prompt enforcement
    animatediff_num_frames: int = 16        # 16 frames = 2s video @ 8fps native
    animatediff_width: int = 768            # 768x512 = Widescreen 16:9 cinema ratio (+50% detail)
    animatediff_height: int = 512

    # Real-ESRGAN AI upscaling config
    realesrgan_anime_model: str = "RealESRGAN_x4plus_anime_6B"
    realesrgan_general_model: str = "RealESRGAN_x4plus"
    realesrgan_tile_size: int = 512         # Tile size for VRAM management


# Global configuration instance
config = VelarisConfig()

# Ensure required directories exist
os.makedirs(config.outputs_dir, exist_ok=True)
os.makedirs(config.temp_dir, exist_ok=True)
os.makedirs(config.weights_dir, exist_ok=True)
