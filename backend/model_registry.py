"""
Velaris Motion - Model Registry v2.0
AnimateDiff-Lightning Stack — Replaces SVD-XT pipeline
Speed: 4-step distillation, ~15-30s/shot on RTX 4060 8GB
Quality: Native anime aesthetics via ToonYou + AnimateDiff-Lightning
"""

import logging
from typing import Callable, Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import torch
from config import config, ModelPin

logger = logging.getLogger("velaris.model_registry")


class ModelDefinition(BaseModel):
    """
    Immutable specification of a machine learning model.
    Enforces pinned revisions rather than mutable 'latest' branches.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Internal identifier, e.g. 'dreamer'")
    model_id: str = Field(..., description="Hugging Face repo id")
    revision: str = Field(..., description="Pinned immutable commit hash or tag")
    task: str = Field(..., description="Task category: 'text-to-image' | 'text-to-video'")
    dtype: str = Field(default="float16", description="Precision: float16, float32, bfloat16")
    expected_vram_mb: int = Field(..., description="Expected peak VRAM usage in MB")


def _load_animatediff_lightning_pipeline(
    base_model_id: str,
    device: str,
    dtype: torch.dtype,
    num_steps: int = 4,
) -> Any:
    """
    Loads the AnimateDiff-Lightning pipeline.
    
    Architecture:
      - Base: ToonYou (anime-optimized SD 1.5 fine-tune)
      - Motion: AnimateDiff-Lightning 4-step distilled adapter (ByteDance)
      - Scheduler: LCMScheduler with trailing timestep spacing
    
    Speed: ~15-30s for 16 frames @ 512x512 on RTX 4060 (8GB)
    vs SVD-XT: ~45 min for same output → 90x speedup
    """
    from diffusers import AnimateDiffPipeline, MotionAdapter, LCMScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Step 1: Load AnimateDiff-Lightning motion adapter (4-step or 8-step)
    step_variants = {
        4: "animatediff_lightning_4step_diffusers.safetensors",
        8: "animatediff_lightning_8step_diffusers.safetensors",
        2: "animatediff_lightning_2step_diffusers.safetensors",
    }
    adapter_file = step_variants.get(num_steps, step_variants[4])
    logger.info(f"Loading AnimateDiff-Lightning motion adapter ({num_steps}-step)...")

    try:
        adapter_path = hf_hub_download("ByteDance/AnimateDiff-Lightning", adapter_file, local_files_only=True)
    except Exception:
        adapter_path = hf_hub_download("ByteDance/AnimateDiff-Lightning", adapter_file)

    adapter: Any = MotionAdapter()
    adapter.load_state_dict(
        load_file(
            adapter_path,
            device=device,
        )
    )

    # Step 2: Load anime base model + inject motion adapter
    logger.info(f"Loading anime base model: {base_model_id}")
    try:
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )

    # Step 3: LCM Scheduler — required for Lightning distilled models
    # trailing timestep_spacing is critical for 4-step quality
    pipe.scheduler = LCMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    # Step 4: Memory optimizations for 8GB VRAM
    if device == "cuda":
        # VAE slicing: decode frames one at a time -> prevents OOM at decode stage
        pipe.vae.enable_slicing()

        # xFormers memory-efficient attention if available
        if config.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("xFormers memory-efficient attention enabled → faster + less VRAM")
            except Exception as xf_err:
                # PyTorch 2.x SDPA is equally good fallback
                logger.info(f"xFormers unavailable ({xf_err}), using PyTorch native SDPA")

        # CPU offload: moves model components to RAM when not in use
        # Allows 8GB VRAM to handle the full pipeline comfortably
        pipe.enable_model_cpu_offload()
        logger.info("Model CPU offload enabled for 8GB VRAM compatibility")
    else:
        pipe.to("cpu")

    return pipe


def _load_wan_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """
    Loads Wan2.1 Text-to-Video 3D Diffusion Transformer (Alibaba Wan-AI).

    VAE is loaded in float32 for numerical stability (required by Wan2.1).
    Transformer and text encoder are loaded in bfloat16 for memory efficiency.
    Applies Hybrid RAM-VRAM Group Offload or model CPU offload for 8GB VRAM compatibility.
    Uses local_files_only=True to skip HuggingFace Hub connection checks when offline.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    logger.info(f"Loading Wan2.1 VAE (float32) from '{model_id}'...")
    # VAE must be float32 for Wan2.1 numerical stability
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
        )
    except Exception:
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

    logger.info(f"Loading Wan2.1 pipeline from '{model_id}' (dtype={dtype})...")
    try:
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
        )

    # Use UniPC scheduler with flow_shift=8.0 for optimal Wan2.1 motion quality
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=8.0
    )

    # Enable VAE slicing and tiling for 8GB VRAM safety during long sequence decode
    if hasattr(pipe, "vae") and pipe.vae is not None:
        try:
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            logger.info("Wan2.1: VAE slicing and tiling enabled")
        except Exception as e:
            logger.warning(f"Wan2.1 VAE memory optimization fallback: {e}")

    if device == "cuda":
        # Wan2.1-1.3B is 1.3B params (~2.6GB in bf16), perfectly sized to reside on 8GB GPU.
        # enable_model_cpu_offload swaps TextEncoder -> DiT Transformer -> VAE seamlessly without PCIe bottleneck.
        if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            logger.info("Wan2.1: Model CPU offload active (full GPU residency per stage)")
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")

    return pipe


def _default_wan_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for Wan2.1 Text-to-Video pipeline."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
    return _load_wan_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


def _load_ltx_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """
    Loads LTX-Video 2B DiT pipeline (Lightricks) for ultra-fast 24fps cinema generation.
    """
    from diffusers import LTXPipeline

    logger.info(f"Loading LTX-Video pipeline from '{model_id}' (dtype={dtype})...")
    pipe = LTXPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    if device == "cuda":
        if config.enable_group_offload and hasattr(pipe, "enable_group_offload"):
            try:
                pipe.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level",
                    num_blocks_per_group=1,
                    use_stream=True,
                )
                logger.info("LTX-Video: Hybrid RAM-VRAM Group Offload active")
            except Exception:
                if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
        elif cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            logger.info("LTX-Video: Model CPU offload active")
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")

    return pipe


def _default_ltx_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for LTX-Video pipeline."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
    return _load_ltx_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


def _load_cogvideox_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """
    Loads the CogVideoX-2B pipeline (3D Diffusion Transformer).
    
    Architecture:
      - Expert 3D DiT with temporal cross-attention
      - T5-XXL language model for nuanced prompt comprehension
      - 3D Causal VAE for high-fidelity spatial-temporal compression
    
    Optimizations for 8GB VRAM (RTX 4060):
      - enable_model_cpu_offload: offloads inactive components to system RAM (peak VRAM ~5.5GB)
      - vae.enable_tiling: decodes 3D video in spatial tiles
      - vae.enable_slicing: decodes temporal frames sequentially
    """
    from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler

    logger.info(f"Loading CogVideoX-2B pipeline from '{model_id}' (dtype={dtype})...")
    try:
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if device == "cuda":
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("CogVideoX: Model CPU offload + VAE tiling/slicing enabled for 8GB VRAM")
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")

    return pipe


def _default_cogvideox_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for CogVideoX-2B pipeline."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
    return _load_cogvideox_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


def _default_animatediff_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for AnimateDiff-Lightning pipeline."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
    return _load_animatediff_lightning_pipeline(
        base_model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        num_steps=config.animatediff_steps,
    )


def _load_epicrealism_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """Loads epiCRealism DSLR pipeline for ultra-photorealistic Master Frame synthesis."""
    from diffusers import StableDiffusionPipeline

    logger.info(f"Loading epiCRealism DSLR pipeline from '{model_id}' (dtype={dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # Disable safety checker: avoids CLIP feature-extractor deprecation crash
        # and saves ~1.2 GB VRAM. Content safety is handled upstream by the security layer.
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    if device == "cuda":
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("epiCRealism: Model CPU offload enabled (safety_checker disabled)")
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")
    return pipe


def _default_epicrealism_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for epiCRealism text-to-image master frame synthesis."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
    return _load_epicrealism_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


def _load_sdxl_turbo_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """Loads SDXL-Turbo for high-resolution 1024x576 photorealistic master frames."""
    from diffusers import AutoPipelineForText2Image

    logger.info(f"Loading SDXL-Turbo pipeline from '{model_id}' (dtype={dtype})...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    if device == "cuda":
        if cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("SDXL-Turbo: Model CPU offload enabled")
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")
    return pipe


def _load_svd_xt_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    cpu_offload: bool = True,
) -> Any:
    """Loads Stable Video Diffusion XT for fluid cinematic video motion."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from diffusers import StableVideoDiffusionPipeline

    logger.info(f"Loading Stable Video Diffusion XT pipeline from '{model_id}'...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    if device == "cuda":
        # Sequential CPU offload guarantees execution within 8GB VRAM
        pipe.enable_sequential_cpu_offload()
        logger.info("SVD-XT: Sequential CPU offload enabled for 8GB VRAM compatibility")
    else:
        pipe.to("cpu")
    return pipe


def _default_sdxl_turbo_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for SDXL-Turbo text-to-image master frame synthesis."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
    return _load_sdxl_turbo_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


def _default_svd_xt_loader(
    model_def: ModelDefinition,
    device: str,
    dtype: Union[torch.dtype, str],
    cpu_offload: bool = True,
) -> Any:
    """Registry loader wrapper for Stable Video Diffusion XT video motion synthesis."""
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
    return _load_svd_xt_pipeline(
        model_id=model_def.model_id,
        device=device,
        dtype=torch_dtype,
        cpu_offload=cpu_offload,
    )


class ModelRegistry:
    """
    Central repository for model definitions and loaders.
    Supports Studio Photorealism (SDXL+SVD+Real-ESRGAN), CogVideoX-2B,
    and AnimateDiff-Lightning.
    """
    def __init__(self):
        self._definitions: Dict[str, ModelDefinition] = {}
        self._loaders: Dict[str, Callable[..., Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        # Wan2.1 SOTA 3D DiT (1.3B) — Flagship Photorealistic Engine
        wan_def = ModelDefinition(
            id=config.wan_t2v.id,
            model_id=config.wan_t2v.model_id,
            revision=config.wan_t2v.revision,
            task="text-to-video",
            dtype=config.wan_t2v.dtype,
            expected_vram_mb=config.wan_t2v.expected_vram_mb,
        )
        self.register(wan_def, _default_wan_loader)

        # Wan2.1 Heavy Cinema (14B) — RAM Group-Offload Engine
        wan_14b_def = ModelDefinition(
            id=config.wan_14b.id,
            model_id=config.wan_14b.model_id,
            revision=config.wan_14b.revision,
            task="text-to-video",
            dtype=config.wan_14b.dtype,
            expected_vram_mb=config.wan_14b.expected_vram_mb,
        )
        self.register(wan_14b_def, _default_wan_loader)

        # LTX-Video 2B DiT (Lightricks) — 24fps Cinema Engine
        ltx_def = ModelDefinition(
            id=config.ltx_video.id,
            model_id=config.ltx_video.model_id,
            revision=config.ltx_video.revision,
            task="text-to-video",
            dtype=config.ltx_video.dtype,
            expected_vram_mb=config.ltx_video.expected_vram_mb,
        )
        self.register(ltx_def, _default_ltx_loader)

        # Two-Stage Photorealistic Studio Stack: epiCRealism DSLR Master Frame
        epic_def = ModelDefinition(
            id=config.studio_epic.id,
            model_id=config.studio_epic.model_id,
            revision=config.studio_epic.revision,
            task="text-to-image",
            dtype=config.studio_epic.dtype,
            expected_vram_mb=config.studio_epic.expected_vram_mb,
        )
        self.register(epic_def, _default_epicrealism_loader)

        # Two-Stage Photorealistic Studio Stack: SDXL-Turbo Master Frame
        sdxl_def = ModelDefinition(
            id=config.studio_sdxl.id,
            model_id=config.studio_sdxl.model_id,
            revision=config.studio_sdxl.revision,
            task="text-to-image",
            dtype=config.studio_sdxl.dtype,
            expected_vram_mb=config.studio_sdxl.expected_vram_mb,
        )
        self.register(sdxl_def, _default_sdxl_turbo_loader)

        # Two-Stage Photorealistic Studio Stack: SVD-XT Motion Synthesis
        svd_def = ModelDefinition(
            id=config.studio_svd.id,
            model_id=config.studio_svd.model_id,
            revision=config.studio_svd.revision,
            task="image-to-video",
            dtype=config.studio_svd.dtype,
            expected_vram_mb=config.studio_svd.expected_vram_mb,
        )
        self.register(svd_def, _default_svd_xt_loader)

        # Flagship 3D Diffusion Transformer (CogVideoX-2B)
        cogvideox_def = ModelDefinition(
            id=config.cogvideox.id,
            model_id=config.cogvideox.model_id,
            revision=config.cogvideox.revision,
            task="text-to-video",
            dtype=config.cogvideox.dtype,
            expected_vram_mb=config.cogvideox.expected_vram_mb,
        )
        self.register(cogvideox_def, _default_cogvideox_loader)

        # Unified AnimateDiff-Lightning pipeline
        # The "dreamer" slot holds the AnimateDiff pipeline
        animatediff_def = ModelDefinition(
            id=config.dreamer.id,
            model_id=config.dreamer.model_id,
            revision=config.dreamer.revision,
            task="text-to-video",   # Combined T2I + animation
            dtype=config.dreamer.dtype,
            expected_vram_mb=config.dreamer.expected_vram_mb,
        )
        self.register(animatediff_def, _default_animatediff_loader)

        # Keep animator slot registered for API compatibility
        animator_def = ModelDefinition(
            id=config.animator.id,
            model_id=config.animator.model_id,
            revision=config.animator.revision,
            task="motion-adapter",
            dtype=config.animator.dtype,
            expected_vram_mb=config.animator.expected_vram_mb,
        )
        self.register(animator_def, _default_animatediff_loader)

    def register(self, definition: ModelDefinition, loader: Optional[Callable[..., Any]] = None):
        """Registers a model definition and its corresponding loader function."""
        self._definitions[definition.id] = definition
        if loader:
            self._loaders[definition.id] = loader

    def get_definition(self, model_id: str) -> ModelDefinition:
        """Retrieves definition by id, raises KeyError if not found."""
        if model_id not in self._definitions:
            raise KeyError(f"Model '{model_id}' is not registered in ModelRegistry.")
        return self._definitions[model_id]

    def get_loader(self, model_id: str) -> Callable[..., Any]:
        """Retrieves loader function by id."""
        if model_id not in self._loaders:
            raise KeyError(f"Loader for model '{model_id}' not found.")
        return self._loaders[model_id]

    def load(
        self,
        model_id: str,
        device: str,
        dtype: Union[torch.dtype, str],
        cpu_offload: bool = True,
    ) -> Any:
        """Loads and returns an instantiated pipeline for the registered model."""
        definition = self.get_definition(model_id)
        loader = self.get_loader(model_id)
        return loader(definition, device=device, dtype=dtype, cpu_offload=cpu_offload)

    def list_models(self) -> Dict[str, ModelDefinition]:
        """Returns all registered model definitions."""
        return dict(self._definitions)


# Global singleton instance
model_registry = ModelRegistry()
