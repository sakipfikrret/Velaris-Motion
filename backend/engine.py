"""
Velaris Motion - Engine v2.0
AnimateDiff-Lightning Architecture — Single-pipeline text-to-video
Replaces: Dreamer (SD1.5) + SVD-XT two-stage pipeline
New:      AnimateDiff-Lightning (4-step, ~15-30s/shot, native anime)
"""

import os
import time
import random
import logging
import shutil
from pathlib import Path
from typing import Optional, Any, Dict, List, Literal, cast
import torch
from PIL import Image
import numpy as np

from config import config
from security import SafeGenerationRequest
from model_registry import model_registry, ModelRegistry
from vram_manager import vram_manager, VRAMManager, VRAMState
from job_queue import Job, GenerationStage, JobStatus
from reproducibility import GenerationMetadata
from interpolation import interpolate_video, is_ffmpeg_available
from audio_synthesizer import generate_audio_for_job, mux_audio_to_video
from director import plan_storyboard, StoryboardPlan, RESOLUTION_MAP
from upscaler import upscale_video_to_quality, concatenate_video_shots, is_ffmpeg_available as upscaler_ffmpeg_available

logger = logging.getLogger("velaris.engine")


class HybridEngine:
    """
    Velaris Motion Engine v2.0 — AnimateDiff-Lightning
    
    Architecture change from v1.x:
      OLD: [Dreamer SD1.5 text-to-image] → [SVD-XT image-to-video] (2 models, ~45 min/shot)
      NEW: [AnimateDiff-Lightning text-to-video] (1 model, ~15-30s/shot)
    
    Key improvements:
      - 90x faster generation (4-step distillation vs 12-20 SVD-XT steps)
      - Native anime aesthetics (ToonYou base model)
      - No temporal flickering (AnimateDiff temporal attention, not SVD-XT)
      - xFormers memory-efficient attention
      - Real-ESRGAN AI upscaling (replaces Lanczos)
    """
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        vram_mgr: Optional[VRAMManager] = None,
        mock_mode: bool = False,
    ):
        self.registry = registry or model_registry
        self.vram = vram_mgr or vram_manager
        self.mock_mode = mock_mode or (os.environ.get("VELARIS_MOCK_MODE", "0") == "1")

        if torch.cuda.is_available() and not config.force_cpu:
            self.device = "cuda"
            self.dtype = torch.float16
            self.cpu_offload = config.enable_model_cpu_offload
            logger.info("Velaris Engine v2.0 initialized on CUDA (fp16, AnimateDiff-Lightning).")
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            self.cpu_offload = False
            logger.info("Velaris Engine v2.0 initialized on CPU (fp32 fallback).")

    def _check_cancellation(self, job: Job):
        if job.is_cancelled:
            self.vram.safe_cleanup()
            raise InterruptedError("Generation was cancelled by user request.")

    def execute_job(self, job: Job):
        """
        Executes a queued generation job through the AnimateDiff-Lightning pipeline.
        Single unified model: text prompt → animated video frames in one pass.
        """
        start_time = time.time()
        video_filename = f"{job.job_id}.mp4"
        video_path = config.outputs_dir / video_filename
        metadata_path = config.outputs_dir / f"{job.job_id}_metadata.json"

        # Deterministic seed
        seed = job.request.seed if job.request.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            self._check_cancellation(job)

            if self.mock_mode:
                self._execute_mock_pipeline(job, seed, video_path, metadata_path, start_time)
                return

            # =========================================================================
            # DIRECTOR MODE: Multi-scene storyboard
            # =========================================================================
            use_director = getattr(job.request, "use_director", False)
            quality = cast(
                Literal["draft", "hd", "fhd", "4k"],
                getattr(job.request, "quality", "fhd")
            )
            duration_mode = cast(
                Literal["auto", "short", "medium", "long", "cinematic"],
                getattr(job.request, "duration_mode", "auto")
            )
            target_duration: Optional[int] = getattr(job.request, "target_duration", None)

            if use_director:
                self._execute_director_pipeline(
                    job, seed, generator, video_path, metadata_path, start_time,
                    quality=quality, duration_mode=duration_mode, target_duration=target_duration
                )
                return

            # =========================================================================
            # SINGLE-SHOT PIPELINE
            # Engine dispatch: Studio Photoreal (SDXL+SVD+Real-ESRGAN) vs CogVideoX vs AnimateDiff
            # =========================================================================
            req_engine = getattr(job.request, "engine", "auto")
            clean_prompt = job.request.prompt.strip()
            prompt_lower = clean_prompt.lower()
            anime_keywords = ["anime", "manga", "cartoon", "illustration", "ghibli", "drawn", "sketch"]

            if req_engine == "animatediff" or (req_engine == "auto" and any(k in prompt_lower for k in anime_keywords)):
                selected_engine = "animatediff"
                active_model_id = config.dreamer.id
                active_dtype = self.dtype
                num_frames = getattr(job.request, "num_frames", config.animatediff_num_frames)
            elif req_engine == "cloud" or (req_engine == "auto" and config.active_engine == "cloud"):
                selected_engine = "cloud"
                active_model_id = "cloud"
                active_dtype = self.dtype
                num_frames = 0
            elif req_engine == "ltx" or (req_engine == "auto" and config.active_engine == "ltx"):
                selected_engine = "ltx"
                active_model_id = config.ltx_video.id
                active_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                num_frames = config.ltx_num_frames
            elif req_engine == "wan" or (req_engine == "auto" and config.active_engine == "wan"):
                selected_engine = "wan"
                active_model_id = config.wan_t2v.id
                active_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                num_frames = config.wan_num_frames
            elif req_engine == "cogvideox" or (req_engine == "auto" and config.active_engine == "cogvideox"):
                selected_engine = "cogvideox"
                active_model_id = config.cogvideox.id
                active_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                num_frames = config.cogvideox_num_frames
            else:
                # Default flagship: Studio Photorealistic Stack
                selected_engine = "studio"
                active_model_id = config.studio_sdxl.id
                active_dtype = self.dtype
                num_frames = config.studio_svd_num_frames

            num_segments = getattr(job.request, "num_segments", 1)
            all_frames: List[Any] = []

            # Aspect Ratio configuration
            req_ar = getattr(job.request, "aspect_ratio", "16:9")
            ar_spec = config.aspect_ratio_dims.get(req_ar, config.aspect_ratio_dims["16:9"])
            stage1_w, stage1_h = ar_spec["width"], ar_spec["height"]

            # Eco Mode: Protect GPU thermals & keep laptop responsive
            eco_mode = getattr(job.request, "eco_mode", False) or config.eco_mode
            if eco_mode and torch.cuda.is_available():
                try:
                    torch.cuda.set_per_process_memory_fraction(config.eco_mode_memory_fraction)
                    logger.info(f"Eco Mode active: VRAM capped at {config.eco_mode_memory_fraction*100:.0f}%, thermals protected")
                except Exception:
                    pass

            if selected_engine == "cloud":
                # =============================================================
                # CLOUD STUDIO PIPELINE (Google Veo 2 / Kling 1.5 / Wan 14B API)
                # =============================================================
                job.update_stage(
                    GenerationStage.GENERATING_IMAGE,
                    0.20,
                    f"Cloud Studio: Submitting prompt to {config.cloud_model}..."
                )
                from cloud_provider import cloud_provider
                cloud_provider.generate_video(
                    prompt=clean_prompt,
                    output_path=video_path,
                    model_name=config.cloud_model,
                    aspect_ratio=req_ar,
                    mock_mode=self.mock_mode,
                )

                duration = round(time.time() - start_time, 2)
                metadata = GenerationMetadata(
                    job_id=job.job_id,
                    prompt=job.request.prompt,
                    seed=seed,
                    models={"pipeline": {"model_id": config.cloud_model, "architecture": "Cloud Cinema API"}},
                    resolution={"width": ar_spec["target_w"], "height": ar_spec["target_h"]},
                    steps={"cloud_steps": 50, "num_frames": 120},
                    motion_bucket_id=0,
                    fps=24,
                    scheduler="CloudFlowMatching",
                    dtype="bfloat16",
                    device="cloud",
                    gpu_device_name="Cloud H100 Cluster",
                    vram_peak_mb=0.0,
                    generation_duration_seconds=duration,
                )
                metadata.save(metadata_path)
                job.video_path = str(video_path)
                job.video_url = f"/videos/{video_filename}"
                job.metadata_path = str(metadata_path)
                job.metadata = metadata.model_dump()
                job.update_stage(GenerationStage.EXPORTING, 1.0, f"Cloud video generated successfully via {config.cloud_model}!")
                return

            if selected_engine == "studio":
                # Check for direct image-to-video upload
                init_img_path = getattr(job, "initial_image_path", None)
                if init_img_path and Path(init_img_path).exists():
                    job.update_stage(
                        GenerationStage.GENERATING_IMAGE,
                        0.20,
                        f"Studio [1/3]: Loaded initial image for direct animation ({Path(init_img_path).name})"
                    )
                    master_img = Image.open(init_img_path).convert("RGB")
                    master_frame_path = config.outputs_dir / f"{job.job_id}_master.jpg"
                    master_img.save(str(master_frame_path), quality=95)
                    logger.info(f"Direct animation master frame loaded: {master_frame_path.name}")
                else:
                    # -------------------------------------------------------------
                    # STUDIO STAGE 1: Photorealistic Master Frame Synthesis (epiCRealism / SDXL)
                    # -------------------------------------------------------------
                    master_model_type = getattr(config, "studio_master_model", "epicrealism")
                    active_master_id = config.studio_epic.id if master_model_type == "epicrealism" else config.studio_sdxl.id
                    active_label = "epiCRealism DSLR" if master_model_type == "epicrealism" else "SDXL-Turbo"

                    job.update_stage(GenerationStage.LOADING_MODEL, 0.10, f"Studio [1/3]: Loading {active_label} Master Frame pipeline...")
                    self._check_cancellation(job)

                    self.vram.prepare_for_model(active_master_id, auto_unload=True)
                    pipe_master = self.registry.load(
                        active_master_id,
                        device=self.device,
                        dtype=active_dtype,
                        cpu_offload=self.cpu_offload,
                    )
                    self.vram.set_model_loaded(active_master_id, pipe_master)
                    self._check_cancellation(job)

                    job.update_stage(
                        GenerationStage.GENERATING_IMAGE,
                        0.20,
                        f"Studio [1/3]: Synthesizing Master Frame with {active_label} ('{clean_prompt[:40]}...')"
                    )
                    studio_prompt, studio_neg = self._build_studio_prompts(job.request)

                    with self.vram.managed_inference(active_master_id):
                        if master_model_type == "epicrealism":
                            actual_steps = config.studio_epic_steps
                            actual_guidance = config.studio_epic_guidance_scale
                            synth_w = 768 if stage1_w >= stage1_h else 432
                            synth_h = 432 if stage1_w >= stage1_h else 768
                            if stage1_w == stage1_h:
                                synth_w = synth_h = 512
                            raw_master = pipe_master(
                                prompt=studio_prompt,
                                negative_prompt=studio_neg,
                                num_inference_steps=actual_steps,
                                guidance_scale=actual_guidance,
                                width=synth_w,
                                height=synth_h,
                                generator=generator,
                            ).images[0]
                            master_img = raw_master.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
                        else:
                            actual_sdxl_guidance = config.studio_sdxl_guidance_scale if config.studio_sdxl_guidance_scale > 0 else 1.5
                            actual_sdxl_steps = max(config.studio_sdxl_steps, 2)
                            master_img = pipe_master(
                                prompt=studio_prompt,
                                negative_prompt=studio_neg,
                                num_inference_steps=actual_sdxl_steps,
                                guidance_scale=actual_sdxl_guidance,
                                width=stage1_w,
                                height=stage1_h,
                                generator=generator,
                            ).images[0]

                    # Save master frame to outputs
                    master_frame_path = config.outputs_dir / f"{job.job_id}_master.jpg"
                    master_img.save(str(master_frame_path), quality=95)
                    logger.info(f"Studio Master Frame generated: {master_frame_path.name}")

                    # Unload master model to leave maximum VRAM for SVD-XT
                    self.vram.unload_current_model()
                    self._check_cancellation(job)

                # -------------------------------------------------------------
                # STUDIO STAGE 2: SVD-XT Fluid Video Motion Synthesis
                # -------------------------------------------------------------
                job.update_stage(
                    GenerationStage.ANIMATING,
                    0.40,
                    f"Studio [2/3]: SVD-XT synthesizing {config.studio_svd_num_frames} fluid motion frames..."
                )
                self.vram.prepare_for_model(config.studio_svd.id, auto_unload=True)
                pipe_svd = self.registry.load(
                    config.studio_svd.id,
                    device=self.device,
                    dtype=active_dtype,
                    cpu_offload=self.cpu_offload,
                )
                self.vram.set_model_loaded(config.studio_svd.id, pipe_svd)
                self._check_cancellation(job)

                svd_input_img = master_img.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)

                with self.vram.managed_inference(config.studio_svd.id):
                    actual_motion_bucket = getattr(job.request, "motion_bucket_id", config.studio_svd_motion_bucket_id) or config.studio_svd_motion_bucket_id
                    actual_svd_steps = getattr(job.request, "num_inference_steps", config.studio_svd_steps) or config.studio_svd_steps
                    actual_noise_aug = getattr(config, "studio_svd_noise_aug_strength", 0.00)

                    svd_frames = pipe_svd(
                        svd_input_img,
                        num_frames=config.studio_svd_num_frames,
                        num_inference_steps=actual_svd_steps,
                        motion_bucket_id=actual_motion_bucket,
                        noise_aug_strength=actual_noise_aug,
                        fps=config.studio_svd_fps,
                        decode_chunk_size=1,
                        generator=generator,
                    ).frames[0]

                all_frames = svd_frames
                self.vram.unload_current_model()
                self._check_cancellation(job)

            else:
                # Direct DiT (Wan2.1 / LTX-Video / CogVideoX) or AnimateDiff-Lightning path
                if selected_engine == "wan":
                    engine_label = "Wan2.1 3D DiT (Alibaba)"
                elif selected_engine == "ltx":
                    engine_label = "LTX-Video 24fps DiT (Lightricks)"
                elif selected_engine == "cogvideox":
                    engine_label = "CogVideoX-2B 3D DiT"
                else:
                    engine_label = "AnimateDiff-Lightning"

                job.update_stage(GenerationStage.LOADING_MODEL, 0.10, f"Loading {engine_label} pipeline...")
                self._check_cancellation(job)

                self.vram.prepare_for_model(active_model_id, auto_unload=True)
                pipe = self.registry.load(
                    active_model_id,
                    device=self.device,
                    dtype=active_dtype,
                    cpu_offload=self.cpu_offload,
                )
                self.vram.set_model_loaded(active_model_id, pipe)
                self._check_cancellation(job)

                with self.vram.managed_inference(active_model_id):
                    if selected_engine == "wan":
                        job.update_stage(
                            GenerationStage.GENERATING_IMAGE,
                            0.30,
                            f"Wan2.1 DiT: '{clean_prompt[:50]}...' ({config.wan_steps} steps, {config.wan_num_frames} frames)"
                        )
                        _req_steps = getattr(job.request, "num_inference_steps", None)
                        actual_steps = _req_steps if (_req_steps and _req_steps > 8) else config.wan_steps
                        _req_guidance = getattr(job.request, "guidance_scale", None)
                        actual_guidance = _req_guidance if (_req_guidance and _req_guidance >= 2.0) else config.wan_guidance_scale
                        enhanced_prompt, neg_prompt = self._build_cinematic_prompts(job.request)

                        if req_ar == "9:16":
                            wan_w, wan_h = 480, 832
                        elif req_ar == "1:1":
                            wan_w, wan_h = 624, 624
                        else:
                            wan_w, wan_h = config.wan_width, config.wan_height

                        result = pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=neg_prompt,
                            height=wan_h,
                            width=wan_w,
                            num_frames=config.wan_num_frames,
                            num_inference_steps=actual_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                        )
                        all_frames = result.frames[0]

                    elif selected_engine == "ltx":
                        job.update_stage(
                            GenerationStage.GENERATING_IMAGE,
                            0.30,
                            f"LTX-Video DiT: '{clean_prompt[:50]}...' ({config.ltx_steps} steps, {config.ltx_num_frames} frames)"
                        )
                        _req_steps = getattr(job.request, "num_inference_steps", None)
                        actual_steps = _req_steps if (_req_steps and _req_steps > 8) else config.ltx_steps
                        _req_guidance = getattr(job.request, "guidance_scale", None)
                        actual_guidance = _req_guidance if (_req_guidance and _req_guidance >= 2.0) else config.ltx_guidance_scale
                        enhanced_prompt, neg_prompt = self._build_cinematic_prompts(job.request)

                        result = pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=neg_prompt,
                            width=config.ltx_width,
                            height=config.ltx_height,
                            num_frames=config.ltx_num_frames,
                            num_inference_steps=actual_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                        )
                        all_frames = result.frames[0]

                    elif selected_engine == "cogvideox":
                        job.update_stage(
                            GenerationStage.GENERATING_IMAGE,
                            0.30,
                            f"CogVideoX-2B DiT: '{clean_prompt[:50]}...' ({config.cogvideox_steps} steps, {config.cogvideox_num_frames} frames)"
                        )
                        _req_steps = getattr(job.request, "num_inference_steps", None)
                        actual_steps = _req_steps if (_req_steps and _req_steps > 8) else config.cogvideox_steps
                        _req_guidance = getattr(job.request, "guidance_scale", None)
                        actual_guidance = _req_guidance if (_req_guidance and _req_guidance >= 2.0) else config.cogvideox_guidance_scale
                        enhanced_prompt, neg_prompt = self._build_cinematic_prompts(job.request)

                        result = pipe(
                            prompt=enhanced_prompt,
                            negative_prompt=neg_prompt,
                            num_frames=config.cogvideox_num_frames,
                            num_inference_steps=actual_steps,
                            guidance_scale=actual_guidance,
                            generator=generator,
                            width=config.cogvideox_width,
                            height=config.cogvideox_height,
                        )
                        all_frames = result.frames[0]
                    else:
                        # AnimateDiff-Lightning path
                        job.update_stage(
                            GenerationStage.GENERATING_IMAGE,
                            0.30,
                            f"AnimateDiff-Lightning: '{clean_prompt[:50]}...' ({config.animatediff_steps} steps, {num_frames} frames)"
                        )
                        enhanced_prompt, neg_prompt = self._build_anime_prompts(job.request)

                        for seg_i in range(num_segments):
                            self._check_cancellation(job)

                            if num_segments > 1:
                                seg_pct = 0.30 + (0.50 * (seg_i / num_segments))
                                job.update_stage(
                                    GenerationStage.ANIMATING,
                                    seg_pct,
                                    f"Generating segment {seg_i + 1}/{num_segments} ({num_frames} frames)..."
                                )

                            actual_steps = getattr(job.request, "num_inference_steps", None) or config.animatediff_steps
                            actual_guidance = getattr(job.request, "guidance_scale", None) or config.animatediff_guidance_scale

                            result = pipe(
                                prompt=enhanced_prompt,
                                negative_prompt=neg_prompt,
                                num_frames=num_frames,
                                num_inference_steps=actual_steps,
                                guidance_scale=actual_guidance,
                                generator=generator,
                                width=config.animatediff_width,
                                height=config.animatediff_height,
                            )
                            seg_frames = result.frames[0]

                            if seg_i == 0:
                                all_frames.extend(seg_frames)
                            else:
                                all_frames.extend(seg_frames[1:])  # Skip duplicate boundary frame

                # Unload pipeline
                self.vram.unload_current_model()

            frames = all_frames
            self._check_cancellation(job)

            # STAGE 3: Unload pipeline to free VRAM for post-processing
            self.vram.unload_current_model()

            # STAGE 4: Export + post-processing
            job.update_stage(GenerationStage.EXPORTING, 0.82, "Encoding and exporting video...")
            self._check_cancellation(job)

            from diffusers.utils import export_to_video
            export_fps = config.ltx_fps if selected_engine == "ltx" else (config.wan_fps if selected_engine == "wan" else job.request.fps)
            export_to_video(frames, str(video_path), fps=export_fps)

            # Frame interpolation: only apply when native framerate is below target fps (skip for native 24fps LTX)
            interpolation_applied = False
            if (
                selected_engine != "ltx"
                and job.request.interpolate
                and config.enable_interpolation
                and is_ffmpeg_available()
            ):
                job.update_stage(
                    GenerationStage.EXPORTING, 0.86,
                    f"RIFE/FFmpeg interpolation to {config.interpolation_target_fps}fps..."
                )
                interpolated_path = video_path.with_stem(video_path.stem + "_interp")
                try:
                    interpolate_video(video_path, interpolated_path, target_fps=config.interpolation_target_fps)
                    video_path.unlink(missing_ok=True)
                    interpolated_path.rename(video_path)
                    interpolation_applied = True
                    logger.info(f"Interpolation: {config.interpolation_target_fps}fps cinema quality")
                except Exception as interp_err:
                    logger.warning(f"Interpolation failed (non-fatal): {interp_err}")
                    if interpolated_path.exists():
                        interpolated_path.unlink(missing_ok=True)

            # AI upscaling: Real-ESRGAN or FFmpeg Lanczos fallback
            quality_sq = cast(
                Literal["draft", "hd", "fhd", "4k"],
                getattr(job.request, "quality", "draft")
            )
            if quality_sq != "draft" and upscaler_ffmpeg_available():
                job.update_stage(
                    GenerationStage.EXPORTING, 0.90,
                    f"AI Upscaling to {quality_sq.upper()} ({RESOLUTION_MAP[quality_sq]['label']})..."
                )
                upscaled_path = video_path.with_stem(video_path.stem + f"_{quality_sq}")
                try:
                    upscale_video_to_quality(
                        video_path,
                        upscaled_path,
                        quality=quality_sq,
                        target_width=ar_spec["target_w"] if selected_engine == "studio" else None,
                        target_height=ar_spec["target_h"] if selected_engine == "studio" else None,
                    )
                    video_path.unlink(missing_ok=True)
                    upscaled_path.rename(video_path)
                    logger.info(f"Upscaled to {quality_sq.upper()}")
                except Exception as up_err:
                    logger.warning(f"Upscaling failed (non-fatal): {up_err}")
                    if upscaled_path.exists():
                        upscaled_path.unlink(missing_ok=True)

            # Audio synthesis
            audio_path = None
            if job.request.audio_mode != "none":
                total_duration = len(frames) / job.request.fps
                job.update_stage(
                    GenerationStage.EXPORTING, 0.94,
                    f"Generating {job.request.audio_mode} audio ({total_duration:.1f}s)..."
                )
                audio_path = generate_audio_for_job(
                    prompt=job.request.prompt,
                    audio_mode=job.request.audio_mode,
                    duration_seconds=total_duration,
                    temp_dir=config.temp_dir,
                    job_id=job.job_id,
                )
                if audio_path and shutil.which("ffmpeg"):
                    muxed_path = video_path.with_stem(video_path.stem + "_av")
                    try:
                        mux_audio_to_video(video_path, audio_path, muxed_path, video_duration=total_duration)
                        video_path.unlink(missing_ok=True)
                        muxed_path.rename(video_path)
                        logger.info("Audio muxed successfully.")
                    except Exception as mux_err:
                        logger.warning(f"Audio mux failed (non-fatal): {mux_err}")
                        if muxed_path.exists():
                            muxed_path.unlink(missing_ok=True)
                    finally:
                        if audio_path and audio_path.exists():
                            audio_path.unlink(missing_ok=True)

            # Save metadata
            duration = round(time.time() - start_time, 2)
            vram_stats = self.vram.get_memory_stats()
            # Resolve final output resolution from upscale preset or native engine size
            if selected_engine == "wan":
                _native_w, _native_h = config.wan_width, config.wan_height
            elif selected_engine == "ltx":
                _native_w, _native_h = config.ltx_width, config.ltx_height
            elif selected_engine == "studio":
                _native_w, _native_h = stage1_w, stage1_h
            elif selected_engine == "cogvideox":
                _native_w, _native_h = config.cogvideox_width, config.cogvideox_height
            else:
                _native_w, _native_h = config.animatediff_width, config.animatediff_height

            _fallback_res = {"width": _native_w, "height": _native_h}
            if quality_sq != "draft":
                if selected_engine == "studio":
                    final_res = {"width": ar_spec["target_w"], "height": ar_spec["target_h"]}
                elif quality_sq in RESOLUTION_MAP:
                    final_res = RESOLUTION_MAP[quality_sq]
                else:
                    final_res = _fallback_res
            else:
                final_res = _fallback_res

            if selected_engine == "wan":
                pipeline_meta = {
                    "model_id": config.wan_t2v.model_id,
                    "architecture": "Wan2.1 (3D Diffusion Transformer, Flow Matching)",
                    "steps": str(config.wan_steps),
                    "guidance_scale": str(config.wan_guidance_scale),
                    "memory_offload": "Hybrid RAM-VRAM Group Offload",
                }
                steps_meta = {
                    "wan_steps": config.wan_steps,
                    "num_frames": len(frames),
                }
                sched_meta = "FlowMatchEulerDiscreteScheduler"
            elif selected_engine == "ltx":
                pipeline_meta = {
                    "model_id": config.ltx_video.model_id,
                    "architecture": "LTX-Video 2B DiT (Lightricks)",
                    "steps": str(config.ltx_steps),
                    "guidance_scale": str(config.ltx_guidance_scale),
                }
                steps_meta = {
                    "ltx_steps": config.ltx_steps,
                    "num_frames": len(frames),
                }
                sched_meta = "LTXDPMScheduler"
            elif selected_engine == "studio":
                master_model_type = getattr(config, "studio_master_model", "epicrealism")
                master_model_name = config.studio_epic.model_id if master_model_type == "epicrealism" else config.studio_sdxl.model_id
                pipeline_meta = {
                    "model_id": f"{master_model_name} + {config.studio_svd.model_id}",
                    "architecture": f"Studio Two-Stage Photorealism ({master_model_type.upper()} + SVD-XT)",
                    "master_model": master_model_name,
                    "svd_steps": str(config.studio_svd_steps),
                    "upscaler": "Real-ESRGAN Compact SRVGGNet",
                }
                steps_meta = {
                    "master_steps": config.studio_epic_steps if master_model_type == "epicrealism" else config.studio_sdxl_steps,
                    "svd_steps": config.studio_svd_steps,
                    "num_frames": len(frames),
                }
                sched_meta = "EulerDiscreteScheduler + SVD-XT"
            elif selected_engine == "cogvideox":
                pipeline_meta = {
                    "model_id": config.cogvideox.model_id,
                    "architecture": "CogVideoX-2B (3D Diffusion Transformer)",
                    "steps": str(config.cogvideox_steps),
                    "guidance_scale": str(config.cogvideox_guidance_scale),
                }
                steps_meta = {
                    "cogvideox_steps": config.cogvideox_steps,
                    "num_frames": len(frames),
                }
                sched_meta = "CogVideoXDPMScheduler (trailing timestep)"
            else:
                pipeline_meta = {
                    "model_id": config.dreamer.model_id,
                    "motion_adapter": config.animator.model_id,
                    "steps": str(config.animatediff_steps),
                    "architecture": "AnimateDiff-Lightning",
                }
                steps_meta = {
                    "animatediff_steps": config.animatediff_steps,
                    "num_frames": num_frames,
                    "num_segments": num_segments,
                }
                sched_meta = "LCMScheduler (trailing timestep)"

            metadata = GenerationMetadata(
                job_id=job.job_id,
                prompt=job.request.prompt,
                seed=seed,
                models={"pipeline": pipeline_meta},
                resolution={"width": final_res["width"], "height": final_res["height"]},
                steps=steps_meta,
                motion_bucket_id=0,
                fps=job.request.fps,
                interpolated=interpolation_applied,
                interpolated_fps=config.interpolation_target_fps if interpolation_applied else None,
                audio_mode=job.request.audio_mode,
                scheduler=sched_meta,
                dtype=str(active_dtype),
                device=self.device,
                gpu_device_name=vram_stats.get("device_name"),
                vram_peak_mb=vram_stats.get("peak_vram_mb", 0.0),
                generation_duration_seconds=duration,
            )
            metadata.save(metadata_path)

            job.video_path = str(video_path)
            job.video_url = f"/videos/{video_filename}"
            job.metadata_path = str(metadata_path)
            job.metadata = metadata.model_dump()

        except InterruptedError:
            logger.info(f"Job {job.job_id} cancelled during execution.")
            self.vram.safe_cleanup()
            raise
        except Exception as exc:
            logger.error(f"Error during job execution {job.job_id}: {exc}", exc_info=True)
            self.vram.safe_cleanup()
            raise exc

    def _build_studio_prompts(self, request: SafeGenerationRequest):
        """
        Builds domain-aware cinematic and photorealistic prompts for Master Frame synthesis.
        Intelligently identifies whether the subject is wildlife/animal, landscape/nature,
        portrait/character, or urban/cinematic, preventing prompt pollution (e.g. adding feathers
        or eyes to mountain landscapes).
        """
        import re
        raw = request.prompt.strip()
        lower_raw = raw.lower()

        # Build natural subject phrase without awkward double articles
        if re.match(r'^(a|an|the)\s+', raw, flags=re.IGNORECASE):
            subject_phrase = raw
            clean_p = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
        else:
            subject_phrase = f"a {raw}"
            clean_p = raw

        photo_keywords = ["photograph", "shot on", "lens", "f/", "8k", "dslr", "35mm", "portra"]
        has_photo_specs = any(k in lower_raw for k in photo_keywords)

        if has_photo_specs:
            enhanced_prompt = f"Award-winning documentary photography, 35mm film, natural optical depth, soft lighting: {clean_p}"
        else:
            # Check domain
            wildlife_words = ["eagle", "bird", "hawk", "falcon", "lion", "tiger", "wolf", "bear", "fox", "deer", "animal", "creature", "cat", "dog", "horse"]
            landscape_words = ["mountain", "valley", "forest", "lake", "ocean", "sea", "river", "desert", "sky", "clouds", "sunset", "sunrise", "hills", "canyon", "waterfall", "nature", "landscape", "tree", "pine"]
            portrait_words = ["woman", "man", "person", "girl", "boy", "face", "portrait", "warrior", "samurai", "model"]

            if any(w in lower_raw for w in wildlife_words):
                enhanced_prompt = (
                    f"A stunning National Geographic wildlife documentary photograph of {subject_phrase}, "
                    f"centered in frame, anatomically accurate, pristine fur or plumage texture, "
                    f"natural eye catchlight, scenic environmental backdrop, natural sunlight, 85mm f/1.4 lens"
                )
            elif any(w in lower_raw for w in landscape_words):
                enhanced_prompt = (
                    f"A breathtaking landscape photograph of {clean_p}, "
                    f"vast scale, atmospheric depth, soft natural morning haze, gentle golden lighting, "
                    f"crisp geological and foliage detail, shot on 35mm lens, Kodak Portra 400, natural color dynamic range"
                )
            elif any(w in lower_raw for w in portrait_words):
                enhanced_prompt = (
                    f"Authentic cinematic portrait photograph of {subject_phrase}, "
                    f"natural skin texture with visible pores, soft catchlights, gentle depth of field, "
                    f"warm natural rim light, 85mm f/1.4 lens, master cinematography"
                )
            else:
                enhanced_prompt = (
                    f"Professional cinematic documentary photograph of {subject_phrase}, "
                    f"centered composition, natural lighting, sharp optical focus, soft background roll-off, 35mm photography"
                )

        default_neg = (
            "multiple heads, two heads, twin, double, duplicate head, extra beaks, two beaks, extra eyes, mutated, "
            "split face, four eyes, three eyes, cloned face, extra faces, extra wings, extra limbs, extra legs, deformed, "
            "cartoon, anime, illustration, painting, drawing, 3d render, sketch, cgi, plastic, synthetic skin, "
            "hdr, oversharpened, high contrast, oversaturated, crushed shadows, blown highlights, artificial edge halos, "
            "bad anatomy, bad proportions, blurry, pixelated, grainy, noise, compression artifacts, watermark, signature, text"
        )
        neg_prompt = request.negative_prompt if request.negative_prompt else default_neg
        return enhanced_prompt, neg_prompt

    def _build_cinematic_prompts(self, request: SafeGenerationRequest):
        """
        Builds quality-enhanced prompts for CogVideoX 3D DiT photorealism.
        T5-XXL understands natural descriptive cinematic language.
        """
        clean_p = request.prompt.strip()
        enhanced_prompt = (
            f"Cinematic film shot, 4k photorealistic, rich detailed textures, natural lighting, "
            f"masterpiece, sharp focus, natural movement: {clean_p}"
        )
        default_neg = (
            "cartoon, anime, illustration, 3d render, plastic, synthetic, smooth, airbrushed, "
            "oversaturated, deformed, bad anatomy, duplicate, extra heads, extra limbs, "
            "blurry, grain, noise, watermark, low quality, artifacts"
        )
        neg_prompt = request.negative_prompt if request.negative_prompt else default_neg
        return enhanced_prompt, neg_prompt

    def _build_anime_prompts(self, request: SafeGenerationRequest):
        """
        Builds quality-enhanced prompts for AnimateDiff anime generation.
        Automatically injects anime quality boosters and strong negative prompts.
        """
        clean_p = request.prompt.strip()

        is_anime = any(k in clean_p.lower() for k in ["anime", "manga", "cartoon", "ghibli", "comic"])

        if is_anime:
            anime_boosters = ["masterpiece", "best quality", "anime", "highly detailed"]
            has_quality = any(k in clean_p.lower() for k in anime_boosters)
            if not has_quality:
                enhanced_prompt = f"masterpiece, best quality, highly detailed anime, solo, {clean_p}"
            else:
                if "group" not in clean_p.lower() and "multiple" not in clean_p.lower() and "solo" not in clean_p.lower():
                    enhanced_prompt = f"solo, {clean_p}"
                else:
                    enhanced_prompt = clean_p

            default_neg = (
                "worst quality, low quality, bad anatomy, bad proportions, "
                "extra heads, two heads, multiple heads, duplicate heads, cloned head, "
                "extra wings, extra limbs, extra arms, extra legs, deformed, "
                "blurry, out of focus, watermark, signature, text, ugly, duplicate, "
                "morbid, mutilated, mutation, poorly drawn face, cloned face, "
                "gross proportions, poorly drawn hands, missing fingers, extra fingers, "
                "lowres, jpeg artifacts, compression artifacts"
            )
        else:
            # Photorealistic wildlife & cinematic nature default
            real_boosters = ["photorealistic", "raw photo", "8k", "masterpiece"]
            has_quality = any(k in clean_p.lower() for k in real_boosters)
            if not has_quality:
                enhanced_prompt = f"masterpiece, best quality, photorealistic, 8k uhd, highly detailed, sharp focus, natural lighting, solo, {clean_p}"
            else:
                if "group" not in clean_p.lower() and "multiple" not in clean_p.lower() and "solo" not in clean_p.lower():
                    enhanced_prompt = f"solo, {clean_p}"
                else:
                    enhanced_prompt = clean_p

            default_neg = (
                "worst quality, low quality, cartoon, anime, illustration, painting, drawing, 3d render, sketch, "
                "bad anatomy, bad proportions, extra heads, two heads, multiple heads, duplicate heads, cloned head, "
                "extra wings, extra limbs, extra legs, deformed, blurry, grain, noisy, artifacts, out of focus, "
                "watermark, signature, text, ugly, duplicate, morbid, mutilated, mutation, lowres, jpeg artifacts"
            )

        neg_prompt = getattr(request, "negative_prompt", None) or default_neg
        return enhanced_prompt, neg_prompt

    # =========================================================================
    # DIRECTOR PIPELINE: Multi-scene storyboard
    # =========================================================================
    def _execute_director_pipeline(
        self,
        job: Job,
        seed: int,
        generator,
        final_video_path: Path,
        metadata_path: Path,
        start_time: float,
        quality: Literal["draft", "hd", "fhd", "4k"] = "fhd",
        duration_mode: Literal["auto", "short", "medium", "long", "cinematic"] = "auto",
        target_duration: Optional[int] = None,
    ):
        """
        Director pipeline: Plans multi-scene storyboard, generates each shot
        with AnimateDiff-Lightning, concatenates, upscales to target quality.
        
        Expected throughput on RTX 4060:
          ~15-30s/shot × N shots = practical long-form video production
        """
        job.update_stage(GenerationStage.LOADING_MODEL, 0.05, "AI Director: Planning storyboard...")
        plan: StoryboardPlan = plan_storyboard(
            prompt=job.request.prompt,
            duration_mode=duration_mode,
            target_duration=target_duration,
            quality=quality,
        )
        logger.info(
            f"Director: {plan.total_shots} shots planned, "
            f"{plan.target_duration_seconds:.1f}s total, quality={quality}"
        )

        num_frames_animatediff = getattr(job.request, "num_frames", config.animatediff_num_frames)
        shot_paths: List[Path] = []

        # Determine which engine to use (same logic as single-shot)
        req_engine = getattr(job.request, "engine", "auto")
        prompt_lower = job.request.prompt.lower()
        anime_kw = ["anime", "manga", "cartoon", "illustration", "ghibli", "drawn", "sketch"]
        if req_engine == "animatediff" or (req_engine == "auto" and any(k in prompt_lower for k in anime_kw)):
            dir_engine = "animatediff"
            dir_model_id = config.dreamer.id
            dir_dtype = self.dtype
            dir_width, dir_height = config.animatediff_width, config.animatediff_height
            dir_num_frames = num_frames_animatediff
        else:
            dir_engine = "cogvideox"
            dir_model_id = config.cogvideox.id
            dir_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            dir_width, dir_height = config.cogvideox_width, config.cogvideox_height
            dir_num_frames = config.cogvideox_num_frames

        dir_label = "CogVideoX-2B DiT" if dir_engine == "cogvideox" else "AnimateDiff-Lightning"
        job.update_stage(GenerationStage.LOADING_MODEL, 0.08, f"Loading {dir_label} for Director...")
        self.vram.prepare_for_model(dir_model_id, auto_unload=True)
        pipe = self.registry.load(
            dir_model_id, device=self.device, dtype=dir_dtype, cpu_offload=self.cpu_offload
        )
        self.vram.set_model_loaded(dir_model_id, pipe)

        for shot_i, scene in enumerate(plan.scenes):
            self._check_cancellation(job)
            overall_pct = 0.10 + 0.72 * (shot_i / plan.total_shots)

            job.update_stage(
                GenerationStage.ANIMATING,
                overall_pct,
                f"Director Shot {shot_i + 1}/{plan.total_shots}: '{scene.prompt[:40]}...'"
            )

            with self.vram.managed_inference(dir_model_id):
                clean_sc = scene.prompt.strip()

                if dir_engine == "cogvideox":
                    enhanced_sc = (
                        f"Cinematic film shot, 4k photorealistic, rich detailed textures, "
                        f"natural lighting, masterpiece, sharp focus, natural movement: {clean_sc}"
                    )
                    neg_prompt = (
                        getattr(job.request, "negative_prompt", None)
                        or "cartoon, anime, illustration, 3d render, plastic, synthetic, "
                           "deformed, bad anatomy, blurry, grain, watermark, low quality"
                    )
                    dir_steps = config.cogvideox_steps
                    dir_guidance = config.cogvideox_guidance_scale
                    dir_num_frames = config.cogvideox_num_frames
                else:
                    if not any(k in clean_sc.lower() for k in ["masterpiece", "best quality", "anime"]):
                        enhanced_sc = f"masterpiece, best quality, highly detailed anime, {clean_sc}"
                    else:
                        enhanced_sc = clean_sc
                    neg_prompt = (
                        getattr(job.request, "negative_prompt", None)
                        or "worst quality, low quality, bad anatomy, deformed, blurry, "
                           "watermark, signature, text, lowres, jpeg artifacts"
                    )
                    dir_steps = config.animatediff_steps
                    dir_guidance = config.animatediff_guidance_scale
                    dir_num_frames = num_frames_animatediff

                result = pipe(
                    prompt=enhanced_sc,
                    negative_prompt=neg_prompt,
                    num_frames=dir_num_frames,
                    num_inference_steps=dir_steps,
                    guidance_scale=dir_guidance,
                    generator=generator,
                    width=dir_width,
                    height=dir_height,
                )
                shot_frames = result.frames[0]

            # Export shot to temp
            from diffusers.utils import export_to_video
            shot_path = config.temp_dir / f"{job.job_id}_shot_{shot_i:02d}.mp4"
            export_to_video(shot_frames, str(shot_path), fps=job.request.fps)
            shot_paths.append(shot_path)
            logger.info(f"Director: Shot {shot_i + 1}/{plan.total_shots} done → {shot_path.name}")

        # Unload pipeline
        self.vram.unload_current_model()

        # Concatenate all shots
        job.update_stage(GenerationStage.EXPORTING, 0.83, f"Concatenating {plan.total_shots} shots...")
        concat_path = config.temp_dir / f"{job.job_id}_concat.mp4"
        concatenate_video_shots(shot_paths, concat_path)
        for sp in shot_paths:
            sp.unlink(missing_ok=True)

        # Frame interpolation
        interpolation_applied = False
        if job.request.interpolate and config.enable_interpolation and is_ffmpeg_available():
            job.update_stage(GenerationStage.EXPORTING, 0.87,
                f"Interpolating to {config.interpolation_target_fps}fps...")
            interp_path = concat_path.with_stem(concat_path.stem + "_interp")
            try:
                interpolate_video(concat_path, interp_path, target_fps=config.interpolation_target_fps)
                concat_path.unlink(missing_ok=True)
                concat_path = interp_path
                interpolation_applied = True
            except Exception as e:
                logger.warning(f"Interpolation failed (non-fatal): {e}")

        # AI Upscaling
        if quality != "draft" and upscaler_ffmpeg_available():
            safe_quality = cast(Literal["draft", "hd", "fhd", "4k"], quality)
            job.update_stage(GenerationStage.EXPORTING, 0.91,
                f"Upscaling to {safe_quality.upper()}...")
            upscaled_path = concat_path.with_stem(concat_path.stem + f"_{safe_quality}")
            try:
                upscale_video_to_quality(concat_path, upscaled_path, quality=safe_quality)
                concat_path.unlink(missing_ok=True)
                concat_path = upscaled_path
            except Exception as e:
                logger.warning(f"Upscaling failed (non-fatal): {e}")

        # Audio
        audio_path = None
        total_duration = plan.target_duration_seconds
        if job.request.audio_mode != "none":
            job.update_stage(GenerationStage.EXPORTING, 0.95,
                f"Generating {job.request.audio_mode} audio ({total_duration:.1f}s)...")
            audio_path = generate_audio_for_job(
                prompt=job.request.prompt,
                audio_mode=job.request.audio_mode,
                duration_seconds=total_duration,
                temp_dir=config.temp_dir,
                job_id=job.job_id,
            )
            if audio_path and upscaler_ffmpeg_available():
                muxed_path = concat_path.with_stem(concat_path.stem + "_av")
                try:
                    mux_audio_to_video(concat_path, audio_path, muxed_path, video_duration=total_duration)
                    concat_path.unlink(missing_ok=True)
                    concat_path = muxed_path
                except Exception as e:
                    logger.warning(f"Audio mux failed (non-fatal): {e}")
                finally:
                    if audio_path and audio_path.exists():
                        audio_path.unlink(missing_ok=True)

        shutil.move(str(concat_path), str(final_video_path))

        # Save metadata
        duration = round(time.time() - start_time, 2)
        vram_stats = self.vram.get_memory_stats()
        safe_quality_key = quality if quality in RESOLUTION_MAP else "fhd"
        res = RESOLUTION_MAP.get(safe_quality_key, RESOLUTION_MAP["fhd"])
        pipeline_meta = {
            "model_id": config.cogvideox.model_id if dir_engine == "cogvideox" else config.dreamer.model_id,
            "engine": dir_engine,
            "architecture": "CogVideoX-2B DiT" if dir_engine == "cogvideox" else "AnimateDiff-Lightning",
        }
        if dir_engine == "animatediff":
            pipeline_meta["motion_adapter"] = config.animator.model_id
            pipeline_meta["steps"] = str(config.animatediff_steps)
        else:
            pipeline_meta["steps"] = str(config.cogvideox_steps)

        metadata = GenerationMetadata(
            job_id=job.job_id,
            prompt=job.request.prompt,
            seed=seed,
            models={"pipeline": pipeline_meta},
            resolution={"width": res["width"], "height": res["height"]},
            steps={
                "steps": config.cogvideox_steps if dir_engine == "cogvideox" else config.animatediff_steps,
                "num_frames": dir_num_frames,
                "total_shots": plan.total_shots,
            },
            motion_bucket_id=0,
            fps=job.request.fps,
            interpolated=interpolation_applied,
            interpolated_fps=config.interpolation_target_fps if interpolation_applied else None,
            audio_mode=job.request.audio_mode,
            scheduler="CogVideoXDPMScheduler" if dir_engine == "cogvideox" else "LCMScheduler (trailing timestep)",
            dtype=str(self.dtype),
            device=self.device,
            gpu_device_name=vram_stats.get("device_name"),
            vram_peak_mb=vram_stats.get("peak_vram_mb", 0.0),
            generation_duration_seconds=duration,
        )
        metadata.save(metadata_path)

        video_filename = final_video_path.name
        job.video_path = str(final_video_path)
        job.video_url = f"/videos/{video_filename}"
        job.metadata_path = str(metadata_path)
        job.metadata = metadata.model_dump()

    # =========================================================================
    # MOCK PIPELINE (CI / GPU-less testing)
    # =========================================================================
    def _execute_mock_pipeline(
        self,
        job: Job,
        seed: int,
        video_path: Path,
        metadata_path: Path,
        start_time: float,
    ):
        """Mock pipeline for CI testing — generates synthetic gradient video."""
        import cv2

        stages = [
            (GenerationStage.LOADING_MODEL, 0.15, "Mock: Loading AnimateDiff-Lightning..."),
            (GenerationStage.GENERATING_IMAGE, 0.50, "Mock: Generating anime frames (4 steps)..."),
            (GenerationStage.EXPORTING, 0.90, "Mock: Encoding video..."),
        ]
        for stage, progress, msg in stages:
            self._check_cancellation(job)
            job.update_stage(stage, progress, msg)
            time.sleep(0.05)

        width, height = 512, 512
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, float(job.request.fps), (width, height))

        rng = np.random.default_rng(seed)
        base_color = rng.integers(30, 200, size=3, dtype=np.int32)

        for i in range(16):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            current_color = ((base_color + i * 4) % 255).astype(np.uint8)
            frame[:] = current_color
            cv2.circle(frame, (width // 2, height // 2), 60 + i * 3, (255, 255, 255), -1)
            cv2.putText(
                frame,
                f"Velaris v2.0 AnimateDiff | Seed {seed} | Frame {i+1}/16",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            out.write(frame)
        out.release()

        duration = round(time.time() - start_time, 2)
        vram_stats = self.vram.get_memory_stats()
        metadata = GenerationMetadata(
            job_id=job.job_id,
            prompt=job.request.prompt,
            seed=seed,
            models={"pipeline": {"model_id": config.dreamer.model_id, "architecture": "AnimateDiff-Lightning-Mock"}},
            resolution={"width": width, "height": height},
            steps={"animatediff_steps": config.animatediff_steps, "num_frames": 16},
            motion_bucket_id=0,
            fps=job.request.fps,
            scheduler="MockScheduler",
            dtype=str(self.dtype),
            device=self.device,
            gpu_device_name="MockDevice",
            vram_peak_mb=128.0,
            generation_duration_seconds=duration,
        )
        metadata.save(metadata_path)

        job.video_path = str(video_path)
        job.video_url = f"/videos/{video_path.name}"
        job.metadata_path = str(metadata_path)
        job.metadata = metadata.model_dump()


# Global default engine instance
engine = HybridEngine()
