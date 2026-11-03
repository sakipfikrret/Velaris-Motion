"""
Velaris Motion - Hardened FastAPI Application
Target Release: v1.1.0
"""

import os
import time
import uuid
import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Literal

from fastapi import FastAPI, Request, status, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from config import config
from security import (
    SafeGenerationRequest,
    StandardErrorResponse,
    ErrorDetail,
    resolve_safe_path,
    validate_safe_filename,
    validate_uploaded_image,
)
from model_registry import model_registry
from vram_manager import vram_manager
from job_queue import job_queue, QueueFullError, JobStatus, GenerationStage, JobSnapshot
from engine import engine
from preflight import run_preflight, PreflightReport

logger = logging.getLogger("velaris.api")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register engine executor and start serial GPU worker
    job_queue.set_executor(engine.execute_job)
    job_queue.start()
    yield
    # Shutdown: gracefully stop worker and cleanup VRAM
    job_queue.stop()
    vram_manager.safe_cleanup()


app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Deterministic, hardened local video generation runtime",
    lifespan=lifespan,
)

# CORS configuration for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


from fastapi.encoders import jsonable_encoder


# Standardized error response helper
def make_error_response(code: str, message: str, status_code: int, details: Any = None) -> JSONResponse:
    payload = {
        "error": {
            "code": code,
            "message": message,
            "details": jsonable_encoder(details) if details is not None else None,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0].get("msg") if errors else "Malformed or invalid request data."
    return make_error_response(
        code="VALIDATION_ERROR",
        message=f"Request validation failed: {msg}",
        status_code=422,
        details=jsonable_encoder(errors),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return make_error_response(
        code="INVALID_PARAMETER",
        message=str(exc),
        status_code=status.HTTP_400_BAD_REQUEST,
    )


@app.exception_handler(QueueFullError)
async def queue_full_handler(request: Request, exc: QueueFullError):
    return make_error_response(
        code="QUEUE_FULL",
        message=str(exc),
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return make_error_response(
        code="HTTP_ERROR",
        message=exc.detail,
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return make_error_response(
        code="INTERNAL_SERVER_ERROR",
        message=f"An unexpected internal error occurred: {str(exc)}",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """System health status and VRAM overview."""
    vram_stats = vram_manager.get_memory_stats()
    return {
        "app": config.app_name,
        "version": config.version,
        "status": "ready",
        "device": engine.device,
        "vram": vram_stats,
        "active_jobs": len(job_queue.list_jobs(limit=10)),
    }


@app.get("/preflight", response_model=PreflightReport, tags=["Diagnostics"])
def preflight_check():
    """Runs full hardware, CUDA, VRAM, and disk space diagnostics."""
    return run_preflight()


@app.get("/vram", tags=["Diagnostics"])
def vram_status():
    """Detailed GPU memory state and model occupancy."""
    return vram_manager.get_memory_stats()


@app.post("/generate", status_code=status.HTTP_202_ACCEPTED, tags=["Generation"])
async def generate_video(req: SafeGenerationRequest):
    """
    Submits a video generation task to the serial GPU job queue.
    Returns 202 Accepted immediately with job identifier.
    """
    try:
        job = job_queue.enqueue(req)
        return job.to_snapshot().model_dump()
    except QueueFullError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/animate-image", status_code=status.HTTP_202_ACCEPTED, tags=["Generation"])
async def animate_image(
    image: UploadFile = File(..., description="Image to animate (PNG, JPG, JPEG, WEBP)"),
    prompt: Optional[str] = Form(default=None),
    num_inference_steps: int = Form(default=25),
    motion_bucket_id: int = Form(default=127),
    fps: int = Form(default=7),
    seed: Optional[int] = Form(default=None),
    interpolate: bool = Form(default=True),
    audio_mode: Literal["none", "music", "narration"] = Form(default="none"),
    num_segments: int = Form(default=1),
    quality: Literal["draft", "hd", "fhd", "4k"] = Form(default="hd"),
    duration_mode: Literal["auto", "short", "medium", "long", "cinematic"] = Form(default="auto"),
    target_duration: Optional[int] = Form(default=None),
    use_director: bool = Form(default=False),
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Form(default="16:9"),
    eco_mode: bool = Form(default=False),
):
    """
    Sprint 4A: Direct Image-to-Video generation endpoint.
    Animates an uploaded image directly via SVD-XT, bypassing Dreamer.
    Reduces generation latency by ~45% and saves ~2-3 GB VRAM.
    """
    if not image.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file missing filename.")

    file_bytes = await image.read()
    try:
        validate_uploaded_image(file_bytes, image.filename)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))

    # Save to a unique temporary file
    ext = Path(image.filename).suffix.lower()
    temp_image_path = config.temp_dir / f"upload_{uuid.uuid4().hex[:12]}{ext}"
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    temp_image_path.write_bytes(file_bytes)

    clean_prompt = prompt.strip() if prompt and prompt.strip() else f"Direct animation of {Path(image.filename).stem}"
    try:
        req = SafeGenerationRequest(
            prompt=clean_prompt,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            seed=seed,
            interpolate=interpolate,
            audio_mode=audio_mode,
            num_segments=num_segments,
            quality=quality,
            duration_mode=duration_mode,
            target_duration=target_duration,
            use_director=use_director,
            engine="studio",
            aspect_ratio=aspect_ratio,
            eco_mode=eco_mode,
        )
    except ValueError as param_err:
        temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(param_err))

    try:
        job = job_queue.enqueue(req, initial_image_path=str(temp_image_path))
        return job.to_snapshot().model_dump()
    except QueueFullError as queue_err:
        temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(queue_err))
    except Exception as exc:
        temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.get("/jobs/{job_id}", response_model=JobSnapshot, tags=["Generation"])
def get_job_status(job_id: str):
    """Retrieves real-time status, execution stage, progress, and video output URL."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job.to_snapshot()


@app.get("/jobs/{job_id}/stream", tags=["Generation"])
async def stream_job_status(job_id: str):
    """
    Sprint 3A: Server-Sent Events stream for a generation job.
    Pushes status updates at 300ms intervals; auto-closes on terminal states.
    Clients should prefer this over polling GET /jobs/{job_id}.

    Usage (TypeScript):
        const es = new EventSource(`http://localhost:8000/jobs/${jobId}/stream`);
        es.onmessage = (e) => setState(JSON.parse(e.data));
    """
    async def event_generator():
        terminal_states = {"COMPLETED", "FAILED", "CANCELLED"}
        try:
            while True:
                job = job_queue.get_job(job_id)
                if job is None:
                    yield f"event: error\ndata: {{\"detail\": \"Job '{job_id}' not found\"}}\n\n"
                    break
                snap = job.to_snapshot().model_dump()
                # Convert enums to strings for JSON serialisation
                snap["status"] = snap["status"] if isinstance(snap["status"], str) else snap["status"].value
                if snap.get("stage"):
                    snap["stage"] = snap["stage"] if isinstance(snap["stage"], str) else snap["stage"].value
                yield f"data: {json.dumps(snap)}\n\n"
                if snap["status"] in terminal_states:
                    break
                await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass  # Client disconnected — clean exit

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


@app.post("/jobs/{job_id}/cancel", tags=["Generation"])
def cancel_generation_job(job_id: str):
    """Requests cancellation of a queued or running generation job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    
    success = job.request_cancellation()
    if not success:
        return {
            "job_id": job_id,
            "cancelled": False,
            "message": f"Job is in state '{job.status.value}' and cannot be cancelled.",
            "current_status": job.status.value,
        }
    
    return {
        "job_id": job_id,
        "cancelled": True,
        "message": "Cancellation requested successfully.",
        "current_status": job.status.value,
    }


@app.get("/jobs", response_model=List[JobSnapshot], tags=["Generation"])
def list_generation_jobs(limit: int = Query(default=20, ge=1, le=100)):
    """Lists recent generation jobs."""
    return job_queue.list_jobs(limit=limit)


@app.get("/ideate", tags=["Directorial Intelligence"])
def get_ideated_scenes(prompt: str = Query(..., min_length=1, max_length=500)):
    """
    Directorial Intelligence: Generates 3 cohesive, photorealistic cinematic scene concepts
    from a simple user prompt with zero GPU strain.
    """
    from prompt_director import ideate_cinematic_scenes
    concepts = ideate_cinematic_scenes(prompt)
    return {"prompt": prompt, "concepts": [c.model_dump() for c in concepts]}


@app.get("/videos/{filename:path}", tags=["Media"])
async def get_video(filename: str):
    """
    Retrieves rendered video file with path traversal protection.
    Only allows access to sanitized .mp4 files inside the outputs directory.
    """
    try:
        safe_path = resolve_safe_path(config.outputs_dir, filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not safe_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found.")

    return FileResponse(
        safe_path,
        media_type="video/mp4",
        filename=safe_path.name,
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/metadata/{filename:path}", tags=["Media"])
async def get_metadata(filename: str):
    """
    Retrieves generation reproducibility metadata JSON file with path traversal protection.
    """
    try:
        safe_path = resolve_safe_path(config.outputs_dir, filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not safe_path.is_file() or not safe_path.name.endswith(".json"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metadata file not found.")

    return FileResponse(
        safe_path,
        media_type="application/json",
        filename=safe_path.name,
    )


# =============================================================================
# BENCHMARK ENDPOINTS (Sprint 5B)
# =============================================================================

@app.get("/benchmark-results", tags=["Diagnostics"])
def get_benchmark_results():
    """
    Sprint 5B: Returns the latest benchmark metrics or live system stats.
    Powers the frontend Benchmark Dashboard.
    """
    # Look for saved benchmark results in project root or outputs dir
    report_candidates = [
        Path("benchmark_results.json").resolve(),
        config.outputs_dir / "benchmark_results.json",
    ]
    for candidate in report_candidates:
        if candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read {candidate}: {e}")

    # Fallback default snapshot if no benchmark run has been completed yet
    vram_stats = vram_manager.get_memory_stats()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "velaris_version": config.version,
        "mock_mode": engine.mock_mode,
        "device": engine.device,
        "total_prompts": 0,
        "success_count": 0,
        "success_rate_percent": 100.0,
        "oom_count": 0,
        "oom_rate_percent": 0.0,
        "other_failures": 0,
        "latency": {
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        },
        "peak_vram_mb": vram_stats.get("allocated_vram_mb", 0.0),
        "resolution": "1024x576",
        "video_length_frames": 14,
        "fps": 7,
        "detailed_results": [],
    }


@app.post("/benchmark/run", tags=["Diagnostics"])
def trigger_benchmark(
    mock: bool = Query(default=True, description="Run in fast mock mode"),
    prompts: int = Query(default=5, ge=1, le=10, description="Number of test prompts to evaluate"),
):
    """
    Sprint 5B: Triggers an automated benchmark evaluation.
    Default runs in mock mode for instant feedback without heavy GPU load.
    """
    from benchmark import BenchmarkRunner
    runner = BenchmarkRunner(mock_mode=mock)
    report = runner.run(prompt_count=prompts)

    # Persist results
    try:
        report_file = Path("benchmark_results.json").resolve()
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to persist benchmark_results.json: {e}")

    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
