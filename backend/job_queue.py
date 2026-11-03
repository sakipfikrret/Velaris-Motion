"""
Velaris Motion - Serial Job Queue, Concurrency Control & State Machine
Target Release: v1.1.0
"""

import time
import uuid
import queue
import threading
from enum import Enum
from typing import Dict, Optional, Any, List, Callable
from pydantic import BaseModel, Field
from security import SafeGenerationRequest
from config import config


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GenerationStage(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    GENERATING_IMAGE = "generating_image"
    LOADING_VIDEO_MODEL = "loading_video_model"
    ANIMATING = "animating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobSnapshot(BaseModel):
    """Serializable snapshot of a job's current status and metrics."""
    job_id: str
    status: JobStatus
    stage: GenerationStage
    prompt: str
    progress: float
    message: str
    video_url: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_img2vid: bool = False


class Job:
    """Represents a discrete generation request managed by the serial runtime."""
    def __init__(
        self,
        request: SafeGenerationRequest,
        job_id: Optional[str] = None,
        initial_image_path: Optional[str] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.request = request
        self.initial_image_path = initial_image_path
        self.status = JobStatus.QUEUED
        self.stage = GenerationStage.QUEUED
        self.progress = 0.0
        self.message = "Queued in runtime queue"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.video_path: Optional[str] = None
        self.video_url: Optional[str] = None
        self.metadata_path: Optional[str] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self._cancel_event = threading.Event()
        self._lock = threading.RLock()

    def request_cancellation(self) -> bool:
        """Signals cancellation request."""
        with self._lock:
            if self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            if self.status == JobStatus.QUEUED:
                self.status = JobStatus.CANCELLED
                self.stage = GenerationStage.CANCELLED
                self.message = "Cancelled while queued"
                self.completed_at = time.time()
                self._cancel_event.set()
                return True
            if self.status == JobStatus.RUNNING:
                self.status = JobStatus.CANCELLING
                self.message = "Cancellation requested..."
                self._cancel_event.set()
                return True
            return False

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def update_stage(self, stage: GenerationStage, progress: float, message: str):
        with self._lock:
            if self.is_cancelled and stage not in (GenerationStage.CANCELLED, GenerationStage.FAILED):
                return
            self.stage = stage
            self.progress = round(progress, 2)
            self.message = message

    def to_snapshot(self) -> JobSnapshot:
        with self._lock:
            duration = None
            if self.started_at:
                end_time = self.completed_at or time.time()
                duration = round(end_time - self.started_at, 2)

            return JobSnapshot(
                job_id=self.job_id,
                status=self.status,
                stage=self.stage,
                prompt=self.request.prompt,
                progress=self.progress,
                message=self.message,
                video_url=self.video_url,
                created_at=self.created_at,
                started_at=self.started_at,
                completed_at=self.completed_at,
                duration_seconds=duration,
                error=self.error,
                metadata=self.metadata,
                is_img2vid=self.initial_image_path is not None,
            )


class QueueFullError(Exception):
    """Raised when the job queue exceeds maximum capacity."""
    pass


class JobQueue:
    """
    Serial job queue dedicated to a single GPU runtime.
    Guarantees request 2 cannot disrupt GPU state while request 1 is executing.
    """
    def __init__(self, max_size: int = config.max_queue_size):
        self.max_size = max_size
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._executor_callback: Optional[Callable[[Job], None]] = None

    def set_executor(self, executor: Callable[[Job], None]):
        """Sets the backend execution worker callback."""
        self._executor_callback = executor

    def start(self):
        """Starts the background worker thread."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="Velaris-Serial-GPU-Worker",
                daemon=True
            )
            self._worker_thread.start()

    def stop(self):
        """Stops the worker thread."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def enqueue(
        self,
        request: SafeGenerationRequest,
        job_id: Optional[str] = None,
        initial_image_path: Optional[str] = None,
    ) -> Job:
        """Enqueues a new generation job (Text-to-Video or direct Image-to-Video)."""
        with self._lock:
            # Check pending jobs in queue
            active_count = sum(
                1 for j in self._jobs.values() 
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING)
            )
            if active_count >= self.max_size:
                raise QueueFullError(f"Job queue is full ({active_count}/{self.max_size}). Please wait for jobs to finish.")

            job = Job(request, job_id=job_id, initial_image_path=initial_image_path)
            self._jobs[job.job_id] = job
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                raise QueueFullError(f"Job queue buffer full.")

            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> List[JobSnapshot]:
        with self._lock:
            sorted_jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )
            return [j.to_snapshot() for j in sorted_jobs[:limit]]

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        return job.request_cancellation()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job: Job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with job._lock:
                if job.is_cancelled or job.status == JobStatus.CANCELLED:
                    job.status = JobStatus.CANCELLED
                    job.stage = GenerationStage.CANCELLED
                    self._queue.task_done()
                    continue

                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                job.update_stage(GenerationStage.QUEUED, 0.05, "Job picked up by GPU worker")

            try:
                if self._executor_callback:
                    self._executor_callback(job)
                else:
                    raise RuntimeError("No executor registered for job queue.")

                with job._lock:
                    if job.status == JobStatus.CANCELLING or job.is_cancelled:
                        job.status = JobStatus.CANCELLED
                        job.stage = GenerationStage.CANCELLED
                        job.message = "Generation cancelled successfully"
                    elif job.status == JobStatus.RUNNING:
                        job.status = JobStatus.COMPLETED
                        job.stage = GenerationStage.COMPLETED
                        job.progress = 1.0
                        job.message = "Generation completed successfully"
            except Exception as e:
                with job._lock:
                    job.status = JobStatus.FAILED
                    job.stage = GenerationStage.FAILED
                    job.error = str(e)
                    job.message = f"Failed: {str(e)}"
            finally:
                with job._lock:
                    job.completed_at = time.time()
                self._queue.task_done()


# Global singleton instance
job_queue = JobQueue()
