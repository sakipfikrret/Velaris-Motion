"""
Unit tests for serial JobQueue, state machine, and cancellation.
"""

import time
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from security import SafeGenerationRequest
from job_queue import JobQueue, Job, JobStatus, GenerationStage, QueueFullError


def test_job_queue_serial_execution():
    jq = JobQueue(max_size=5)
    executed_order = []

    def mock_executor(job: Job):
        executed_order.append(job.request.prompt)
        time.sleep(0.05)

    jq.set_executor(mock_executor)
    jq.start()

    req1 = SafeGenerationRequest(prompt="First Prompt")
    req2 = SafeGenerationRequest(prompt="Second Prompt")

    job1 = jq.enqueue(req1)
    job2 = jq.enqueue(req2)

    assert job1.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    assert job2.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    # Wait for completion
    timeout = 5.0
    start = time.time()
    while time.time() - start < timeout:
        if job1.status == JobStatus.COMPLETED and job2.status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)

    jq.stop()

    assert job1.status == JobStatus.COMPLETED
    assert job2.status == JobStatus.COMPLETED
    # FIFO order guaranteed
    assert executed_order == ["First Prompt", "Second Prompt"]


def test_job_cancellation_while_queued():
    jq = JobQueue(max_size=5)
    started = []

    def slow_executor(job: Job):
        started.append(job.job_id)
        time.sleep(0.3)

    jq.set_executor(slow_executor)
    jq.start()

    req1 = SafeGenerationRequest(prompt="Active Job")
    req2 = SafeGenerationRequest(prompt="Job to Cancel")

    job1 = jq.enqueue(req1)
    job2 = jq.enqueue(req2)

    # Cancel job2 immediately
    cancelled = jq.cancel_job(job2.job_id)
    assert cancelled is True
    assert job2.status == JobStatus.CANCELLED
    assert job2.stage == GenerationStage.CANCELLED

    # Wait for job1 to complete
    time.sleep(0.5)
    jq.stop()

    assert job1.status == JobStatus.COMPLETED
    assert job2.status == JobStatus.CANCELLED
    # job2 was never picked up by executor
    assert job2.job_id not in started


def test_queue_full_raises_error():
    jq = JobQueue(max_size=2)
    # Don't start worker to leave queue full
    req = SafeGenerationRequest(prompt="Fill queue")
    j1 = jq.enqueue(req)
    j2 = jq.enqueue(req)

    with pytest.raises(QueueFullError):
        jq.enqueue(req)
