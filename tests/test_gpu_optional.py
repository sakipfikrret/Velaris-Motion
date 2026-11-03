"""
Optional Real-GPU test suite.
Only executes when explicitly requested via RUN_GPU_TESTS=1 environment variable
or if pytest is executed with the custom marker.
"""

import os
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from vram_manager import vram_manager, VRAMState
from preflight import run_preflight

gpu_available = torch.cuda.is_available()
run_gpu_env = os.environ.get("RUN_GPU_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not (gpu_available and run_gpu_env),
    reason="Optional real-GPU suite: Set RUN_GPU_TESTS=1 with a working CUDA GPU to run."
)


def test_real_gpu_allocation_and_cleanup():
    """Validates real CUDA allocation and rigorous VRAM reclamation."""
    assert torch.cuda.is_available()
    vram_manager.safe_cleanup()

    initial_stats = vram_manager.get_memory_stats()
    assert initial_stats["cuda_available"] is True
    initial_alloc = initial_stats["allocated_vram_mb"]

    # Allocate a dummy 100MB tensor on CUDA
    tensor = torch.empty((100, 1024, 1024), dtype=torch.uint8, device="cuda")
    vram_manager._update_peak()
    
    mid_stats = vram_manager.get_memory_stats()
    assert mid_stats["allocated_vram_mb"] >= initial_alloc + 95.0

    # Clean up tensor
    del tensor
    vram_manager.safe_cleanup()

    final_stats = vram_manager.get_memory_stats()
    # Should be back to negligible allocations
    assert final_stats["allocated_vram_mb"] <= initial_alloc + 5.0
