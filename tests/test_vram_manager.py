"""
Unit tests for VRAM Manager state machine, exclusive model residency, and exception recovery.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from vram_manager import VRAMManager, VRAMState


def test_vram_manager_initial_state():
    mgr = VRAMManager(device="cpu")
    assert mgr.state == VRAMState.UNLOADED
    assert mgr.current_model is None
    stats = mgr.get_memory_stats()
    assert stats["state"] == "unloaded"
    assert stats["current_model"] is None


def test_vram_manager_lifecycle():
    mgr = VRAMManager(device="cpu")
    mock_pipeline = object()

    # Step 1: Prepare model
    mgr.prepare_for_model("dreamer")
    assert mgr.state == VRAMState.LOADING
    assert mgr.current_model == "dreamer"

    # Step 2: Confirm loaded
    mgr.set_model_loaded("dreamer", mock_pipeline)
    assert mgr.state == VRAMState.LOADED
    assert mgr.current_model == "dreamer"

    # Step 3: Inference context
    with mgr.managed_inference("dreamer"):
        assert mgr.state == VRAMState.INFERRING

    assert mgr.state == VRAMState.LOADED

    # Step 4: Unload
    mgr.unload_current_model()
    assert mgr.state == VRAMState.UNLOADED
    assert getattr(mgr, "current_model") is None


def test_vram_manager_rejects_dual_loading_without_auto_unload():
    mgr = VRAMManager(device="cpu")
    mgr.prepare_for_model("dreamer")
    mgr.set_model_loaded("dreamer", object())

    # Attempting to load another model without auto_unload must raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot load model 'animator'"):
        mgr.prepare_for_model("animator", auto_unload=False)


def test_vram_manager_auto_unloads_previous_model():
    mgr = VRAMManager(device="cpu")
    mgr.prepare_for_model("dreamer")
    mgr.set_model_loaded("dreamer", object())

    # With auto_unload=True (default), previous model is safely unloaded
    mgr.prepare_for_model("animator", auto_unload=True)
    assert mgr.state == VRAMState.LOADING
    assert mgr.current_model == "animator"


def test_vram_manager_exception_recovery_in_managed_inference():
    mgr = VRAMManager(device="cpu")
    mgr.prepare_for_model("dreamer")
    mgr.set_model_loaded("dreamer", object())

    with pytest.raises(ZeroDivisionError):
        with mgr.managed_inference("dreamer"):
            raise ZeroDivisionError("Simulated inference failure")

    # Exception recovery should have cleaned up references and reset state to UNLOADED
    assert mgr.state == VRAMState.UNLOADED
    assert getattr(mgr, "current_model") is None
