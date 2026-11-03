"""
Unit tests for preflight diagnostic checks.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from preflight import (
    check_python_version,
    check_pytorch_cuda,
    check_vram,
    check_disk_space,
    check_model_registry,
    run_preflight,
)


def test_check_python_version():
    item = check_python_version()
    assert item.status in ("PASS", "WARN")
    assert item.value != ""


def test_check_model_registry():
    item = check_model_registry()
    assert item.status == "PASS"
    assert "Dreamer" in item.value
    assert "Animator" in item.value


def test_run_preflight_report():
    report = run_preflight()
    assert report.overall_status in ("PASS", "WARN", "FAIL")
    assert len(report.checks) >= 5
