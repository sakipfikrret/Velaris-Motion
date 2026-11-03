"""
Velaris Motion - Preflight Diagnostics & Hardware Compatibility Checker
Target Release: v1.1.0
"""

import sys
import os
import shutil
import platform
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import torch
from config import config
from model_registry import model_registry
from interpolation import is_ffmpeg_available, get_ffmpeg_version


class DiagnosticItem(BaseModel):
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    value: str
    message: str
    remediation: Optional[str] = None


class PreflightReport(BaseModel):
    overall_status: str  # "PASS", "WARN", "FAIL"
    platform: str
    checks: List[DiagnosticItem]
    can_generate: bool


def check_python_version() -> DiagnosticItem:
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor in (10, 11, 12):
        return DiagnosticItem(
            name="Python Version",
            status="PASS",
            value=ver_str,
            message="Supported Python version detected (3.10 - 3.12).",
        )
    elif version.major == 3 and version.minor > 12:
        return DiagnosticItem(
            name="Python Version",
            status="WARN",
            value=ver_str,
            message="Python > 3.12 detected. Some PyTorch/xFormers wheels may lack prebuilt binaries.",
            remediation="Recommend using Python 3.10 or 3.11 for maximum wheel compatibility.",
        )
    else:
        return DiagnosticItem(
            name="Python Version",
            status="FAIL",
            value=ver_str,
            message="Unsupported Python version. Velaris requires Python 3.10+.",
            remediation="Install Python 3.10 from https://python.org and add to PATH.",
        )


def check_pytorch_cuda() -> DiagnosticItem:
    torch_ver = torch.__version__
    cuda_available = torch.cuda.is_available()
    torch_version_mod = getattr(torch, "version", None)
    cuda_build = getattr(torch_version_mod, "cuda", None) if torch_version_mod else None

    if cuda_available and cuda_build:
        return DiagnosticItem(
            name="PyTorch & CUDA",
            status="PASS",
            value=f"PyTorch {torch_ver} (CUDA {cuda_build})",
            message="CUDA acceleration is fully enabled and ready.",
        )
    elif cuda_build and not cuda_available:
        return DiagnosticItem(
            name="PyTorch & CUDA",
            status="FAIL",
            value=f"PyTorch {torch_ver} (CUDA {cuda_build} built, but device unavailable)",
            message="PyTorch was built with CUDA, but no NVIDIA GPU driver was detected.",
            remediation="Ensure NVIDIA drivers are up to date and your GPU is properly enabled in Device Manager.",
        )
    else:
        return DiagnosticItem(
            name="PyTorch & CUDA",
            status="WARN",
            value=f"PyTorch {torch_ver} (CPU only)",
            message="CPU fallback mode. Video generation will run very slowly without an NVIDIA GPU.",
            remediation="Run force_gpu_install.bat to install PyTorch with CUDA 12.1 support.",
        )


def check_vram() -> DiagnosticItem:
    if not torch.cuda.is_available():
        return DiagnosticItem(
            name="VRAM Capacity",
            status="WARN",
            value="N/A (CPU)",
            message="No GPU VRAM available. Will use system RAM.",
            remediation="An 8GB+ NVIDIA GPU (e.g. RTX 3060/4060) is strongly recommended.",
        )

    try:
        device_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        total_mb = total_bytes / (1024 * 1024)
        free_mb = free_bytes / (1024 * 1024)

        if total_mb >= 7500:
            return DiagnosticItem(
                name="VRAM Capacity",
                status="PASS",
                value=f"{device_name} ({total_mb:.0f} MB total, {free_mb:.0f} MB free)",
                message="Sufficient VRAM detected for sequential Dreamer and SVD-XT pipelines.",
            )
        elif total_mb >= 5500:
            return DiagnosticItem(
                name="VRAM Capacity",
                status="WARN",
                value=f"{device_name} ({total_mb:.0f} MB total, {free_mb:.0f} MB free)",
                message="Constrained VRAM (6GB-8GB). CPU offloading is required.",
                remediation="Ensure background GPU applications (games, browser hardware accel) are closed.",
            )
        else:
            return DiagnosticItem(
                name="VRAM Capacity",
                status="FAIL",
                value=f"{device_name} ({total_mb:.0f} MB total)",
                message="VRAM is below the recommended minimum (6GB) for local video diffusion.",
                remediation="Consider running in CPU compatibility mode or upgrading to an 8GB+ GPU.",
            )
    except Exception as e:
        return DiagnosticItem(
            name="VRAM Capacity",
            status="WARN",
            value="Error querying VRAM",
            message=str(e),
        )


def check_disk_space() -> DiagnosticItem:
    try:
        check_path = config.outputs_dir.resolve()
        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024 ** 3)
        
        if free_gb >= 10.0:
            return DiagnosticItem(
                name="Disk Space",
                status="PASS",
                value=f"{free_gb:.1f} GB free",
                message="Sufficient disk space for HuggingFace model cache and video outputs.",
            )
        elif free_gb >= 2.0:
            return DiagnosticItem(
                name="Disk Space",
                status="WARN",
                value=f"{free_gb:.1f} GB free",
                message="Disk space is low (<10 GB). If models are not already cached, downloading may fail.",
                remediation="Free up disk space on the system drive if downloading new models.",
            )
        else:
            return DiagnosticItem(
                name="Disk Space",
                status="FAIL",
                value=f"{free_gb:.1f} GB free",
                message="Critically low disk space (<2 GB). Generation outputs or caching will fail.",
                remediation="Free up at least 10 GB of disk space before proceeding.",
            )
    except Exception as e:
        return DiagnosticItem(
            name="Disk Space",
            status="WARN",
            value="Unknown",
            message=f"Could not verify disk space: {e}",
        )


def check_xformers() -> DiagnosticItem:
    try:
        import importlib
        xformers = importlib.import_module("xformers")
        xformers_ver = getattr(xformers, "__version__", "installed")
        return DiagnosticItem(
            name="xFormers (Memory Efficient Attention)",
            status="PASS",
            value=f"xFormers {xformers_ver} installed",
            message="Optional memory efficient attention is available.",
        )
    except ImportError:
        return DiagnosticItem(
            name="xFormers (Memory Efficient Attention)",
            status="PASS",
            value="Not installed (Optional)",
            message="PyTorch 2.0+ SDPA (Scaled Dot-Product Attention) native fallback active.",
        )


def check_model_registry() -> DiagnosticItem:
    models = model_registry.list_models()
    if "dreamer" in models and "animator" in models:
        d = models["dreamer"]
        a = models["animator"]
        has_studio = "studio_sdxl" in models and "studio_svd" in models
        studio_note = ", Studio: SDXL+SVD ready" if has_studio else ""
        return DiagnosticItem(
            name="Model Registry",
            status="PASS",
            value=f"Dreamer: {d.model_id} (rev: {d.revision[:7]}), Animator: {a.model_id} (rev: {a.revision[:7]}){studio_note}",
            message="Core models and Studio Photoreal pipelines registered with pinned immutable revisions.",
        )
    else:
        return DiagnosticItem(
            name="Model Registry",
            status="FAIL",
            value="Missing definitions",
            message="Core models are not properly registered.",
            remediation="Verify backend/model_registry.py defaults.",
        )


def check_directory_permissions() -> DiagnosticItem:
    try:
        test_file = config.outputs_dir / ".write_test"
        with open(test_file, "w") as f:
            f.write("test")
        if test_file.exists():
            test_file.unlink()
        return DiagnosticItem(
            name="Directory Permissions",
            status="PASS",
            value="Writable",
            message=f"Output directory '{config.outputs_dir.name}' is verified writable.",
        )
    except Exception as e:
        return DiagnosticItem(
            name="Directory Permissions",
            status="FAIL",
            value="Read-only or restricted",
            message=f"Cannot write to output directory: {e}",
            remediation="Ensure user permissions allow write access to workspace.",
        )


def check_ffmpeg() -> DiagnosticItem:
    """Verifies FFmpeg is installed and accessible for frame interpolation."""
    if is_ffmpeg_available():
        version = get_ffmpeg_version()
        return DiagnosticItem(
            name="FFmpeg (Frame Interpolation)",
            status="PASS",
            value=version,
            message="FFmpeg available. RIFE-style 24fps frame interpolation is enabled.",
        )
    else:
        return DiagnosticItem(
            name="FFmpeg (Frame Interpolation)",
            status="WARN",
            value="Not installed",
            message="FFmpeg not found on PATH. Frame interpolation will be skipped (videos output at 7fps).",
            remediation="Install FFmpeg from https://ffmpeg.org/download.html and add to system PATH.",
        )


def check_torch_compile() -> DiagnosticItem:
    """Checks if torch.compile is available for UNet inference acceleration."""
    if not torch.cuda.is_available():
        return DiagnosticItem(
            name="torch.compile (Inference Acceleration)",
            status="WARN",
            value="Skipped (CPU mode)",
            message="torch.compile acceleration only applies to CUDA GPUs.",
        )
    if hasattr(torch, "compile"):
        return DiagnosticItem(
            name="torch.compile (Inference Acceleration)",
            status="PASS",
            value=f"Available (PyTorch {torch.__version__})",
            message="torch.compile will reduce UNet overhead by ~20-35% after first warm-up run.",
        )
    else:
        return DiagnosticItem(
            name="torch.compile (Inference Acceleration)",
            status="WARN",
            value="Not available",
            message="torch.compile requires PyTorch 2.0+. Falling back to eager mode.",
            remediation="Upgrade PyTorch: pip install torch>=2.0.0",
        )


def run_preflight() -> PreflightReport:
    """Executes all pre-flight hardware and software checks."""
    checks = [
        check_python_version(),
        check_pytorch_cuda(),
        check_vram(),
        check_disk_space(),
        check_xformers(),
        check_model_registry(),
        check_directory_permissions(),
        check_ffmpeg(),
        check_torch_compile(),
    ]

    has_fail = any(c.status == "FAIL" for c in checks)
    has_warn = any(c.status == "WARN" for c in checks)

    if has_fail:
        overall = "FAIL"
        can_generate = False
    elif has_warn:
        overall = "WARN"
        can_generate = True
    else:
        overall = "PASS"
        can_generate = True

    return PreflightReport(
        overall_status=overall,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        checks=checks,
        can_generate=can_generate,
    )


def print_cli_report(report: PreflightReport):
    """Prints a clear, formatted terminal output for CLI or startup scripts."""
    print("=" * 65)
    print("   VELARIS MOTION - HARDWARE & RUNTIME PRE-FLIGHT CHECK   ")
    print(f"   Target: v{config.version} | System: {report.platform}")
    print("=" * 65)
    
    for c in report.checks:
        status_symbol = {
            "PASS": "[ OK ]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]"
        }.get(c.status, "[ -- ]")
        
        print(f"\n{status_symbol} {c.name}")
        print(f"       Value:   {c.value}")
        print(f"       Details: {c.message}")
        if c.remediation:
            print(f"       Action:  --> {c.remediation}")

    print("\n" + "=" * 65)
    if report.overall_status == "PASS":
        print("   RESULT: ALL SYSTEMS GO. Ready for local video generation.")
    elif report.overall_status == "WARN":
        print("   RESULT: SYSTEM READY WITH WARNINGS. Generation may proceed.")
    else:
        print("   RESULT: PRE-FLIGHT CHECK FAILED. Please resolve issues above.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    report = run_preflight()
    print_cli_report(report)
    sys.exit(0 if report.can_generate else 1)
