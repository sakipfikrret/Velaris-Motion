"""
Unit tests for backend/interpolation.py - Sprint 1C.
Tests FFmpeg availability detection, error handling, and the interpolation
pipeline without requiring an actual GPU or model weights.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from interpolation import (
    is_ffmpeg_available,
    get_ffmpeg_version,
    interpolate_video,
)

# Helper: patch _get_ffmpeg_exe to simulate "no FFmpeg found anywhere"
_NO_FFMPEG = patch("interpolation._get_ffmpeg_exe", return_value="")
# Helper: patch _get_ffmpeg_exe to simulate "FFmpeg found at /usr/bin/ffmpeg"
def _with_ffmpeg(path: str = "/usr/bin/ffmpeg"):
    return patch("interpolation._get_ffmpeg_exe", return_value=path)


# ---------------------------------------------------------------------------
# is_ffmpeg_available()
# ---------------------------------------------------------------------------

def test_ffmpeg_available_returns_bool():
    result = is_ffmpeg_available()
    assert isinstance(result, bool)


def test_ffmpeg_available_when_present(tmp_path):
    """When shutil.which finds ffmpeg, is_ffmpeg_available returns True."""
    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        assert is_ffmpeg_available() is True


def test_ffmpeg_not_available_when_absent():
    """When FFmpeg is absent, is_ffmpeg_available returns False."""
    with patch("interpolation._get_ffmpeg_exe", return_value=""):
        assert is_ffmpeg_available() is False


# ---------------------------------------------------------------------------
# get_ffmpeg_version()
# ---------------------------------------------------------------------------

def test_get_ffmpeg_version_not_installed():
    with patch("interpolation._get_ffmpeg_exe", return_value=""):
        assert get_ffmpeg_version() == "Not installed"


def test_get_ffmpeg_version_returns_string():
    """Returns a non-empty string when FFmpeg is available."""
    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 6.0 Copyright (c) ...\n"
        with patch("subprocess.run", return_value=mock_result):
            version = get_ffmpeg_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_get_ffmpeg_version_strips_copyright():
    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 6.1.1 Copyright (c) 2000-2023 ...\n"
        with patch("subprocess.run", return_value=mock_result):
            version = get_ffmpeg_version()
    assert "Copyright" not in version
    assert "6.1.1" in version


# ---------------------------------------------------------------------------
# interpolate_video()
# ---------------------------------------------------------------------------

def test_interpolate_raises_when_ffmpeg_missing(tmp_path):
    """If FFmpeg is not available, interpolate_video raises RuntimeError."""
    src = tmp_path / "input.mp4"
    src.touch()
    with patch("interpolation._get_ffmpeg_exe", return_value=""):
        with pytest.raises(RuntimeError, match="FFmpeg is not available"):
            interpolate_video(src, tmp_path / "output.mp4")


def test_interpolate_raises_when_input_missing(tmp_path):
    """If input file does not exist, interpolate_video raises FileNotFoundError."""
    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        with pytest.raises(FileNotFoundError):
            interpolate_video(
                tmp_path / "nonexistent.mp4",
                tmp_path / "output.mp4",
            )


def test_interpolate_calls_ffmpeg_with_correct_filter(tmp_path):
    """Verifies FFmpeg is called with the minterpolate filter and correct fps."""
    src = tmp_path / "input.mp4"
    src.touch()
    dst = tmp_path / "output.mp4"

    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            interpolate_video(src, dst, target_fps=24)

    called_args = mock_run.call_args[0][0]  # positional arg: the command list
    cmd_str = " ".join(called_args)

    assert "minterpolate" in cmd_str
    assert "fps=24" in cmd_str
    assert "libx264" in cmd_str
    assert str(src) in cmd_str
    assert str(dst) in cmd_str


def test_interpolate_raises_on_ffmpeg_failure(tmp_path):
    """If FFmpeg exits non-zero, a RuntimeError with stderr is raised."""
    src = tmp_path / "input.mp4"
    src.touch()

    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg"],
                stderr="Invalid data found when processing input",
            )
            with pytest.raises(RuntimeError, match="Frame interpolation failed"):
                interpolate_video(src, tmp_path / "output.mp4")


def test_interpolate_returns_output_path(tmp_path):
    """On success, interpolate_video returns the output path."""
    src = tmp_path / "input.mp4"
    src.touch()
    dst = tmp_path / "output.mp4"

    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = interpolate_video(src, dst, target_fps=24)

    assert result == dst


def test_interpolate_custom_crf_and_preset(tmp_path):
    """Verifies custom CRF and preset are passed through to FFmpeg."""
    src = tmp_path / "input.mp4"
    src.touch()

    with patch("interpolation.shutil.which", return_value="/usr/bin/ffmpeg"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            interpolate_video(src, tmp_path / "out.mp4", quality_crf=23, preset="medium")

    cmd_str = " ".join(mock_run.call_args[0][0])
    assert "-crf 23" in cmd_str
    assert "medium" in cmd_str
