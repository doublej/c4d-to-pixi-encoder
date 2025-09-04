import os
from pathlib import Path

import pytest

from main import build_ffmpeg_individual_cmd, OutputFormat, Quality


def test_png_to_avif_command_includes_alpha_extract(tmp_path: Path):
    # Arrange
    src = tmp_path / "frame.png"
    # We don't need a real image file; only the command is under test
    src.write_bytes(b"fake")
    dst = tmp_path / "out.avif"
    q = Quality.from_name("high", OutputFormat.AVIF)

    # Act
    cmd = build_ffmpeg_individual_cmd(src, dst, q, OutputFormat.AVIF, crop_rect=None)
    joined = " ".join(cmd)

    # Assert
    # Ensure the alpha-preserving chain is present
    assert "-filter_complex" in cmd
    assert "format=rgba" in joined
    assert "alphaextract" in joined
    # Ensure color and alpha are mapped as two streams
    assert "-map [c]".replace(" ", "") in joined.replace(" ", "")
    assert "-map [a]".replace(" ", "") in joined.replace(" ", "")
    # Ensure AVIF still picture flag is present
    assert "-still-picture" in cmd

