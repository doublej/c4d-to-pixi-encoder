"""Subprocess and external command utilities."""

import subprocess
from pathlib import Path


def run_subprocess(cmd: list[str], *, log: bool = True, timeout: int | None = None) -> tuple[int, str]:
    """Run subprocess command with proper error handling.

    Args:
        cmd: Command and arguments list
        log: Whether to log the command
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (return_code, stderr_output)
    """
    if log:
        print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, str(e)


def write_ffconcat_file(frame_paths: list[Path], target_dir: Path) -> Path:
    """Write FFmpeg concat demuxer file for frame sequence.

    Args:
        frame_paths: List of frame paths in order
        target_dir: Directory to write concat file

    Returns:
        Path to created concat file
    """
    concat_path = target_dir / "input.ffconcat"

    with open(concat_path, "w") as f:
        f.write("ffconcat version 1.0\n")
        for frame_path in frame_paths:
            escaped = str(frame_path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
            f.write("duration 0.033\n")  # 30fps default

        # Repeat last frame for duration
        if frame_paths:
            escaped = str(frame_paths[-1]).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    return concat_path
