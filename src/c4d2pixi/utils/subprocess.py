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



