"""
Core utility functions for SS Image Processor.

This module contains essential processing utilities that don't fit
into more specific domains. These are primarily I/O operations
and subprocess management functions.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import List, Tuple


def write_ffconcat_file(frame_paths: List[Path], target_dir: Path) -> Path:
    """Write an ffconcat list file enumerating frames in order.

    Args:
        frame_paths (List[Path]): Sequence of frame file paths.
        target_dir (Path): Directory where the list file will be written.

    Returns:
        Path: Path to the created ffconcat file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    list_path = target_dir / "frames.ffconcat"
    with list_path.open("w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in frame_paths:
            posix = p.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{posix}'\n")
    return list_path


def run_subprocess(cmd: List[str], *, log: bool = True) -> Tuple[int, str]:
    """Run a subprocess command, returning (exit_code, pretty_cmd).

    Args:
        cmd (List[str]): Command and arguments, split.
        log (bool): Whether to log ffmpeg commands to stderr.

    Returns:
        Tuple[int, str]: Return code and a shell-quoted pretty string of the command.
    """
    pretty = " ".join(shlex.quote(c) for c in cmd)
    # Optionally log ffmpeg commands to stderr before execution for traceability
    if log:
        try:
            tool_name = Path(cmd[0]).name.lower() if cmd else ""
            if tool_name == "ffmpeg":
                print(f"[ffmpeg] {pretty}", file=sys.stderr, flush=True)
        except Exception:
            # Never let logging failures affect execution
            pass
    try:
        res = run(cmd, check=True)
        return res.returncode, pretty
    except CalledProcessError as e:
        return e.returncode, pretty