"""
FFmpeg command building module for SS Image Processor.

This module consolidates all FFmpeg command construction logic,
separating it from the main processing pipeline for better maintainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
from core_types import OutputFormat, Quality
from image_utils import check_alpha_tiff

# FFmpeg constants
DEFAULT_FRAME_RATE = 30


class FFmpegCommandBuilder:
    """Builder class for constructing FFmpeg commands."""
    
    @staticmethod
    def build_animated_cmd(
        list_file: Path,
        out_path: Path,
        quality: Quality,
        threads: int,
        fmt: OutputFormat,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
        has_alpha: bool = True,
    ) -> List[str]:
        """Create ffmpeg command for an animated image sequence, with optional crop and alpha optimization."""
        base = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-an",
            "-vsync", "0",
            "-r", str(DEFAULT_FRAME_RATE),
        ]
        codec_args = fmt.animated_codec_args(threads, has_alpha)
        if crop_rect is not None:
            args = codec_args.copy()
            if "-vf" in args:
                i = args.index("-vf")
                if i + 1 < len(args):
                    existing = args[i + 1]
                    x, y, w, h = crop_rect
                    crop = f"crop={w}:{h}:{x}:{y}"
                    args[i + 1] = f"{crop},{existing}"
                    codec_args = args
        return base + codec_args + quality.ffmpeg_args + [str(out_path)]

    @staticmethod
    def build_individual_cmd(
        src: Path,
        dst: Path,
        quality: Quality,
        fmt: OutputFormat,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[str]:
        """Create ffmpeg command for a single frame.

        For AVIF+TIFF, use a tested filter_complex that separates color and alpha,
        encoding color with the configured CRF and alpha losslessly.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)

        if fmt is OutputFormat.AVIF and src.suffix.lower() in {".tif", ".tiff"}:
            color_crf = FFmpegCommandBuilder._extract_crf(quality.ffmpeg_args) or "26"
            # Detect whether TIFF has an alpha channel; if not, avoid alphaextract
            try:
                has_alpha = check_alpha_tiff(src)
            except Exception:
                has_alpha = False

            if not has_alpha:
                # No alpha → use optimized settings: aomav1_crf30_s6_420p_aqmode1
                codec_args = fmt.still_codec_args(has_alpha=False)
                base = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-nostdin",
                    "-i", str(src),
                    "-an",
                    "-frames:v", "1",
                ] + codec_args
                vf = None
                # if crop_rect is not None:
                #     x, y, w, h = crop_rect
                #     vf = f"crop={w}:{h}:{x}:{y}"
                if vf is not None:
                    base.extend(["-vf", vf])
                base.append(str(dst))
                return base
            else:
                # Has alpha → split and encode color+alpha (alpha lossless)
                crop_expr = None
                if crop_rect is not None:
                    x, y, w, h = crop_rect
                    crop_expr = f"[0:v]crop={w}:{h}:{x}:{y}"
                # Ensure we always have an alpha plane for extraction
                head = (crop_expr or "[0:v]") + ",format=rgba"
                filter_chain = f"{head},split=2[c][asrc];[asrc]alphaextract[a]"

                return [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-nostdin",
                    "-i", str(src),
                    "-filter_complex", filter_chain,
                    "-map", "[c]",
                    "-map", "[a]",
                    "-frames:v", "1",
                    "-c:v", "libaom-av1",
                    "-still-picture", "1",
                    "-pix_fmt:0", "yuv444p12le",
                    "-crf:0", color_crf,
                    "-pix_fmt:1", "gray12le",
                    "-crf:1", "0",
                    str(dst),
                ]

        # AVIF + PNG: preserve transparency using explicit alpha extraction
        if fmt is OutputFormat.AVIF and src.suffix.lower() in {".png"}:
            color_crf = FFmpegCommandBuilder._extract_crf(quality.ffmpeg_args) or "26"
            # Build filter chain: always ensure an alpha plane exists via format=rgba
            # then optionally crop, split color/alpha and extract alpha.
            chain_base = "[0:v]format=rgba"
            if crop_rect is not None:
                x, y, w, h = crop_rect
                chain_base = f"{chain_base},crop={w}:{h}:{x}:{y}"
            filter_chain = f"{chain_base},split=2[c][asrc];[asrc]alphaextract[a]"

            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-i", str(src),
                "-filter_complex", filter_chain,
                "-map", "[c]",
                "-map", "[a]",
                "-frames:v", "1",
                "-c:v", "libaom-av1",
                "-still-picture", "1",
                # Use 8-bit for PNG inputs to avoid unnecessary bit-depth inflation
                "-pix_fmt:0", "yuv444p",
                "-crf:0", color_crf,
                "-pix_fmt:1", "gray",
                # Keep alpha lossless to preserve edges cleanly
                "-crf:1", "0",
                str(dst),
            ]

        base = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-i", str(src),
            "-an",
        ]
        codec_args = fmt.still_codec_args()
        # Merge crop filter with existing -vf if needed
        if crop_rect is not None:
            # codec_args like ['-vf','format=rgba', ...]; merge into format chain
            args = codec_args.copy()
            if "-vf" in args:
                i = args.index("-vf")
                if i + 1 < len(args):
                    existing = args[i + 1]
                    x, y, w, h = crop_rect
                    crop = f"crop={w}:{h}:{x}:{y}"
                    args[i + 1] = f"{crop},{existing}"
                    codec_args = args
        return base + codec_args + quality.ffmpeg_args + [str(dst)]

    @staticmethod
    def _extract_crf(args: List[str]) -> Optional[str]:
        """Extract CRF value from FFmpeg arguments."""
        for i, a in enumerate(args):
            if a == "-crf" and i + 1 < len(args):
                return args[i + 1]
        return None