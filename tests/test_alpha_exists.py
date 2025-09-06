from pathlib import Path

import numpy as np
import cv2

from image_utils import check_alpha_exists, check_alpha_png, check_alpha_tiff


def write_image(path: Path, array: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), array)
    assert ok, f"failed to write {path}"


def test_check_alpha_png_rgb_and_rgba(tmp_path: Path):
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, :3] = 100
    rgba[:, :, 3] = 255
    p_rgb = tmp_path / "rgb.png"
    p_rgba = tmp_path / "rgba.png"
    write_image(p_rgb, rgb)
    write_image(p_rgba, rgba)
    assert check_alpha_png(p_rgb) is False
    assert check_alpha_png(p_rgba) is True
    # Through top-level helper
    assert check_alpha_exists(p_rgb) is False
    assert check_alpha_exists(p_rgba) is True


def test_check_alpha_tiff_channels(tmp_path: Path):
    # 3-channel TIFF -> no alpha
    rgb = np.full((6, 9, 3), 128, dtype=np.uint8)
    p_rgb = tmp_path / "rgb.tiff"
    write_image(p_rgb, rgb)
    assert check_alpha_tiff(p_rgb) is False
    assert check_alpha_exists(p_rgb) is False

    # 4-channel TIFF -> has alpha
    rgba = np.zeros((6, 9, 4), dtype=np.uint8)
    rgba[:, :, :3] = 10
    rgba[:, :, 3] = 255
    p_rgba = tmp_path / "rgba.tif"
    write_image(p_rgba, rgba)
    assert check_alpha_tiff(p_rgba) is True
    assert check_alpha_exists(p_rgba) is True

