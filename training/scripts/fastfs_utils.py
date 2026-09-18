#!/usr/bin/env python3
"""Minimal Fast-FoundationStereo integration used by dataset conversion.

Fast-FoundationStereo itself and its checkpoint are external dependencies.
This module only defines image rectification and inference glue.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def import_ffs(root: Path) -> tuple[Any, Any, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Fast-FoundationStereo repository not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE, vis_disparity

    return InputPadder, AMP_DTYPE, vis_disparity


def set_default_normalize(model: torch.nn.Module) -> None:
    try:
        from omegaconf import open_dict

        if "normalize" not in model.args:
            with open_dict(model.args):
                model.args.normalize = True
    except Exception:
        if not hasattr(model.args, "normalize"):
            setattr(model.args, "normalize", True)


def _crop(processing: dict[str, Any]) -> dict[str, int]:
    crop = processing.get("crop") or {}
    return {key: int(crop[key]) for key in ("x", "y", "width", "height")}


def build_image_undistort_like_maps(
    calibration: dict[str, Any], processing: dict[str, Any]
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    input_size = (int(calibration["image_width"]), int(calibration["image_height"]))
    output_size = (
        int(processing["rectified_image_size_per_eye"]["width"]),
        int(processing["rectified_image_size_per_eye"]["height"]),
    )
    k_left = np.asarray(calibration["K_left"], dtype=np.float64)
    d_left = np.asarray(calibration["D_left"], dtype=np.float64).reshape(-1, 1)
    k_right = np.asarray(calibration["K_right"], dtype=np.float64)
    d_right = np.asarray(calibration["D_right"], dtype=np.float64).reshape(-1, 1)
    r_left = np.asarray(calibration["R1"], dtype=np.float64)
    r_right = np.asarray(calibration["R2"], dtype=np.float64)
    p_left = np.asarray(processing["rectification"]["P1"], dtype=np.float64)
    p_right = np.asarray(processing["rectification"]["P2"], dtype=np.float64)
    left_maps = cv2.fisheye.initUndistortRectifyMap(
        k_left, d_left, r_left, p_left, output_size, cv2.CV_32FC1
    )
    right_maps = cv2.fisheye.initUndistortRectifyMap(
        k_right, d_right, r_right, p_right, output_size, cv2.CV_32FC1
    )
    crop = _crop(processing)
    camera = {
        "input_width_per_eye": input_size[0],
        "input_height_per_eye": input_size[1],
        "rectified_width_per_eye": output_size[0],
        "rectified_height_per_eye": output_size[1],
        "crop": crop,
        "rotation_deg": float(processing.get("rotation_deg", 0.0)),
        "fx_crop": float(p_left[0, 0]),
        "fy_crop": float(p_left[1, 1]),
        "cx_crop": float(p_left[0, 2]) - float(crop["x"]),
        "cy_crop": float(p_left[1, 2]) - float(crop["y"]),
        "baseline_m": abs(float(p_right[0, 3]) / float(p_left[0, 0])),
    }
    return camera, left_maps, right_maps


def _rotate_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-9:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def rectify_crop(
    frame: np.ndarray,
    left_maps: tuple[np.ndarray, np.ndarray],
    right_maps: tuple[np.ndarray, np.ndarray],
    camera: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    if width % 2:
        raise RuntimeError(f"Side-by-side stereo frame has odd width: {width}x{height}")
    left, right = frame[:, : width // 2], frame[:, width // 2 :]
    expected = (int(camera["input_height_per_eye"]), int(camera["input_width_per_eye"]))
    if left.shape[:2] != expected or right.shape[:2] != expected:
        raise RuntimeError(
            f"Top stereo eye size mismatch: left={left.shape[:2]} right={right.shape[:2]} "
            f"expected={expected}"
        )
    left = cv2.remap(left, left_maps[0], left_maps[1], cv2.INTER_LINEAR)
    right = cv2.remap(right, right_maps[0], right_maps[1], cv2.INTER_LINEAR)
    crop = camera["crop"]
    x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
    rotation = float(camera.get("rotation_deg", 0.0))
    return (
        _rotate_keep_size(left[y : y + h, x : x + w], rotation),
        _rotate_keep_size(right[y : y + h, x : x + w], rotation),
    )


def _make_tensor(image_bgr: np.ndarray, scale: float) -> torch.Tensor:
    if scale != 1.0:
        image_bgr = cv2.resize(
            image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(image_rgb, device="cuda").float()[None].permute(0, 3, 1, 2)


def _disparity_to_depth(
    disparity: np.ndarray, fx: float, baseline_m: float, max_depth_m: float
) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth[valid] = float(fx * baseline_m) / disparity[valid]
    depth[(depth <= 0.0) | (depth > max_depth_m)] = np.nan
    return depth


def colorize_depth(depth: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (1.0 - depth[valid] / max_depth_m) * 255, 0, 255
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def infer_depth(
    model: torch.nn.Module,
    input_padder: Any,
    amp_dtype: Any,
    left: np.ndarray,
    right: np.ndarray,
    camera: dict[str, Any],
    scale: float,
    valid_iters: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    left_tensor = _make_tensor(left, float(scale))
    right_tensor = _make_tensor(right, float(scale))
    padder = input_padder(left_tensor.shape, divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    torch.cuda.synchronize()
    tensor_done = time.perf_counter()
    with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype), torch.no_grad():
        disparity_tensor = model.forward(
            left_padded,
            right_padded,
            iters=int(valid_iters),
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    torch.cuda.synchronize()
    model_done = time.perf_counter()
    height, width = left_tensor.shape[-2:]
    disparity = (
        padder.unpad(disparity_tensor.float())
        .detach()
        .cpu()
        .numpy()
        .reshape(height, width)
        .clip(0, None)
        .astype(np.float32)
    )
    depth = _disparity_to_depth(
        disparity,
        float(camera["fx_crop"]) * float(scale),
        float(camera["baseline_m"]),
        float(max_depth_m),
    )
    depth_done = time.perf_counter()
    if float(scale) != 1.0:
        size = (disparity.shape[1], disparity.shape[0])
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    timing = {
        "tensor_ms": (tensor_done - start) * 1000.0,
        "model_ms": (model_done - tensor_done) * 1000.0,
        "depth_ms": (depth_done - model_done) * 1000.0,
        "total_ms": (depth_done - start) * 1000.0,
    }
    return depth, left, right, timing
