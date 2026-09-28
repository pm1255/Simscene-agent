"""Small, dependency-light 3D Gaussian Splatting visual export.

The seven-layer pipeline keeps the observed/physics mesh unchanged and writes a
second visual representation.  This module creates a 3DGS-compatible PLY from
the registered RGB-D observations and a Nerfstudio-compatible camera manifest.
It is deliberately called a *bootstrap* export: it is a valid Gaussian set
initialised from observations, not a claim that a CUDA trainer was run.  A
trained splat can replace the same directory later without touching L5/L6.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import dump, project


_SH_C0 = 0.28209479177387814


def _sigmoid_inverse(x: float) -> float:
    x = float(np.clip(x, 1e-5, 1 - 1e-5))
    return math.log(x / (1 - x))


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float,
                      max_points: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Keep one averaged colored point per voxel, with deterministic capping."""
    if len(points) == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    keys = np.floor(points / float(voxel)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, inverse, colors.astype(np.float64))
    means = (sums / np.maximum(counts[:, None], 1)).astype(np.uint8)
    centers = (unique.astype(np.float64) + 0.5) * float(voxel)
    if len(centers) > max_points:
        # Deterministic spatial stride rather than random sampling keeps the
        # exported preview stable across machines and reruns.
        stride = int(math.ceil(len(centers) / max_points))
        centers, means = centers[::stride], means[::stride]
    return centers.astype(np.float32), means


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray, scale: float) -> None:
    """Write the standard Gaussian-Splatting vertex attributes as ASCII PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = """ply
format ascii 1.0
element vertex {count}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
""".format(count=len(points))
    rgb = colors.astype(np.float32) / 255.0
    sh = (rgb - 0.5) / _SH_C0
    log_scale = math.log(max(float(scale), 1e-5))
    opacity = _sigmoid_inverse(0.86)
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for p, c in zip(points, sh):
            f.write(
                "%.7f %.7f %.7f 0 0 0 %.7f %.7f %.7f %.7f %.7f %.7f %.7f 1 0 0 0\n"
                % (*p, *c, opacity, log_scale, log_scale, log_scale)
            )


def _write_transforms(out: Path, frames: list[dict]) -> None:
    """Write a Nerfstudio-style camera manifest using CV-to-GL conversion."""
    train = [f for f in frames if f.get("split", "train") == "train"]
    if not train:
        train = frames
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    first = train[0]
    h, w = first["depth"].shape
    manifest = {
        "camera_model": "OPENCV",
        "fl_x": float(first["K"][0, 0]),
        "fl_y": float(first["K"][1, 1]),
        "cx": float(first["K"][0, 2]),
        "cy": float(first["K"][1, 2]),
        "w": int(w),
        "h": int(h),
        "coordinate_convention": "world_z_up_camera_cv_to_nerfstudio_gl",
        "frames": [],
    }
    for f in train:
        transform = np.asarray(f["T"], dtype=float) @ cv_to_gl
        manifest["frames"].append({
            "file_path": "../../L1_observe/capture/" + f["rgb_path"],
            "transform_matrix": transform.tolist(),
        })
    dump(out / "transforms.json", manifest)


def _render_preview(out: Path, points: np.ndarray, colors: np.ndarray,
                    frame: dict, width: int = 896, height: int = 672,
                    splat_scale: float = 0.035) -> None:
    """Render a modest CPU Gaussian preview for the hosted demo.

    This is only a visual sanity check.  Interactive viewers and CUDA trainers
    consume the exported PLY/transform manifest for the high-quality result.
    """
    K = np.asarray(frame["K"], dtype=float).copy()
    src_h, src_w = frame["depth"].shape
    sx, sy = width / src_w, height / src_h
    K[0] *= sx
    K[1] *= sy
    uv, z = project(points.astype(float), K, np.asarray(frame["T"], dtype=float))
    valid = (z > 0.05) & (uv[:, 0] >= -4) & (uv[:, 0] < width + 4) & (uv[:, 1] >= -4) & (uv[:, 1] < height + 4)
    uv, z, colors = uv[valid], z[valid], colors[valid]
    # Paint far points first.  The small alpha disks approximate isotropic
    # Gaussian splats while remaining dependency-free for CI and GitHub Pages.
    order = np.argsort(z)[::-1]
    canvas = np.full((height, width, 3), 238, np.uint8)
    alpha = np.zeros((height, width), np.float32)
    for i in order:
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        radius = max(1, min(10, int(round(K[0, 0] * splat_scale / max(float(z[i]), 0.1)))))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = ((xx - uv[i, 0]) ** 2 + (yy - uv[i, 1]) ** 2) / max(radius * radius, 1)
        a = np.exp(-2.0 * rr).astype(np.float32) * 0.32
        old = alpha[y0:y1, x0:x1]
        use = a * (1.0 - old)
        canvas[y0:y1, x0:x1] = np.clip(
            canvas[y0:y1, x0:x1].astype(np.float32) * (1 - use[..., None])
            + colors[i].astype(np.float32) * use[..., None], 0, 255
        ).astype(np.uint8)
        alpha[y0:y1, x0:x1] = np.clip(old + a, 0, 1)
    Image.fromarray(canvas).save(out / "gaussian_preview.png")


def build_gaussian_visual(out: Path, frames: list[dict], points: np.ndarray,
                          colors: np.ndarray, voxel: float = 0.035) -> dict:
    """Export both a 3DGS-compatible visual and a reproducible preview."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep the visual representation independent from L3/L5 meshes.  The
    # points are observed RGB-D samples, so no hidden geometry is fabricated.
    pts, cols = _voxel_downsample(points, colors, voxel=max(voxel, 0.012))
    _write_ply(out / "scene.splat.ply", pts, cols, scale=max(voxel * 0.8, 0.012))
    cooked_frames = []
    for f in frames:
        cooked_frames.append({
            "K": np.asarray(f["K"], dtype=float),
            "T": np.asarray(f["T"], dtype=float),
            "depth": np.asarray(f["depth"]),
            "split": f.get("split", "train"),
            "rgb_path": f["rgb_path"],
        })
    _write_transforms(out, cooked_frames)
    _render_preview(out, pts, cols, cooked_frames[0])
    metadata = {
        "status": "passed",
        "provider": "rgbd_gaussian_bootstrap",
        "format": "3DGS-compatible ASCII PLY",
        "trained": False,
        "point_count": int(len(pts)),
        "voxel_m": float(voxel),
        "source": "registered L1 RGB-D observations",
        "collision_source": "L5 collision proxies remain independent",
        "preview": "gaussian_preview.png",
        "training_upgrade": {
            "command": "ns-train splatfacto --data transforms.json",
            "requires": "CUDA PyTorch with Nerfstudio or an equivalent gsplat trainer",
            "replace": "scene.splat.ply and keep transforms.json",
        },
        "limitations": [
            "bootstrap splats retain observed surfaces only",
            "moving robot pixels must be masked before training a static scene",
            "the hosted preview is a CPU sanity render, not a trained novel-view render",
        ],
    }
    dump(out / "visual.json", metadata)
    return metadata
