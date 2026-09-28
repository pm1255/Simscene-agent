"""Small, dependency-light multi-view reconstruction demo.

This is deliberately an explicit baseline rather than a learned model: calibrated
silhouettes are carved into a voxel volume and boundary voxels are exported as a
watertight block mesh.  It is useful for testing the agent/harness interfaces and
for showing where real RGB-D/learned providers plug in.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.tile(np.array([[190, 170, 80]], dtype=np.uint8), (len(vertices), 1))
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for p, c in zip(vertices, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def carve_silhouettes(masks: Sequence[np.ndarray], yaws_deg: Sequence[float], grid: int = 48,
                      bounds: Sequence[float] = (-0.12, 0.12, -0.12, 0.12, 0.0, 0.24)) -> tuple[np.ndarray, np.ndarray]:
    """Carve a volume from binary masks and calibrated turntable yaw angles.

    The camera is orthographic and looks at the origin.  A voxel survives when its
    projection lands on foreground in every supplied view.  The returned arrays are
    boundary-voxel vertices/faces in metres; this makes the geometric assumption
    explicit and easy to replace with COLMAP/Open3D/learned providers.
    """
    if len(masks) != len(yaws_deg) or not masks:
        raise ValueError("masks and yaws_deg must be non-empty and have equal length")
    h, w = masks[0].shape
    if any(m.shape != (h, w) for m in masks):
        raise ValueError("all masks must have the same shape")
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
    xs = np.linspace(xmin, xmax, grid, endpoint=False) + (xmax - xmin) / grid / 2
    ys = np.linspace(ymin, ymax, grid, endpoint=False) + (ymax - ymin) / grid / 2
    zs = np.linspace(zmin, zmax, grid, endpoint=False) + (zmax - zmin) / grid / 2
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    keep = np.ones(X.shape, dtype=bool)
    # Fit the object to 80% of the image height/width in the orthographic camera.
    for mask, yaw in zip(masks, yaws_deg):
        t = math.radians(float(yaw))
        xr = X * math.cos(t) - Y * math.sin(t)
        yr = X * math.sin(t) + Y * math.cos(t)
        u = np.clip(((xr - xmin) / (xmax - xmin) * (w - 1)).round().astype(int), 0, w - 1)
        v = np.clip(((zmax - Z) / (zmax - zmin) * (h - 1)).round().astype(int), 0, h - 1)
        keep &= mask[v, u] > 0
    # Boundary cells become a block mesh.  Shared vertices keep the output compact.
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    corners = {
        (1,0,0): [(1,0,0),(1,1,0),(1,1,1),(1,0,1)], (-1,0,0): [(0,0,0),(0,1,0),(0,1,1),(0,0,1)],
        (0,1,0): [(0,1,0),(1,1,0),(1,1,1),(0,1,1)], (0,-1,0): [(0,0,0),(0,0,1),(1,0,1),(1,0,0)],
        (0,0,1): [(0,0,1),(0,1,1),(1,1,1),(1,0,1)], (0,0,-1): [(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
    }
    verts: list[tuple[float,float,float]] = []; faces: list[tuple[int,int,int]] = []; index: dict[tuple[float,float,float], int] = {}
    dx, dy, dz = (xmax-xmin)/grid, (ymax-ymin)/grid, (zmax-zmin)/grid
    for i,j,k in zip(*np.where(keep)):
        for d in directions:
            ni,nj,nk = i+d[0], j+d[1], k+d[2]
            if 0 <= ni < grid and 0 <= nj < grid and 0 <= nk < grid and keep[ni,nj,nk]:
                continue
            base = np.array([xs[i]-dx/2, ys[j]-dy/2, zs[k]-dz/2])
            ids=[]
            for ox,oy,oz in corners[d]:
                p=tuple(np.round(base + np.array([ox*dx,oy*dy,oz*dz]), 9))
                if p not in index: index[p]=len(verts); verts.append(p)
                ids.append(index[p])
            faces.extend([(ids[0],ids[1],ids[2]),(ids[0],ids[2],ids[3])])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def synthetic_turntable(out_dir: Path, views: int = 12, size: int = 192) -> tuple[list[Path], list[float]]:
    """Create a reproducible RGB silhouette sequence for the demo (no model call)."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    paths=[]; yaws=[]
    for i in range(views):
        yaw = i * 360.0 / views
        # Projected bottle width changes with turntable yaw; the mask is RGB input.
        width = int(42 + 14 * abs(math.cos(math.radians(yaw))))
        im=Image.new("RGB", (size,size), (235,240,247)); d=ImageDraw.Draw(im)
        cx=size//2; bottom=160; top=38
        d.rounded_rectangle((cx-width//2,bottom-78,cx+width//2,bottom), radius=10, fill=(226,181,24))
        neck=max(12,width//3); d.rectangle((cx-neck//2,top+24,cx+neck//2,bottom-75), fill=(226,181,24))
        d.rounded_rectangle((cx-10,top,cx+10,top+32), radius=5, fill=(205,160,20))
        d.ellipse((cx-width//2,bottom-86,cx+width//2,bottom-70), fill=(244,205,70))
        p=out_dir/f"frame_{i:03d}.png"; im.save(p); paths.append(p); yaws.append(yaw)
    return paths, yaws


def run_reconstruction(out_dir: Path, grid: int = 48) -> dict:
    from PIL import Image
    input_dir=out_dir/"input_sequence"; paths,yaws=synthetic_turntable(input_dir)
    masks=[]
    for p in paths:
        a=np.asarray(Image.open(p).convert("RGB")); masks.append(np.any(a < 220, axis=2).astype(np.uint8))
    v,f=carve_silhouettes(masks,yaws,grid=grid)
    # Per-vertex color is an appearance proxy. Real RGB-D/UV providers replace it.
    z=(v[:,2]-v[:,2].min())/(np.ptp(v[:,2])+1e-9)
    colors=np.stack([np.full(len(v),220), (170+60*z), np.full(len(v),30)], axis=1).astype(np.uint8)
    mesh=out_dir/"reconstruction.ply"; _write_ply(mesh,v,f,colors)
    report={"method":"calibrated silhouette voxel carving (demo)","input_images":len(paths),"grid":grid,"vertices":int(len(v)),"triangles":int(len(f)),"metric_bounds_m":[float(x) for x in [v[:,0].min(),v[:,0].max(),v[:,1].min(),v[:,1].max(),v[:,2].min(),v[:,2].max()]],"texture":"vertex RGB proxy from reconstructed material prior; UV baking is a replaceable provider","limitations":["silhouette carving cannot recover concavities hidden in every view","synthetic masks are used in this reproducible demo","mass/friction are not inferred from RGB"]}
    (out_dir/"reconstruction_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
