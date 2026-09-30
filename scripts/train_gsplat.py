#!/usr/bin/env python3
"""Train an RGB-D Gaussian scene with the CUDA gsplat rasterizer.

This is the GPU upgrade for the L4 visual branch. It keeps the physics mesh
independent: only Gaussian parameters are optimized here. The input contract
is the same capture.json used by the seven-layer pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image, ImageDraw

from simscene_agent.seven.io import backproject

SH_C0 = 0.28209479177387814


def inverse_sigmoid(x: float) -> float:
    x = float(np.clip(x, 1e-5, 1.0 - 1e-5))
    return math.log(x / (1.0 - x))


def uniform_voxel_cloud(
    points: np.ndarray, colors: np.ndarray, voxel: float, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel average followed by deterministic uniform spatial subsampling."""
    if len(points) == 0:
        return points.reshape(0, 3), colors.reshape(0, 3)
    keys = np.floor(points / float(voxel)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, inverse, colors.astype(np.float64))
    means = (sums / np.maximum(counts[:, None], 1.0)).astype(np.uint8)
    centers = (unique.astype(np.float64) + 0.5) * float(voxel)
    if len(centers) > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(centers), size=max_points, replace=False))
        centers, means = centers[keep], means[keep]
    return centers.astype(np.float32), means


def load_capture(capture: Path, frame_stride: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    metadata = json.loads((capture / "capture.json").read_text())
    frames: list[dict] = []
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for frame_no, record in enumerate(metadata["frames"]):
        if frame_no % max(frame_stride, 1) != 0:
            continue
        rgb = np.asarray(Image.open(capture / record["rgb"]).convert("RGB"), dtype=np.uint8)
        depth = np.load(capture / record["depth"]).astype(np.float32)
        K = np.asarray(record["K"], dtype=np.float32)
        T = np.asarray(record["T_world_camera"], dtype=np.float32)
        frames.append({"index": frame_no, "rgb": rgb, "depth": depth, "K": K, "T": T})
        valid = depth > 0
        points = backproject(depth, K, T)
        all_points.append(points[valid].astype(np.float32))
        all_colors.append(rgb[valid])
    if not frames:
        raise RuntimeError(f"capture has no frames after stride={frame_stride}: {capture}")
    return frames, np.concatenate(all_points), np.concatenate(all_colors)


class GaussianModel(nn.Module):
    def __init__(self, points: np.ndarray, colors: np.ndarray, voxel: float):
        super().__init__()
        self.means = nn.Parameter(torch.from_numpy(points.copy()))
        quat = torch.zeros((len(points), 4), dtype=torch.float32)
        quat[:, 0] = 1.0
        self.quats = nn.Parameter(quat)
        self.log_scales = nn.Parameter(
            torch.full((len(points), 3), math.log(max(voxel * 0.65, 0.003)), dtype=torch.float32)
        )
        self.opacity_logits = nn.Parameter(
            torch.full((len(points),), inverse_sigmoid(0.25), dtype=torch.float32)
        )
        rgb = np.clip(colors.astype(np.float32) / 255.0, 1e-3, 1.0 - 1e-3)
        self.color_logits = nn.Parameter(torch.from_numpy(np.log(rgb / (1.0 - rgb))))

    def forward(self, viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int):
        from gsplat import rasterization

        return rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.log_scales.exp(),
            opacities=self.opacity_logits.sigmoid(),
            colors=self.color_logits.sigmoid(),
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
            packed=False,
            near_plane=0.03,
            far_plane=100.0,
            radius_clip=0.0,
            render_mode="RGB",
        )


def write_ply(path: Path, model: GaussianModel) -> None:
    """Write common 3DGS PLY fields (log scales and logit opacity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        points = model.means.detach().cpu().numpy()
        scales = model.log_scales.detach().cpu().numpy()
        opacity = model.opacity_logits.detach().cpu().numpy()
        rgb = model.color_logits.sigmoid().detach().cpu().numpy()
    sh = (rgb - 0.5) / SH_C0
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
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for p, c, o, s in zip(points, sh, opacity, scales):
            handle.write(
                "%.7f %.7f %.7f 0 0 0 %.7f %.7f %.7f %.7f %.7f %.7f %.7f 1 0 0 0\n"
                % (*p, *c, o, *s)
            )


def make_camera(frame: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    T_c2w = torch.from_numpy(frame["T"]).to(device=device, dtype=torch.float32)
    view = torch.linalg.inv(T_c2w)
    K = torch.from_numpy(frame["K"]).to(device=device, dtype=torch.float32)
    return view, K


def render_views(model: GaussianModel, frames: list[dict], out: Path, device: torch.device) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rendered: list[Image.Image] = []
    sample_ids = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int).tolist()
    for rank, frame_id in enumerate(sample_ids):
        frame = frames[frame_id]
        h, w = frame["depth"].shape
        view, K = make_camera(frame, device)
        with torch.no_grad():
            renders, _alphas, _meta = model(view[None], K[None], width=w, height=h)
        render = renders[0, ..., :3].clamp(0, 1).mul(255).byte().cpu().numpy()
        name = f"view_{rank:02d}_frame_{frame['index']:04d}.png"
        Image.fromarray(render).save(out / name)
        pair = Image.new("RGB", (w * 2, h + 22), "white")
        pair.paste(Image.fromarray(frame["rgb"]), (0, 0))
        pair.paste(Image.fromarray(render), (w, 0))
        draw = ImageDraw.Draw(pair)
        draw.text((4, h + 4), f"input {frame['index']:04d}", fill="black")
        draw.text((w + 4, h + 4), "gsplat trained", fill="black")
        rendered.append(pair.resize((w * 4, (h + 22) * 4), Image.Resampling.NEAREST))
    sheet = Image.new("RGB", (rendered[0].width, sum(im.height for im in rendered)), "white")
    y = 0
    for image in rendered:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(out / "trained_vs_input_sheet.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("train_gsplat.py requires CUDA; use the CPU preview only for local smoke tests")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    frames, raw_points, raw_colors = load_capture(args.capture, args.frame_stride)
    points, colors = uniform_voxel_cloud(raw_points, raw_colors, args.voxel, args.max_points)
    model = GaussianModel(points, colors, args.voxel).to(device)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        train_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        train_model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rank_frames = list(range(rank, len(frames), world_size)) or [0]
    initial = torch.from_numpy(points).to(device)
    start = time.time()
    last_loss = float("inf")
    for step in range(args.steps):
        frame = frames[rank_frames[step % len(rank_frames)]]
        h, w = frame["depth"].shape
        view, K = make_camera(frame, device)
        target = torch.from_numpy(frame["rgb"].astype(np.float32) / 255.0).to(device)
        renders, _alphas, _meta = train_model(view[None], K[None], width=w, height=h)
        prediction = renders[0, ..., :3].clamp(0, 1)
        loss = (prediction - target).abs().mean()
        loss = loss + 1e-5 * (model.means - initial).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if rank == 0 and (step % 100 == 0 or step == args.steps - 1):
            print(f"step={step} loss={last_loss:.6f} points={len(points)}", flush=True)

    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        write_ply(args.out / "scene_trained_gsplat.splat.ply", model)
        render_views(model, frames, args.out, device)
        metadata = {
            "status": "passed",
            "trained": True,
            "trainer": "gsplat_cuda_ddp",
            "points": int(len(points)),
            "steps": int(args.steps),
            "world_size": world_size,
            "frame_stride": int(args.frame_stride),
            "voxel_m": float(args.voxel),
            "last_loss": last_loss,
            "seconds": round(time.time() - start, 2),
            "output": "scene_trained_gsplat.splat.ply",
            "physics_note": "Gaussian output is visual only; L3 mesh and L5 collision assets remain unchanged.",
            "limitations": [
                "RGB-D initialized cloud; unseen surfaces cannot be recovered",
                "this first trainer uses fixed point count and no opacity-based densification",
                "moving objects should be masked for a static-scene training run",
            ],
        }
        (args.out / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
