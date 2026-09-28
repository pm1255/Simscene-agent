"""Contracts shared by all seven stages. Metres, world Z-up, camera CV axes."""
import json
import hashlib
from pathlib import Path
import numpy as np


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera(position, target):
    pos = np.asarray(position, dtype=float)
    forward = np.asarray(target) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, forward], axis=1)
    T[:3, 3] = pos
    return T


def backproject(depth, K, T):
    v, u = np.indices(depth.shape)
    rays = np.stack([(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)], -1)
    return (rays * depth[..., None]) @ T[:3, :3].T + T[:3, 3]


def project(points, K, T):
    local = (points-T[:3, 3]) @ T[:3, :3]
    z = local[:, 2]
    uv = local[:, :2] / np.maximum(z[:, None], 1e-8)
    uv = uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def ray_boxes(origin, rays, boxes):
    """Unnormalised camera rays: returned t is camera-Z depth, not ray range."""
    depth = np.full(rays.shape[:-1], np.inf)
    ids = np.zeros(depth.shape, np.uint16)
    for b in boxes:
        lo, hi = np.asarray(b['bounds'])
        parallel = np.abs(rays) < 1e-10
        outside = parallel & ((origin < lo) | (origin > hi))
        safe = np.where(parallel, 1., rays)
        a, c = (lo-origin)/safe, (hi-origin)/safe
        low = np.where(parallel, -np.inf, np.minimum(a, c))
        high = np.where(parallel, np.inf, np.maximum(a, c))
        near, far = low.max(-1), high.min(-1)
        hit = np.where(near > 1e-5, near, far)
        valid = (~outside.any(-1)) & (far >= np.maximum(near, 0)) & (hit > 1e-5)
        use = valid & (hit < depth)
        depth[use] = hit[use]
        ids[use] = b['id']
    depth[~np.isfinite(depth)] = 0
    return depth.astype(np.float32), ids
