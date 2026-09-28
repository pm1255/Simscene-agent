from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class Mesh:
    vertices: List[Vec3]
    faces: List[Tuple[int, int, int]]

    def bounds(self) -> Tuple[Vec3, Vec3]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def extent(self) -> Vec3:
        lo, hi = self.bounds()
        return hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]


def box_mesh(size: Vec3, center: Vec3 = (0.0, 0.0, 0.0)) -> Mesh:
    sx, sy, sz = (v / 2.0 for v in size)
    cx, cy, cz = center
    vertices = [
        (cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return Mesh(vertices, faces)


def cylinder_mesh(radius: float, height: float, segments: int = 24) -> Mesh:
    vertices: List[Vec3] = []
    for z in (-height / 2, height / 2):
        for i in range(segments):
            a = 2 * math.pi * i / segments
            vertices.append((radius * math.cos(a), radius * math.sin(a), z))
    vertices.extend([(0.0, 0.0, -height / 2), (0.0, 0.0, height / 2)])
    bottom, top = 2 * segments, 2 * segments + 1
    faces: List[Tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([(i, j, segments + j), (i, segments + j, segments + i)])
        faces.append((bottom, j, i))
        faces.append((top, segments + i, segments + j))
    return Mesh(vertices, faces)


def validate_mesh(mesh: Mesh) -> dict:
    degenerate = 0
    for a, b, c in mesh.faces:
        pa, pb, pc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
        ab = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        ac = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        area2 = math.sqrt(sum(x * x for x in cross))
        if area2 < 1e-10:
            degenerate += 1
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "degenerate_faces": degenerate,
        "finite_vertices": all(math.isfinite(x) for v in mesh.vertices for x in v),
        "bounds": {"min": mesh.bounds()[0], "max": mesh.bounds()[1]},
        "valid": degenerate == 0 and all(math.isfinite(x) for v in mesh.vertices for x in v),
    }


def aabb_for_object(position: Iterable[float], dimensions: Iterable[float]) -> dict:
    p = list(position)
    d = list(dimensions)
    return {
        "min": [p[i] - d[i] / 2 for i in range(3)],
        "max": [p[i] + d[i] / 2 for i in range(3)],
    }


def aabb_overlap(a: dict, b: dict) -> bool:
    return all(a["min"][i] < b["max"][i] and b["min"][i] < a["max"][i] for i in range(3))
