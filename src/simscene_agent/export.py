from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from .geometry import Mesh
from .schemas import AssetSpec, SceneSpec


def write_obj(path: Path, mesh: Mesh, color: str = "#b8b8b8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# SimScene generated mesh; color={color}\n")
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_usda(path: Path, scene: SceneSpec) -> None:
    """Write a small, readable USD-like manifest for interoperability demos.

    It intentionally keeps the source scene metadata explicit; production adapters can
    map this representation to full UsdPhysics schemas for a chosen simulator.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#usda 1.0", "(", '    doc = "SimScene generated scene manifest"', ")", "", 'def Xform "Scene" {']
    for obj in scene.objects:
        p = ", ".join(f"{v:.4f}" for v in obj.position)
        lines.extend([
            f'    def Xform "{obj.object_id}" {{',
            f'        string assetId = "{obj.asset_id}"',
            f'        double3 xformOp:translate = ({p})',
            f'        custom string role = "{next((a.role for a in scene.assets if a.asset_id == obj.asset_id), "static")}"',
            "    }",
        ])
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_urdf(path: Path, asset: AssetSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    links = [f'  <link name="{asset.asset_id}_body">', '    <inertial><mass value="1.0"/></inertial>', "  </link>"]
    for part in asset.parts:
        if part.get("movable"):
            links.append(f'  <link name="{part["id"]}"/>')
            joint = part.get("joint", {})
            links.append(
                f'  <joint name="{part["id"]}_joint" type="{joint.get("type", "revolute")}">'
                f'<parent link="{asset.asset_id}_body"/><child link="{part["id"]}"/>'
                f'<axis xyz="{" ".join(map(str, joint.get("axis", [0, 1, 0])))}"/>\n  </joint>'
            )
    path.write_text("<?xml version=\"1.0\"?>\n<robot name=\"%s\">\n%s\n</robot>\n" % (asset.asset_id, "\n".join(links)), encoding="utf-8")
