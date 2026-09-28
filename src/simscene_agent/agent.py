from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .export import write_json, write_obj, write_urdf, write_usda
from .geometry import aabb_for_object, aabb_overlap, box_mesh, cylinder_mesh, validate_mesh
from .routing import route_asset
from .schemas import AssetSpec, Constraint, SceneObject, SceneSpec, TaskSpec
from .synth import generate_sequence


class SceneAgent:
    """Inspectable baseline controller.

    This baseline intentionally uses deterministic procedural assets. A GPT/VLM or
    image-to-3D provider can replace `plan_scene` and `build_asset`, while the scene
    schema, validators, and exporters stay unchanged.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def plan_scene(self, scene_id: str, description: str) -> SceneSpec:
        if scene_id == "kitchen":
            assets = [
                AssetSpec("cabinet_01", "cabinet", "articulated", "procedural", [1.8, .55, 1.7], "#b98555", parts=[
                    {"id": "door", "semantic": "door", "movable": True, "joint": {"type": "revolute", "axis": [0, 1, 0], "range_deg": [0, 90]}},
                    {"id": "shelf", "semantic": "shelf", "movable": False},
                ], physics={"mass_kg": 35, "static_friction": .5, "dynamic_friction": .4}, affordances=["open", "contain"]),
                AssetSpec("table_01", "table", "static", "procedural", [1.8, .8, .75], "#8b5e3c", physics={"mass_kg": 20, "static_friction": .6}, affordances=["support"]),
                AssetSpec("cup_01", "cup", "graspable", "procedural", [.12, .12, .14], "#60a5fa", physics={"mass_kg": .3, "static_friction": .4}, affordances=["grasp", "contain"]),
                AssetSpec("robot_01", "mobile_manipulator", "agent", "procedural", [.45, .45, 1.0], "#94a3b8"),
            ]
            objects = [SceneObject("cabinet_01", "cabinet_01", [1.8, 1.2, .85]), SceneObject("table_01", "table_01", [0.4, .1, .375]), SceneObject("cup_01", "cup_01", [0.4, .1, .85]), SceneObject("robot_01", "robot_01", [-1.2, .1, .5])]
            constraints = [Constraint("supported_by", "cup_01", "table_01", hard=True), Constraint("joint_openable", "cabinet_01", value=90, hard=True), Constraint("reachable", "robot_01", "cup_01", hard=True), Constraint("placeable", "cup_01", "cabinet_01", hard=True)]
            return SceneSpec(scene_id, description, [4.0, 3.0, 2.5], assets, objects, constraints, {"route": "procedural baseline; replaceable by image-to-3d"})
        if scene_id == "office":
            assets = [AssetSpec("desk_01", "desk", "static", "procedural", [2.2, .8, .75], "#c08457", physics={"mass_kg": 25}, affordances=["support"]), AssetSpec("monitor_01", "monitor", "static", "procedural", [1.0, .2, .65], "#94a3b8", physics={"mass_kg": 4}), AssetSpec("shelf_01", "shelf", "static", "procedural", [1.0, .45, 2.1], "#a78bfa", physics={"mass_kg": 18}, affordances=["contain"]), AssetSpec("robot_01", "mobile_manipulator", "agent", "procedural", [.45, .45, 1.0], "#f87171")]
            objects = [SceneObject("desk_01", "desk_01", [0.0, 0.0, .375]), SceneObject("monitor_01", "monitor_01", [0.0, 0.0, 1.05]), SceneObject("shelf_01", "shelf_01", [1.65, 1.0, 1.05]), SceneObject("robot_01", "robot_01", [-1.0, .0, .5])]
            constraints = [Constraint("supported_by", "monitor_01", "desk_01"), Constraint("reachable", "robot_01", "monitor_01")]
            return SceneSpec(scene_id, description, [4.0, 3.0, 2.5], assets, objects, constraints, {"route": "procedural baseline; reachability-ready"})
        assets = [AssetSpec("shelf_01", "shelf", "static", "procedural", [1.8, .45, 2.0], "#a16207", physics={"mass_kg": 18}, affordances=["contain"])]
        objects = [SceneObject("shelf_01", "shelf_01", [0.0, 0.0, 1.0])]
        for i, x in enumerate((-.55, -.2, .2, .55), 1):
            assets.append(AssetSpec(f"object_{i}", "object", "static", "procedural", [.18, .18, .22], "#60a5fa" if i % 2 else "#fb7185", physics={"mass_kg": .2}))
            objects.append(SceneObject(f"object_{i}", f"object_{i}", [x, 0.0, 1.35]))
        constraints = [Constraint("contained_by", f"object_{i}", "shelf_01") for i in range(1, 5)]
        return SceneSpec(scene_id, description, [3.0, 2.0, 2.5], assets, objects, constraints, {"route": "procedural baseline; containment-ready"})

    def build_asset(self, asset: AssetSpec, asset_dir: Path) -> dict:
        if asset.category == "cup":
            visual = cylinder_mesh(asset.dimensions[0] / 2, asset.dimensions[2])
            collision = cylinder_mesh(asset.dimensions[0] / 2, asset.dimensions[2], segments=12)
        else:
            visual = box_mesh(asset.dimensions)
            collision = box_mesh(asset.dimensions)
        report = validate_mesh(visual)
        write_obj(asset_dir / f"{asset.asset_id}.obj", visual, asset.color)
        write_obj(asset_dir / f"{asset.asset_id}_collision.obj", collision, asset.color)
        write_json(asset_dir / f"{asset.asset_id}.json", {"asset": asset.to_dict(), "geometry": report, "representation": {"visual_mesh": f"{asset.asset_id}.obj", "collision_mesh": f"{asset.asset_id}_collision.obj", "texture": None, "note": "procedural baseline; plug in image-to-3d and texture providers"}})
        write_urdf(asset_dir / f"{asset.asset_id}.urdf", asset)
        return report

    def validate_scene(self, scene: SceneSpec) -> dict:
        reports = []
        boxes = {}
        for obj in scene.objects:
            asset = next(a for a in scene.assets if a.asset_id == obj.asset_id)
            boxes[obj.object_id] = aabb_for_object(obj.position, asset.dimensions)
        for i, a in enumerate(scene.objects):
            for b in scene.objects[i + 1:]:
                if a.object_id.startswith("robot") or b.object_id.startswith("robot"):
                    continue
                if aabb_overlap(boxes[a.object_id], boxes[b.object_id]):
                    # Deliberately allow support contact, but flag other intersections.
                    support = any(c.kind in {"supported_by", "contained_by"} and c.subject == a.object_id and c.object == b.object_id for c in scene.constraints) or any(c.kind in {"supported_by", "contained_by"} and c.subject == b.object_id and c.object == a.object_id for c in scene.constraints)
                    if not support:
                        reports.append({"type": "aabb_overlap", "objects": [a.object_id, b.object_id]})
        missing = []
        for c in scene.constraints:
            ids = {o.object_id for o in scene.objects}
            if c.subject not in ids or (c.object and c.object not in ids):
                missing.append(c.to_dict())
        return {"passed": not reports and not missing, "collisions": reports, "missing_constraint_entities": missing, "constraint_count": len(scene.constraints)}

    def run(self, scene_id: str, description: str, frames: int = 8) -> Path:
        root = self.output_dir / scene_id
        root.mkdir(parents=True, exist_ok=True)
        scene = self.plan_scene(scene_id, description)
        assets_dir = root / "assets"
        asset_reports = {}
        for asset in scene.assets:
            asset_reports[asset.asset_id] = self.build_asset(asset, assets_dir)
        scene_report = self.validate_scene(scene)
        write_json(root / "scene.json", scene.to_dict())
        write_json(root / "validation.json", {"asset_reports": asset_reports, "scene_report": scene_report, "agent": {"controller": "deterministic-baseline", "next_extension": "LLM/VLM tool policy"}})
        write_usda(root / "scene.usda", scene)
        generate_sequence(scene_id, root, frames=frames)
        return root

    def run_all(self, frames: int = 8) -> List[Path]:
        configs = {
            "kitchen": "A kitchen with a hinged cabinet, a table, a cup, and a mobile manipulator.",
            "office": "An office with a desk, monitor, shelf, and a reachability path.",
            "shelf": "A shelf scene with four contained objects and support constraints.",
        }
        return [self.run(k, v, frames=frames) for k, v in configs.items()]
