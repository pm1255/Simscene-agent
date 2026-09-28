from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AssetSpec:
    asset_id: str
    category: str
    role: str = "static"
    source: str = "procedural"
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: str = "#b8b8b8"
    parts: List[Dict[str, Any]] = field(default_factory=list)
    physics: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)
    confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneObject:
    object_id: str
    asset_id: str
    position: List[float]
    rotation_z: float = 0.0
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Constraint:
    kind: str
    subject: str
    object: Optional[str] = None
    value: Any = None
    hard: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    task_id: str
    description: str
    requirements: List[Constraint] = field(default_factory=list)
    target_simulator: str = "generic"
    robot: Optional[Dict[str, Any]] = None
    budget: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "requirements": [r.to_dict() for r in self.requirements],
            "target_simulator": self.target_simulator,
            "robot": self.robot,
            "budget": self.budget,
        }


@dataclass
class SceneSpec:
    scene_id: str
    description: str
    room_size: List[float]
    assets: List[AssetSpec] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "description": self.description,
            "room_size": self.room_size,
            "assets": [a.to_dict() for a in self.assets],
            "objects": [o.to_dict() for o in self.objects],
            "constraints": [c.to_dict() for c in self.constraints],
            "metadata": self.metadata,
        }
