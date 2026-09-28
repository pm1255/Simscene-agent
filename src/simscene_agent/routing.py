from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RouteDecision:
    method: str
    reason: str
    required_outputs: list[str]


def route_asset(role: str, evidence: Dict, requirements: Dict) -> RouteDecision:
    """Deterministic baseline router; an LLM can propose overrides, but checks remain local."""
    multi_view = bool(evidence.get("multi_view"))
    articulated = bool(requirements.get("requires_joint")) or role == "articulated"
    internal = bool(requirements.get("requires_interior"))
    background = role == "background"
    if background and not requirements.get("requires_collision", False):
        return RouteDecision("3dgs_or_visual_mesh", "background object without physical interaction", ["visual"])
    if articulated:
        method = "multi_view_part_reconstruction" if multi_view else "single_view_articulated_generation"
        return RouteDecision(method, "articulation requires part structure and a kinematic hypothesis", ["visual", "parts", "joints", "collision"])
    if internal:
        return RouteDecision("parametric_or_mesh_generation_with_interior", "container function requires explicit free volume", ["visual", "parts", "interior", "collision"])
    if multi_view:
        return RouteDecision("multi_view_mesh_reconstruction", "observations provide geometric evidence", ["visual", "mesh", "collision"])
    return RouteDecision("single_view_to_mesh", "single-view input requires a learned shape prior", ["visual", "mesh", "collision"])
