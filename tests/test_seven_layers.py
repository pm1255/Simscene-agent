import json
from pathlib import Path

import pytest

from simscene_agent.seven.pipeline import run_seven_layers


@pytest.mark.parametrize("scene", ["living_room", "office", "corridor"])
def test_all_seven_layers_complete_with_physics_rollout(tmp_path: Path, scene: str):
    summary_path = run_seven_layers(tmp_path / "seven", scene)
    summary = json.loads(summary_path.read_text())

    assert summary["status"] == "passed"
    assert summary["layer_count"] == 7
    assert all(check["passed"] for check in summary["checks"])

    simulation = json.loads((summary_path.parent / "L5_simulation" / "simulation.json").read_text())
    assert simulation["runtime"]["njnt"] >= 1
    assert simulation["runtime"]["floor_contacts"] > 0

    rollout = json.loads((summary_path.parent / "L7_verify" / "navigation_rollout.json").read_text())
    assert rollout["passed"]
    assert rollout["obstacle_contact_events"] == 0

    visual = json.loads((summary_path.parent / "L4_visual" / "visual.json").read_text())
    assert visual["provider"] == "rgbd_gaussian_bootstrap"
    assert visual["trained"] is False
    assert visual["point_count"] > 0
    assert (summary_path.parent / "L4_visual" / "scene.splat.ply").exists()
    assert (summary_path.parent / "L4_visual" / "gaussian_preview.png").exists()
