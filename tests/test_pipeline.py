import json
from pathlib import Path

from simscene_agent.agent import SceneAgent
from simscene_agent.geometry import box_mesh, validate_mesh


def test_box_mesh_is_valid():
    report = validate_mesh(box_mesh((1, 2, 3)))
    assert report["valid"]
    assert report["triangles"] == 12


def test_examples_generate(tmp_path: Path):
    root = SceneAgent(tmp_path).run("kitchen", "test", frames=2)
    assert (root / "scene.json").exists()
    assert (root / "scene.usda").exists()
    assert len(list((root / "image_sequence").glob("*.svg"))) == 2
    report = json.loads((root / "validation.json").read_text())
    assert "scene_report" in report
