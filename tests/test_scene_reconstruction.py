import json
from pathlib import Path
from simscene_agent.scene_reconstruction import run_scene_reconstruction

def test_whole_scene_reconstruction_has_navigation_path(tmp_path: Path):
    report = run_scene_reconstruction(tmp_path / "scene")
    assert report["task"] == "whole_scene_rgbd_reconstruction_for_navigation"
    assert report["fused_surface_voxels"] > 1000
    assert report["navigation"]["path_found"]
    assert (tmp_path / "scene" / "scene_surface.ply").exists()
    assert (tmp_path / "scene" / "scene_mesh.ply").exists()
    assert json.loads((tmp_path / "scene" / "scene_reconstruction_report.json").read_text())["input_frames"] == 10
