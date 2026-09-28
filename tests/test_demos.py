import json
from pathlib import Path
from simscene_agent.reconstruction import run_reconstruction
from simscene_agent.physics_demo import run_physics_demo

def test_reconstruction_and_physics_demos(tmp_path: Path):
    recon = run_reconstruction(tmp_path / "recon", grid=20)
    physics = run_physics_demo(tmp_path / "physics", steps=60)
    assert recon["vertices"] > 0 and recon["triangles"] > 0
    assert physics["stable_support_passed"]
    assert physics["door_joint_limit_passed"]
    assert json.loads((tmp_path / "recon" / "reconstruction_report.json").read_text())["input_images"] == 12
