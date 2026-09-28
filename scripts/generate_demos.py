from pathlib import Path
from simscene_agent.reconstruction import run_reconstruction
from simscene_agent.physics_demo import run_physics_demo
from simscene_agent.scene_reconstruction import run_scene_reconstruction
from simscene_agent.seven.pipeline import run_seven_layers

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]/"examples"/"generated"
    print("reconstruction",run_reconstruction(root/"reconstruction"))
    print("physics",run_physics_demo(root/"physics"))
    print("scene",run_scene_reconstruction(root/"scene_reconstruction"))
    print("seven_layers",run_seven_layers(root/"seven_layer_scene", "living_room"))
