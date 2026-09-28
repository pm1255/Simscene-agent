from pathlib import Path
from simscene_agent.agent import SceneAgent

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "runs"
    for path in SceneAgent(root).run_all(frames=8):
        print(f"generated {path}")
