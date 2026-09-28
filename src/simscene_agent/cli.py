from __future__ import annotations

import argparse
from pathlib import Path

from .agent import SceneAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inspectable sim-ready scene examples.")
    parser.add_argument("--scene", choices=["kitchen", "office", "shelf", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    agent = SceneAgent(args.out)
    paths = agent.run_all(args.frames) if args.scene == "all" else [agent.run(args.scene, args.scene, args.frames)]
    for path in paths:
        print(path)

if __name__ == "__main__":
    main()
