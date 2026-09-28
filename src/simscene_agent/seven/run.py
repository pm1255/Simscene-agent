from pathlib import Path
from .pipeline import run_seven_layers
if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path('examples/generated/seven_layer_scene')); p.add_argument('--scene',default='living_room',choices=['living_room','office','corridor']); a=p.parse_args()
    print(run_seven_layers(a.out,a.scene))
