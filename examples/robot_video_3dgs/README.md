# Real robot video → Gaussian visual + Mesh physics

The first real-data target for this repository is one episode from the public
[DROID dataset](https://droid-dataset.github.io/). DROID records a Franka Panda
arm with two exterior ZED stereo cameras and a wrist camera, and releases camera
calibration and robot state data. The official repository also documents a
small `droid_100` sample for debugging.

The raw episode is intentionally not vendored in this repository. Download it
from the official source and keep its CC-BY-4.0 attribution. The open-source
code in this repository remains Apache-2.0.

The adapter contract for a real episode is:

```text
rgb/                 # exterior RGB frames, time ordered
depth/               # stereo or RGB-D depth in metres
masks/robot/         # robot foreground masks
camera.json          # K and world-from-camera pose per frame
robot_state.json     # joint positions and timestamps
```

The seven-layer route then produces two independent outputs:

```text
L4_visual/scene.splat.ply       # visual Gaussian representation
L5_simulation/collision/*.obj   # static-scene collision meshes
L5_simulation/robot.urdf        # articulated robot collision/visual model
L5_simulation/scene.xml         # MuJoCo scene
L7_verify/robot_replay.mp4      # replay and verification evidence
```

Robot pixels must be masked before training a static background Gaussian. The
robot is rendered from its URDF using recorded joint states; this prevents
motion ghosts while preserving exact articulated geometry for simulation. The
current checked-in demos use the same dual-output interface with synthetic
calibrated RGB-D fixtures. Once CUDA PyTorch plus Nerfstudio/gsplat is installed,
`L4_visual/transforms.json` is the training input and the trained splat can
replace the bootstrap PLY without changing L5–L7.
