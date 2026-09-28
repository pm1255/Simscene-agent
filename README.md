# SimScene Agent

SimScene Agent 是一个面向**整场景重建、仿真导入和导航验证**的可审计 agent / harness 基线。输入是一段带相机标定的 RGB-D 场景序列，输出是可查看的房间级网格、带 RGB 纹理的部件资产、碰撞与关节模型、场景图、占据栅格和经过仿真复核的导航路径。

仓库的主 demo 严格按七层执行。每层都写出 JSON 合同和可复查的中间产物，L7 只接受独立检查通过的结果。

## 快速运行

```bash
cd simscene-agent
python3 -m venv .venv
.venv/bin/pip install numpy scipy scikit-image trimesh pillow mujoco pytest
PYTHONPATH=src .venv/bin/python -m simscene_agent.seven.run --out examples/generated/seven_layer_scene --scene living_room
```

同一条七层流程还提供办公室和走廊布局：

```bash
PYTHONPATH=src .venv/bin/python -m simscene_agent.seven.run --out examples/generated/seven_layer_office --scene office
PYTHONPATH=src .venv/bin/python -m simscene_agent.seven.run --out examples/generated/seven_layer_corridor --scene corridor
```

## Demo 项目

打开 [Demo 结果目录](https://github.com/pm1255/Simscene-agent/tree/main/examples/generated) 查看三个完整七层场景：

- [客厅场景](https://github.com/pm1255/Simscene-agent/tree/main/examples/generated/seven_layer_scene)：[导航预览](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_scene/L6_layout/navigation_path.png)，[七层验收摘要](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_scene/run_summary.json)
- [办公室场景](https://github.com/pm1255/Simscene-agent/tree/main/examples/generated/seven_layer_office)：[导航预览](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_office/L6_layout/navigation_path.png)，[七层验收摘要](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_office/run_summary.json)
- [走廊场景](https://github.com/pm1255/Simscene-agent/tree/main/examples/generated/seven_layer_corridor)：[导航预览](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_corridor/L6_layout/navigation_path.png)，[七层验收摘要](https://github.com/pm1255/Simscene-agent/blob/main/examples/generated/seven_layer_corridor/run_summary.json)

每个目录都对应同一条 L1–L7 流程，包含重建表面、部件纹理、MuJoCo 场景、碰撞代理、场景图、导航栅格和独立验收报告。仓库内提供 `scripts/generate_demos.py` 重新生成这些结果。

![七层导航路径](examples/generated/seven_layer_scene/L6_layout/navigation_path.png)

## 七层的实际职责

1. **L1 观测与坐标校验**：读取 RGB、米制深度、mask、内参和位姿，检查输入完整性与实例身份。
2. **L2 证据路由与预算**：按深度、位姿、mask 和用途选择重建、纹理、碰撞和补全 provider。
3. **L3 几何与外观融合**：把多帧深度投影到统一世界坐标，做 projective TSDF 融合并用 marching cubes 提取表面。
4. **L4 部件、补全与纹理**：按实例/部件生成独立 OBJ，利用可见性检查把输入 RGB 投影为纹理 atlas；隐藏几何明确标记为先验。
5. **L5 碰撞、质量、关节与仿真资产**：生成闭合碰撞代理、质量/摩擦先验和可加载的 MuJoCo XML，执行接触与关节检查。
6. **L6 场景图、占据图与导航**：从碰撞代理建立占据栅格、膨胀障碍区域，输出空间关系并用 A* 求导航路径。
7. **L7 独立验收与失败归因**：重新检查合同，在 MuJoCo 中回放路径，验收接触、终点误差和有限状态并输出修复建议。

## 结果目录

每个场景目录都会写出以下可复查结果：`run_summary.json`、`L0_spec.json`、`L1_observe/`、`L2_route/routes.json`、`L3_geometry/scene_surface.ply`、`L4_parts/`、`L5_simulation/scene.xml`、`L6_layout/scene_graph.json`、`L6_layout/navigation_path.png`、`L7_verify/verification.json`。

## 输入与替换 provider

输入合同位于 `L1_observe/capture/capture.json`，每帧包含 `rgb`、`depth`、`mask`、`K` 和 `T_world_camera`。默认 L3 provider 是 projective TSDF + marching cubes，L4 纹理来自输入 RGB 的可见性检查投影。3D Gaussian Splatting 可以作为外观或新视角 provider，但仿真碰撞仍需要独立的闭合代理。

## 测试

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

## License

Apache-2.0（详见 `LICENSE`）。
