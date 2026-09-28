"""Whole-room RGB-D reconstruction and navigation baseline.

The demo uses a calibrated pinhole RGB-D sequence rendered from a small room made
of boxes.  It then *reconstructs* a colored surface point cloud by back-projecting
depth, voxelizes the observations, and derives an inflated 2-D occupancy map and
an A* path.  The provider is intentionally simple and inspectable; real captures
can replace ``render_sequence`` while keeping the fusion and navigation harness.
"""
from __future__ import annotations

import heapq, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[int, int, int]
    navigable: bool = False


ROOM = (8.0, 6.0, 3.0)
OBJECTS = [
    Box("floor", (4.0, 3.0, -0.05), (8.0, 6.0, 0.10), (205, 198, 180), True),
    Box("north_wall", (4.0, 5.95, 1.5), (8.0, 0.10, 3.0), (225, 230, 236)),
    Box("south_wall", (4.0, 0.05, 1.5), (8.0, 0.10, 3.0), (225, 230, 236)),
    Box("west_wall", (0.05, 3.0, 1.5), (0.10, 6.0, 3.0), (225, 230, 236)),
    Box("east_wall", (7.95, 3.0, 1.5), (0.10, 6.0, 3.0), (225, 230, 236)),
    Box("table", (2.1, 2.0, 0.68), (1.8, 1.1, 1.36), (143, 91, 52)),
    Box("sofa", (5.6, 1.45, 0.48), (2.0, 1.35, 0.96), (94, 112, 145)),
    Box("cabinet", (5.8, 4.25, 1.0), (1.25, 0.65, 2.0), (170, 122, 70)),
    Box("pillar", (3.9, 4.0, 1.1), (0.55, 0.55, 2.2), (120, 128, 136)),
]


def _look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - position; forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    return right, up, forward


def _camera_rays(position: np.ndarray, target: np.ndarray, width: int, height: int, focal: float) -> np.ndarray:
    right, up, forward = _look_at(position, target)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x=(u-(width-1)/2)/focal; y=-((v-(height-1)/2)/focal)
    rays=forward[None,None,:] + x[...,None]*right[None,None,:] + y[...,None]*up[None,None,:]
    return rays/np.linalg.norm(rays,axis=2,keepdims=True)


def _ray_box(origin: np.ndarray, rays: np.ndarray, box: Box) -> np.ndarray:
    lo=np.asarray(box.center)-np.asarray(box.size)/2; hi=np.asarray(box.center)+np.asarray(box.size)/2
    inv=1.0/np.where(np.abs(rays)<1e-8, np.sign(rays)*1e-8+1e-8, rays)
    t1=(lo-origin)*inv; t2=(hi-origin)*inv
    near=np.max(np.minimum(t1,t2),axis=2); far=np.min(np.maximum(t1,t2),axis=2)
    return np.where((far>=np.maximum(near,0)) & (far>0), np.maximum(near,0), np.inf)


def render_sequence(out_dir: Path, width: int = 160, height: int = 120, frames: int = 10) -> list[dict]:
    """Render calibrated RGB-D observations of the room and persist each frame."""
    from PIL import Image
    seq=out_dir/"input_rgbd"; seq.mkdir(parents=True,exist_ok=True)
    records=[]; target=np.array([4.0,3.0,1.0])
    for i in range(frames):
        # Cameras stay inside the room so the walls do not occlude the entire view.
        a=2*math.pi*i/frames + 0.15; pos=np.array([4+2.4*math.cos(a),3+1.7*math.sin(a),2.4])
        rays=_camera_rays(pos,target,width,height,145.0); depth=np.full((height,width),np.inf); rgb=np.zeros((height,width,3),dtype=np.uint8)
        for box in OBJECTS:
            hit=_ray_box(pos,rays,box); update=hit<depth; depth[update]=hit[update]; rgb[update]=box.color
        valid=np.isfinite(depth); depth[~valid]=0
        Image.fromarray(rgb).save(seq/f"rgb_{i:03d}.png")
        np.save(seq/f"depth_{i:03d}.npy",depth.astype(np.float32))
        records.append({"rgb":f"rgb_{i:03d}.png","depth":f"depth_{i:03d}.npy","position_m":pos.tolist(),"target_m":target.tolist(),"fx":145.0,"fy":145.0,"cx":(width-1)/2,"cy":(height-1)/2})
    (out_dir/"calibration.json").write_text(json.dumps({"width":width,"height":height,"frames":records},indent=2)+"\n")
    return records


def _backproject(depth: np.ndarray, record: dict) -> tuple[np.ndarray, np.ndarray]:
    h,w=depth.shape; u,v=np.meshgrid(np.arange(w),np.arange(h)); valid=depth>0
    # Camera-local pinhole coordinates. Camera orientation is reconstructed by look-at.
    x=(u-record["cx"])/record["fx"]*depth; y=-(v-record["cy"])/record["fy"]*depth; z=depth
    pos=np.asarray(record["position_m"]); right,up,forward=_look_at(pos,np.asarray(record["target_m"]))
    points=pos + x[...,None]*right + y[...,None]*up + z[...,None]*forward
    return points[valid], valid


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w",encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p,c in zip(points,colors): f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w",encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for p,c in zip(vertices,colors): f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for a,b,c in faces: f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def _astar(occupied: np.ndarray, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
    h,w=occupied.shape; q=[(0,start)]; came={}; cost={start:0}; dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    def heur(p): return abs(p[0]-goal[0])+abs(p[1]-goal[1])
    while q:
        _,cur=heapq.heappop(q)
        if cur==goal:
            path=[]
            while cur in came: path.append(cur); cur=came[cur]
            return [start]+path[::-1]
        for di,dj in dirs:
            nxt=(cur[0]+di,cur[1]+dj)
            if not(0<=nxt[0]<w and 0<=nxt[1]<h) or occupied[nxt[1],nxt[0]]: continue
            nc=cost[cur]+1
            if nc<cost.get(nxt,10**9): cost[nxt]=nc; came[nxt]=cur; heapq.heappush(q,(nc+heur(nxt),nxt))
    return []


def run_scene_reconstruction(out_dir: Path, cell: float = 0.10, source_dir: Path | None = None) -> dict:
    from PIL import Image, ImageDraw
    from .geometry import box_mesh
    from .export import write_obj
    out_dir.mkdir(parents=True,exist_ok=True)
    if source_dir is None:
        records=render_sequence(out_dir)
    else:
        import shutil
        shutil.copytree(Path(source_dir)/"input_rgbd", out_dir/"input_rgbd", dirs_exist_ok=True)
        calibration=json.loads((Path(source_dir)/"calibration.json").read_text())
        (out_dir/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
        records=calibration["frames"]
    all_points=[]; all_colors=[]
    for rec in records:
        depth=np.load(out_dir/"input_rgbd"/rec["depth"]); rgb=np.asarray(Image.open(out_dir/"input_rgbd"/rec["rgb"]))
        pts,valid=_backproject(depth,rec); all_points.append(pts); all_colors.append(rgb[valid])
    points=np.concatenate(all_points); colors=np.concatenate(all_colors)
    # Fuse repeated observations into one colored voxel surface point cloud.
    ijk=np.floor(points/cell).astype(int); unique,inv=np.unique(ijk,axis=0,return_inverse=True)
    fused=np.zeros((len(unique),3)); fused_col=np.zeros((len(unique),3)); counts=np.bincount(inv)
    np.add.at(fused,inv,points); np.add.at(fused_col,inv,colors); fused/=counts[:,None]; fused_col=np.clip(fused_col/counts[:,None],0,255).astype(np.uint8)
    _write_ply(out_dir/"scene_surface.ply",fused,fused_col)
    # Turn the fused occupied voxels into a lightweight editable cube mesh. This
    # is an observed-surface proxy; production adapters can replace it with TSDF
    # marching cubes while keeping the same scene coordinate frame.
    occupied={tuple(v):i for i,v in enumerate(unique)}; corners={}
    directions=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    face_corners={(1,0,0):[(1,0,0),(1,1,0),(1,1,1),(1,0,1)],(-1,0,0):[(0,0,0),(0,0,1),(0,1,1),(0,1,0)],(0,1,0):[(0,1,0),(0,1,1),(1,1,1),(1,1,0)],(0,-1,0):[(0,0,0),(1,0,0),(1,0,1),(0,0,1)],(0,0,1):[(0,0,1),(1,0,1),(1,1,1),(0,1,1)],(0,0,-1):[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]}
    mv=[]; mc=[]; mf=[]; mind={}
    for idx,color in zip(unique,fused_col):
        for d in directions:
            if tuple(idx+np.asarray(d)) in occupied: continue
            ids=[]
            for corner in face_corners[d]:
                key=tuple((idx+np.asarray(corner))*cell)
                if key not in mind: mind[key]=len(mv); mv.append(key); mc.append(color)
                ids.append(mind[key])
            mf.extend([(ids[0],ids[1],ids[2]),(ids[0],ids[2],ids[3])])
    _write_mesh_ply(out_dir/"scene_mesh.ply",np.asarray(mv),np.asarray(mf),np.asarray(mc,dtype=np.uint8))
    # A navigation map is derived only from fused geometry above the floor.
    nx,ny=int(ROOM[0]/cell),int(ROOM[1]/cell); obstacle=np.zeros((ny,nx),dtype=bool)
    high=(fused[:,2]>0.18)&(fused[:,2]<2.3)
    for x,y in fused[high,:2]:
        ix,iy=int(x/cell),int(y/cell)
        if 0<=ix<nx and 0<=iy<ny: obstacle[iy,ix]=True
    # Inflate for a 0.35m radius robot; this is the navigation safety margin.
    radius=max(1,int(0.35/cell)); inflated=obstacle.copy()
    for y,x in zip(*np.where(obstacle)):
        inflated[max(0,y-radius):min(ny,y+radius+1),max(0,x-radius):min(nx,x+radius+1)]=True
    start=(int(0.8/cell),int(0.8/cell)); goal=(int(6.8/cell),int(4.8/cell)); path=_astar(inflated,start,goal)
    nav=np.where(inflated,0,255).astype(np.uint8); Image.fromarray(nav).resize((nx*4,ny*4),resample=Image.Resampling.NEAREST).save(out_dir/"navigation_occupancy.png")
    navrgb=np.stack([nav,nav,nav],axis=2); d=ImageDraw.Draw(Image.fromarray(navrgb).resize((nx*4,ny*4),resample=Image.Resampling.NEAREST)); navim=Image.fromarray(navrgb).resize((nx*4,ny*4),resample=Image.Resampling.NEAREST); d=ImageDraw.Draw(navim)
    if path:
        d.line([(x*4+2,y*4+2) for x,y in path],fill=(220,45,45),width=5); d.ellipse((start[0]*4-5,start[1]*4-5,start[0]*4+5,start[1]*4+5),fill=(20,150,70)); d.ellipse((goal[0]*4-5,goal[1]*4-5,goal[0]*4+5,goal[1]*4+5),fill=(40,80,220))
    navim.save(out_dir/"navigation_path.png")
    (out_dir/"navigation_path.json").write_text(json.dumps({"cell_size_m":cell,"start_xy_cell":start,"goal_xy_cell":goal,"path_xy_cells":path},indent=2)+"\n")
    report={"task":"whole_scene_rgbd_reconstruction_for_navigation","input_frames":len(records),"input_resolution":[160,120],"raw_points":int(len(points)),"fused_surface_voxels":int(len(fused)),"mesh_vertices":int(len(mv)),"mesh_triangles":int(len(mf)),"voxel_size_m":cell,"room_size_m":ROOM,"outputs":["scene_surface.ply","scene_mesh.ply","navigation_occupancy.png","navigation_path.png"],"navigation":{"robot_radius_m":0.35,"start_xy_m":[start[0]*cell,start[1]*cell],"goal_xy_m":[goal[0]*cell,goal[1]*cell],"path_cells":len(path),"path_found":bool(path)},"limitations":["synthetic calibrated RGB-D renderer is used for reproducibility","semantic labels come from the scene manifest, not an open-vocabulary detector","voxel fusion produces an observed surface; unseen interiors need a learned or procedural completion provider","RGB-D alone does not identify mass or friction"]}
    (out_dir/"scene_reconstruction_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out_dir/"scene_manifest.json").write_text(json.dumps({"room_size_m":ROOM,"objects":[{"name":b.name,"center_m":b.center,"size_m":b.size,"color_rgb":b.color,"navigable":b.navigable} for b in OBJECTS]},indent=2)+"\n")
    asset_dir=out_dir/"assets"; asset_dir.mkdir(exist_ok=True)
    for b in OBJECTS:
        write_obj(asset_dir/f"{b.name}.obj",box_mesh(b.size,b.center),"#%02x%02x%02x"%b.color)
    return report
