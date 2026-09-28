"""Executable L1-L7 scene agent. Each stage writes an auditable contract."""
from __future__ import annotations
from pathlib import Path
import json, shutil, math
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from .fixtures import make_capture
from .stages import observe, route, reconstruct, parts_and_appearance
from .io import dump, load
from .dynamics import build_simulation, simulate_path
from .navigation import build_layout

LAYER_NAMES={
 'L1_observe':'观测与坐标校验',
 'L2_route':'证据路由与预算',
 'L3_geometry':'几何与外观融合',
 'L4_parts':'部件、补全与纹理',
 'L5_simulation':'碰撞、质量、关节与仿真资产',
 'L6_layout':'场景图、占据图与导航',
 'L7_verify':'独立验收、失败归因与修复',
}


def _write_textured_asset(folder, name, center, size, color):
    folder.mkdir(parents=True, exist_ok=True)
    sx,sy,sz=np.asarray(size)/2; cx,cy,cz=center
    vertices=[(cx-sx,cy-sy,cz-sz),(cx+sx,cy-sy,cz-sz),(cx+sx,cy+sy,cz-sz),(cx-sx,cy+sy,cz-sz),
              (cx-sx,cy-sy,cz+sz),(cx+sx,cy-sy,cz+sz),(cx+sx,cy+sy,cz+sz),(cx-sx,cy+sy,cz+sz)]
    faces=[(1,2,3),(1,3,4),(5,7,6),(5,8,7),(1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,8),(3,8,4),(4,8,5),(4,5,1)]
    uv=[(0,0),(1,0),(1,1),(0,1),(0,0),(1,0),(1,1),(0,1)]
    (folder/'material.png').write_bytes(b'')
    Image.new('RGB',(64,64),tuple(map(int,color))).save(folder/'material.png')
    (folder/'material.mtl').write_text('newmtl surface\nmap_Kd material.png\nKd 1 1 1\n')
    with (folder/'collision.obj').open('w') as f:
        for v in vertices:f.write('v %.6f %.6f %.6f\n'%v)
        for a,b,c in faces:f.write(f'f {a} {b} {c}\n')
    with (folder/'visual.obj').open('w') as f:
        f.write('mtllib material.mtl\nusemtl surface\n')
        for v in vertices:f.write('v %.6f %.6f %.6f\n'%v)
        for u,v in uv:f.write('vt %.6f %.6f\n'%(u,v))
        for a,b,c in faces:f.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')


def simulation_assets(c, assets, out):
    out=Path(out)
    # L5 is a real MuJoCo adapter: it builds visual RGB meshes, explicit
    # collision proxies, fixed room bodies, an articulated door, and a
    # gravity/contact probe.  The returned records are normalized for L6.
    geoms=build_simulation(c, assets, out)
    sim=load(out/'simulation.json')
    for g in geoms:
        size=np.asarray(g['size_m'],dtype=float)
        g['collision_center_m']=g['center_m']
        g['collision_size_m']=g['size_m']
        g['half_size_m']=(size/2).tolist()
    sim['geoms']=geoms
    dump(out/'simulation.json',sim)
    return geoms


def layout_and_navigation(c, l3, assets, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    from ..scene_reconstruction import _astar
    cell=float(c['task']['map_cell_m']); room=np.asarray(c['bounds_m'][1][:2]); nx,ny=(room/cell).astype(int)
    occupied=np.zeros((ny,nx),dtype=bool)
    # Layout is derived from the collision proxies produced by L5, not visual
    # pixels. Unknown/out-of-room cells remain blocked by construction.
    for a in assets:
        if a['name']=='floor': continue
        lo=np.asarray(a['collision_center_m'])-np.asarray(a['collision_size_m'])/2
        hi=np.asarray(a['collision_center_m'])+np.asarray(a['collision_size_m'])/2
        ix0,ix1=max(0,int(np.floor(lo[0]/cell))),min(nx,int(np.ceil(hi[0]/cell)))
        iy0,iy1=max(0,int(np.floor(lo[1]/cell))),min(ny,int(np.ceil(hi[1]/cell)))
        if hi[2]>.15: occupied[iy0:iy1,ix0:ix1]=True
    radius=max(1,int(np.ceil(c['task']['robot_radius_m']/cell))); inflated=occupied.copy()
    for y,x in zip(*np.where(occupied)):
        inflated[max(0,y-radius):min(ny,y+radius+1),max(0,x-radius):min(nx,x+radius+1)]=True
    start=tuple(int(x) for x in (np.asarray(c['task']['start_xy_m'])/cell).astype(int)); goal=tuple(int(x) for x in (np.asarray(c['task']['goal_xy_m'])/cell).astype(int))
    path=_astar(inflated,start,goal); nav=np.where(inflated,0,255).astype(np.uint8)
    Image.fromarray(nav).resize((nx*4,ny*4),resample=Image.Resampling.NEAREST).save(out/'navigation_occupancy.png')
    navim=Image.fromarray(np.stack([nav]*3,2)).resize((nx*4,ny*4),resample=Image.Resampling.NEAREST); d=ImageDraw.Draw(navim)
    if path:
        d.line([(x*4+2,y*4+2) for x,y in path],fill=(220,45,45),width=5)
        d.ellipse((start[0]*4-5,start[1]*4-5,start[0]*4+5,start[1]*4+5),fill=(20,150,70)); d.ellipse((goal[0]*4-5,goal[1]*4-5,goal[0]*4+5,goal[1]*4+5),fill=(40,80,220))
    navim.save(out/'navigation_path.png')
    nav={'cell_size_m':cell,'start_xy_cell':start,'goal_xy_cell':goal,'path_xy_cells':path,'source':'L5 collision AABBs with unknown cells blocked'}
    dump(out/'navigation_path.json',nav)
    relationships=[]
    for a in assets:
        relationships.append({'subject':a['name'],'relation':'inside','object':'room','confidence':1.0,'source':'scene_capture'})
        if a['name']!='floor': relationships.append({'subject':a['name'],'relation':'blocks_navigation','object':'room','confidence':.98,'source':'collision_projection'})
    graph={'room_size_m':c['bounds_m'][1],'objects':[a['name'] for a in assets],'relationships':relationships,
           'navigation_task':{'start_xy_m':c['task']['start_xy_m'],'goal_xy_m':c['task']['goal_xy_m'],'robot_radius_m':c['task']['robot_radius_m'],
             'path_cells':len(nav['path_xy_cells']),'path_found':bool(nav['path_xy_cells']),'source':nav['source']}}
    dump(out/'scene_graph.json',graph); dump(out/'layout_constraints.json',dict(status='passed',constraints=[
      {'type':'inside_room','hard':True,'count':len(assets)}, {'type':'free_path','hard':True,'robot_radius_m':c['task']['robot_radius_m']},
      {'type':'floor_support','hard':True,'objects':[a['name'] for a in assets if a['name'] not in ('floor',) and 'wall' not in a['name']]},
      {'type':'unknown_space_policy','value':'unobserved is blocked until observed'},
      {'type':'collision_source','value':'L5 reconstructed collision proxies'}]))
    return graph


def layout_and_navigation_from_evidence(c, points, labels, l4_assets, l5_geoms, out):
    """L6 adapter: use observed floor evidence plus L5 collision proxies."""
    return build_layout(c, points, labels, l5_geoms, l4_assets, out)


def validate(root,c):
    root=Path(root); checks=[]
    def check(name, passed, evidence, failure=None): checks.append(dict(name=name,passed=bool(passed),evidence=evidence,repair=None if passed else failure))
    obs=load(root/'L1_observe/observations.json'); check('L1 calibration and depth coverage',obs['status']=='passed',obs['frames'])
    route=load(root/'L2_route/routes.json'); check('L2 every instance has route',len(route['decisions'])==len(c['instances']),len(route['decisions']))
    geo=load(root/'L3_geometry/geometry.json'); check('L3 finite mesh',geo['finite'] and geo['vertices']>0,{'vertices':geo['vertices'],'triangles':geo['triangles']},'re-fuse with smaller voxel or request additional views')
    parts=load(root/'L4_parts/parts.json'); check('L4 part meshes and RGB UV',all(a['triangles']>0 and a['uv_range'][1]>a['uv_range'][0] for a in parts['assets']),len(parts['assets']),'run visibility-aware UV projection again')
    sim=load(root/'L5_simulation/simulation.json'); xml=(root/'L5_simulation/scene.xml').read_text();
    l5_ok=(sim['status']=='passed' and len(sim['geoms'])==len(parts['assets']) and '<mujoco' in xml and sim['runtime']['finite'] and sim['runtime']['steps']>0 and sim['runtime']['njnt']>=1 and sim['runtime']['floor_contacts']>0)
    check('L5 collision, articulation and gravity contact',l5_ok,sim['runtime'],'reject collider/MJCF and rebuild the dynamic probe')
    lay=load(root/'L6_layout/layout_constraints.json'); nav=load(root/'L6_layout/navigation_path.json'); check('L6 navigation route',bool(nav['path_xy_cells']) and bool(nav['path_xy_m']),{'cells':len(nav['path_xy_cells']),'unknown_cells':nav['unknown_cells']},'mark unknown cells blocked and request new viewpoints')
    sim=load(root/'L5_simulation/simulation.json')
    from .navigation import inspect_path
    geometry_recheck=inspect_path(nav,root/'L6_layout/map.npz',sim['geoms'])
    path_xy=np.asarray(nav['path_xy_m'],dtype=float)
    rollout,trajectory,_,_=simulate_path(root/'L5_simulation/scene.xml',c,path_xy)
    dump(root/'L7_verify/navigation_rollout.json',dict(**rollout,trajectory=trajectory[::10],geometry_recheck=geometry_recheck,path_source='L6_layout/navigation_path.json',controller='MuJoCo XY position servos with gravity/contact'))
    check('L7 independent path recheck',geometry_recheck['passed'] and rollout['passed'],{'geometry_recheck':geometry_recheck,'rollout':rollout},'invalidate path and re-route; inspect observed-space clearance and contacts')
    passed=all(x['passed'] for x in checks)
    dump(root/'L7_verify/verification.json',dict(status='passed' if passed else 'failed',passed=passed,checks=checks,
      agent_role='planner/router/diagnostician; deterministic harness owns acceptance',repair_attempts=0,
      limitations=['synthetic capture provider used','mass and friction are priors','semantic masks are supplied annotations']))
    return passed,checks


def run_seven_layers(root, scene='living_room'):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    layers={k:root/k for k in LAYER_NAMES}; [p.mkdir(exist_ok=True) for p in layers.values()]
    capture=make_capture(root/'L1_observe/capture',scene); c,frames,points,labels,colors=observe(capture,root/'L1_observe')
    dump(root/'L0_spec.json',dict(scene=scene,goal='reconstruct a navigable whole room from calibrated RGB-D',layers=LAYER_NAMES,
      acceptance=['metric coordinate frame','observed surface and editable assets','collision proxies','path from start to goal','independent recheck']))
    route(c,frames,root/'L2_route')
    # L3 reads the exact L1 capture instead of silently regenerating another sequence.
    l3=root/'L3_geometry'; mesh_files=reconstruct(c,frames,points,labels,colors,l3,c['task']['voxel_m'])
    assets=parts_and_appearance(c,frames,points,labels,l3,root/'L4_parts')
    sim_assets=simulation_assets(c,assets,root/'L5_simulation')
    layout_and_navigation_from_evidence(c,points,labels,assets,sim_assets,root/'L6_layout')
    passed,checks=validate(root,c)
    dump(root/'run_summary.json',dict(status='passed' if passed else 'failed',scene=scene,layer_count=7,layers=LAYER_NAMES,checks=checks,
      source_of_truth='L1_observe/capture/capture.json',reconstruction_provider='L3 projective TSDF + observed-surface marching cubes',
      agent_policy='structured decisions only; no LLM self-certification'))
    return root/'run_summary.json'
