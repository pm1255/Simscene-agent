"""L6 occupancy and layout. No fixture geometry is consulted."""
import heapq
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
from .io import dump


def astar(blocked,start,goal):
    h,w=blocked.shape
    def valid(p): return 0<=p[0]<w and 0<=p[1]<h and not blocked[p[1],p[0]]
    if not valid(start) or not valid(goal): return []
    costs={start:0}; prev={}; queue=[(0,start)]
    while queue:
        _,cur=heapq.heappop(queue)
        if cur==goal:
            path=[cur]
            while cur in prev: cur=prev[cur]; path.append(cur)
            return path[::-1]
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(cur[0]+dx,cur[1]+dy); cost=costs[cur]+1
            if valid(nxt) and cost<costs.get(nxt,float('inf')):
                costs[nxt]=cost; prev[nxt]=cur
                heapq.heappush(queue,(cost+abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1]),nxt))
    return []


def build_layout(c,points,labels,geoms,assets,out,policy='astar'):
    floor_id=next(a['id'] for a in assets if a['name']=='floor')
    ground=points[labels==floor_id]; origin=np.min(ground[:,:2],axis=0)
    cell=float(c['task']['map_cell_m']); extent=np.max(ground[:,:2],axis=0)-origin
    nx,ny=np.ceil(extent/cell).astype(int); y,x=np.indices((ny,nx))
    centers=origin+np.stack([x+.5,y+.5],-1)*cell
    # A cell is observed only when it contains a measured floor sample.
    known=np.zeros((ny,nx),bool); ind=np.floor((ground[:,:2]-origin)/cell).astype(int)
    ok=(ind[:,0]>=0)&(ind[:,0]<nx)&(ind[:,1]>=0)&(ind[:,1]<ny)
    known[ind[ok,1],ind[ok,0]]=True
    occupied=np.zeros_like(known)
    for g in geoms:
        if g['name']=='floor': continue
        half=np.asarray(g['size_m'])/2; center=np.asarray(g['center_m'])
        # Geoms entirely above the robot do not occupy its swept vertical interval.
        if center[2]-half[2]>c['task']['robot_height_m']+.04 or center[2]+half[2]<.02: continue
        occupied |= np.all(np.abs(centers-center[:2])<=half[:2]+cell/2,axis=2)
    blocked=occupied|(~known)
    # Pad the boundary: leaving the measured map is always forbidden.
    clearance=distance_transform_edt(np.pad(~blocked,1,constant_values=False))[1:-1,1:-1]*cell
    margin=float(c['task']['robot_radius_m'])+np.sqrt(2)*cell/2+.04
    inflated=blocked|(clearance<=margin)
    def cellof(p): return tuple(int(i) for i in np.floor((np.asarray(p)-origin)/cell))
    start,goal=cellof(c['task']['start_xy_m']),cellof(c['task']['goal_xy_m'])
    cells=astar(inflated,start,goal) if policy=='astar' else [start,goal]
    coords=[(origin+(np.asarray(p)+.5)*cell).tolist() for p in cells]
    if coords: coords=[c['task']['start_xy_m'],*coords,c['task']['goal_xy_m']]
    np.savez_compressed(out/'map.npz',known=known,occupied=occupied,blocked=inflated,clearance=clearance,origin=origin,cell=cell)
    image=np.full((ny,nx,3),235,np.uint8); image[~known]=[145,151,165]; image[occupied]=[28,41,56]
    image[inflated&known&(~occupied)]=[194,201,209]
    Image.fromarray(np.where(inflated,0,255).astype(np.uint8)).resize((nx*7,ny*7),Image.Resampling.NEAREST).save(out/'navigation_occupancy.png')
    im=Image.fromarray(image).resize((nx*7,ny*7),Image.Resampling.NEAREST); draw=ImageDraw.Draw(im)
    if cells: draw.line([(int(p[0])*7+3,int(p[1])*7+3) for p in cells],fill=(224,86,58),width=4)
    for p,color in [(start,(20,175,140)),(goal,(48,95,195))]: draw.ellipse([p[0]*7-3,p[1]*7-3,p[0]*7+9,p[1]*7+9],fill=color)
    im.save(out/'navigation_path.png')
    nav=dict(status='candidate',policy=policy,cell_size_m=cell,origin_xy_m=origin.tolist(),
             path_xy_cells=cells,path_xy_m=coords,start_xy_m=c['task']['start_xy_m'],goal_xy_m=c['task']['goal_xy_m'],
             robot_radius_m=c['task']['robot_radius_m'],robot_height_m=c['task']['robot_height_m'],
             unknown_cells=int((~known).sum()),known_cells=int(known.sum()),inflation_m=margin,
             length_m=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum()) if len(coords)>1 else 0.)
    dump(out/'navigation_path.json',nav)
    relations=[]
    for a in assets:
        relations.append(dict(subject=a['name'],relation=a['relation'],object=a['parent'],source=a['parent_source']))
    dump(out/'scene_graph.json',dict(objects=assets,relations=relations,
        layout_policy='Preserve measured world poses; repair navigation without moving furniture.',
        free_space_policy='Only observed floor cells; unknown and outside are blocked.',
        coordinates='metres, world Z up',navigation=nav))
    dump(out/'layout_constraints.json',dict(status='passed',constraints=[
        {'type':'inside_observed_floor','hard':True,'unknown_policy':'blocked'},
        {'type':'free_path','hard':True,'robot_radius_m':c['task']['robot_radius_m']},
        {'type':'collision_source','value':'L5 reconstructed collision proxies'},
        {'type':'navigation_policy','value':'A* on inflated observed-space occupancy'}]))
    return nav


def inspect_path(nav,map_file,geoms):
    """Independent continuous cylinder-vs-box and observed-area acceptance."""
    points=nav['path_xy_m']; errors=[]; clearance=float('inf'); samples=[]
    if len(points)<2: return dict(passed=False,errors=['no_path'],minimum_clearance_m=None,samples=0)
    for a,b in zip(points[:-1],points[1:]):
        n=max(2,int(np.ceil(np.linalg.norm(np.array(b)-a)/.01)))
        samples.extend(np.linspace(a,b,n))
    samples=np.asarray(samples)
    data=np.load(map_file,allow_pickle=False); cells=np.floor((samples-data['origin'])/float(data['cell'])).astype(int)
    h,w=data['known'].shape
    inside=(cells[:,0]>=0)&(cells[:,0]<w)&(cells[:,1]>=0)&(cells[:,1]<h)
    if not inside.all(): errors.append('outside_observed_map')
    good=cells[inside]
    if not data['known'][good[:,1],good[:,0]].all(): errors.append('unobserved_floor')
    if data['blocked'][good[:,1],good[:,0]].any(): errors.append('insufficient_observed_clearance')
    for g in geoms:
        if g['name']=='floor': continue
        center,half=np.array(g['center_m']),np.array(g['size_m'])/2
        if center[2]-half[2]>nav['robot_height_m']+.04 or center[2]+half[2]<.02: continue
        gap=np.maximum(np.abs(samples-center[:2])-half[:2],0)
        distance=np.linalg.norm(gap,axis=1)-nav['robot_radius_m']
        minimum=float(distance.min()); clearance=min(clearance,minimum)
        if minimum<0: errors.append(f'collision:{g["name"]}')
    if np.linalg.norm(np.array(points[0])-nav['start_xy_m'])>.001: errors.append('wrong_start')
    if np.linalg.norm(np.array(points[-1])-nav['goal_xy_m'])>.001: errors.append('wrong_goal')
    return dict(passed=not errors,errors=errors,minimum_clearance_m=None if not np.isfinite(clearance) else clearance,samples=len(samples))
