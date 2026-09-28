"""Synthetic acquisition ONLY. The reconstruction pipeline never imports this file.

Geometry is used by this renderer and a held-out evaluator, not by L1-L6.
Masks and part names are supplied annotations, not a claimed segmentation model.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .io import camera, backproject, ray_boxes, dump


def make_capture(root, kind='living_room'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sequence = root / 'input_rgbd'
    sequence.mkdir(parents=True, exist_ok=True)
    boxes=[]
    def add(name, center, size, color, parent='room', role='fixed'):
        c, s = np.array(center), np.array(size)
        boxes.append(dict(id=len(boxes)+1, name=name, parent=parent, role=role,
                          bounds=[(c-s/2).tolist(), (c+s/2).tolist()], color=color))
    add('floor', [3, 2.5, -.06], [6, 5, .12], [185,169,140])
    add('wall_north', [3, 5.04, 1.4], [6,.08,2.8], [194,211,209])
    add('wall_south', [3, -.04, 1.4], [6,.08,2.8], [219,222,228])
    add('wall_west', [-.04, 2.5, 1.4], [.08,5,2.8], [219,222,228])
    add('wall_east', [6.04, 2.5, 1.4], [.08,5,2.8], [219,222,228])
    # Separate tabletop and legs preserve under-table space in the collision model.
    tx,ty = {'living_room':(1.65,2.2),'office':(1.7,3.35),'corridor':(2.0,2.7)}[kind]
    add('table_top',[tx,ty,.78],[1.35,.9,.10],[151,101,64],'table')
    for i,(dx,dy) in enumerate([(-.53,-.33),(.53,-.33),(-.53,.33),(.53,.33)]):
        add(f'table_leg_{i}',[tx+dx,ty+dy,.36],[.10,.10,.72],[77,79,82],'table')
    sx,sy = {'living_room':(4.65,1.6),'office':(4.55,2.5),'corridor':(4.5,2.5)}[kind]
    add('sofa_seat',[sx,sy,.4],[1.5,.85,.65],[60,103,130],'sofa')
    add('sofa_back',[sx,sy+.45,.73],[1.5,.16,1.3],[60,103,130],'sofa')
    for i,dx in enumerate([-.78,.78]):
        add(f'sofa_arm_{i}',[sx+dx,sy,.54],[.16,.9,.94],[49,88,119],'sofa')
    # Cabinet has an interior and a separate door. Joint is a declared hypothesis.
    cx,cy=4.45,4.3
    add('cabinet_back',[cx,cy+.25,.95],[1.15,.08,1.9],[165,128,90],'cabinet')
    for i,dx in enumerate([-.57,.57]):
        add(f'cabinet_side_{i}',[cx+dx,cy,.95],[.08,.5,1.9],[165,128,90],'cabinet')
    for i,z in enumerate([.06,.92,1.86]):
        add(f'cabinet_shelf_{i}',[cx,cy,z],[1.06,.5,.08],[165,128,90],'cabinet')
    add('cabinet_door',[cx,cy-.3,.96],[1.06,.06,1.72],[173,132,86],'cabinet','hinge_candidate')
    if kind=='office':
        # Leave a robot-width route around the divider after L6 inflation.
        add('partition',[3.1,2.25,.6],[.12,.9,1.2],[91,113,112])
    if kind=='corridor':
        # Staggered dividers leave a 0.75 m opening for the navigation task.
        add('partition_a',[2.5,3.8,1.1],[.12,1.8,2.2],[192,204,211])
        add('partition_b',[3.75,1.25,1.1],[.12,1.8,2.2],[192,204,211])
    # 36 views from clear interior stations, deliberately overlapping; six held out.
    stations=[(.65,.65),(3,.65),(5.5,.65),(.65,2.5),(3,2.4),(5.5,3.1),(.65,4.45),(3,4.5),(5.5,4.6)]
    K=np.array([[140.,0,111.5],[0,140.,83.5],[0,0,1.]])
    h,w=168,224; v,u=np.indices((h,w))
    local=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones((h,w))],-1)
    frames=[]
    for station in stations:
        for heading in range(4):
            pos=np.array([*station,2.35])
            a=heading*np.pi/2+.28
            T=camera(pos, pos+[np.cos(a)*3,np.sin(a)*3,-1.8])
            depth,ids=ray_boxes(pos,local@T[:3,:3].T,boxes)
            pts=backproject(depth,K,T); rgb=np.full((h,w,3),225,np.uint8)
            for b in boxes:
                m=ids==b['id']; p=pts[m]
                # A genuine spatially varying RGB texture to test projection/UV retention.
                grain=.88+.10*np.sin(p[:,0]*32+p[:,1]*4+p[:,2]*13)
                if b['name']=='floor':
                    grain=.82+.16*((np.floor(p[:,0]*2)+np.floor(p[:,1]*2))%2)
                rgb[m]=np.clip(np.asarray(b['color'])[None,:]*grain[:,None],0,255)
            i=len(frames); stem=f'{i:03d}'
            Image.fromarray(rgb).save(sequence/f'rgb_{stem}.png')
            Image.fromarray(ids).save(sequence/f'mask_{stem}.png')
            np.save(sequence/f'depth_{stem}.npy',depth)
            frames.append(dict(rgb=f'input_rgbd/rgb_{stem}.png',depth=f'input_rgbd/depth_{stem}.npy',mask=f'input_rgbd/mask_{stem}.png',
                               K=K.tolist(),T_world_camera=T.tolist(),split='test' if i%6==5 else 'train'))
    annotations=[{k:b[k] for k in ('id','name','parent','role')} for b in boxes]
    goals={'living_room':[5.45,4.55],'office':[5.45,3.4],'corridor':[3.0,2.4]}
    dump(root/'capture.json',dict(schema_version=1,source='synthetic_annotated_rgbd',scene=kind,
        depth_convention='camera_z_m',coordinates='world_z_up_camera_x_right_y_down_z_forward',
        bounds_m=[[-.2,-.2,-.2],[6.2,5.2,3.0]],frames=frames,instances=annotations,
        task=dict(start_xy_m=[.7,.7],goal_xy_m=goals[kind],robot_radius_m=.22,robot_height_m=.65,
                  map_cell_m=.08,voxel_m=.065,require_articulation=True),
        assumptions=['Metric calibrated poses, instance and part masks are supplied annotations.',
                     'Cabinet hinge type is a semantic prior; no motion recovery from static RGB.']))
    dump(root/'calibration.json',dict(width=w,height=h,frames=[dict(rgb=r['rgb'].split('/',1)[1],depth=r['depth'].split('/',1)[1],
        position_m=np.asarray(r['T_world_camera'])[:3,3].tolist(),target_m=[3,2.5,1],fx=r['K'][0][0],fy=r['K'][1][1],cx=r['K'][0][2],cy=r['K'][1][2]) for r in frames]))
    dump(root/'evaluation_truth.json',dict(boxes=boxes))
    thumbs=Image.new('RGB',(w*4,h*2))
    for j,i in enumerate([0,4,8,12,16,20,24,28]):
        thumbs.paste(Image.open(root/frames[i]['rgb']),(j%4*w,j//4*h))
    thumbs.save(root/'contact_sheet.jpg')
    return root/'capture.json'
