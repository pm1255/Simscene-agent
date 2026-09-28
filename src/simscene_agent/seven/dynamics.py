"""L5 simulator adapter. Runtime checks require moving bodies and real contacts."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
from .io import dump


def numbers(v):
    return ' '.join(f'{float(x):.8g}' for x in v)


def build_simulation(c, assets, out):
    out=Path(out); (out/'collision').mkdir(parents=True,exist_ok=True)
    root=ET.Element('mujoco',model=c['scene'])
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='.005',gravity='0 0 -9.81',integrator='implicitfast')
    visual=ET.SubElement(root,'visual'); ET.SubElement(visual,'global',offwidth='960',offheight='720')
    default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',friction='0.6 0.005 0.0001',condim='3')
    asset_xml=ET.SubElement(root,'asset'); world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='3 2.5 6',dir='0 0 -1',diffuse='.8 .8 .8')
    acts=ET.SubElement(root,'actuator'); geoms=[]; joint_names=[]
    for a in assets:
        name=a['name']; center=np.asarray(a['collision_center_m']); size=np.asarray(a['collision_size_m'])
        if not np.isfinite(size).all() or np.any(size<=0): raise ValueError(f'Invalid collider: {name}')
        origin=np.array(a['origin_m']); pos=center-origin
        proxy=trimesh.creation.box(extents=size); proxy.apply_translation(pos)
        proxy.export(out/'collision'/f'{name}.obj')
        body=ET.SubElement(world,'body',name=name,pos=numbers(origin))
        mass=float(max(.2,np.prod(size)*180)) if a['hinge'] else 0.
        if a['hinge']:
            jname=f'{name}_hinge'; joint_names.append(jname)
            ET.SubElement(body,'joint',name=jname,type='hinge',axis=numbers(a['hinge']['axis']),
                          range=numbers(a['hinge']['range_rad']),limited='true',damping='3')
            ET.SubElement(acts,'position',name=f'{name}_servo',joint=jname,kp='45',kv='10',
                          ctrlrange=numbers(a['hinge']['range_rad']),ctrllimited='true')
        # Visual RGB mesh and collision box are explicitly different geometries.
        mesh_path=out.parent/'L4_parts'/a['mesh']
        tex_path=mesh_path.parent/'material_0.png'
        ET.SubElement(asset_xml,'texture',name=f'{name}_rgb',type='2d',file=f'../L4_parts/{name}/material_0.png')
        ET.SubElement(asset_xml,'material',name=f'{name}_material',texture=f'{name}_rgb',specular='0.05',shininess='0.1')
        ET.SubElement(asset_xml,'mesh',name=f'{name}_visual',file=f'../L4_parts/{name}/visual.obj',inertia='shell')
        ET.SubElement(body,'geom',name=f'{name}_visual',type='mesh',mesh=f'{name}_visual',
                      material=f'{name}_material',group='1',contype='0',conaffinity='0',mass='0')
        ET.SubElement(body,'geom',name=f'{name}_collision',type='box',pos=numbers(pos),size=numbers(size/2),
                      group='3',mass=str(mass),rgba='.4 .5 .6 .25')
        geoms.append(dict(name=name,center_m=center.tolist(),size_m=size.tolist(),origin_m=origin.tolist(),
                          mass_kg=mass,mass_source='density_prior_180kg_m3' if mass else 'fixed_to_world',
                          friction=[.6,.005,.0001],friction_source='simulation_prior',
                          collision_file=f'collision/{name}.obj',watertight=bool(proxy.is_watertight),
                          positive_volume=bool(proxy.volume>0)))
    # An actuated cylindrical navigation probe with vertical motion under gravity.
    # It is a holonomic surrogate, not a claimed differential-drive robot controller.
    robot=ET.SubElement(world,'body',name='navigator',pos='0 0 0')
    for axis,vec in [('x','1 0 0'),('y','0 1 0'),('z','0 0 1')]:
        ET.SubElement(robot,'joint',name=f'nav_{axis}',type='slide',axis=vec,damping='1')
        if axis!='z': ET.SubElement(acts,'position',name=f'nav_{axis}',joint=f'nav_{axis}',kp='700',kv='90')
    ET.SubElement(robot,'geom',name='navigator_collision',type='cylinder',mass='8',
                  size=numbers([c['task']['robot_radius_m'],c['task']['robot_height_m']/2]),rgba='.15 .75 .55 1',
                  friction='0.08 0.001 0.0001')
    ET.indent(root); ET.ElementTree(root).write(out/'scene.xml',encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml')); data=mujoco.MjData(model)
    initialize(model,data,c,z_extra=.15)
    start_z=float(data.qpos[model.joint('nav_z').qposadr[0]])
    floor_id=model.geom('floor_collision').id; robot_id=model.geom('navigator_collision').id
    floor_contacts=0
    for _ in range(400):
        mujoco.mj_step(model,data)
        floor_contacts+=sum({int(x.geom1),int(x.geom2)}=={floor_id,robot_id} for x in data.contact)
    end_z=float(data.qpos[model.joint('nav_z').qposadr[0]])
    hinge_tests=[]
    for name in joint_names:
        addr=model.joint(name).qposadr[0]; act=model.actuator(name.removesuffix('_hinge')+'_servo').id
        data.ctrl[act]=1.0
        for _ in range(600): mujoco.mj_step(model,data)
        angle=float(data.qpos[addr]); hinge_tests.append(dict(joint=name,target_rad=1.,actual_rad=angle,passed=abs(angle-1)<.08))
        data.ctrl[act]=0
    finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    runtime=dict(mujoco_version=mujoco.__version__,ngeom=model.ngeom,nbody=model.nbody,njnt=model.njnt,
                 steps=400+600*len(hinge_tests),finite=finite,start_probe_z_m=start_z,end_probe_z_m=end_z,
                 floor_contacts=int(floor_contacts),hinge_tests=hinge_tests,
                 gravity_contact_passed=bool(floor_contacts>0 and abs(end_z-c['task']['robot_height_m']/2)<.02))
    passed=finite and runtime['gravity_contact_passed'] and all(j['passed'] for j in hinge_tests)
    if c['task'].get('require_articulation') and not hinge_tests: passed=False
    dump(out/'simulation.json',dict(status='passed' if passed else 'failed',backend='MuJoCo',runtime=runtime,
         geoms=geoms,probe_model='holonomic cylinder with XY servos and gravity Z',
         limits='Static room furniture; only declared hinge hypotheses and probe are dynamic.'))
    return geoms


def initialize(model,data,c,z_extra=0.):
    mujoco.mj_resetData(model,data)
    for name,value in zip(['nav_x','nav_y','nav_z'],[*c['task']['start_xy_m'],c['task']['robot_height_m']/2+z_extra]):
        data.qpos[model.joint(name).qposadr[0]]=value
        if name!='nav_z': data.ctrl[model.actuator(name).id]=value
    mujoco.mj_forward(model,data)


def simulate_path(xml_path,c,path_xy,render=False):
    model=mujoco.MjModel.from_xml_path(str(xml_path)); data=mujoco.MjData(model); initialize(model,data,c)
    xyaddr=[model.joint(n).qposadr[0] for n in ['nav_x','nav_y']]
    zaddr=model.joint('nav_z').qposadr[0]; robot=model.geom('navigator_collision').id
    floor=model.geom('floor_collision').id; collision_events=[]; trajectory=[]; frame_states=[]
    # 5mm target spacing, 20ms per sample: a 0.25m/s controller command.
    waypoint=np.asarray([*c['task']['start_xy_m']]); targets=[]
    for end in path_xy:
        end=np.asarray(end); count=max(1,int(np.ceil(np.linalg.norm(end-waypoint)/.005)))
        targets.extend(np.linspace(waypoint,end,count+1)[1:]); waypoint=end
    targets.extend([waypoint]*100)
    for i,target in enumerate(targets):
        for a,val in zip(['nav_x','nav_y'],target): data.ctrl[model.actuator(a).id]=val
        for _ in range(4):
            mujoco.mj_step(model,data)
            for contact in data.contact:
                pair={int(contact.geom1),int(contact.geom2)}
                if robot in pair and floor not in pair and contact.dist<-.0005:
                    collision_events.append(dict(time_s=float(data.time),penetration_m=float(-contact.dist)))
        trajectory.append([float(data.time),*data.qpos[xyaddr].tolist(),float(data.qpos[zaddr])])
        if i%25==0: frame_states.append(data.qpos.copy())
    final_error=float(np.linalg.norm(data.qpos[xyaddr]-c['task']['goal_xy_m']))
    report=dict(steps=len(targets)*4,finite=bool(np.isfinite(data.qpos).all()),
                obstacle_contact_events=len(collision_events),final_goal_error_m=final_error,
                passed=bool(not collision_events and final_error<.12 and np.isfinite(data.qpos).all()))
    return report,trajectory,model,frame_states
