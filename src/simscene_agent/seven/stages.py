"""L1-L4: validate observations, select an executable provider, reconstruct, and texture."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh
from .io import load, dump, sha, backproject, project


def observe(manifest_path, out):
    manifest_path=Path(manifest_path).resolve(); root=manifest_path.parent
    c=load(manifest_path)
    if c.get('depth_convention') != 'camera_z_m':
        raise ValueError('L1 requires camera_z_m depth; ray range must be explicitly converted.')
    if c.get('coordinates') != 'world_z_up_camera_x_right_y_down_z_forward':
        raise ValueError('Unsupported camera/world coordinate convention')
    if not c.get('frames') or not c.get('instances'):
        raise ValueError('Frames and annotated instance/part identities are required for this provider')
    ids={i['id'] for i in c['instances']}; names=[i['name'] for i in c['instances']]
    if len(ids)!=len(c['instances']) or len(set(names))!=len(names): raise ValueError('Duplicate instances')
    frames=[]; counts={str(i):0 for i in ids}; points=[]; labels=[]; colors=[]; hashes={}
    for r in c['frames']:
        files={k:(root/r[k]).resolve() for k in ('rgb','depth','mask')}
        if any(not p.is_relative_to(root) for p in files.values()): raise ValueError('Input path escapes capture directory')
        rgb=np.array(Image.open(files['rgb']).convert('RGB')); depth=np.load(files['depth'],allow_pickle=False)
        mask=np.array(Image.open(files['mask'])); K=np.asarray(r['K']); T=np.asarray(r['T_world_camera'])
        if depth.ndim!=2 or rgb.shape[:2]!=depth.shape or mask.shape!=depth.shape: raise ValueError('RGB/depth/mask shapes differ')
        if not np.isfinite(depth).all() or np.any(depth<0): raise ValueError('Invalid metric depth')
        if K.shape!=(3,3) or T.shape!=(4,4) or K[0,0]<=0 or K[1,1]<=0: raise ValueError('Invalid calibration shape')
        if not np.isfinite(T).all() or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-5):
            raise ValueError('Camera pose rotation is not rigid/right-handed')
        if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(K[2],[0,0,1]): raise ValueError('Invalid homogeneous calibration')
        if not set(np.unique(mask)) <= ids|{0}: raise ValueError('Unknown mask instance')
        valid=(depth>0)&(mask>0)
        if valid.mean()<.05: raise ValueError('Insufficient depth coverage')
        p=backproject(depth,K,T)
        if r.get('split','train')=='train':
            points.append(p[valid]); labels.append(mask[valid]); colors.append(rgb[valid])
            for i in ids: counts[str(i)]+=int((mask[valid]==i).sum())
        frames.append(dict(rgb=rgb,depth=depth,mask=mask,K=K,T=T,points=p,
                           split=r.get('split','train'),rgb_path=r['rgb']))
        for pth in files.values(): hashes[pth.name]=sha(pth)
    if len(points)<2: raise ValueError('At least two training frames required')
    # Tiny thin parts may contribute only a few pixels in a wide room view;
    # the later mesh/UV checks still reject parts with no usable triangles.
    missing=[i for i,n in counts.items() if n<8]
    if missing: raise ValueError(f'Parts with insufficient observations: {missing}; acquire more views')
    dump(out/'observations.json',dict(status='passed',source=c['source'],frames=len(frames),
        train_frames=len(points),counts=counts,depth_convention=c['depth_convention'],input_hashes=hashes,
        segmentation='supplied_instance_and_part_masks',pose_source='supplied_metric_calibration',
        limitations=c.get('assumptions',[])))
    return c,frames,np.concatenate(points),np.concatenate(labels),np.concatenate(colors)


def route(c,frames,out):
    decisions=[]
    for inst in c['instances']:
        n=sum(int(((f['mask']==inst['id'])&(f['depth']>0)).any()) for f in frames if f['split']=='train')
        if n<2: raise ValueError(f"Insufficient multiview evidence for {inst['name']}")
        decisions.append(dict(instance=inst['name'],views=n,provider='projective_tsdf_rgbd',
          structure='annotated_part_reconstruction',appearance='mesh_rgb_projection + gaussian_bootstrap',
          collision='per_part_AABB_fit_with_minimum_thickness_prior',
          articulation='vertical_hinge_hypothesis' if inst['role']=='hinge_candidate' else 'fixed',
          reason='Registered metric depth and poses are available; surface evidence takes precedence over generation.'))
    dump(out/'routes.json',dict(status='passed',policy='deterministic_evidence_router',decisions=decisions,
      unavailable_routes={'single_rgb':'no image-to-3D model configured; blocked, never silently substituted',
                          'unknown_poses':'SLAM/SfM provider not configured',
                          'trained_3dgs':'CUDA splat trainer is optional; bootstrap Gaussian export is always written',
                          '3dgs_collision':'3DGS is visual-only; L5 requires closed collision geometry'}))
    return decisions


def triangulate_part(frames, iid):
    """Fallback for thin observed parts erased at TSDF resolution: depth-grid triangles."""
    vv=[]; ff=[]; offset=0
    ranked=sorted([f for f in frames if f['split']=='train'],key=lambda f:int((f['mask']==iid).sum()),reverse=True)[:3]
    for f in ranked:
        p=f['points'][::2,::2]; mask=f['mask'][::2,::2]; dep=f['depth'][::2,::2]; h,w=dep.shape
        idx=np.arange(h*w).reshape(h,w)
        fs=np.concatenate([np.stack([idx[:-1,:-1],idx[:-1,1:],idx[1:,1:]],-1).reshape(-1,3),
                           np.stack([idx[:-1,:-1],idx[1:,1:],idx[1:,:-1]],-1).reshape(-1,3)])
        verts=p.reshape(-1,3)
        ok=np.all(mask.reshape(-1)[fs]==iid,axis=1)&np.all(dep.reshape(-1)[fs]>0,axis=1)
        edge=np.linalg.norm(verts[fs]-np.roll(verts[fs],1,axis=1),axis=2).max(1)
        fs=fs[ok&(edge<.3)]
        vv.append(verts); ff.append(fs+offset); offset+=len(verts)
    m=trimesh.Trimesh(np.concatenate(vv),np.concatenate(ff),process=True)
    m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    return m


def reconstruct(c,frames,points,labels,colors,out,voxel=None):
    voxel=float(voxel or c['task']['voxel_m']); lo,hi=np.array(c['bounds_m'])
    axes=[np.arange(lo[i],hi[i]+voxel,voxel) for i in range(3)]
    shape=tuple(len(a) for a in axes)
    if np.prod(shape)>3_000_000: raise ValueError('TSDF budget exceeded: increase voxel size')
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    field=np.ones(len(grid),np.float32); weights=np.zeros(len(grid),np.uint16); mu=voxel*3
    for frame in frames:
        if frame['split']!='train': continue
        uv,z=project(grid,frame['K'],frame['T']); h,w=frame['depth'].shape
        ui=np.rint(uv[:,0]).astype(int); vi=np.rint(uv[:,1]).astype(int)
        ix=np.flatnonzero((z>0)&(ui>=0)&(ui<w)&(vi>=0)&(vi<h))
        d=frame['depth'][vi[ix],ui[ix]]; sdf=d-z[ix]
        good=(d>0)&(sdf>=-mu); ix=ix[good]; val=np.clip(sdf[good]/mu,-1,1)
        field[ix]=(field[ix]*weights[ix]+val)/(weights[ix]+1); weights[ix]+=1
    # Mask prevents extracting a fabricated surface at an entirely unobserved boundary.
    v,f,_,_=marching_cubes(field.reshape(shape),level=0,spacing=(voxel,)*3,
                          mask=(weights.reshape(shape)>0),allow_degenerate=False)
    v+=lo; dist,near=cKDTree(points[::2]).query(v)
    # Reject triangles far from any observed sample (including truncation boundaries).
    keep=np.all(dist[f]<2*voxel,axis=1)
    m=trimesh.Trimesh(v,f[keep],process=True); m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
    _,nearest=cKDTree(points).query(m.triangles_center); face_ids=labels[nearest]
    mesh_files={}; methods={}
    for inst in c['instances']:
        sub=m.submesh([np.flatnonzero(face_ids==inst['id'])],append=True)
        if not isinstance(sub,trimesh.Trimesh) or len(sub.faces)<8:
            sub=triangulate_part(frames,inst['id']); methods[inst['name']]='depth_grid_thin_part_fallback'
        else: methods[inst['name']]='tsdf_marching_cubes'
        if len(sub.faces)<2: raise ValueError(f"No observed mesh for {inst['name']}")
        name=f"{inst['name']}.ply"; sub.export(out/name); mesh_files[inst['name']]=name
    m.visual.vertex_colors=colors[cKDTree(points).query(m.vertices)[1]]
    m.export(out/'scene_surface.ply')
    dump(out/'geometry.json',dict(status='passed',method='projective_TSDF_zero_isosurface',
      voxel_m=voxel,truncation_m=mu,observed_voxels=int((weights>0).sum()),vertices=len(m.vertices),triangles=len(m.faces),
      finite=bool(np.isfinite(m.vertices).all()),watertight=bool(m.is_watertight),
      watertight_required=False,reason='Observed room surfaces are open; closed collision proxies are constructed in L5.',
      per_part_method=methods,mesh_files=mesh_files))
    return mesh_files


def texture_part(mesh,frames,iid):
    """Choose a real RGB frame per face with depth occlusion and instance checks.

    Triangle islands use camera-projected UV coordinates into the RGB contact atlas.
    This retains RGB variation, not just a constant color or invented UV layout.
    """
    train=[f for f in frames if f['split']=='train']; h,w=train[0]['depth'].shape
    atlas=Image.new('RGB',(w*6,h*int(np.ceil(len(train)/6))))
    for j,f in enumerate(train): atlas.paste(Image.fromarray(f['rgb']),(j%6*w,j//6*h))
    tris=mesh.vertices[mesh.faces]; samples=np.concatenate([tris,tris.mean(1)[:,None,:]],axis=1)
    best=np.full(len(tris),np.inf); views=np.full(len(tris),-1); selected_uv=np.zeros((len(tris),3,2))
    for j,fr in enumerate(train):
        uv,z=project(samples.reshape(-1,3),fr['K'],fr['T']); uv=uv.reshape(-1,4,2); z=z.reshape(-1,4)
        px=np.rint(uv).astype(int); valid=np.all((z>0)&(px[:,:,0]>=0)&(px[:,:,0]<w)&(px[:,:,1]>=0)&(px[:,:,1]<h),axis=1)
        x=np.clip(px[:,:,0],0,w-1); y=np.clip(px[:,:,1],0,h-1)
        d=fr['depth'][y,x]; err=np.abs(d-z)
        valid &= np.all((d>0)&(err<.16),axis=1)&(fr['mask'][y[:,3],x[:,3]]==iid)
        score=err.mean(1)+.002*z[:,3]; use=valid&(score<best)
        best[use]=score[use]; views[use]=j; selected_uv[use]=uv[use,:3]
    keep=views>=0
    v=tris[keep].reshape(-1,3); f=np.arange(len(v)).reshape(-1,3)
    uv=selected_uv[keep]; view=views[keep]
    uv[:,:,0]+=((view%6)*w)[:,None]; uv[:,:,1]+=((view//6)*h)[:,None]
    uv[:,:,0]=(uv[:,:,0]+.5)/atlas.width; uv[:,:,1]=1-(uv[:,:,1]+.5)/atlas.height
    visual=trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),image=atlas)
    textured=trimesh.Trimesh(v,f,visual=visual,process=False)
    return textured,atlas,int((~keep).sum())


def parts_and_appearance(c,frames,points,labels,l3,out):
    assets=[]; scene=trimesh.Scene(); files=load(l3/'geometry.json')['mesh_files']
    for inst in c['instances']:
        name=inst['name']; raw=trimesh.load(l3/files[name],force='mesh'); mesh,atlas,discarded=texture_part(raw,frames,inst['id'])
        if len(mesh.faces)<1: raise ValueError(f'No visibility-verified textured faces for {name}')
        obs=points[labels==inst['id']]; bounds=np.stack([obs.min(0),obs.max(0)])
        # Latent board thickness cannot be measured from a single visible face.
        # Explicit fixed thickness prior extends inward away from the room.
        lo,hi=bounds.copy(); center=(lo+hi)/2; size=np.maximum(hi-lo,.055)
        if name=='floor': center[2]=(lo[2]+hi[2])/2-.04; size[2]=.08
        for wall,axis,sign in [('wall_north',1,1),('wall_south',1,-1),('wall_east',0,1),('wall_west',0,-1)]:
            if name==wall: center[axis]+=sign*.04; size[axis]=.08
        observed_bounds=[lo.tolist(),hi.tolist()]
        pivot=None
        if inst['role']=='hinge_candidate': pivot=[float(lo[0]),float(center[1]),float(lo[2])]
        origin=np.array(pivot if pivot is not None else center)
        local=mesh.copy(); local.vertices-=origin
        folder=out/name; folder.mkdir(parents=True,exist_ok=True)
        # OBJ UV + PNG + MTL; GLB embeds the same RGB atlas for a portable preview.
        from trimesh.exchange.obj import export_obj
        obj,resources=export_obj(local,return_texture=True,write_texture=False,mtl_name='material.mtl')
        (folder/'visual.obj').write_text(obj)
        for file,data in resources.items():
            if isinstance(data,str): (folder/file).write_text(data)
            else: (folder/file).write_bytes(data)
        scene.add_geometry(mesh,node_name=name,geom_name=name)
        material=mesh.visual.material
        assets.append(dict(**inst,mesh=f'{name}/visual.obj',origin_m=origin.tolist(),
            observed_bounds_m=observed_bounds,collision_center_m=center.tolist(),collision_size_m=size.tolist(),
            color_rgb=np.median(np.concatenate([fr['rgb'][fr['mask']==inst['id']] for fr in frames]),axis=0).astype(int).tolist(),
            triangles=len(mesh.faces),discarded_occluded_faces=discarded,uv_range=[float(mesh.visual.uv.min()),float(mesh.visual.uv.max())],
            relation='revolute_hypothesis' if pivot else 'fixed_part',
            hinge=None if pivot is None else dict(axis=[0,0,-1],range_rad=[0,1.25],pivot_m=pivot,
                source='vertical hinge at observed left edge: semantic hypothesis, not recovered motion'),
            prior='minimum board thickness 0.055m; architecture thickness 0.08m',
            parent_source='supplied annotation',appearance_source='RGB atlas projected with depth visibility tests'))
    scene.export(out/'scene.glb')
    dump(out/'parts.json',dict(status='passed',assets=assets,
        segmentation='supplied instance/part masks lifted to 3D, nearest-observation face assignment',
        texture='per-triangle projective UV atlas sampled from captured RGB, no PBR inference',
        hidden_geometry_completed=False,relations_inferred_from_motion=False))
    return assets
