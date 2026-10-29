#!/usr/bin/env python3
"""Locate geometric reflection residuals; independent float64 plane images.

This is an offline diagnostic, not a replacement SDK reference or acceptance
threshold. No corrected geometry or channel is written into the dataset.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['WIWO_GPU']='-1'
import argparse,importlib.util,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path.cwd()/'src'))
from wiwo_ma.platform.dataset_audit import sha_file,save_json


def exact_plane_images(source,target,triangles,depth):
    source=np.asarray(source,float);target=np.asarray(target,float);tri=np.asarray(triangles,float);n=len(source)
    normals=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);norm=np.linalg.norm(normals,axis=-1)
    normals=np.divide(normals,norm[:,:,None],out=np.zeros_like(normals),where=norm[:,:,None]>0)
    im=source.copy();images=[im.copy()]
    for d in range(3):
        active=depth>d;normal=normals[:,d];p=tri[:,d,0]
        im[active]-=2*np.sum((im[active]-p[active])*normal[active],axis=-1)[:,None]*normal[active];images.append(im.copy())
    current=target.copy();vertices=np.zeros((n,3,3));inside=np.ones(n,bool)
    for d in reversed(range(3)):
        active=np.flatnonzero(depth>d);normal=normals[active,d];p=tri[active,d,0];direction=images[d+1][active]-current[active]
        ratio=np.sum((p-current[active])*normal,axis=-1)/np.sum(direction*normal,axis=-1)
        point=current[active]+ratio[:,None]*direction;vertices[active,d]=point;current[active]=point
        v0,v1,v2=tri[active,d,0],tri[active,d,1],tri[active,d,2];e1,e2=v1-v0,v2-v0;w=point-v0
        aa=np.sum(e1*e1,axis=-1);bb=np.sum(e1*e2,axis=-1);cc=np.sum(e2*e2,axis=-1);dd=np.sum(w*e1,axis=-1);ee=np.sum(w*e2,axis=-1)
        determinant=aa*cc-bb*bb;u=(cc*dd-bb*ee)/determinant;v=(aa*ee-bb*dd)/determinant
        inside[active]&=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
    previous=source.copy();length=np.zeros(n)
    for d in range(3):
        active=depth>d;length[active]+=np.linalg.norm(vertices[active,d]-previous[active],axis=-1);previous[active]=vertices[active,d]
    length+=np.linalg.norm(target-previous,axis=-1)
    return vertices,length,inside


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('frozenpost',a.run/'executed_sources/audit_platform_sdk16_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    records=[];summary=[]
    for i in [278,279,770,771,1433,1434,1566,1567]:
        folder=a.run/'references'/str(i)
        if not (folder/'complete.json').exists():continue
        z=m.read_state(folder,(np.arange(64)-31.5)*30000);depth=z['depth'];exact,length,inside=exact_plane_images(z['source_positions'],z['target_positions'],z['primitive_triangles'],depth)
        previous=z['source_positions'].astype(float);all_angles=[]
        for d in range(3):
            ix=np.flatnonzero(depth>d);vertex=z['vertices'][ix,d].astype(float);normal=z['normals'][ix,d].astype(float);normal/=np.linalg.norm(normal,axis=-1)[:,None]
            incoming=vertex-previous[ix];l0=np.linalg.norm(incoming,axis=-1);incoming/=l0[:,None];nextpoint=z['target_positions'][ix].astype(float)
            if d<2:
                more=depth[ix]>d+1;nextpoint[more]=z['vertices'][ix,d+1][more]
            outgoing=nextpoint-vertex;l1=np.linalg.norm(outgoing,axis=-1);outgoing/=l1[:,None]
            ideal=incoming-2*np.sum(incoming*normal,axis=1)[:,None]*normal;angle=np.arccos(np.clip(np.sum(ideal*outgoing,axis=1),-1,1));all_angles.extend(angle.tolist())
            # This0.001rad cut is a diagnostic inventory cut, NOT a physical or
            # numerical acceptance threshold and never filters CSI/ref paths.
            for local in np.flatnonzero(angle>.001):
                k=ix[local];actual=z['vertices'][k,:depth[k]].astype(float);clean=exact[k,:depth[k]]
                records.append(dict(frame=i,path_index=int(k),user=int(z['user'][k]),tx_port=int(z['tx_port'][k]),depth=int(depth[k]),interaction=d,
                    specular_angle_residual_rad=float(angle[local]),incoming_length_m=float(l0[local]),outgoing_length_m=float(l1[local]),
                    coefficient_abs=float(abs(z['coefficient'][k])),word=str(z['words'][k]),
                    actual_vertices=actual.tolist(),independent_exact_plane_vertices=clean.tolist(),
                    maximum_vertex_difference_m=float(np.linalg.norm(actual-clean,axis=-1).max()),
                    exact_optical_length_vs_SDK_delay_m=float(length[k]-float(z['delay_s'][k])*299792458),
                    exact_reflection_points_inside_triangles=bool(inside[k]),
                    SDK_point_offset_scale_at_vertex_m=float((1+abs(vertex[local]).max())*2**-24*100),
                    no_dataset_or_reference_modified=True))
            previous[ix]=vertex
        summary.append(dict(index=i,path_count=len(depth),max_angle_residual_rad=max(all_angles),
            exact_plane_all_points_inside_count=int(inside.sum()),
            scope='Exact same stored primitive sequence, finite triangle inclusion checked; visibility of altered paths not independently retraced, not a FullRT substitute'))
    save_json(a.out/'geometric_outlier_paths.json',dict(inventory_cut_rad=.001,acceptance_threshold_changed=False,paths=records,states=summary,
        diagnosis='Short ground/vehicle reflection segment amplifies coordinate-scaled SDK origin/target offsets. Source mechanism identified; no claim of bitwise reimplementation or universal SDK physical certification.',
        source_locations=['sionna/rt/constants.py:EPSILON_FLOAT = dr.epsilon(mi.Float)*100','sionna/rt/utils/ray_tracing.py:offset_p','sionna/rt/path_solvers/image_method.py:ImageMethod._... spawn_ray_to(vertex_offset, offset_p(ray_target_offset,...))'],
        epsilon_Float32=2**-24,scope='Fresh SDK geometric numerical residual, not a newly discovered CSI storage mismatch'))
    (a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save_json(a.out/'manifest.json',dict(new_GPU_seconds=0,files={str(p.relative_to(a.out)):sha_file(p) for p in a.out.iterdir() if p.is_file()}))
    print('Short-segment geometric path records',len(records),'new_GPU_seconds0')


if __name__=='__main__':main()
