"""Self-contained configurable presets, no RT computation or teacher lookup."""
from copy import deepcopy
import math
import numpy as np
from .config import SCHEMA,validate


def reflector_example():
    return validate(dict(schema=SCHEMA,scene=dict(name='reflector'),radio=dict(carrier_hz=3.5e9),
        timing=dict(duration_s=2.,csi_dt_s=.01,mechanics_dt_s=.002,control_dt_s=.1),
        base_stations=[dict(id='bs0',position_m=[0,0,2],panels=[dict(id='panel0',position_m=[0,0,0],size_m=[.02,.2,.2],array=dict(kind='UPA',rows=2,cols=2,spacing_m=.04))])],
        users=[dict(id='ue0',position_m=[4,1,1.5],trajectory=[dict(time_s=0.,position_m=[4,1,1.5]),dict(time_s=2.,position_m=[4,1.5,1.5])])],
        actors=[dict(id='blocker',position_m=[2,-1,1],size_m=[.2,.8,2.],trajectory=[dict(time_s=0.,position_m=[2,-1,1]),dict(time_s=2.,position_m=[2,1,1])])],
        actions=[dict(time_s=.1,bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[.05,0,0],orientation_deg=[5,0,0]),
            dict(time_s=1.,bs_id='bs0',panel_id='panel0',type='target_pose',position_m=[-.03,0,0],orientation_deg=[-3,0,0])]))


def munich_example(users=40,carrier_hz=28e9):
    # The legacy roof location is retained, but the preset is self-contained:
    # spherical mounting points, explicit coordinates and seeded UE routes.
    center=np.array([8.,21.,32.]);panels=[];spacing=299792458./carrier_hz/2
    golden=(1+np.sqrt(5.))/2
    for i in range(16):
        z=1-2*(i+.5)/16;phi=2*np.pi*i/golden;direction=np.array([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
        # Panels are mounted on independent bounded 3D gantry+gimbal stages.
        yaw=np.degrees(np.arctan2(direction[1],direction[0]));pitch=-np.degrees(np.arcsin(direction[2]))
        position=(direction*1.1).tolist();size=max(.09,spacing*2.3)
        panels.append(dict(id=f'panel{i:02d}',position_m=position,orientation_deg=[float(yaw),float(pitch),0.],size_m=[.015,size,size],
            array=dict(kind='UPA',rows=2,cols=2,spacing_m=spacing),mechanics=dict(mechanism='gantry_gimbal',max_rotation_rad=.4)))
    rng=np.random.default_rng(20261024);ue=[]
    for i in range(users):
        # Explicit open-street corridor waypoints. Position/paths are not
        # claimed surveyed ground truth; every location is saved in the config.
        start=np.array([float(rng.uniform(5,95)),float(rng.uniform(55,85)),1.5]);delta=np.array([.15*(-1 if i%2 else 1),.1,0.])*15
        ue.append(dict(id=f'ue{i:02d}',position_m=start.tolist(),trajectory=[dict(time_s=0.,position_m=start.tolist()),dict(time_s=15.,position_m=(start+delta).tolist()),dict(time_s=30.,position_m=start.tolist())]))
    actions=[]
    for i in [0,3,7,12]:
        p=panels[i];initial=np.asarray(p['position_m']);angles=np.asarray(p['orientation_deg'])
        actions.append(dict(time_s=1.,bs_id='bs0',panel_id=p['id'],type='target_pose',position_m=(initial+[.04,0,0]).tolist(),orientation_deg=(angles+[2,0,0]).tolist()))
        actions.append(dict(time_s=15.,bs_id='bs0',panel_id=p['id'],type='target_pose',position_m=(initial+[-.03,0,.02]).tolist(),orientation_deg=(angles+[-2,0,0]).tolist()))
    actions.sort(key=lambda a:a['time_s'])
    return validate(dict(schema=SCHEMA,scene=dict(name='munich'),radio=dict(carrier_hz=carrier_hz),
        timing=dict(duration_s=30.,csi_dt_s=.01,mechanics_dt_s=.002,control_dt_s=.1),
        base_stations=[dict(id='bs0',position_m=center.tolist(),panels=panels)],users=ue,
        actors=[dict(id='vehicle',position_m=[35,52,1],size_m=[3,1.7,2],trajectory=[dict(time_s=0.,position_m=[35,52,1]),dict(time_s=30.,position_m=[50,52,1])])],actions=actions))


def physical_munich_example():
    """Short admission scenario, not an SDK-validated dataset or road survey.

    A bounded delivery-van box crosses an actual BS→UE corridor near the UE,
    where the descending rooftop ray is below its 2.6m roof. Its 8m traverse
    in 4s peaks at 3.75m/s and 2.887m/s². Geometry preflight is mandatory.
    Legacy presets/configurations remain available for reproduction.
    """
    c=munich_example();duration=4.
    c['timing'].update(duration_s=duration,csi_dt_s=.04)
    spec=dict(model='bounded_quintic',max_speed_m_s=.5,max_acceleration_m_s2=1.,
              max_angular_speed_rad_s=.8,max_angular_acceleration_rad_s2=1.5)
    for i,u in enumerate(c['users']):
        p=np.array(u['position_m']);q=p+np.array([.05*(-1 if i%2 else 1),.03,0])
        u.update(motion=deepcopy(spec),trajectory=[dict(time_s=0.,position_m=p.tolist(),orientation_deg=[0,0,0]),
                   dict(time_s=duration,position_m=q.tolist(),orientation_deg=[0,0,0])])
    r=np.array(c['users'][0]['position_m']);direction=(np.array(c['base_stations'][0]['position_m'])-r)[:2];direction/=np.linalg.norm(direction)
    side=np.array([-direction[1],direction[0],0.]);center=r+np.r_[direction*2.8,0.];center[2]=1.3
    yaw=float(np.degrees(np.arctan2(side[1],side[0])));a=center-4*side;b=center+4*side
    c['actors']=[dict(id='delivery_van',position_m=a.tolist(),orientation_deg=[yaw,0,0],size_m=[3,1.7,2.6],material='metal',ground_z_m=0.,
        motion={**spec,'max_speed_m_s':4.,'max_acceleration_m_s2':3.},
        trajectory=[dict(time_s=0.,position_m=a.tolist(),orientation_deg=[yaw,0,0]),dict(time_s=duration,position_m=b.tolist(),orientation_deg=[yaw,0,0])])]
    # A second, spatially separated bounded actor exercises joint geometry updates.
    p=np.array(c['users'][1]['position_m'])+[3,0,-.9];q=p+[.6,0,0]
    c['actors'].append(dict(id='wooden_crate_cart_proxy',position_m=p.tolist(),size_m=[.6,.6,1.2],material='wood',ground_z_m=0.,motion=deepcopy(spec),
        trajectory=[dict(time_s=0.,position_m=p.tolist()),dict(time_s=duration,position_m=q.tolist())]))
    c['actions']=[dict(time_s=.1,bs_id='bs0',panel_id='panel00',type='target_pose',position_m=(np.array(c['base_stations'][0]['panels'][0]['position_m'])+[.03,0,0]).tolist(),orientation_deg=(np.array(c['base_stations'][0]['panels'][0]['orientation_deg'])+[1,0,0]).tolist())]
    c['physics_validation']=dict(require_preflight=True,require_LOS_cycle=True)
    return validate(c)


def continuous_munich_example():
    """Small physical integration gate: 4s/401 frames, two moving panels.

    A translating metallic screen on a rail crosses the direct ray corridor.
    This deliberately slow scenario supports 100Hz sampling at 3.5GHz under
    the documented specular path-length derivative bound. It is not a van.
    Actual city clearance and LOS transitions still require CPU preflight.
    """
    c=physical_munich_example();c['users']=c['users'][:2]
    c['radio']['carrier_hz']=3.5e9;c['timing']['csi_dt_s']=.01
    c['base_stations'][0]['panels']=[c['base_stations'][0]['panels'][i] for i in (0,3)]
    actions=[]
    for i,p in enumerate(c['base_stations'][0]['panels']):
        p['array']=dict(kind='UPA',rows=1,cols=2,spacing_m=299792458/3.5e9/2)
        p['size_m']=[.015,.12,.08]
        p['mechanics'].update(max_speed_m_s=.03,max_acceleration_m_s2=.12,max_angular_speed_rad_s=.06,max_angular_acceleration_rad_s2=.24)
        point=np.array(p['position_m']);angles=np.array(p['orientation_deg'])
        for t,sign in ((.1,1),(2.6,-1)):
            actions.append(dict(time_s=t,bs_id='bs0',panel_id=p['id'],type='target_pose',position_m=(point+sign*np.array([.01*(-1 if i else 1),0,.008])).tolist(),orientation_deg=(angles+sign*np.array([.4*(-1 if i else 1),.2,0])).tolist()))
    for u in c['users']:u['motion'].update(max_speed_m_s=.05,max_acceleration_m_s2=.1)
    a=c['actors'][0];p=np.array(a['trajectory'][0]['position_m']);q=np.array(a['trajectory'][-1]['position_m']);center=(p+q)/2;side=(q-p)/np.linalg.norm(q-p)
    a.update(id='rail_mounted_metal_screen',position_m=(center-.4*side).tolist(),size_m=[.3,.6,2.6])
    a['trajectory'][0]['position_m']=a['position_m'];a['trajectory'][-1]['position_m']=(center+.4*side).tolist()
    for a in c['actors']:a['motion'].update(max_speed_m_s=.4,max_acceleration_m_s2=.6)
    c['actions']=sorted(actions,key=lambda a:a['time_s']);c['dataset']['capture_complete_paths']=True
    return validate(c)


def continuous_fullscale_example():
    """CPU-qualified native physical-port candidate. Loading authorizes no GPU.

    Reuse the already qualified 401-state trajectory byte-for-byte instead of
    inventing another large experiment or changing historical inputs.
    """
    from .config import ROOT,load
    return load(ROOT/'configs/platform/munich_continuous_full64_4s_NOT_authorized.json')
