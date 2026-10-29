"""Versioned SI-unit physical system configuration, independent of CUDA/UI."""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[3]
SCHEMA='wiwo-native-platform-v1'
MECHANISMS={'fixed':([],[]),'rail_x':([0],[]),'rail_y':([1],[]),'rail_z':([2],[]),
    'planar_xy':([0,1],[]),'cartesian':([0,1,2],[]),'gimbal':([],[0,1,2]),'gantry_gimbal':([0,1,2],[0,1,2])}
HARDWARE_MODES=('fixed_array','single_ma','independent_ma','panel_6dma','multi_panel_6dma')


def mechanical_axes(mechanics):
    """Allowed BS-local translations and fixed BS-local rotation generators.

    A single hinge and SO(3) are closed rotation groups. An arbitrary two-axis
    rotation-vector mask is not a two-axis serial gimbal and is rejected.
    Optional restrictions never expand the underlying mechanism's freedoms.
    Missing keys preserve historical configurations and fingerprints.
    """
    base_t,base_r=MECHANISMS[mechanics['mechanism']]
    result=[]
    for key,allowed in (('translation_axes',base_t),('rotation_axes',base_r)):
        axes=mechanics.get(key,allowed)
        if not isinstance(axes,(list,tuple)) or any(type(k) is not int or k not in allowed for k in axes) or len(set(axes))!=len(axes):
            raise ValueError(key+' must be unique BS-local axes permitted by the mechanism')
        if key=='rotation_axes' and len(axes)==2:raise ValueError('Two-axis serial gimbals require joint kinematics; use locked, one fixed-axis hinge, or full SO(3)')
        result.append(list(axes))
    return tuple(result)


def hardware_mode(bs):
    p=bs['panels']
    if 'hardware_mode' in bs:return bs['hardware_mode']
    if all(x['mechanics']['mechanism']=='fixed' for x in p):return 'fixed_array'
    if all(len(array_positions(x['array']))==1 for x in p):return 'single_ma' if len(p)==1 else 'independent_ma'
    return 'panel_6dma' if len(p)==1 else 'multi_panel_6dma'


def finite_vec(value,name,n=3):
    a=np.asarray(value,float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(name+' requires finite SI values')
    return a.tolist()


def positive(value,name,zero=False):
    if isinstance(value,bool) or not np.isfinite(value) or value<0 or not zero and value==0:raise ValueError(name+' must be finite and '+('nonnegative' if zero else 'positive'))
    return float(value)


def integer(value,name,minimum=1):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError(name+' requires an integer >= '+str(minimum))
    return value


def array_positions(array):
    """Metre offsets, array broadside+x, row=z and column=y like Sionna UPA."""
    kind=array.get('kind','UPA');rows=array.get('rows',1);cols=array.get('cols',1);spacing=array.get('spacing_m',.05)
    integer(rows,'rows');integer(cols,'cols');positive(spacing,'spacing_m')
    if kind not in ('single','ULA','UPA'):raise ValueError('Array kind must be single, ULA or UPA')
    if kind=='single' and rows*cols!=1 or kind=='ULA' and rows!=1:raise ValueError('Invalid single/ULA dimensions')
    return np.array([[0.,(col-(cols-1)/2)*spacing,((rows-1)/2-row)*spacing] for col in range(cols) for row in range(rows)],float)


def normalize_motion(node):
    node['position_m']=finite_vec(node.get('position_m',[0,0,0]),'position_m')
    node['orientation_deg']=finite_vec(node.get('orientation_deg',[0,0,0]),'orientation_deg')
    route=node.setdefault('trajectory',[])
    last=-1.
    for point in route:
        t=positive(point['time_s'],'trajectory.time_s',True)
        if t<=last:raise ValueError('Trajectory timestamps must strictly increase')
        last=t;point['position_m']=finite_vec(point['position_m'],'trajectory.position_m')
        point['orientation_deg']=finite_vec(point.get('orientation_deg',node['orientation_deg']),'trajectory.orientation_deg')
    if route and route[0]['time_s']!=0:raise ValueError('Trajectory must start at physical time0')
    if route and (route[0]['position_m']!=node['position_m'] or route[0]['orientation_deg']!=node['orientation_deg']):raise ValueError('Initial pose and trajectory time0 must agree')
    from .motion import validate_motion
    validate_motion(node)


def validate(config):
    c=deepcopy(config)
    if c.get('schema')!=SCHEMA:raise ValueError('Explicit platform schema required')
    scene=c.setdefault('scene',{'name':'munich'})
    if scene.get('name') not in ('munich','etoile','reflector'):raise ValueError('Unknown supported scene')
    if scene.get('xml_path'):
        path=(ROOT/scene['xml_path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Scene XML must exist inside project')
    radio=c.setdefault('radio',{})
    for key,default in [('carrier_hz',3.5e9),('subcarrier_spacing_hz',30000.),('bandwidth_hz',1.92e6)]:radio[key]=positive(radio.get(key,default),'radio.'+key)
    radio['num_subcarriers']=integer(radio.get('num_subcarriers',64),'num_subcarriers')
    if radio['num_subcarriers']*radio['subcarrier_spacing_hz']>radio['bandwidth_hz']*(1+1e-12):raise ValueError('Subcarrier grid exceeds configured bandwidth')
    for side in ('tx','rx'):
        radio.setdefault(side+'_pattern','tr38901' if side=='tx' else 'iso')
        if radio[side+'_pattern'] not in ('iso','dipole','hw_dipole','tr38901'):raise ValueError('Unsupported SDK antenna pattern')
        radio.setdefault(side+'_polarization','V')
        if radio[side+'_polarization'] not in ('V','H','VH','cross'):raise ValueError('Unsupported SDK polarization')
    rt=c.setdefault('rt',{})
    for key,default,lower in [('max_depth',3,0),('samples_per_src',100000,1),('max_num_paths_per_src',1000000,1),('seed',42,0),('tx_device_chunk',4,1),('max_display_paths',128,0)]:rt[key]=integer(rt.get(key,default),'rt.'+key,lower)
    for key,default in [('los',True),('specular_reflection',True),('diffuse_reflection',False),('refraction',False),('diffraction',False),('edge_diffraction',False),('diffraction_lit_region',True)]:
        rt.setdefault(key,default)
        if not isinstance(rt[key],bool):raise ValueError('RT mechanism flags must be boolean')
    if rt.get('synthetic_array',False) is not False:raise ValueError('Research mode requires synthetic_array=False; geometry preview emits no CSI')
    rt['synthetic_array']=False
    if rt.get('loop_mode','evaluated') not in ('evaluated',):raise ValueError('Active platform uses unchanged evaluated SDK execution')
    if rt.get('precision','float32')!='float32':raise ValueError('Pinned native SDK provides Float32 propagation; other precisions are not validated')
    rt['loop_mode']='evaluated';rt['precision']='float32'
    timing=c.setdefault('timing',{})
    for key,default in [('duration_s',30.),('csi_dt_s',.01),('mechanics_dt_s',.002),('control_dt_s',.1)]:timing[key]=positive(timing.get(key,default),'timing.'+key)
    for key in ['duration_s','csi_dt_s','control_dt_s']:
        ratio=timing[key]/timing['mechanics_dt_s']
        if abs(ratio-round(ratio))>1e-7:raise ValueError(key+' must lie on integer mechanics ticks')
    if abs(timing['duration_s']/timing['csi_dt_s']-round(timing['duration_s']/timing['csi_dt_s']))>1e-7:raise ValueError('Duration must contain an integral number of CSI intervals')
    if not c.get('base_stations') or not c.get('users'):raise ValueError('At least one BS and one UE required')
    ids=set();panels=0
    for bs in c['base_stations']:
        normalize_motion(bs)
        if bs['id'] in ids:raise ValueError('Duplicate BS identifier')
        ids.add(bs['id'])
        if not bs.get('panels'):raise ValueError('Each BS needs at least one fixed or movable group')
        for panel in bs['panels']:
            panel['id']=str(panel['id'])
            if not panel['id'] or '/' in panel['id']:raise ValueError('Invalid panel identifier')
            panel.setdefault('position_m',[0,0,0]);panel['position_m']=finite_vec(panel['position_m'],'panel BS-local position_m')
            panel.setdefault('orientation_deg',[0,0,0]);panel['orientation_deg']=finite_vec(panel['orientation_deg'],'panel BS-local orientation_deg')
            panel.setdefault('array',dict(kind='UPA',rows=2,cols=2,spacing_m=.04));offsets=array_positions(panel['array'])
            panel.setdefault('size_m',[.015,.12,.12]);panel['size_m']=finite_vec(panel['size_m'],'panel.size_m')
            if min(panel['size_m'])<=0:raise ValueError('Positive panel dimensions required')
            if np.any(np.ptp(offsets,axis=0)>np.asarray(panel['size_m'])+1e-12):raise ValueError('Spatial array does not fit rigid panel')
            m=panel.setdefault('mechanics',{});m.setdefault('mechanism','gantry_gimbal')
            if m['mechanism'] not in MECHANISMS:raise ValueError('Unsupported mechanical hardware')
            mechanical_axes(m)
            p=np.asarray(panel['position_m']);m.setdefault('workspace_min_m',(p-.25).tolist());m.setdefault('workspace_max_m',(p+.25).tolist())
            low=np.asarray(finite_vec(m['workspace_min_m'],'workspace_min_m'));high=np.asarray(finite_vec(m['workspace_max_m'],'workspace_max_m'))
            if np.any(low>high) or np.any(p<low) or np.any(p>high):raise ValueError('Initial panel pose outside workspace')
            for key,default in [('max_speed_m_s',.15),('max_acceleration_m_s2',.4),('max_angular_speed_rad_s',.3),('max_angular_acceleration_rad_s2',.6),('max_rotation_rad',.6)]:m[key]=positive(m.get(key,default),'mechanics.'+key)
            m['command_latency_s']=positive(m.get('command_latency_s',0.),'command_latency_s',True)
            panels+=1
        if len({p['id'] for p in bs['panels']})!=len(bs['panels']):raise ValueError('Duplicate panel within BS')
        mode=hardware_mode(bs)
        if mode not in HARDWARE_MODES:raise ValueError('Unknown physical hardware_mode')
        if mode=='fixed_array' and any(p['mechanics']['mechanism']!='fixed' for p in bs['panels']):raise ValueError('Fixed hardware must use fixed mechanisms')
        if mode in ('single_ma','panel_6dma') and len(bs['panels'])!=1:raise ValueError('Single MA/rigid panel requires one mechanism group')
        if mode in ('single_ma','independent_ma') and any(len(array_positions(p['array']))!=1 for p in bs['panels']):raise ValueError('Independent MA hardware requires one spatial antenna per mechanism')
    if len({u['id'] for u in c['users']})!=len(c['users']):raise ValueError('Duplicate UE identifier')
    for u in c['users']:
        normalize_motion(u);u.setdefault('array',dict(kind='single',rows=1,cols=1,spacing_m=.04));array_positions(u['array'])
    c.setdefault('actors',[])
    if len({a['id'] for a in c['actors']})!=len(c['actors']):raise ValueError('Duplicate actor identifier')
    for a in c['actors']:
        normalize_motion(a);a['size_m']=finite_vec(a['size_m'],'actor.size_m')
        if min(a['size_m'])<=0:raise ValueError('Positive actor dimensions required')
        a.setdefault('material','metal')
        if 'ground_z_m' in a:
            if isinstance(a['ground_z_m'],bool) or not np.isfinite(a['ground_z_m']):raise ValueError('Finite actor ground_z_m required')
        if a['material'] not in ('metal','concrete','glass','wood','brick'):raise ValueError('Actor material must be an explicitly supported ITU material')
    if 'physics_validation' in c:
        for key in ('require_preflight','require_LOS_cycle','allow_legacy_discontinuous_motion'):
            if key in c['physics_validation'] and not isinstance(c['physics_validation'][key],bool):raise ValueError('physics_validation flags must be boolean')
    c.setdefault('actions',[]);last=-1.
    for action in c['actions']:
        t=positive(action['time_s'],'action.time_s',True)
        grid=timing['mechanics_dt_s'] if action.get('type') in ('stop','stay') else timing['control_dt_s']
        if t<last or t>timing['duration_s'] or abs(t/grid-round(t/grid))>1e-7:raise ValueError('Actions must be time-ordered on control ticks (emergency stop/stay may use mechanics ticks)')
        last=t
    c.setdefault('dataset',{});c['dataset'].setdefault('compression','lzf');c['dataset'].setdefault('checkpoint_every_frames',1)
    if 'capture_complete_paths' in c['dataset'] and not isinstance(c['dataset']['capture_complete_paths'],bool):raise ValueError('capture_complete_paths must be boolean')
    if 'independent_reference_frames' in c['dataset']:
        frames=c['dataset']['independent_reference_frames'];count=round(timing['duration_s']/timing['csi_dt_s'])+1
        if not isinstance(frames,list) or len(set(frames))!=len(frames) or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<count for i in frames):raise ValueError('Independent reference indices must be unique frame indices inside the episode')
    if 'independent_reference_extremes' in c['dataset']:
        value=c['dataset']['independent_reference_extremes']
        if type(value) is not int or value not in (0,2):raise ValueError('Independent extremes must be0 or2: weakest row and largest relative CSI transition')
    if c['dataset']['compression'] not in ('lzf','gzip',None):raise ValueError('Unsupported HDF5 compression')
    c.setdefault('seed',42);integer(c['seed'],'seed',0)
    return c


def fingerprint(config):return hashlib.sha256(json.dumps(validate(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):
    p=Path(path);return validate(json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text()))


def frequency_offsets(config):
    n=config['radio']['num_subcarriers'];return (np.arange(n,dtype=float)-(n-1)/2)*config['radio']['subcarrier_spacing_hz']


def estimated_storage(config):
    c=validate(config);pol=lambda x:2 if x in ('VH','cross') else 1
    tx=sum(len(array_positions(p['array'])) for b in c['base_stations'] for p in b['panels'])*pol(c['radio']['tx_polarization'])
    rx=max(len(array_positions(u['array'])) for u in c['users'])*pol(c['radio']['rx_polarization'])
    frames=int(round(c['timing']['duration_s']/c['timing']['csi_dt_s']))+1
    return dict(frames=frames,users=len(c['users']),rx_ports_max=rx,tx_ports=tx,subcarriers=c['radio']['num_subcarriers'],
        CSI_uncompressed_bytes=frames*len(c['users'])*rx*tx*c['radio']['num_subcarriers']*8,
        axes=['time','user','rx_antenna','tx_antenna','subcarrier'],storage_excludes_states_paths_and_HDF5_overhead=True)
