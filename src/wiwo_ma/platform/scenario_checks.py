"""CPU admission checks on actual city triangles and canonical physical ports.

Geometric LOS is not a native SDK CSI validation. Mesh/OBB surface screening
does not certify watertight solid containment, road legality or vehicle IK.
"""
import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .config import validate, fingerprint
from .physics import PhysicalWorld
from .motion import trajectory_diagnostics
from .scene_geometry import scene_xml, load_display_meshes
from .dataset_audit import segment_box
from .storage import atomic_json


def path_quality(points):
    p = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    # SDK EPSILON_FLOAT = 100 * Float32 unit roundoff; offset is position-scaled.
    offset = 100*2**-24*(1+np.max(np.abs(p), axis=-1))
    budget = offset[:-1]+offset[1:]
    return dict(min_segment_m=float(lengths.min()), max_endpoint_offset_ratio=float(np.max(budget/np.maximum(lengths, 1e-30))),
                short_segment_offset_sensitive=bool(np.any(lengths < 100*budget)),
                scope='Diagnostic only; original SDK path, CIR and CSI are unmodified')


def current_state_preflight(config,snapshot):
    """One frozen pose, not a certification of its remaining trajectory."""
    from wiwo_ma.v4.collision import boxes_overlap
    c=validate(config);world=PhysicalWorld(c);world.restore(snapshot);s=world.state();world._check(s)
    geometry=StaticGeometry(c);violations=[]
    for i,(_,panel) in enumerate(world.panels):
        count=geometry.box_surface_intersections(s['positions'][i],s['rotations'][i],panel['size_m'])
        if count:violations.append(dict(panel=panel['id'],reason='panel_static_surface_penetration',triangles=count))
    for i,actor in enumerate(s['actors']):
        ground=c['actors'][i].get('ground_z_m')
        count=geometry.box_surface_intersections(actor['position_m'],actor['rotation_matrix'],actor['size_m'],ground)
        if count:violations.append(dict(actor=actor['id'],reason='actor_static_surface_penetration',triangles=count))
        bottom=float(np.min(actor['vertices'][:,2]))
        if ground is not None and abs(bottom-ground)>2e-4:violations.append(dict(actor=actor['id'],reason='actor_ground_contact'))
        if np.any(np.all(np.abs((s['rx_port_positions']-actor['position_m'])@actor['rotation_matrix'])<np.asarray(actor['size_m'])/2,axis=1)):violations.append(dict(actor=actor['id'],reason='actor_contains_RX'))
        for previous in s['actors'][:i]:
            if boxes_overlap(actor['position_m'],actor['rotation_matrix'],np.asarray(actor['size_m'])/2,previous['position_m'],previous['rotation_matrix'],np.asarray(previous['size_m'])/2):violations.append(dict(actor=actor['id'],reason='actor_collision'))
    from .state_identity import geometry_digest
    return dict(status='Failed' if violations else 'Passed',scope='One frozen CPU pose only; no full-trajectory or SDK numerical certification',
        configuration_sha256=fingerprint(c),physical_time_s=s['timestamp'],initial_geometry_sha256=geometry_digest(s),
        failure_types=sorted({v['reason'] for v in violations}),violations=violations,scene_resources=geometry.resources,
        new_GPU_seconds=0,CSI_numerical_status='Unknown',limitations=['Open city surface screening cannot certify enclosed solid containment','No support structure or self-scattering certification'])


class StaticGeometry:
    def __init__(self, config):
        import mitsuba as mi
        if mi.variant() and mi.variant() != 'scalar_rgb':
            raise RuntimeError('Run preflight in an isolated CPU process, never switch a CUDA solver variant')
        mi.set_variant('scalar_rgb'); self.mi = mi
        xml = scene_xml(config['scene']); d = {'type': 'scene'}
        for i, shape in enumerate(ET.parse(xml).getroot().findall('shape')):
            if shape.find('transform') is not None: raise NotImplementedError('Scene transforms need an explicit geometry adapter')
            file = xml.parent/shape.find("string[@name='filename']").get('value')
            d[str(i)] = dict(type=shape.get('type'), filename=str(file), face_normals=True)
        self.scene = mi.load_dict(d)
        meshes, resources = load_display_meshes(str(xml))
        self.triangles = np.concatenate([m['vertices'][m['faces']] for m in meshes]).astype(float)
        self.lo, self.hi = self.triangles.min(axis=1), self.triangles.max(axis=1)
        self.resources = [dict(path=str(xml), sha256=hashlib.sha256(xml.read_bytes()).hexdigest())]+resources

    def controls(self):
        success=0;tested=0
        for tri in self.triangles:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);length=np.linalg.norm(normal)
            if length<1e-8:continue
            p=tri.mean(axis=0);n=normal/length
            tested+=1;success+=self.blocked(p-.002*n,p+.002*n)
            if tested==12:break
        ceiling=float(self.hi[:,2].max()+10);p=np.array([0.,0.,ceiling])
        empty=not self.blocked(p,p+[0,0,1.])
        return dict(triangle_crossings=tested,positive_hits=int(success),empty_upward_ray_clear=empty,status='Passed' if tested==success==12 and empty else 'Failed')

    def blocked(self, a, b):
        mi = self.mi; delta = np.asarray(b, float)-a; length = float(np.linalg.norm(delta))
        if length <= 2e-4: return True
        ray = mi.Ray3f(mi.Point3f(np.asarray(a)+delta/length*1e-4), mi.Vector3f(delta/length)); ray.maxt = length-2e-4
        return bool(self.scene.ray_test(ray))

    def box_surface_intersections(self, center, rotation, size, ground_z=None):
        half = np.asarray(size)/2; extent = np.abs(rotation)@half
        mask = np.all(self.lo < center+extent-1e-7, axis=1)&np.all(self.hi > center-extent+1e-7, axis=1)
        triangles = self.triangles[mask]
        if ground_z is not None:
            triangles = triangles[np.max(np.abs(triangles[:,:,2]-ground_z), axis=1)>1e-6]
        if not len(triangles): return 0
        tri = (triangles-center)@rotation
        edges = np.roll(tri, -1, axis=1)-tri
        axes = [np.tile(np.eye(3)[i], (len(tri), 1)) for i in range(3)]
        axes.append(np.cross(edges[:,0], edges[:,1]))
        axes += [np.cross(edges[:,j], np.eye(3)[i]) for i in range(3) for j in range(3)]
        overlap = np.ones(len(tri), bool)
        for axis in axes:
            projections = np.sum(tri*axis[:,None,:], axis=-1); radius = np.abs(axis)@half
            nondegenerate = np.linalg.norm(axis, axis=1)>1e-12
            separated = (projections.min(axis=1)>=radius-1e-7)|(projections.max(axis=1)<=-radius+1e-7)
            overlap &= ~(nondegenerate&separated)
        return int(overlap.sum())

    def supported(self, points):
        """Actual static surface within 0.1mm of each bottom-corner height."""
        mi=self.mi
        for p in points:
            ray=mi.Ray3f(mi.Point3f(np.asarray(p)+[0,0,.02]),mi.Vector3f(0,0,-1));ray.maxt=.04
            hit=self.scene.ray_intersect(ray)
            if not hit.is_valid() or abs(float(hit.p.z)-p[2])>1e-4:return False
        return True


def check_motion_contract(config):
    rows = [dict(kind=kind, **trajectory_diagnostics(e)) for kind in ('base_stations','users','actors') for e in config[kind]]
    bad = [r for r in rows if r['velocity_discontinuities']]
    return dict(status='Failed' if bad else 'Passed', entities=rows,
                limitation='C2 waypoint stops and speed/acceleration bounds; no vehicle steering or biomechanical certification')


def preflight(config, output=None):
    start = time.perf_counter(); c = validate(config); world = PhysicalWorld(c)
    motion = check_motion_contract(c); geometry = StaticGeometry(c);controls=geometry.controls()
    opts = c.get('physics_validation', {}); duration = c['timing']['duration_s']; dt = c['timing']['mechanics_dt_s']
    csi_ticks = round(c['timing']['csi_dt_s']/dt)
    violations = []; times = []; blocked = []; sampled = []; snapshots = {}
    from wiwo_ma.v4.collision import boxes_overlap
    for tick in range(round(duration/dt)+1):
        s = world.advance_to(tick*dt) if tick else world.state()
        for i,(bi,panel) in enumerate(world.panels):
            center,r=np.asarray(s['positions'][i]),np.asarray(s['rotations'][i]);m=panel['mechanics'];size=np.asarray(panel['size_m'])
            count=geometry.box_surface_intersections(center,r,size)
            if count:violations.append(dict(tick=tick,panel=i,reason='panel_static_surface_penetration',triangles=count))
            if tick<round(duration/dt):
                bs=c['base_stations'][bi];bm=bs.get('motion',{});radius=np.linalg.norm(size)/2
                if bs['trajectory'] and bm.get('model')!='bounded_quintic':
                    violations.append(dict(tick=tick,panel=i,reason='unbounded_BS_panel_surface_sweep'));continue
                bs_v=bm.get('max_speed_m_s',0.) if bs['trajectory'] else 0.
                bs_w=bm.get('max_angular_speed_rad_s',0.) if bs['trajectory'] else 0.
                reach=np.linalg.norm(np.maximum(np.abs(m['workspace_min_m']),np.abs(m['workspace_max_m'])))+radius
                margin=(bs_v+bs_w*reach+m['max_speed_m_s']+radius*m['max_angular_speed_rad_s'])*dt
                swept=geometry.box_surface_intersections(center,r,size+2*margin)
                if swept:violations.append(dict(tick=tick,panel=i,reason='uncertified_panel_city_swept_clearance',triangles=swept))
        # Every mechanical tick, not only CSI samples. City surface penetration
        # remains a sampled screen; report its resolution rather than claiming CCD.
        for a in s['actors']:
            spec = next(x for x in c['actors'] if x['id']==a['id'])
            ground = spec.get('ground_z_m')
            if ground is not None and np.max(np.abs(np.sort(a['vertices'][:,2])[:4]-ground))>1e-4:
                violations.append(dict(tick=tick, actor=a['id'], reason='ground_contact_or_ground_penetration'))
            if ground is not None and not geometry.supported(a['vertices'][np.argsort(a['vertices'][:,2])[:4]]):
                violations.append(dict(tick=tick, actor=a['id'], reason='no_static_support_at_bottom_corners'))
            count = geometry.box_surface_intersections(a['position_m'], a['rotation_matrix'], a['size_m'], ground)
            if count: violations.append(dict(tick=tick, actor=a['id'], reason='static_surface_penetration', triangles=count))
            if tick<round(duration/dt) and spec.get('motion',{}).get('model')=='bounded_quintic':
                m=spec['motion'];radius=np.linalg.norm(a['size_m'])/2
                # Every intermediate rigid point lies within this inflated
                # start-frame box: |translation| + radius * |rotation angle|.
                margin=(m['max_speed_m_s']+radius*m['max_angular_speed_rad_s'])*dt
                swept=geometry.box_surface_intersections(a['position_m'],a['rotation_matrix'],np.asarray(a['size_m'])+2*margin,ground)
                if swept:violations.append(dict(tick=tick,actor=a['id'],reason='uncertified_swept_city_surface_clearance',triangles=swept))
            if np.any(np.all(np.abs((s['rx_port_positions']-a['position_m'])@a['rotation_matrix'])<np.asarray(a['size_m'])/2,axis=1)):
                violations.append(dict(tick=tick, actor=a['id'], reason='actor_contains_RX'))
        for i,a in enumerate(s['actors']):
            for b in s['actors'][:i]:
                if boxes_overlap(a['position_m'],a['rotation_matrix'],np.asarray(a['size_m'])/2,b['position_m'],b['rotation_matrix'],np.asarray(b['size_m'])/2):
                    violations.append(dict(tick=tick, actor=a['id'], other=b['id'],reason='actor_collision'))
        if tick in (0, round(duration/dt)//2, round(duration/dt)): snapshots[str(tick)] = world.snapshot()
        if tick%csi_ticks: continue
        tx, rx = s['tx_port_positions'], s['rx_port_positions']
        actor_mask = np.zeros((len(rx),len(tx)), bool)
        for a in s['actors']: actor_mask |= segment_box(tx[None],rx[:,None],a['position_m'],a['rotation_matrix'],a['size_m'])
        static = np.array([[geometry.blocked(a,b) for a in tx] for b in rx])
        blocked.append(actor_mask|static); times.append(s['timestamp'])
        sampled.append(dict(time_s=s['timestamp'],static_blocked=int(static.sum()),actor_blocked=int(actor_mask.sum()),geometric_NLOS=int((actor_mask|static).sum())))
    b = np.asarray(blocked); enter = (~b[:-1])&b[1:]; leave = b[:-1]&~b[1:]
    transitions = []
    for i in np.flatnonzero(np.any(enter|leave,axis=(1,2))):
        transitions.append(dict(before_frame=int(i),after_frame=int(i+1),time_s=times[i+1],LOS_to_NLOS=int(enter[i].sum()),NLOS_to_LOS=int(leave[i].sum())))
    # Replay from the midpoint snapshot must reconstruct the same canonical end state.
    final = s; mid = str(round(duration/dt)//2); restored = None
    if mid in snapshots:
        world.restore(snapshots[mid]); restored = world.advance_to(duration)
    restore_ok = restored is not None and all(np.array_equal(final[key],restored[key]) for key in ('tx_port_positions','rx_port_positions','rotations','user_velocities'))
    cycle = bool(np.any(enter)&np.any(leave)); failures = []
    if motion['status']!='Passed': failures.append('unbounded_legacy_velocity_jump')
    if controls['status']!='Passed':failures.append('static_geometry_positive_negative_controls')
    if violations: failures.append('physical_geometry_screen')
    if opts.get('require_LOS_cycle') and not cycle: failures.append('required_LOS_NLOS_LOS_cycle_missing')
    if not restore_ok: failures.append('state_restore')
    report = dict(status='Failed' if failures else 'Passed',scope='CPU kinematics, sampled city-surface screening and geometric LOS only',
        configuration_sha256=fingerprint(c),new_GPU_seconds=0,CSI_numerical_status='Unknown',
        mechanics_ticks=round(duration/dt)+1,CSI_samples=len(times),links_per_sample=int(b[0].size),motion=motion,
        geometry_controls=controls,
        violations=violations,failure_types=failures,LOS_cycle=cycle,LOS_to_NLOS=int(enter.sum()),NLOS_to_LOS=int(leave.sum()),
        transitions=transitions,sampled_visibility=sampled,branch_restore_bitwise=bool(restore_ok),
        limitations=['No native SDK propagation or CSI computed','Open city meshes: enclosed solid containment is not certified',
            'Collision screening at mechanics ticks, not certified continuous collision detection','No ground friction, vehicle steering, suspension or support-structure dynamics'],
        wall_s=time.perf_counter()-start,scene_resources=geometry.resources,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    if output:
        out = Path(output); out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'report.json',report); atomic_json(out/'config.json',c)
        np.savez_compressed(out/'geometric_visibility.npz',timestamps=times,blocked=b)
    return report
