"""One causal physical clock; bounded actuator execution and exact rigid ports.

Reuse the existing acceleration executor and SO(3)/OBB implementations. A
Cartesian gantry plus gimbal is an explicit mechanism, not a robot IK claim.
All SDK/UI port coordinates use the SAME final Float32 world coordinates.
"""
from copy import deepcopy
import math
import numpy as np
from wiwo_ma.mechanics import MechanicalController,euler_to_matrix,matrix_to_euler,rotation_exp,rotation_log
from wiwo_ma.v4.collision import boxes_overlap
from wiwo_ma.v5.actor_mesh import absolute_vertices
from .config import validate,array_positions,mechanical_axes,fingerprint,hardware_mode


def pose_at(entity,t):
    if entity.get('motion',{}).get('model')=='bounded_quintic':
        from .motion import bounded_pose
        return bounded_pose(entity,t)
    route=entity.get('trajectory',[])
    if not route:return np.asarray(entity['position_m']),euler_to_matrix(entity['orientation_deg']),np.zeros(3),np.zeros(3)
    if t>=route[-1]['time_s']:
        q=route[-1];return np.asarray(q['position_m']),euler_to_matrix(q['orientation_deg']),np.zeros(3),np.zeros(3)
    for a,b in zip(route,route[1:]):
        if a['time_s']<=t<b['time_s']:
            dt=b['time_s']-a['time_s'];s=(t-a['time_s'])/dt;p,q=np.asarray(a['position_m']),np.asarray(b['position_m'])
            r0,r1=euler_to_matrix(a['orientation_deg']),euler_to_matrix(b['orientation_deg']);omega=rotation_log(r1@r0.T)/dt
            return p+(q-p)*s,rotation_exp(omega*(t-a['time_s']))@r0,(q-p)/dt,omega
    raise ValueError('Trajectory does not cover time0')


def box_vertices(size):
    return np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],float)*np.asarray(size)/2


BOX_FACES=np.array([[0,4,6],[0,6,2],[1,3,7],[1,7,5],[0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,2,3],[0,3,1],[4,5,7],[4,7,6]],np.uint32)


def constrained_target(config,bs_id,panel_id,displacement,rotation_vector):
    """Exact BS-local target from initial mount and permitted joint generators.

    Explicit relative coordinates preserve locked components exactly. Never
    project an illegal world pose, snap a received target or relax tolerances.
    SO(3) exponential composes with the actual initial orientation, not its
    rounded UI Euler display. A one-axis restriction is a fixed local hinge.
    """
    bs=next(b for b in config['base_stations'] if b['id']==bs_id);p=next(p for p in bs['panels'] if p['id']==panel_id);ta,ra=mechanical_axes(p['mechanics'])
    d=np.asarray(displacement,float);v=np.asarray(rotation_vector,float)
    if d.shape!=(3,) or v.shape!=(3,) or not np.isfinite(d).all() or not np.isfinite(v).all():raise ValueError('Finite relative mechanical target required')
    if any(d[k]!=0 for k in range(3) if k not in ta) or any(v[k]!=0 for k in range(3) if k not in ra):raise ValueError('Nonzero target on a locked mechanical axis')
    r=rotation_exp(v)@euler_to_matrix(p['orientation_deg'])
    return dict(bs_id=bs_id,panel_id=panel_id,type='target_pose',frame='bs_local',position_m=(np.array(p['position_m'])+d).tolist(),orientation_deg=matrix_to_euler(r).tolist())


class PhysicalWorld:
    def __init__(self,config):
        self.config=validate(config);self.configuration=fingerprint(config);self.tick=0;self.action_index=0
        self.controllers=[];self.panels=[];self.keys={};self.auto_stops={};self.executed=[];self.scheduled=[]
        for bi,bs in enumerate(self.config['base_stations']):
            for p in bs['panels']:
                m=p['mechanics'];spec=dict(model='acceleration',workspace_min=m['workspace_min_m'],workspace_max=m['workspace_max_m'],
                    orientation_min_deg=[-180.,-90.,-180.],orientation_max_deg=[180.,90.,180.],
                    max_speed=m['max_speed_m_s'],max_acceleration=m['max_acceleration_m_s2'],
                    max_angular_speed_deg=math.degrees(m['max_angular_speed_rad_s']),max_angular_acceleration_deg=math.degrees(m['max_angular_acceleration_rad_s2']),
                    command_latency=m['command_latency_s'],settling_time=0.,execution_policy='replace')
                self.controllers.append(MechanicalController({'mechanics':spec},p['position_m'],p['orientation_deg']))
                key=(str(bs['id']),p['id'])
                if key in self.keys:raise ValueError('Duplicate mechanical group')
                self.keys[key]=len(self.panels);self.panels.append((bi,p))
        self._apply_due_actions();self._check(self.state())

    @property
    def time(self):return self.tick*self.config['timing']['mechanics_dt_s']

    def state(self):
        bsposes=[pose_at(b,self.time) for b in self.config['base_stations']]
        positions=[];rotations=[];velocity=[];angular=[];targets=[];targetR=[];ports=[];portmap=[];portangles=[];status=[];ids=[];mounts=[]
        pol=2 if self.config['radio']['tx_polarization'] in ('VH','cross') else 1
        for j,((bi,p),ctrl) in enumerate(zip(self.panels,self.controllers)):
            bp,br,bv,bw=bsposes[bi];s=ctrl.state;point=bp+br@s.position;r=br@s.rotation
            positions.append(point);rotations.append(r);velocity.append(bv+br@s.velocity+np.cross(bw,br@s.position));angular.append(bw+br@s.angular_velocity)
            targets.append(bp+br@s.target_position);targetR.append(br@s.target_rotation);mounts.append(bp+br@np.asarray(p['position_m']))
            spatial=absolute_vertices(array_positions(p['array']),point,r)
            for element,xyz in enumerate(spatial):
                for polarization in range(pol):ports.append(xyz);portmap.append((bi,j,element,polarization));portangles.append(np.deg2rad(matrix_to_euler(r)).astype(np.float32))
            status.append(dict(movement_state=s.movement_state,command_id=s.command_id,completed=s.completed,
                remaining_time_s=None if not np.isfinite(s.remaining_time) else s.remaining_time,mechanism=p['mechanics']['mechanism']))
            ids.append((str(self.config['base_stations'][bi]['id']),p['id']))
        users=[];urot=[];rxpositions=[];rxmap=[];rxangles=[];uv=[];uw=[];actors=[]
        rxpol=2 if self.config['radio']['rx_polarization'] in ('VH','cross') else 1
        for i,u in enumerate(self.config['users']):
            p,r,v,w=pose_at(u,self.time);users.append(p);urot.append(r);uv.append(v);uw.append(w)
            for element,xyz in enumerate(absolute_vertices(array_positions(u['array']),p,r)):
                for polarization in range(rxpol):rxpositions.append(xyz);rxmap.append((i,element,polarization));rxangles.append(np.deg2rad(matrix_to_euler(r)).astype(np.float32))
        for a in self.config['actors']:
            p,r,v,w=pose_at(a,self.time);actors.append(dict(id=a['id'],position_m=p,rotation_matrix=r,velocity_m_s=v,angular_velocity_rad_s=w,
                size_m=a['size_m'],vertices=absolute_vertices(box_vertices(a['size_m']),p,r),faces=BOX_FACES.copy(),material=a['material']))
        return dict(timestamp=self.time,mechanics_tick=self.tick,configuration_sha256=self.configuration,
            panel_ids=ids,positions=np.asarray(positions),rotations=np.asarray(rotations),velocities=np.asarray(velocity),angular_velocities=np.asarray(angular),
            target_positions=np.asarray(targets),target_rotations=np.asarray(targetR),mount_positions=np.asarray(mounts),mechanics=status,
            tx_port_positions=np.asarray(ports,np.float32),tx_port_map=np.asarray(portmap,np.int32),tx_port_orientations_rad=np.asarray(portangles,np.float32),
            rx_port_positions=np.asarray(rxpositions,np.float32),rx_port_map=np.asarray(rxmap,np.int32),rx_port_orientations_rad=np.asarray(rxangles,np.float32),
            user_positions=np.asarray(users),user_rotations=np.asarray(urot),user_velocities=np.asarray(uv),user_angular_velocities=np.asarray(uw),
            bs_positions=np.asarray([x[0] for x in bsposes]),bs_rotations=np.asarray([x[1] for x in bsposes]),hardware_modes=[hardware_mode(b) for b in self.config['base_stations']],actors=actors,
            coordinate_contract='Metres,Z-up; ports are final Float32 world coordinates, identical SDK/UI inputs',actions_executed=len(self.executed))

    def submit(self,action):
        """Accept UI actions for the next configured controller tick.

        Preflight a copy with inherited pending commands, without changing the
        current physical pose. Low-level command() also serves emergency stops.
        """
        period=self.config['timing']['control_dt_s'];t=math.ceil((self.time-1e-10)/period)*period
        if abs(t-self.time)<1e-9:return self.command(action)
        if t>self.config['timing']['duration_s']:raise ValueError('No control tick left within horizon')
        old=self.snapshot()
        try:self.advance_to(t);self.command(action)
        finally:self.restore(old)
        self.scheduled.append((t,deepcopy(action)));self.scheduled.sort(key=lambda x:x[0])
        return dict(accepted=True,submitted_time_s=self.time,effective_time_s=t,pending_control_tick=True)

    def submit_many(self,actions):
        """One transaction at one control tick, including joint future clearance.

        Validate the combined motion, not each target against an obsolete set
        of the other panels' commands. A rejected member rolls back everything.
        """
        actions=deepcopy(actions)
        if not actions or len(actions)>len(self.panels):raise ValueError('Nonempty panel action batch required')
        keys=[(str(a['bs_id']),str(a['panel_id'])) for a in actions]
        if len(set(keys))!=len(keys):raise ValueError('At most one command per panel in a transaction')
        old=self.snapshot();period=self.config['timing']['control_dt_s']
        t=math.ceil((self.time-1e-10)/period)*period
        if t>self.config['timing']['duration_s']:raise ValueError('No control tick left within horizon')
        try:
            self.advance_to(t)
            for a in actions:self.command(a,preflight=False)
            accepted=self.snapshot();self._preflight_motion();self.restore(accepted)
            if t>old['tick']*self.config['timing']['mechanics_dt_s']+1e-9:
                self.restore(old)
                self.scheduled.extend((t,a) for a in actions);self.scheduled.sort(key=lambda x:x[0])
            return dict(accepted=True,panels=len(actions),submitted_time_s=old['tick']*self.config['timing']['mechanics_dt_s'],effective_time_s=t,joint_clearance_checked=True)
        except BaseException:self.restore(old);raise

    def _preflight_motion(self):
        ends=[self.time]
        for c in self.controllers:
            trial=deepcopy(c)
            if trial.pending:
                arrival=max(x[0] for x in trial.pending);trial.advance(max(0.,arrival-trial.state.time))
            if trial.velocity_command is None:ends.append(trial.settle_end)
        for j,(tick,_) in self.auto_stops.items():
            m=self.panels[j][1]['mechanics']
            ends.append(tick*self.config['timing']['mechanics_dt_s']+2*max(m['max_speed_m_s']/m['max_acceleration_m_s2'],m['max_angular_speed_rad_s']/m['max_angular_acceleration_rad_s2']))
        dt=self.config['timing']['mechanics_dt_s']
        horizon=min(max(ends),self.config['timing']['duration_s'])
        self.advance_to(min(self.config['timing']['duration_s'],math.ceil((horizon-1e-10)/dt)*dt),script=False)

    def _check(self,s):
        for (bi,p),ctrl in zip(self.panels,self.controllers):
            m=p['mechanics'];a=ctrl.state.position
            if np.any(a<np.asarray(m['workspace_min_m'])-1e-9) or np.any(a>np.asarray(m['workspace_max_m'])+1e-9):raise ValueError('Mechanical workspace violation')
            ta,ra=mechanical_axes(m);delta=a-np.asarray(p['position_m']);rot=rotation_log(ctrl.state.rotation@euler_to_matrix(p['orientation_deg']).T)
            if any(abs(delta[k])>1e-8 for k in range(3) if k not in ta) or any(abs(rot[k])>1e-8 for k in range(3) if k not in ra):raise ValueError('Command exceeds mechanism degrees of freedom')
            if np.linalg.norm(rot)>m['max_rotation_rad']+1e-8:raise ValueError('Mechanical orientation envelope exceeded')
        for i in range(len(self.panels)):
            for j in range(i):
                radius=sum(np.linalg.norm(self.panels[k][1]['size_m'])/2 for k in (i,j))
                if np.linalg.norm(s['positions'][i]-s['positions'][j])>radius+1e-10:continue
                if boxes_overlap(s['positions'][i],s['rotations'][i],np.asarray(self.panels[i][1]['size_m'])/2,
                    s['positions'][j],s['rotations'][j],np.asarray(self.panels[j][1]['size_m'])/2):raise ValueError('Panel collision')
        # Actors are physical obstacles, including when a user reconfigures
        # their location. This sampled OBB check is not city-solid/CCD proof.
        for i,(_,p) in enumerate(self.panels):
            for actor in s['actors']:
                radius=(np.linalg.norm(p['size_m'])+np.linalg.norm(actor['size_m']))/2
                if np.linalg.norm(s['positions'][i]-actor['position_m'])>radius:continue
                if boxes_overlap(s['positions'][i],s['rotations'][i],np.asarray(p['size_m'])/2,actor['position_m'],actor['rotation_matrix'],np.asarray(actor['size_m'])/2):raise ValueError('Panel/actor collision')

    def _advance_tick(self):
        before=self.state();dt=self.config['timing']['mechanics_dt_s'];end=(self.tick+1)*dt
        for c in self.controllers:c.advance(dt,timestamp_end=end)
        self.tick+=1
        for j,(tick,command) in list(self.auto_stops.items()):
            if self.tick>=tick:
                self.controllers[j].command({'type':'stop'});del self.auto_stops[j]
        after=self.state();self._check(after)
        # Conservative whole-interval certificate. All point rotations fit
        # inside their bounding spheres; unproved close motion is rejected.
        for i in range(len(self.panels)):
            for j in range(i):
                moving=np.linalg.norm(after['positions'][i]-before['positions'][i])+np.linalg.norm(after['positions'][j]-before['positions'][j])
                rotating=np.linalg.norm(after['rotations'][i]-before['rotations'][i])+np.linalg.norm(after['rotations'][j]-before['rotations'][j])
                if moving+rotating<1e-14:continue
                radius=sum(np.linalg.norm(self.panels[k][1]['size_m'])/2 for k in (i,j))
                distance=np.linalg.norm(before['positions'][i]-before['positions'][j])
                speed=sum(self.panels[k][1]['mechanics']['max_speed_m_s']+np.linalg.norm(before['velocities'][k]) for k in (i,j))
                if distance-speed*dt<=radius:raise ValueError('Cannot certify collision clearance for moving panels')

    def advance_to(self,t,*,script=True):
        dt=self.config['timing']['mechanics_dt_s'];target=int(round(float(t)/dt))
        if target<self.tick or abs(target*dt-t)>1e-8 or t>self.config['timing']['duration_s']+1e-9:raise ValueError('Forward mechanics-grid timestamp inside horizon required')
        checkpoint=self.snapshot()
        try:
            while self.tick<target:
                self._advance_tick()
                if script:self._apply_due_actions()
        except BaseException:self.restore(checkpoint);raise
        return self.state()

    def _apply_due_actions(self):
        script=self.config['actions']
        while self.action_index<len(script) and script[self.action_index]['time_s']<=self.time+1e-10:
            action=script[self.action_index];self.command(action,preflight=False);self.action_index+=1
        while self.scheduled and self.scheduled[0][0]<=self.time+1e-10:
            _,action=self.scheduled.pop(0);self.command(action,preflight=False)

    def command(self,action,*,preflight=True):
        a=deepcopy(action);key=(str(a['bs_id']),str(a['panel_id']))
        if key not in self.keys:raise ValueError('Unknown BS/panel')
        j=self.keys[key];bi,p=self.panels[j];m=p['mechanics'];kind=a.get('type','target_pose')
        if kind not in ('target_pose','velocity','stop','stay'):raise ValueError('Unsupported mechanical action')
        ctrl=self.controllers[j];old=self.snapshot();base_speed,base_omega=ctrl.max_speed,ctrl.max_angular_speed
        request={'type':kind}
        try:
            if kind=='target_pose':
                point=np.asarray(a.get('position_m',ctrl.state.target_position),float);angles=np.asarray(a.get('orientation_deg',matrix_to_euler(ctrl.state.target_rotation)),float)
                if point.shape!=(3,) or angles.shape!=(3,) or not np.isfinite(point).all() or not np.isfinite(angles).all():raise ValueError('Finite target pose required')
                if a.get('frame','bs_local')=='world':
                    bp,br,_,_=pose_at(self.config['base_stations'][bi],self.time);point=br.T@(point-bp);angles=matrix_to_euler(br.T@euler_to_matrix(angles))
                elif a.get('frame','bs_local')!='bs_local':raise ValueError('Unknown coordinate frame')
                ta,ra=mechanical_axes(m);delta=point-np.asarray(p['position_m']);rot=rotation_log(euler_to_matrix(angles)@euler_to_matrix(p['orientation_deg']).T)
                if any(abs(delta[k])>1e-9 for k in range(3) if k not in ta) or any(abs(rot[k])>1e-9 for k in range(3) if k not in ra):raise ValueError('Target exceeds hardware degrees of freedom')
                if np.any(point<np.asarray(m['workspace_min_m'])) or np.any(point>np.asarray(m['workspace_max_m'])) or np.linalg.norm(rot)>m['max_rotation_rad']:raise ValueError('Target outside mechanical limits')
                request.update(position=point.tolist(),orientation_deg=angles.tolist())
            if kind=='velocity':
                v=np.asarray(a.get('velocity_m_s',[0,0,0]),float);w=np.asarray(a.get('angular_velocity_rad_s',[0,0,0]),float)
                if v.shape!=(3,) or w.shape!=(3,) or not np.isfinite(v).all() or not np.isfinite(w).all():raise ValueError('Finite velocity vectors required')
                ta,ra=mechanical_axes(m)
                if np.linalg.norm(v)>base_speed+1e-12 or np.linalg.norm(w)>base_omega+1e-12 or any(abs(v[k])>1e-12 for k in range(3) if k not in ta) or any(abs(w[k])>1e-12 for k in range(3) if k not in ra):raise ValueError('Velocity command exceeds physical limits')
                duration=float(a.get('duration_s',0));ticks=round(duration/self.config['timing']['mechanics_dt_s'])
                if not np.isfinite(duration) or duration<=0 or abs(ticks*self.config['timing']['mechanics_dt_s']-duration)>1e-8:raise ValueError('Velocity needs finite duration on mechanics grid')
                request.update(velocity=v.tolist(),angular_velocity=w.tolist());self.auto_stops[j]=(self.tick+ticks,ctrl.command_counter+1)
            else:self.auto_stops.pop(j,None)
            ctrl.command(request)
            if kind=='target_pose' and a.get('duration_s') is not None:
                if ctrl.latency>0:raise ValueError('Exact target duration with nonzero latency is not supported; use zero latency or omit duration')
                duration=float(a['duration_s']);minimum=ctrl.settle_end-ctrl.state.time
                if not np.isfinite(duration) or duration<minimum-1e-9:raise ValueError('Requested duration shorter than feasible actuator motion')
                # Exact slower target profile using the EXISTING executor;
                # acceleration remains bounded and brake inheritance retained.
                original=old['controllers'][j];low,high=1e-6,1.
                for _ in range(45):
                    mid=(low+high)/2;ctrl.restore(original);ctrl.max_speed=base_speed*mid;ctrl.max_angular_speed=base_omega*mid;ctrl.command(request)
                    if ctrl.settle_end-ctrl.state.time>duration:low=mid
                    else:high=mid
                ctrl.restore(original);ctrl.max_speed=base_speed*high;ctrl.max_angular_speed=base_omega*high;ctrl.command(request)
                ctrl.max_speed,ctrl.max_angular_speed=base_speed,base_omega
            self._check(self.state());submitted=self.snapshot()
            if preflight:
                self._preflight_motion()
                self.restore(submitted)
            self.executed.append(dict(time_s=self.time,action=a,accepted=True))
            return dict(accepted=True,time_s=self.time,command_id=self.controllers[j].state.command_id)
        except BaseException:self.restore(old);raise

    def snapshot(self):
        return deepcopy(dict(configuration=self.configuration,tick=self.tick,action_index=self.action_index,
            controllers=[c.snapshot() for c in self.controllers],auto_stops=self.auto_stops,executed=self.executed,scheduled=self.scheduled))

    def restore(self,snapshot):
        s=deepcopy(snapshot)
        if s['configuration']!=self.configuration or len(s['controllers'])!=len(self.controllers):raise ValueError('Snapshot configuration does not match')
        self.tick=s['tick'];self.action_index=s['action_index'];self.auto_stops=s['auto_stops'];self.executed=s['executed'];self.scheduled=s.get('scheduled',[])
        for c,value in zip(self.controllers,s['controllers']):c.restore(value)
