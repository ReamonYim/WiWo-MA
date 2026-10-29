"""Pinned original Sionna RT2.2.0, real one-location devices per spatial port.

ULA/UPA offsets are rigid physical coordinates computed by PhysicalWorld.
SDK single-location arrays supply the configured polarization patterns; no
synthetic-array, Image cache, reused CIR, learned model or Doppler extension.
Chunk boundaries/seed are part of the reference definition and saved config.
"""
import importlib.metadata as metadata
import json,time
from pathlib import Path
import numpy as np
from wiwo_ma.scene import load_asset
from wiwo_ma.mechanics import matrix_to_euler
from .config import ROOT,validate,frequency_offsets
from .physics import box_vertices,BOX_FACES


def spatial_devices(state,side,polarizations):
    positions=np.asarray(state[side+'_port_positions'],np.float32);mapping=np.asarray(state[side+'_port_map'],np.int32)
    rows=np.arange(0,len(mapping),polarizations)
    if len(positions)%polarizations or np.any(mapping[rows,-1]!=0):raise ValueError('Invalid physical port/polarization mapping')
    for p in range(polarizations):
        if not np.array_equal(positions[rows],positions[rows+p]):raise ValueError('Polarizations must share a physical location')
    return positions[rows],mapping[rows]


def arrange_cfr(raw,rx_map,tx_device_indices,config):
    """SDK [rxDevice,rxPol,txDevice,txPol,time=1,F] → [UE,Rx,Tx,F]."""
    rxpol=2 if config['radio']['rx_polarization'] in ('VH','cross') else 1
    txpol=2 if config['radio']['tx_polarization'] in ('VH','cross') else 1
    nusers=len(config['users']);counts=np.bincount(rx_map[:,0],minlength=nusers)*rxpol
    nfreq=config['radio']['num_subcarriers'];raw=np.asarray(raw)
    expected=(len(rx_map),rxpol,len(tx_device_indices),txpol,1,nfreq)
    if raw.shape!=expected:raise ValueError('SDK CFR shape changed: '+str(raw.shape)+' expected '+str(expected))
    out=np.zeros((nusers,int(max(counts)),len(tx_device_indices)*txpol,nfreq),np.complex64)
    valid=np.zeros((nusers,int(max(counts))),bool)
    for user in range(nusers):
        selected=np.flatnonzero(rx_map[:,0]==user)
        values=raw[selected,:,:,:,0,:].reshape(len(selected)*rxpol,len(tx_device_indices)*txpol,nfreq)
        out[user,:len(values)]=values;valid[user,:len(values)]=True
    return out,valid


class NativeRT:
    def __init__(self,config,*,check_budget=None,complete_path_sink=None):
        self.config=validate(config);self.check_budget=check_budget or (lambda:None);self.check_budget()
        self.complete_path_sink=complete_path_sink
        start=time.perf_counter()
        import sionna.rt as rt
        import mitsuba as mi
        import drjit as dr
        if rt.__version__!='2.2.0' or not mi.variant().startswith('cuda'):raise RuntimeError('Native research backend requires pinned Sionna2.2.0 CUDA variant')
        self.rt,self.mi,self.dr=rt,mi,dr
        dr.set_flag(dr.JitFlag.SymbolicCalls,False);dr.set_flag(dr.JitFlag.SymbolicLoops,False)
        self.scene,self.asset=load_asset(self.config);r=self.config['radio'];self.scene.frequency=r['carrier_hz']
        self.scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern=r['tx_pattern'],polarization=r['tx_polarization'])
        self.scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern=r['rx_pattern'],polarization=r['rx_polarization'])
        self.solver=rt.PathSolver(deterministic=True);self.solver.loop_mode='evaluated'
        self.tx_names=[];self.rx_names=[];self.actors={}
        if self.config['scene'].get('material_override'):
            material=rt.ITURadioMaterial(name='platform_static_override',itu_type=self.config['scene']['material_override'],thickness=.1)
            self.scene.add(material)
            for obj in self.scene.objects.values():obj.radio_material=material
        for actor in self.config['actors']:
            vertices=box_vertices(actor['size_m']);code=__import__('hashlib').sha256(vertices.tobytes()).hexdigest()[:16]
            path=ROOT/'assets/platform_generated'/('box_'+code+'.ply');path.parent.mkdir(exist_ok=True,parents=True)
            if not path.exists():
                lines=['ply','format ascii 1.0','element vertex 8','property float x','property float y','property float z','element face 12','property list uchar int vertex_indices','end_header']
                lines += [' '.join(map(str,v)) for v in vertices];lines += ['3 '+' '.join(map(str,v)) for v in BOX_FACES]
                path.write_text('\n'.join(lines)+'\n')
            mat=rt.ITURadioMaterial(name='platform_material_'+actor['id'],itu_type=actor['material'],thickness=.01)
            obj=rt.SceneObject(name='platform_actor_'+actor['id'],fname=str(path),radio_material=mat);self.scene.add(obj);self.actors[actor['id']]=obj
        self.initialization_seconds=time.perf_counter()-start;self.calls=0

    def _update_actor(self,state):
        mi,dr=self.mi,self.dr
        for actor in state['actors']:
            obj=self.actors[actor['id']];vertices=np.asarray(actor['vertices'],np.float32)
            obj._transform_vertices(lambda _,v=vertices:mi.Point3f(v.T))
            obj._orientation=mi.Point3f(np.deg2rad(matrix_to_euler(actor['rotation_matrix'])).tolist())
            obj.velocity=mi.Vector3f(np.asarray(actor['velocity_m_s'],np.float32).tolist())
            actual=np.asarray(obj.mi_mesh.vertex_positions_buffer()).reshape(-1,3)
            if not np.array_equal(actual.view(np.uint32),vertices.view(np.uint32)):raise ValueError('SDK actor BVH geometry differs from canonical UI vertices')
        dr.eval();dr.sync_thread()

    def solve(self,state):
        self.check_budget();mi,dr,rt=self.mi,self.dr,self.rt;start=time.perf_counter();cfg=self.config
        if state['configuration_sha256']!=__import__('wiwo_ma.platform.config',fromlist=['fingerprint']).fingerprint(cfg):raise ValueError('State/configuration mismatch')
        txpol=len(self.scene.tx_array.antenna_pattern.patterns);rxpol=len(self.scene.rx_array.antenna_pattern.patterns)
        tx,txmap=spatial_devices(state,'tx',txpol);rx,rxmap=spatial_devices(state,'rx',rxpol)
        for name in self.rx_names:self.scene.remove(name)
        self.rx_names=[]
        for i,(xyz,mapping) in enumerate(zip(rx,rxmap)):
            user=int(mapping[0]);name=f'platform_rx_{i:05d}'
            angles=np.asarray(state['rx_port_orientations_rad'][i*rxpol],np.float32)
            self.scene.add(rt.Receiver(name=name,position=mi.Point3f(xyz.tolist()),orientation=mi.Point3f(angles.tolist())))
            self.rx_names.append(name)
        self._update_actor(state);geometry_done=time.perf_counter();pieces=[];paths_out=[];batches=[];valid=None
        settings={key:cfg['rt'][key] for key in ['max_depth','max_num_paths_per_src','samples_per_src','synthetic_array','los','specular_reflection','diffuse_reflection','refraction','diffraction','edge_diffraction','diffraction_lit_region','seed']}
        for first in range(0,len(tx),cfg['rt']['tx_device_chunk']):
            self.check_budget();begun=time.perf_counter();indices=np.arange(first,min(first+cfg['rt']['tx_device_chunk'],len(tx)))
            for name in self.tx_names:self.scene.remove(name)
            self.tx_names=[]
            for i in indices:
                panel=int(txmap[i,1]);name=f'platform_tx_{i:05d}'
                angles=np.asarray(state['tx_port_orientations_rad'][i*txpol],np.float32)
                self.scene.add(rt.Transmitter(name=name,position=mi.Point3f(tx[i].tolist()),orientation=mi.Point3f(angles.tolist())))
                actual=np.asarray(self.scene.transmitters[name].orientation).reshape(3)
                if not np.array_equal(actual.view(np.uint32),angles.view(np.uint32)):raise ValueError('SDK antenna-pattern orientation differs from canonical Float32 angles')
                self.tx_names.append(name)
            p=self.solver(scene=self.scene,**settings);self.calls+=1
            dr.eval();dr.sync_thread();trace_done=time.perf_counter()
            actual_tx=np.asarray(p.sources).T;actual_rx=np.asarray(p.targets).T
            if not np.array_equal(actual_tx.view(np.uint32),tx[indices].view(np.uint32)) or not np.array_equal(actual_rx.view(np.uint32),rx.view(np.uint32)):
                raise ValueError('Actual SDK endpoint coordinates disagree with physical/UI state')
            raw=p.cfr(frequencies=mi.Float(frequency_offsets(cfg)),num_time_steps=1,normalize_delays=False,normalize=False,out_type='numpy')
            values,valid=arrange_cfr(raw,rxmap,indices,cfg);pieces.append(values)
            if self.complete_path_sink is not None:
                from .path_capture import complete_V_paths
                self.complete_path_sink(int(first),complete_V_paths(p,self.scene,cfg,int(first),rxmap))
            paths_out.extend(self._display_paths(p,indices,rxmap,state,txpol,rxpol))
            dr.sync_thread();batches.append(dict(first_tx_device=int(first),spatial_sources=len(indices),SDK_wall_seconds=trace_done-begun,
                CFR_and_capture_wall_seconds=time.perf_counter()-trace_done,total_wall_seconds=time.perf_counter()-begun))
            self.check_budget()
        H=np.concatenate(pieces,axis=2)
        if not np.isfinite(H).all():raise ValueError('Nonfinite native complex CSI')
        selected=sorted(paths_out,key=lambda p:-(p['coefficient_real']**2+p['coefficient_imag']**2))[:cfg['rt']['max_display_paths']]
        from .scenario_checks import path_quality
        for path in selected:path['geometry_quality']=path_quality(path['points'])
        return dict(H=H,rx_valid=valid,paths=selected,physical_timestamp=state['timestamp'],
            path_quality_scope='Retained display subset only; unretained paths Unknown; no path removed from CSI',
            offset_sensitive_retained_paths=sum(p['geometry_quality']['short_segment_offset_sensitive'] for p in selected),
            label_source='Sionna_RT_2.2.0_native_full_per_port',independent_validation_status='Unknown',synthetic_array=False,
            geometry_identity='Passed_actual_SDK_source_target_and_actor_Float32_operands',
            solver_wall_seconds=time.perf_counter()-start,geometry_update_seconds=geometry_done-start,SDK_calls=len(batches),batches=batches,
            allocator_peak_bytes=int(dr.detail.malloc_watermark(dr.JitBackend.CUDA)),
            initialization_seconds_separate=self.initialization_seconds,frequency_offsets_hz=frequency_offsets(cfg),
            path_capture_scope='Strongest finite subset for display; complete CSI is SDK CFR, no path filtering enters H',
            software={name:metadata.version(name) for name in ['sionna-rt','mitsuba','drjit','numpy','h5py']},scene_hash=self.asset.scene_hash)

    def _display_paths(self,paths,indices,rxmap,state,txpol,rxpol):
        maximum=self.config['rt']['max_display_paths']
        if maximum==0:return []
        b=paths._paths_buffer;n=b.buffer_size
        if not n:return []
        source=np.asarray(b.source_indices,int);target=np.asarray(b.target_indices,int);pathid=np.asarray(paths._path_ind,int)
        cir,tau=paths.cir(normalize_delays=False,num_time_steps=1,out_type='numpy')
        # Capture strongest polarization per physical path; H retains ALL.
        choices=[]
        for rp in range(rxpol):
            for tp in range(txpol):
                coef=cir[target,rp,source,tp,pathid,0];delay=tau[target,rp,source,tp,pathid]
                for i in np.argsort(-np.abs(coef),kind='stable')[:maximum]:choices.append((float(abs(coef[i])),int(i),rp,tp,complex(coef[i]),float(delay[i])))
        choices=sorted(choices,reverse=True)[:maximum];vertices=[];types=[]
        active=self.dr.full(self.mi.Bool,True,n)
        for d in range(self.config['rt']['max_depth']):
            kinds=np.asarray(b.get_interaction_type(self.mi.UInt(d+1),active),int);types.append(kinds)
            vertices.append(np.asarray(b.get_vertex(self.mi.UInt(d+1),active)).T)
        tx,txmap=spatial_devices(state,'tx',txpol);rx,_=spatial_devices(state,'rx',rxpol);out=[]
        for _,i,rp,tp,c,delay in choices:
            src=int(indices[source[i]]);tgt=int(target[i]);points=[tx[src].tolist()]
            interactions=[]
            for v,kind in zip(vertices,types):
                if kind[i]!=0:points.append(v[i].tolist());interactions.append(int(kind[i]))
            points.append(rx[tgt].tolist())
            out.append(dict(user=int(rxmap[tgt,0]),tx_port=src*txpol+tp,rx_spatial_element=int(rxmap[tgt,1]),rx_polarization=rp,
                points=points,interaction_types=interactions,delay_s=delay,coefficient_real=c.real,coefficient_imag=c.imag,timestamp=state['timestamp']))
        return out

    def close(self):
        self.scene=None;self.solver=None
