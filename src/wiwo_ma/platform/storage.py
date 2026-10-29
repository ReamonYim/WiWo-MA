"""Streaming complex HDF5, committed rows and recoverable write-ahead frame.

No pickle/eval. Snapshot types are explicitly whitelisted. CSI, physical state,
actions, paths, snapshot and row hash commit together; recovery truncates only
uncommitted tails. Power-loss durability depends on the filesystem/HDF5 driver.
"""
from dataclasses import fields
import fcntl,hashlib,json,os
from pathlib import Path
import h5py,numpy as np
from wiwo_ma.mechanics import MechanicalState,_Trajectory
from .config import validate,fingerprint,frequency_offsets,estimated_storage


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,complex):return [value.real,value.imag]
    if isinstance(value,float) and not np.isfinite(value):raise ValueError('Nonfinite JSON metadata')
    return value


def time_index(timestamps,t):
    """Floor a physical time, allowing only Float64 tick roundoff at a grid.

    GUI decimal 2.01 and mechanics tick1005*.002 can straddle one ULP.
    Off-grid requests still select the preceding frame, never the nearest one.
    """
    if not np.isfinite(t) or not len(timestamps):raise ValueError('Finite timestamp in nonempty dataset required')
    tolerance=8*np.spacing(max(1.,abs(float(t))))
    return int(np.clip(np.searchsorted(timestamps,float(t)+tolerance,side='right')-1,0,len(timestamps)-1))


def encode(value):
    if isinstance(value,np.ndarray):return {'@':'array','dtype':str(value.dtype),'value':value.tolist()}
    if isinstance(value,np.generic):return encode(value.item())
    if isinstance(value,MechanicalState):return {'@':'MechanicalState','value':encode(vars(value))}
    if isinstance(value,_Trajectory):return {'@':'Trajectory','value':encode(vars(value))}
    if isinstance(value,dict):return {'@':'dict','value':[[encode(k),encode(v)] for k,v in value.items()]}
    if isinstance(value,tuple):return {'@':'tuple','value':[encode(v) for v in value]}
    if isinstance(value,list):return [encode(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        if np.isnan(value):raise ValueError('NaN in mechanical snapshot')
        return {'@':'infinity','value':1 if value>0 else -1}
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError('Unserializable mechanical snapshot '+type(value).__name__)


def decode(value):
    if isinstance(value,list):return [decode(x) for x in value]
    if not isinstance(value,dict):return value
    kind=value.get('@');v=value.get('value')
    if kind=='array':
        dtype=np.dtype(value['dtype'])
        if dtype.kind not in 'biuf':raise ValueError('Only numeric state arrays are supported')
        return np.asarray(v,dtype=dtype)
    if kind=='dict':return {decode(k):decode(x) for k,x in v}
    if kind=='tuple':return tuple(decode(x) for x in v)
    if kind=='infinity':
        if v not in (-1,1):raise ValueError('Invalid infinity tag')
        return float('inf')*v
    if kind=='MechanicalState':
        d=decode(v)
        if set(d)!={f.name for f in fields(MechanicalState)}:raise ValueError('MechanicalState schema mismatch')
        return MechanicalState(**d)
    if kind=='Trajectory':
        d=decode(v);allowed={'start','target','angular','speed','acceleration','initial_velocity','brake_time','stop','distance','direction','accel_time','peak_speed','cruise_time','duration'}
        if set(d)!=allowed:raise ValueError('Trajectory schema mismatch')
        obj=object.__new__(_Trajectory);obj.__dict__.update(d);return obj
    raise ValueError('Unknown snapshot type')


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(plain(value),f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def row_digest(H,text):return hashlib.sha256(np.ascontiguousarray(H,np.complex64).tobytes()+text.encode()).hexdigest()


class CSIWriter:
    def __init__(self,folder,config,*,resume=False):
        self.folder=Path(folder);self.config=validate(config);self.configuration=fingerprint(config);self.pending=self.folder/'pending_frame.npz'
        self.folder.mkdir(parents=True,exist_ok=True);path=self.folder/'csi.h5'
        if path.exists() and not resume:raise FileExistsError(path)
        self.lock=(self.folder/'writer.lock').open('a+b')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.lock.close();raise
        # Earliest format avoids latest-superblock writer-open flags on a
        # killed process. No concurrent writer/reader of this handle is allowed.
        self.file=h5py.File(path,'r+' if path.exists() else 'w',libver='earliest')
        if 'schema' not in self.file.attrs:
            self.file.attrs.update(schema='wiwo-native-csi-v1',configuration_sha256=self.configuration,config_json=json.dumps(self.config,sort_keys=True),
                axes='time,user,rx_antenna,tx_antenna,subcarrier',complex_storage='complex64 real+imag',position_units='m',time_units='s',frequency_units='Hz',
                committed_frames=0,complete=False,label_scope='Every stored native frame independently solved; independent numerical validation Unknown unless separately recorded')
            self.file.create_dataset('frequency_offsets_hz',data=frequency_offsets(self.config))
        elif self.file.attrs['configuration_sha256']!=self.configuration:
            self.file.close();self.lock.close();raise ValueError('Resume configuration mismatch')
        self.count=int(self.file.attrs['committed_frames'])
        for name in ['H','timestamps','frames_json','frame_sha256','label_source','independent_validation']:
            if name in self.file:
                d=self.file[name]
                if len(d)<self.count:raise ValueError('Committed dataset is truncated')
                if len(d)>self.count:d.resize(self.count,axis=0)
        if self.count:
            text=self.file['frames_json'].asstr()[self.count-1];h=self.file['H'][self.count-1]
            if row_digest(h,text)!=self.file['frame_sha256'].asstr()[self.count-1]:raise ValueError('Last committed frame hash mismatch')
        self.flush()

    def flush(self):
        self.file.flush()
        try:
            handle=self.file.id.get_vfd_handle()
            if isinstance(handle,int):os.fsync(handle)
        except (AttributeError,NotImplementedError,RuntimeError):pass

    def stage(self,result,state,snapshot):
        if self.pending.exists():raise FileExistsError('Unresolved pending frame must be recovered first')
        h=np.asarray(result['H'],np.complex64)
        shape=estimated_storage(self.config)
        if h.shape!=(shape['users'],shape['rx_ports_max'],shape['tx_ports'],shape['subcarriers']) or not np.isfinite(h).all():raise ValueError('Invalid complex CSI shape/values')
        if result['physical_timestamp']!=state['timestamp'] or state['configuration_sha256']!=self.configuration:raise ValueError('CSI/state timestamp or configuration mismatch')
        if result['label_source'] not in ('Sionna_RT_2.2.0_native_full_per_port','test_fixture'):raise ValueError('Unknown explicit CSI provenance')
        metadata={k:v for k,v in result.items() if k!='H'}
        text=json.dumps(dict(state=plain(state),snapshot=encode(snapshot),result=plain(metadata)),sort_keys=True,separators=(',',':'),allow_nan=False)
        digest=row_digest(h,text);tmp=self.pending.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,H=h,frame_json=np.array(text),frame_sha256=np.array(digest),index=np.array(self.count));f.flush();os.fsync(f.fileno())
        tmp.replace(self.pending)

    def commit_pending(self):
        if not self.pending.exists():return False
        with np.load(self.pending,allow_pickle=False) as f:h=f['H'];text=str(f['frame_json']);sha=str(f['frame_sha256']);index=int(f['index'])
        if row_digest(h,text)!=sha:raise ValueError('Pending frame hash mismatch')
        if index==self.count-1:
            if self.file['frame_sha256'].asstr()[index]!=sha:raise ValueError('Already-committed pending frame conflicts')
            self.pending.unlink();return True
        if index!=self.count:raise ValueError('Pending frame index mismatch')
        value=json.loads(text);t=value['state']['timestamp']
        if self.count and t<=float(self.file['timestamps'][self.count-1]):raise ValueError('Nonmonotonic or duplicate physical timestamp')
        if self.count and abs(t-float(self.file['timestamps'][self.count-1])-self.config['timing']['csi_dt_s'])>1e-8:raise ValueError('CSI time-grid gap; missing labels cannot be silently skipped')
        if 'H' not in self.file:
            chunks=(1,1,min(h.shape[1],4),min(h.shape[2],64),min(h.shape[3],64))
            self.file.create_dataset('H',shape=(0,*h.shape),maxshape=(None,*h.shape),dtype=np.complex64,chunks=chunks,compression=self.config['dataset']['compression'])
            self.file.create_dataset('timestamps',shape=(0,),maxshape=(None,),dtype='f8',chunks=True)
            for name in ['frames_json','frame_sha256','label_source','independent_validation']:
                self.file.create_dataset(name,shape=(0,),maxshape=(None,),dtype=h5py.string_dtype(),chunks=True)
        for name in ['H','timestamps','frames_json','frame_sha256','label_source','independent_validation']:self.file[name].resize(self.count+1,axis=0)
        self.file['H'][self.count]=h;self.file['timestamps'][self.count]=t;self.file['frames_json'][self.count]=text;self.file['frame_sha256'][self.count]=sha
        self.file['label_source'][self.count]=value['result']['label_source'];self.file['independent_validation'][self.count]=value['result']['independent_validation_status']
        self.flush();self.count+=1;self.file.attrs['committed_frames']=self.count;self.flush()
        atomic_json(self.folder/'checkpoint.json',dict(committed_frames=self.count,last_timestamp=t,last_frame_sha256=sha,configuration_sha256=self.configuration))
        # Readers of running jobs use this atomic one-frame publication, not
        # concurrent access to a non-SWMR HDF5 writer.
        latest=self.folder/'latest_frame.npz';tmp=latest.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,H=h,frame_json=np.array(text),index=np.array(index));f.flush();os.fsync(f.fileno())
        tmp.replace(latest)
        self.pending.unlink();return True

    def latest_snapshot(self):
        if not self.count:return None
        return decode(json.loads(self.file['frames_json'].asstr()[self.count-1])['snapshot'])

    def complete(self):self.file.attrs['complete']=True;self.flush()
    def close(self):
        try:self.flush();self.file.close()
        finally:self.lock.close()


class CSIReader:
    def __init__(self,path):
        self.path=Path(path)
        with h5py.File(self.path,'r') as f:
            if f.attrs.get('schema')!='wiwo-native-csi-v1':raise ValueError('Unsupported native CSI format')
            self.config=json.loads(f.attrs['config_json']);self.count=int(f.attrs['committed_frames']);self.complete=bool(f.attrs['complete']);self.timestamps=f['timestamps'][:self.count]
    def frame(self,index,*,verify=True):
        if not 0<=index<self.count:raise IndexError(index)
        with h5py.File(self.path,'r') as f:
            H=f['H'][index];text=f['frames_json'].asstr()[index]
            digest=f['frame_sha256'].asstr()[index]
            if verify and row_digest(H,text)!=digest:raise ValueError('Frame hash mismatch')
            value=json.loads(text);value.update(H=H,index=index,frame_sha256=digest);return value
    def seek(self,t):
        return self.frame(time_index(self.timestamps,t))
