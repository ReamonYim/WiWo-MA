"""CPU-only admission estimates. Historical observations are never a grant.

No CUDA import, RT query, file mutation, or assumption of physical truth.
Unknown configurations get byte counts, not invented GPU throughput.
"""
import math
import numpy as np
from .config import validate,estimated_storage,array_positions,frequency_offsets,fingerprint
from .motion import segment_bounds,trajectory_diagnostics


def sampling_assessment(config):
    c=validate(config);reasons=[]
    def bound(e,key):
        if not e['trajectory']:return 0.
        if e.get('motion',{}).get('model')!='bounded_quintic':
            reasons.append(e['id']+': legacy motion without C2 bound');return None
        return max((segment_bounds(a,b)[key] for a,b in zip(e['trajectory'],e['trajectory'][1:])),default=0.)
    # This version deliberately does not extrapolate to a moving BS frame.
    if any(b['trajectory'] for b in c['base_stations']):reasons.append('Moving BS bound not implemented')
    speeds=[bound(u,'max_speed_m_s') for u in c['users']]
    rotations=[bound(u,'max_angular_speed_rad_s') for u in c['users']]
    ue=[]
    for u,v,w in zip(c['users'],speeds,rotations):
        ue.append(None if v is None or w is None else v+w*float(np.linalg.norm(array_positions(u['array']),axis=1).max()))
    actors=[]
    for a in c['actors']:
        v=bound(a,'max_speed_m_s');w=bound(a,'max_angular_speed_rad_s')
        actors.append(None if v is None or w is None else v+w*float(np.linalg.norm(a['size_m']))/2)
    # Diffuse/refraction/diffraction need separate optical-path derivative proofs.
    if any(c['rt'][k] for k in ('diffuse_reflection','refraction','diffraction','edge_diffraction')):reasons.append('Non-specular mechanisms need a separate bound')
    tx=max(p['mechanics']['max_speed_m_s']+p['mechanics']['max_angular_speed_rad_s']*float(np.linalg.norm(array_positions(p['array']),axis=1).max()) for b in c['base_stations'] for p in b['panels'])
    rate=None if reasons else tx+max(ue)+2*c['rt']['max_depth']*max(actors,default=0.)
    hz=None if rate is None else (c['radio']['carrier_hz']+max(abs(frequency_offsets(c))))*rate/299792458.
    nyquist=.5/c['timing']['csi_dt_s']
    return dict(status='Unknown' if hz is None else 'Passed_conditional_phase_bound' if nyquist>hz else 'Failed_conditional_phase_sampling',
        CSI_sample_Hz=1/c['timing']['csi_dt_s'],Nyquist_Hz=nyquist,optical_length_rate_bound_m_s=rate,persistent_specular_phase_bound_Hz=hz,
        minimum_sample_Hz_strictly_greater_than=None if hz is None else 2*hz,reasons=reasons,
        definition='Sum of endpoint speed bounds plus 2*Depth*maximum rigid actor point speed; actual C2 segment bounds, actuator speed envelopes; static city/BS',
        timing_diagnostics=timing_diagnostics(c),
        limitations=['Analytic bound, not measured Doppler or a physical certification','Visibility/path birth/death discontinuities need not be bandlimited','Does not model antenna pattern null sensitivity or guarantee all CSI bandwidth'])


def timing_diagnostics(config):
    """Clock/envelope diagnostics, never an integration-error certificate."""
    c=validate(config);dt=c['timing']['mechanics_dt_s'];csidt=c['timing']['csi_dt_s'];control=c['timing']['control_dt_s']
    panels=[p for b in c['base_stations'] for p in b['panels']]
    from .config import mechanical_axes
    active_t=[p['mechanics'] for p in panels if mechanical_axes(p['mechanics'])[0]]
    active_r=[p['mechanics'] for p in panels if mechanical_axes(p['mechanics'])[1]]
    maximum=lambda group,key:max((m[key] for m in group),default=0.)
    motions=[trajectory_diagnostics(e) for key in ('base_stations','users','actors') for e in c[key]]
    jumps=[dict(entity=m['id'],**jump) for m in motions for jump in m['velocity_discontinuities']]
    coincident=abs(control/csidt-round(control/csidt))<1e-7
    return dict(mechanics_dt_s=dt,CSI_dt_s=csidt,control_dt_s=control,
        mechanics_steps_per_CSI=round(csidt/dt),mechanics_steps_per_control=round(control/dt),
        every_control_boundary_on_CSI_grid=coincident,
        max_panel_translation_per_mechanics_tick_m=maximum(active_t,'max_speed_m_s')*dt,
        max_panel_velocity_change_per_tick_m_s=maximum(active_t,'max_acceleration_m_s2')*dt,
        max_panel_rotation_per_mechanics_tick_rad=maximum(active_r,'max_angular_speed_rad_s')*dt,
        max_panel_angular_velocity_change_per_tick_rad_s=maximum(active_r,'max_angular_acceleration_rad_s2')*dt,
        legacy_velocity_jumps=jumps,
        risk_flags=(['Legacy velocity discontinuities invalidate a global C2 motion claim'] if jumps else [])+([] if coincident else ['CSI samples do not land on every control boundary']),
        integration_accuracy='Unknown_without_step_refinement; bounds are maximum increments, not measured solver truncation errors',
        recommendation='For persistent-path Doppler use sample rate >2*phase bound with margin; use event-focused sampling for visibility changes; never interpolate samples as Full RT labels')


def estimate(config):
    c=validate(config);shape=estimated_storage(c);frames=shape['frames'];refs=min(frames,len(c['dataset'].get('independent_reference_frames',[]))+c['dataset'].get('independent_reference_extremes',0))
    chunks=math.ceil(sum(len(array_positions(p['array'])) for b in c['base_stations'] for p in b['panels'])/c['rt']['tx_device_chunk'])
    applicable=(c['scene']['name']=='munich' and shape['tx_ports']==64 and shape['users']==40 and shape['rx_ports_max']==1
        and c['radio']['tx_polarization'] in ('V','H') and c['radio']['rx_polarization'] in ('V','H') and c['rt']['max_depth']==3
        and c['rt']['samples_per_src']==100000 and c['rt']['max_num_paths_per_src']==1000000 and c['rt']['tx_device_chunk']==4
        and c['rt']['los'] and c['rt']['specular_reflection'] and not any(c['rt'][k] for k in ('diffuse_reflection','diffraction','refraction','edge_diffraction')))
    # Historical long-run mean15.95 and three saved K40/64 native observations
    # 21.10655/17.39241/17.66902s, all28GHz. New3.5GHz not timed yet.
    baseline=[15.95,21.10654753446579,17.3924098610878,17.669023260474205]
    total=frames+refs
    wall=[total*min(baseline)+30*(2+refs),total*max(baseline)*1.5+60*(2+refs)+600] if applicable else None
    # 3 saved full64 cases: HDF2,855kB(single-row chunk overhead included),
    # complete paths798–815kB. Conservative extra1.6MiB states/HDF and2x path
    # allowance; two copies including frozen delivery. No compression credit.
    paths=bool(c['dataset'].get('capture_complete_paths'))
    disk=2*(shape['CSI_uncompressed_bytes']+(frames+refs)*int((1.6+(1.6 if paths else 0))*1024**2)+refs*shape['CSI_uncompressed_bytes']//frames)+256*1024**2 if applicable else None
    return dict(schema='native-platform-resource-estimate-v1',status='Partial_historical_estimate' if applicable else 'Unknown_GPU_cost_no_matching_calibration',
        configuration_sha256=fingerprint(c),dimensions=shape,physical_duration_s=c['timing']['duration_s'],native_queries=frames,independent_reference_queries=refs,
        SDK_calls_per_query=chunks,estimated_SDK_calls=total*chunks,estimated_total_wall_s=wall,
        estimated_disk_including_one_frozen_copy_bytes=disk,allocator_limit_recommendation_bytes=12*1024**3 if applicable else None,
        historical_allocator_peak_bytes=3586348800 if applicable else None,historical_NVML_observed_peak_bytes=4961861632 if applicable else None,
        memory_status='Unknown_peak_for_new_fullscale_window; 12GiB is a guard, not a measured peak',
        sampling=sampling_assessment(c),new_GPU_seconds=0,GPU_authorized=False,
        calibration=dict(full64_long_run_mean_s=15.95,three_selected_native_query_s=baseline[1:],carrier_hz=28e9,
            source='docs/platform/CONTINUOUS_NATIVE_WEB_GATE.md; artifacts/platform/validation/physical_native_gate_attempt03/full64_{0,1,2}/native_metadata.json',
            caveat='Historical28GHz, different motion; current frequency/materials/path count and competing jobs can change cost. Not a new measurement or guaranteed ETA'),
        limitations=['Raw CSI bytes exact; state/path/disk and wall estimates conservative but unvalidated for new window',
            'One planned restart and per-reference fresh initialization included; extra retries require remaining authorized budget',
            'All independent budgets remain separate; loading this estimate never authorizes any GPU job'])


def markdown(q):
    d=q['dimensions'];wall=q['estimated_total_wall_s'];disk=q['estimated_disk_including_one_frozen_copy_bytes'];s=q['sampling']
    return ('**CPU资源预估（不授权GPU）**  '+str(d['frames'])+'帧 / '+str(d['tx_ports'])+'真实TX / '+str(d['users'])+'UE  \n'
        +'原始CSI '+format(d['CSI_uncompressed_bytes']/1024**2,'.2f')+' MiB；独立参考 '+str(q['independent_reference_queries'])+'帧。  \n'
        +('历史估计总墙钟 '+format(wall[0]/3600,'.2f')+'–'+format(wall[1]/3600,'.2f')+'小时，含初始化/参考/恢复；新场景尚未计时。' if wall else 'GPU耗时Unknown：此配置未建立可复用的资源预估模型。')+'  \n'
        +('预计含冻结副本 '+format(disk/1024**3,'.2f')+' GiB；显存保护建议12 GiB，实际新峰值Unknown。' if disk else '状态/路径存储及峰值显存Unknown。')+'  \n'
        +'CSI采样 '+format(s['CSI_sample_Hz'],'.1f')+'Hz / Nyquist '+format(s['Nyquist_Hz'],'.1f')+'Hz；'
        +('持续镜面路径解析相位上界 '+format(s['persistent_specular_phase_bound_Hz'],'.2f')+'Hz。' if s['persistent_specular_phase_bound_Hz'] is not None else '相位上界Unknown。')+'  \n'
        +'遮挡/路径出生消失不保证带限。机械预览不计算CSI；此估计不是实测Doppler或GPU许可。  \n'
        +'采样判定：'+s['status']+'；风险：'+str(s['reasons']+s['timing_diagnostics']['risk_flags'])+'。机械步长误差未经步长细化认证。')
