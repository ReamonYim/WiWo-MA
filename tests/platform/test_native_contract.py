import numpy as np,pytest
from wiwo_ma.platform.config import validate,estimated_storage
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.native_rt import spatial_devices,arrange_cfr
from test_physics import config


def test_independent_receive_arrays_dual_polarization_and_tensor_mapping():
    c=config(3);c['radio']['tx_polarization']='VH';c['radio']['rx_polarization']='VH'
    c['users'].append(dict(id='ue1',position_m=[5,2,1.5],array=dict(kind='ULA',rows=1,cols=2,spacing_m=.03)))
    c=validate(c);w=PhysicalWorld(c);s=w.state();tx,tm=spatial_devices(s,'tx',2);rx,rm=spatial_devices(s,'rx',2)
    assert len(tx)==12 and len(rx)==3 and len(s['tx_port_positions'])==24
    # Distinct sentinels for every SDK axis detect silent transpositions.
    raw=np.arange(3*2*4*2*1*64,dtype=np.float32).reshape(3,2,4,2,1,64).astype(np.complex64)
    h,mask=arrange_cfr(raw,rm,np.arange(4),c)
    assert h.shape==(2,4,8,64) and mask.tolist()==[[True,True,False,False],[True]*4]
    np.testing.assert_array_equal(h[1,3,7],raw[2,1,3,1,0])
    np.testing.assert_array_equal(h[0,0,0],raw[0,0,0,0,0])
    assert estimated_storage(c)['tx_ports']==24


def test_coordinate_mismatch_cannot_be_hidden_by_polarization_expansion():
    c=config();c['radio']['tx_polarization']='VH';w=PhysicalWorld(c);s=w.state();s['tx_port_positions'][1,0]+=.01
    with pytest.raises(ValueError,match='physical location'):spatial_devices(s,'tx',2)
