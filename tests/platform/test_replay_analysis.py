import numpy as np
import pytest
from wiwo_ma.platform.storage import CSIWriter
from wiwo_ma.platform.physics import PhysicalWorld
from wiwo_ma.platform.replay_analysis import analyze_link
from test_physics import config


@pytest.mark.parametrize('amplitude',[1e-5,0.])
def test_headless_analysis_reads_real_saved_values_without_normalization(tmp_path,amplitude):
    c=config();world=PhysicalWorld(c);writer=CSIWriter(tmp_path,c)
    for i in range(7):
        world.advance_to(i*.01);h=np.ones((1,1,4,64),np.complex64)*np.complex64(np.exp(2j*np.pi*3*i*.01)*amplitude)
        writer.stage(dict(H=h,physical_timestamp=world.time,label_source='test_fixture',independent_validation_status='Unknown',paths=[]),world.state(),world.snapshot());writer.commit_pending()
    writer.close()
    with pytest.raises(ValueError,match='closed'):analyze_link(tmp_path/'csi.h5')
    writer=CSIWriter(tmp_path,c,resume=True);writer.complete();writer.close()
    q=analyze_link(tmp_path/'csi.h5');assert q['original_frames']==q['sampled_frames']==7 and q['stride']==1
    np.testing.assert_allclose(q['amplitude_dB'],-100 if amplitude else -600,atol=2e-6)
    if amplitude:np.testing.assert_allclose(q['ACF_magnitude'],1.,atol=1e-12)
    else:assert q['ACF_magnitude']==[None]*7 and q['ACF_status'].startswith('Unknown')
    assert q['first_0p5_crossing_s'] is None
    with pytest.raises(ValueError,match='index'):analyze_link(tmp_path/'csi.h5',user=1)
