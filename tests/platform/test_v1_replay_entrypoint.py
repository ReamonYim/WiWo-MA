import sys
import pytest
from wiwo_ma.platform.storage import CSIWriter
from wiwo_ma.platform.physics import PhysicalWorld
from test_session_and_failure_recovery import config,frame


@pytest.mark.parametrize('complete',[False,True])
def test_replay_command_embedded_config_no_authorization_and_rejects_live_HDF(tmp_path,monkeypatch,complete):
    import wiwo_ma.platform.cli as cli
    import wiwo_ma.platform.web as web
    c=config();w=PhysicalWorld(c);writer=CSIWriter(tmp_path,c);writer.stage(*frame(w));writer.commit_pending()
    if complete:writer.complete()
    writer.close();seen={}
    monkeypatch.setattr(web,'serve',lambda c,**kw:seen.update(configuration=c,**kw))
    monkeypatch.setattr(sys,'argv',['wiwo-ma-platform','replay','--dataset',str(tmp_path/'csi.h5'),'--port','8134'])
    if complete:
        cli.main();assert seen['configuration']==w.config and seen['port']==8134 and seen['replay']==str(tmp_path/'csi.h5') and 'authorization' not in seen
    else:
        with pytest.raises(ValueError,match='closed complete'):cli.main()


def test_environment_metadata_reports_explicit_local_OptiX_override_without_loading_it(tmp_path,monkeypatch):
    from wiwo_ma.platform.environment import inspect_environment
    library=tmp_path/'libnvoptix.so';library.write_text('explicit_CPU_test_not_a_real_library')
    monkeypatch.setenv('WIWO_OPTIX_LIBRARY',str(library))
    report=inspect_environment();assert report['optix_library_exists'] and report['optix_library_path']==str(library)
    assert report['native_runtime_compatibility'].startswith('Unknown') and report['new_GPU_seconds']==0
