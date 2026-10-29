"""Environment inventory without importing Sionna/Mitsuba or creating CUDA."""
import importlib.metadata as metadata
import platform,sys,os,hashlib
from pathlib import Path
from .config import ROOT


def inspect_environment():
    packages={}
    for name in ('sionna-rt','mitsuba','drjit','numpy','scipy','h5py','viser','PyYAML','matplotlib','playwright'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    missing=[k for k,v in packages.items() if v is None and k!='playwright']
    from .gui_delivery import UPSTREAM_HTML_SHA256
    try:
        frontend=Path(metadata.distribution('viser').locate_file('viser/client/build/index.html'))
        frontend_hash=hashlib.sha256(frontend.read_bytes()).hexdigest()
    except (metadata.PackageNotFoundError,FileNotFoundError):frontend_hash=None
    frontend_ok=packages['viser']=='1.1.1' and frontend_hash==UPSTREAM_HTML_SHA256
    optix=Path(os.environ.get('WIWO_OPTIX_LIBRARY',os.environ.get('DRJIT_LIBOPTIX_PATH','')))
    return dict(status='Passed_metadata_only' if not missing and packages['sionna-rt']=='2.2.0' and frontend_ok else 'Failed_dependency_metadata_or_frontend_build',
        python=sys.version,executable=sys.executable,system=platform.platform(),packages=packages,missing=missing,
        native_runtime_compatibility='Unknown_this_CPU_command_does_not_initialize_CUDA',optix_library_exists=optix.is_file(),optix_library_path=str(optix) if str(optix)!='.' else 'system discovery; not checked',
        ordered_GUI_adapter=dict(status='Passed_pinned_upstream_build' if frontend_ok else 'Failed_unreviewed_frontend',upstream_html_sha256=frontend_hash),
        launch='WIWO_GPU=-1 scripts/platform/run serve --config <config> --port 8080',
        browser='Forward the Web port and its adjacent API port over SSH',new_GPU_seconds=0)
