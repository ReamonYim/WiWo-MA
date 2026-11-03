"""Pinned SDK material coefficients, frequency validity, and backend-free reads."""
import hashlib
from importlib import metadata
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from wiwo_ma.platform import itu_catalog


@pytest.mark.parametrize('name,permittivity,conductivity',[
    ('concrete',5.24,.10337344451390922),
    ('brick',3.91,.02806223402275108),
    ('wood',1.99,.014169738514166533),
    ('glass',6.31,.01429637620386651),
    ('metal',1.,1e7),
    ('asphalt_concrete',4.83,.04550496759868662),
])
def test_actual_SDK_material_parameters_at_2_8_GHz(name,permittivity,conductivity):
    result=itu_catalog.material_parameters(name,2.8e9)
    assert result['itu_type']==name
    assert result['frequency_hz']==2.8e9
    assert result['relative_permittivity']==pytest.approx(permittivity,rel=1e-12)
    assert result['conductivity_S_per_m']==pytest.approx(conductivity,rel=1e-12)
    assert result['valid_range_GHz'][0]<=2.8<=result['valid_range_GHz'][1]
    assert set(result['coefficients'])=={'a','b','c','d'}


@pytest.mark.parametrize('name,frequency',[
    ('rubber',2.8e9),('floorboard',2.8e9),('clear_acrylic',2.8e9),('brick',70e9),
])
def test_unknown_material_and_unsupported_frequency_are_rejected(name,frequency):
    with pytest.raises(ValueError,match='Unknown Sionna ITU material|is not defined'):
        itu_catalog.material_parameters(name,frequency)


@pytest.mark.parametrize('frequency',[0.,-1.,math.nan,math.inf,-math.inf,None,'2.8e9',True])
def test_frequency_requires_a_finite_positive_number(frequency):
    with pytest.raises(ValueError,match='finite positive'):
        itu_catalog.material_parameters('wood',frequency)


def test_overlap_selection_preserves_SDK_first_valid_band_order():
    result=itu_catalog.material_parameters('wood',150e9)
    assert result['valid_range_GHz']==[110.,330.]
    assert result['relative_permittivity']==1.82
    assert result['coefficients']==dict(a=1.82,b=0.,c=.004,d=1.0761)


def test_actual_catalog_records_installed_version_and_exact_source_hash():
    catalog=itu_catalog.load_catalog()
    distribution=metadata.distribution('sionna-rt')
    source=Path(distribution.locate_file('sionna/rt/radio_materials/itu.py')).read_bytes()
    assert catalog['version']==distribution.version=='2.2.0'
    assert catalog['source_path']=='sionna/rt/radio_materials/itu.py'
    assert catalog['source_sha256']==hashlib.sha256(source).hexdigest()
    assert len(catalog['table'])==19
    assert catalog['table']['concrete'][(1.,100.)]==(5.24,0.,.0462,.7822)


def test_unreviewed_distribution_version_is_rejected_before_reading_source(monkeypatch):
    monkeypatch.setattr(itu_catalog.metadata,'distribution',lambda _:SimpleNamespace(version='2.1.0'))
    with pytest.raises(RuntimeError,match='requires sionna-rt 2.2.0'):
        itu_catalog.load_catalog()


def test_each_catalog_read_reflects_source_updates_without_a_global_cache(tmp_path,monkeypatch):
    distribution=metadata.distribution('sionna-rt')
    source=Path(distribution.locate_file('sionna/rt/radio_materials/itu.py')).read_bytes()
    path=tmp_path/'itu.py'
    path.write_bytes(source)
    snapshot=SimpleNamespace(version='2.2.0',locate_file=lambda _:path)
    monkeypatch.setattr(itu_catalog.metadata,'distribution',lambda _:snapshot)
    first=itu_catalog.load_catalog()
    path.write_bytes(source+b'\n# Source-update checksum fixture\n')
    second=itu_catalog.load_catalog()
    assert first['source_sha256']!=second['source_sha256']
    assert first['table']==second['table']


def test_catalog_and_parameter_reads_do_not_import_any_RT_backend():
    code='''
import sys
from wiwo_ma.platform.itu_catalog import load_catalog,material_parameters
catalog=load_catalog()
assert material_parameters('glass',2.8e9,catalog)['relative_permittivity']==6.31
assert not any(name=='sionna' or name.startswith(('sionna.','mitsuba','drjit')) for name in sys.modules)
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
