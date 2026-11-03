"""Read pinned Sionna ITU models without initializing an RT backend.

The installed SDK source is the authority for material names, frequency bands,
and coefficients. No SDK module is imported or executed to read this table.
"""
import ast
import hashlib
from importlib import metadata
import math
from numbers import Real
from pathlib import Path


SDK_VERSION = '2.2.0'
SOURCE_MODULE = 'sionna/rt/radio_materials/itu.py'
TABLE_NAME = 'ITU_MATERIALS_PROPERTIES'


def _validate_table(table):
    if not isinstance(table,dict) or not table:
        raise RuntimeError('Installed Sionna ITU table is missing or invalid')
    for name,bands in table.items():
        if not isinstance(name,str) or not isinstance(bands,dict) or not bands:
            raise RuntimeError('Installed Sionna ITU table has an invalid material')
        for band,coefficients in bands.items():
            if not isinstance(band,tuple) or len(band)!=2:
                raise RuntimeError('Installed Sionna ITU frequency band is invalid')
            if not isinstance(coefficients,tuple) or len(coefficients)!=4:
                raise RuntimeError('Installed Sionna ITU coefficients are invalid')
            values=(*band,*coefficients)
            if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x) for x in values):
                raise RuntimeError('Installed Sionna ITU table contains invalid numbers')
            if band[0]<0 or band[0]>band[1]:
                raise RuntimeError('Installed Sionna ITU frequency band is invalid')


def load_catalog():
    """Read a fresh, SHA256-identified snapshot of the installed SDK table."""
    distribution=metadata.distribution('sionna-rt')
    if distribution.version!=SDK_VERSION:
        raise RuntimeError('ITU catalog requires sionna-rt '+SDK_VERSION+
                           '; installed version is '+distribution.version)
    path=Path(distribution.locate_file(SOURCE_MODULE))
    source=path.read_bytes()
    module=ast.parse(source,filename=SOURCE_MODULE)
    assignments=[]
    for node in module.body:
        if isinstance(node,ast.Assign) and any(
                isinstance(target,ast.Name) and target.id==TABLE_NAME for target in node.targets):
            assignments.append(node.value)
    if len(assignments)!=1:
        raise RuntimeError('Installed Sionna ITU table must have one literal assignment')
    try:
        table=ast.literal_eval(assignments[0])
    except (ValueError,TypeError,SyntaxError) as error:
        raise RuntimeError('Installed Sionna ITU table is not a literal mapping') from error
    _validate_table(table)
    return dict(version=distribution.version,source_sha256=hashlib.sha256(source).hexdigest(),
                source_path=SOURCE_MODULE,table=table)


def material_parameters(itu_type,frequency_hz,catalog=None):
    """Evaluate the SDK's first applicable band at a finite positive frequency.

    ``catalog`` may be a snapshot returned by :func:`load_catalog`; it is not a
    custom-material parameter override. Band insertion order matches the SDK.
    """
    if isinstance(frequency_hz,bool) or not isinstance(frequency_hz,Real):
        raise ValueError('Frequency must be a finite positive number in Hz')
    frequency_hz=float(frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz<=0:
        raise ValueError('Frequency must be a finite positive number in Hz')
    catalog=load_catalog() if catalog is None else catalog
    if catalog.get('version')!=SDK_VERSION:
        raise ValueError('ITU catalog version must be '+SDK_VERSION)
    table=catalog.get('table')
    if not isinstance(table,dict):
        raise ValueError('ITU catalog is missing the SDK material table')
    if not isinstance(itu_type,str) or itu_type not in table:
        raise ValueError('Unknown Sionna ITU material: '+str(itu_type))
    frequency_ghz=frequency_hz/1e9
    for band,coefficients in table[itu_type].items():
        if band[0]<=frequency_ghz<=band[1]:
            a,b,c,d=coefficients
            return dict(itu_type=itu_type,frequency_hz=frequency_hz,
                        relative_permittivity=float(a*frequency_ghz**b),
                        conductivity_S_per_m=float(c*frequency_ghz**d),
                        valid_range_GHz=list(band),coefficients=dict(a=a,b=b,c=c,d=d))
    raise ValueError('Sionna ITU material '+itu_type+' is not defined at '+
                     str(frequency_hz)+' Hz')
