"""External-map integration tests are optional only in a clean checkout."""
from pathlib import Path
import pytest

from wiwo_ma.platform.town02_resources import resource_status


def pytest_addoption(parser):
    parser.addoption('--require-town02',action='store_true',default=False,
                     help='Fail if the complete external Town02 import is unavailable')


@pytest.fixture(scope='session')
def require_town02(request):
    report=resource_status(Path(__file__).resolve().parents[2])
    if report['status']=='not_imported' and not request.config.getoption('--require-town02'):
        pytest.skip('External Town02 map is not imported. '+report['hint'])
    if report['status']!='available':
        pytest.fail('Town02 integration prerequisites failed: '+str(report),pytrace=False)
    return report
