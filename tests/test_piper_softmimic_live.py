import numpy as np
import pytest
from pathlib import Path
import shutil
import subprocess

from sim.piper_softmimic.live import Clock, Session, controls


def test_live_browser_controls_and_force_projection():
    root=Path(__file__).resolve().parents[1]
    if not shutil.which('node') or not (root/'outputs/piper_compliance/softmimic_official/replay/data.json').exists():
        pytest.skip('Requires Node and exported replay assets')
    subprocess.run(['node','--experimental-vm-modules',str(root/'tests/piper_soft_live_ui.cjs')],cwd=root,check=True,capture_output=True,text=True)


@pytest.mark.parametrize('force',[[float('nan'),0,0],[0,0,float('inf')],[5,5,0],[0,0],[[0,0,0]],'bad'])
def test_force_rejects_invalid_and_over_limit(force):
    with pytest.raises((ValueError,TypeError)):controls({'force':force})


def test_force_allows_user_magnitude_and_world_vector():
    np.testing.assert_array_equal(controls({'force':[3,0,-4]}),[3,0,-4])
    with pytest.raises(ValueError):controls({'paused':'false'})


def test_clock_has_no_catchup_push_after_disconnect_or_pause():
    clock=Clock();assert clock.ticks(0)==0
    assert clock.ticks(.1)==5
    assert clock.ticks(4)==0
    assert clock.ticks(4.1)==5
    assert clock.ticks(4.2,True)==0
    assert clock.ticks(4.22)==1


def test_only_owner_controls_worlds_and_expired_session_can_be_replaced():
    session=Session();session.authorize('create',{},0);token=session.token
    with pytest.raises(RuntimeError):session.authorize('create',{},1)
    with pytest.raises(KeyError):session.authorize('step',{'session':'wrong'},1)
    session.authorize('step',{'session':token},1)
    with pytest.raises(KeyError):session.authorize('step',{'session':token},12)
    session.authorize('create',{},12);assert session.token!=token
    session.authorize('close',{'session':session.token},13);assert session.token is None
