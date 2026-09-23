from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from autopilot.navigation.follow import FollowDriver
from autopilot.navigation.path_tracker import PathTracker
from autopilot.navigation.steering_controller import SteeringController


def test_stationary_noise_does_not_refresh_course():
    path = PathTracker([])
    path.push_pose(10, 0, 0)
    path.push_pose(10.5, 20, 0)
    assert path.motion_heading(10.5) == pytest.approx(90)
    for t in (11, 11.5, 12, 12.5):
        path.push_pose(t, 20, 0)
    assert path.mh_t == 10.5
    assert path.motion_heading(12.5) is None
    assert not path.push_pose(11, 100, 100)


def test_capture_gap_requires_new_motion():
    path = PathTracker([])
    path.push_pose(10, 0, 0)
    path.push_pose(10.5, 20, 0)
    path.push_pose(15, 50, 50)
    assert path.motion_heading(15) is None
    path.push_pose(15.5, 50, 30)
    assert path.motion_heading(15.5) == pytest.approx(0)


@pytest.mark.parametrize('sign', [-1, 1])
def test_large_error_has_bounded_pulse_and_needs_new_observation(sign):
    ctrl = SteeringController(hold_max=8, settle_t=0.1)
    assert ctrl.step(10, sign * 90, 0, 0, 10) == sign
    for t in (10.1, 10.2, 10.4):
        assert ctrl.step(t, sign * 90, 0, 0, 10) == sign
    assert ctrl.step(10.5, sign * 90, 0, 0, 10) == 0
    assert ctrl.step(10.7, sign * 90, 0, 0, 10) == 0
    assert ctrl.step(10.71, sign * 90, 0, 0, 10.71) == sign


def test_sign_reversal_and_missing_heading_release_immediately():
    ctrl = SteeringController()
    assert ctrl.step(10, 90, 0, 0, 10) == 1
    assert ctrl.step(10.1, -90, 0, 0, 10.1) == 0
    assert ctrl.step(12, 90, 0, None, 12) == 0
    assert ctrl.step(12, 90, 0, 0, 10) == 0


def test_follower_never_uses_registration_rotation_as_vehicle_heading(monkeypatch):
    from autopilot.navigation import follow

    monkeypatch.setattr(follow.time, 'time', lambda: 10)
    loc = SimpleNamespace(latest={'ts': 10, 'map_px': (0, 0), 'pose': {'th': 90}})
    kb = MagicMock()
    driver = FollowDriver(loc, [(0, -1000), (0, -2000)], kb=kb)
    driver._wait = lambda _: driver._stop_ev.set()
    driver.run()
    assert driver.state == 'wait_heading'
    kb.set_state.assert_not_called()
    kb.release_all.assert_called()
    kb.close.assert_called()


def test_north_up_map_with_eastward_motion_drives_straight(monkeypatch):
    from autopilot.navigation import follow

    monkeypatch.setattr(follow.time, 'time', lambda: 10.5)
    loc = SimpleNamespace(latest={'ts': 10.5, 'map_px': (20, 0), 'pose': {'th': 0}})
    kb = MagicMock()
    driver = FollowDriver(loc, [(1000, 0), (2000, 0)], kb=kb)
    driver.path.push_pose(10, 0, 0)
    driver._wait = lambda _: driver._stop_ev.set()
    driver.run()
    assert driver.state == 'run'
    assert driver.last['heading'] == pytest.approx(90)
    assert driver.last['err'] == pytest.approx(0)
    keys = kb.set_state.call_args.args[0]
    assert not keys.get('A') and not keys.get('D')
