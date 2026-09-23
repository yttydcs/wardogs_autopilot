"""Exercise input transitions and shutdown without sending desktop keystrokes."""

import ctypes
from unittest.mock import MagicMock

import pytest

from autopilot.hardware import software_keyboard as sk


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(sk.sys, "platform", "win32")
    api = MagicMock()
    api.SendInput.return_value = 1
    monkeypatch.setattr(sk.ctypes, "WinDLL", lambda *a, **kw: api, raising=False)
    # Watchdog timing is driven explicitly below rather than sleeping in tests.
    monkeypatch.setattr(sk.threading.Thread, "start", lambda self: None)
    d = sk.SoftwareKeyDriver()
    events = []

    def send(count, ptr, size):
        event = ctypes.cast(ptr, ctypes.POINTER(sk._Input)).contents
        assert count == 1
        assert size == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        assert event.type == 1
        events.append((event.payload.keyboard.scan, event.payload.keyboard.flags))
        return 1

    api.SendInput.side_effect = send
    yield d, events, api
    api.SendInput.side_effect = send
    d.close()


def test_hold_switch_and_release(driver):
    d, events, _ = driver
    d.set_state({"W": True, "D": True})
    d.set_state({"W": True, "D": True})
    assert events == [(0x11, 8), (0x20, 8)]
    d.set_state({"A": True})
    assert events[2:] == [(0x11, 10), (0x20, 10), (0x1E, 8)]
    assert d.held() == ["A"]
    d.release_all()
    assert not d.held()
    assert events[-1] == (0x1E, 10)


def test_close_blocks_late_navigation_tick(driver):
    d, events, _ = driver
    d.set_state({"W": True})
    d.close()
    d.set_state({"W": True, "SPACE": True})
    d.close()
    assert events == [(0x11, 8), (0x11, 10)]


def test_navigation_stop_closes_software_driver(driver):
    from autopilot.navigation.follow import FollowDriver

    d, events, _ = driver
    follower = FollowDriver(loc=None, pts=[(0, 0), (100, 100)], kb=d)
    d.set_state({"W": True})
    follower.stop()
    d.set_state({"W": True})
    assert events == [(0x11, 8), (0x11, 10)]
    assert follower._stop_ev.is_set()


def test_watchdog_releases_stale_command(driver, monkeypatch):
    d, events, _ = driver
    d.set_state({"W": True})
    monkeypatch.setattr(sk.time, "monotonic", lambda: d._last_command + 0.3)
    d._stop = MagicMock()
    d._stop.wait.side_effect = [False, True]
    d._watchdog()
    assert events[-1] == (0x11, 10)
    assert not d.held()


def test_failed_second_key_releases_first(driver):
    d, _, api = driver
    api.SendInput.side_effect = [1, 0, 1]
    with pytest.raises(OSError):
        d.set_state({"W": True, "D": True})
    assert not d.held()
    with pytest.raises(OSError):
        d.set_state({"W": True})


def test_failed_release_retained_for_retry(driver):
    d, _, api = driver
    d.set_state({"W": True})
    api.SendInput.side_effect = [0, 1]
    with pytest.raises(OSError):
        d.release_all()
    assert d.held() == ["W"]
    d.release_all()
    assert not d.held()


def test_config_and_ui_select_software_driver(monkeypatch):
    from pydantic import ValidationError

    from autopilot.common.config import AppConfig, NavigatorConfig
    from autopilot.ui.tabs.routes_tab import RoutesTab

    fake = object()
    monkeypatch.setattr(sk, "SoftwareKeyDriver", lambda: fake)
    tab = MagicMock(app_cfg=AppConfig())
    assert RoutesTab._make_kb(tab, "COM6") is fake
    with pytest.raises(ValidationError):
        NavigatorConfig(key_source="typo")


def test_ui_keeps_arduino_option(monkeypatch):
    from autopilot.common.config import AppConfig, NavigatorConfig
    from autopilot.hardware import arduino_keyboard
    from autopilot.ui.tabs.routes_tab import RoutesTab

    factory = MagicMock()
    monkeypatch.setattr(arduino_keyboard, "ArduinoKeyDriver", factory)
    tab = MagicMock(app_cfg=AppConfig(navigator=NavigatorConfig(key_source="arduino")))
    assert RoutesTab._make_kb(tab, "COM9") is factory.return_value
    factory.assert_called_once_with("COM9")
