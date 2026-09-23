"""Diagnostics remain available without successful localization or map rendering."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from autopilot.ui.tabs.map_tab import MapTab


def make_tab(loc):
    tab = SimpleNamespace(
        get_loc=lambda: loc,
        _last_loc=None,
        map_name="zestafona",
        cfg={"capture": {"mmap_roi": [42, 993, 340, 304]}},
        dbg_text=MagicMock(),
        canvas_widget=SimpleNamespace(disp=None),
        clipboard_clear=MagicMock(),
        clipboard_append=MagicMock(),
    )
    tab._diagnostic_text = lambda: MapTab._diagnostic_text(tab)
    tab._update_diagnostics = lambda: MapTab._update_diagnostics(tab)
    return tab


def test_copy_without_localization_is_not_empty():
    tab = make_tab(None)
    MapTab.copy_debug(tab)
    payload = json.loads(tab.clipboard_append.call_args.args[0])
    assert payload["phase"] == "waiting for localization"
    assert payload["pose"] is None
    assert payload["capture"]["mmap_roi"] == [42, 993, 340, 304]


def test_failed_match_updates_text_without_canvas():
    item = {"pose": None, "diag": {"reject": "index_no_match", "kp_mm": 1188}}
    loc = SimpleNamespace(latest=item, phase="searching pose", error=None, attempt=5)
    tab = make_tab(loc)
    MapTab.update_loc(tab, item)
    payload = json.loads(tab.dbg_text.insert.call_args.args[1])
    assert payload["diag"]["reject"] == "index_no_match"
    assert payload["attempt"] == 5
    MapTab.copy_debug(tab)
    assert json.loads(tab.clipboard_append.call_args.args[0]) == payload


def test_copy_includes_worker_error_before_first_frame():
    tab = make_tab(SimpleNamespace(latest=None, error="capture failed", phase="starting"))
    MapTab.copy_debug(tab)
    assert json.loads(tab.clipboard_append.call_args.args[0])["error"] == "capture failed"
