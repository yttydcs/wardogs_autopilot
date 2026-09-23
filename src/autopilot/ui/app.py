"""Studio: Visual tuning and control dashboard for WARDOGS autopilot.

Coordinates the main window, map selector toolbar, global hotkeys,
and the three primary tabs: Capture Zone (ROI), Map Diagnostics, and Routes.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

from .. import PROJECT_ROOT
from ..common.config import AppConfig
from ..common.log import get_logger
from ..hardware.screen_capture import ScreenCapture
from ..vision import locator
from ..vision.tracker import LiveLocator
from .hotkeys import _HK_F6, _HK_F7, HotkeyManager
from .map_renderer import crop_map_viewport
from .tabs.map_tab import MapTab
from .tabs.roi_tab import RoiTab
from .tabs.routes_tab import RoutesTab
from .theme import ThemeManager
from .window import WindowState

logger = get_logger("ui.app")

DARK_BG = "#242424"
DARK_CANVAS = "#1e1e1e"
RGB_CANVAS = (30, 30, 30)


class App(tk.Tk):
    """Main dashboard application window coordinating UI components and services."""

    def __init__(self, cfg: dict[str, Any] | AppConfig) -> None:
        super().__init__()
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        else:
            self.cfg = cfg
            try:
                self.app_cfg = AppConfig(**cfg)
            except Exception:
                self.app_cfg = AppConfig()

        self.title("WARDOGS minimap studio")
        self._init_window_geometry()

        self._map_store = locator.get_store()
        self._map_name = self._cfg_map_name()
        self._map_size = 32768
        self._thumb = 8

        cap_cfg = self.app_cfg.capture
        self._cap = ScreenCapture(monitor=cap_cfg.monitor, roi=cap_cfg.mmap_roi)
        self._mask = locator.make_mask()
        # Share the live settings dictionary with the UI and capture producer.
        # Passing AppConfig makes a snapshot, so newly selected ROIs are ignored.
        self._loc_thread = LiveLocator(cfg=self.cfg, mask=self._mask)
        self._loc_thread.start()

        self._hotkeys = HotkeyManager(self, self._on_global_hotkey)
        self._hotkeys.start()

        self._map8: np.ndarray | None = None
        self._map_pyr: dict[int, np.ndarray] | None = None
        self._last_loc: dict[str, Any] | None = None

        self._build_ui()

        self._theme = ThemeManager(self, self.roi_tab.status_lbl)
        self._theme.apply()
        self._theme.start_poll()

        # Start map loading and main UI poll loop
        threading.Thread(target=self._load_map, daemon=True).start()
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_window_geometry(self) -> None:
        self._win_state = WindowState(self, self.cfg, self._save_cfg)
        geom = self._win_state.restore(self.cfg)
        if geom:
            self.geometry(geom)
        else:
            self.geometry("1030x932")
        self._win_state.bind_configure()

    def _build_ui(self) -> None:
        # Top toolbar
        self._toolbar = ttk.Frame(self)
        self._toolbar.pack(fill="x", padx=6, pady=(4, 0))

        ttk.Label(self._toolbar, text="Active map:").pack(side="left", padx=(0, 4))
        self._map_sel = ttk.Combobox(
            self._toolbar,
            values=locator.available_maps(),
            state="readonly",
            width=14,
        )
        self._map_sel.pack(side="left")
        self._map_sel.set(self._map_name)
        self._map_sel.bind("<<ComboboxSelected>>", self._on_map_changed)

        self._map_size_lbl = ttk.Label(self._toolbar, text="", foreground="#8a8a8a")
        self._map_size_lbl.pack(side="left", padx=(8, 0))

        # Notebook tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.roi_tab = RoiTab(
            self.nb,
            cfg=self.cfg,
            save_cfg_fn=self._save_cfg,
            screen_cap_supplier=lambda: self._cap,
            loc_thread_supplier=lambda: self._loc_thread,
            on_map_rebuilt=self._on_map_rebuilt,
        )
        self.nb.add(self.roi_tab, text="Capture zone")

        self.map_tab = MapTab(
            self.nb,
            cfg=self.cfg,
            save_cfg_fn=self._save_cfg,
            loc_thread_supplier=lambda: self._loc_thread,
            on_pick_roi=self.roi_tab.pick_roi,
        )
        self.nb.add(self.map_tab, text="Map")

        self.routes_tab = RoutesTab(
            self.nb,
            app_cfg=self.app_cfg,
            loc_thread_supplier=lambda: self._loc_thread,
            map_name_supplier=lambda: self._map_name,
            map_store_supplier=lambda: self._map_store,
        )
        self.nb.add(self.routes_tab, text="Routes")

    def _cfg_map_name(self) -> str:
        m = self.cfg.get("map")
        if isinstance(m, dict) and m.get("name"):
            return str(m["name"])
        return "zestafona"

    def _on_map_changed(self, _e: Any = None) -> None:
        new_name = self._map_sel.get().strip()
        if not new_name or new_name == self._map_name:
            return
        self._apply_map(new_name)

    def _apply_map(self, name: str) -> None:
        self._map_name = name
        self.cfg.setdefault("map", {})["name"] = name
        self._save_cfg()
        sz = locator.full_map_size(name)
        self._map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
        self._map_size_lbl.config(text=f"{self._map_size}x{self._map_size}")
        self.map_tab.map_name = name
        self.routes_tab.preset_reload()
        threading.Thread(target=self._load_map, daemon=True).start()

    def _on_map_rebuilt(self, name: str) -> None:
        if name == self._map_name:
            threading.Thread(target=self._load_map, daemon=True).start()

    def _load_map(self) -> None:
        name = self._map_name
        sz = locator.full_map_size(name)
        self._map_size = sz[0] if isinstance(sz, (tuple, list)) else (sz or 32768)
        map8 = locator.color_map()
        pyr = locator.load_previews()
        self._map8 = map8
        self._map_pyr = pyr

        self.map_tab.canvas_widget._map8 = map8
        self.map_tab.canvas_widget._map_pyr = pyr
        self.map_tab.canvas_widget._map_size = self._map_size
        self.routes_tab.canvas_widget._map8 = map8
        self.routes_tab.canvas_widget._map_pyr = pyr
        self.routes_tab.canvas_widget._map_size = self._map_size

        def on_loaded() -> None:
            self.map_tab.canvas_widget.set_map(
                map8, pyr, map_size=self._map_size, thumb=self._thumb
            )
            self.routes_tab.canvas_widget.set_map(
                map8, pyr, map_size=self._map_size, thumb=self._thumb
            )
            self._map_size_lbl.config(text=f"{self._map_size}x{self._map_size}")

        try:
            self.after(0, on_loaded)
        except Exception:
            pass

    def _poll(self) -> None:
        """Periodic UI update loop."""
        try:
            self.roi_tab.update_preview()
            latest = self._loc_thread.latest
            self._last_loc = latest
            self.map_tab.update_loc(latest)
            if self.routes_tab.canvas_widget.disp is not None:
                self.routes_tab.routes_refresh()
            self.routes_tab.sync_driver_state()
        except Exception as exc:
            logger.debug("Poll exception: %s", exc)

        self.after(50, self._poll)

    def _on_global_hotkey(self, key_id: int) -> None:
        if key_id == _HK_F6:
            self.routes_tab.follow_toggle()
        elif key_id == _HK_F7:
            self.routes_tab.emergency_stop()

    def _save_cfg(self) -> None:
        try:
            from ..common.config import AppConfig

            app_cfg = AppConfig.load("config.json")
            app_cfg.capture = self.cfg.get("capture", app_cfg.capture)
            app_cfg.locator = self.cfg.get("locator", app_cfg.locator)
            app_cfg.map = self.cfg.get("map", app_cfg.map)
            app_cfg.navigator = self.cfg.get("navigator", app_cfg.navigator)
            app_cfg.debug = self.cfg.get("debug", app_cfg.debug)
            app_cfg.save("config.json")
        except Exception:
            import json

            with open(os.path.join(PROJECT_ROOT, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)

    def _on_close(self) -> None:
        self._win_state.save_now()
        self._hotkeys.stop()
        self.routes_tab.emergency_stop()
        self._loc_thread.stop()
        self.destroy()

    # ---------- Backward compatibility proxies for tests & legacy callers ----------
    @property
    def _roi_vars(self) -> dict[str, tk.StringVar]:
        return self.roi_tab.roi_vars

    @property
    def _roi_status(self) -> ttk.Label:
        return self.roi_tab.status_lbl

    def _apply_roi(self) -> None:
        self.roi_tab.apply_roi()

    def _pick_roi(self) -> None:
        self.roi_tab.pick_roi()

    @property
    def _tune_vars(self) -> dict[str, tk.StringVar]:
        return self.map_tab.tune_vars

    @property
    def _tune_status(self) -> ttk.Label:
        return self.map_tab.tune_status

    def _apply_tune(self) -> None:
        self.map_tab.apply_tune()

    def _get_map_cache_status(self, name: str) -> tuple[str, str]:
        return self.roi_tab._get_map_cache_status(name)

    def _preset_path(self, name: str) -> str:
        return self.routes_tab.preset_mgr.preset_path(name)

    def _preset_dir(self) -> str:
        return str(self.routes_tab.preset_mgr.presets_dir)

    def _save_debug_frame(self) -> None:
        self.map_tab.save_debug_frame()

    def _bg_crop(self, s: float, ru: float, rv: float, rw: float, rh: float) -> np.ndarray:
        pyr = self._map_pyr or ({self._map8.shape[0]: self._map8} if self._map8 is not None else {})
        return crop_map_viewport(s, ru, rv, rw, rh, pyr, self._map_size, self._thumb, RGB_CANVAS)

    def _bg_update(self, name: str, s: float, ox: float, oy: float) -> None:
        if name == "map":
            self.map_tab.canvas_widget._update_background(s, ox, oy)
        else:
            self.routes_tab.canvas_widget._update_background(s, ox, oy)

    @property
    def _map_canvas(self) -> tk.Canvas:
        return self.map_tab.canvas_widget.canvas

    @property
    def _routes_canvas(self) -> tk.Canvas:
        return self.routes_tab.canvas_widget.canvas

    @property
    def _map_disp(self) -> tuple[float, float, float] | None:
        return self.map_tab.canvas_widget.disp

    @_map_disp.setter
    def _map_disp(self, val: tuple[float, float, float] | None) -> None:
        self.map_tab.canvas_widget._disp = val

    @property
    def _map_fit_scale(self) -> float:
        return self.map_tab.canvas_widget.fit_scale

    @_map_fit_scale.setter
    def _map_fit_scale(self, val: float) -> None:
        self.map_tab.canvas_widget._fit_scale = val

    @property
    def _map_pending(self) -> tuple[float, float, float] | None:
        return self.map_tab.canvas_widget._pending

    @_map_pending.setter
    def _map_pending(self, val: tuple[float, float, float] | None) -> None:
        self.map_tab.canvas_widget._pending = val

    @property
    def _map_zoom_job(self) -> str | None:
        return self.map_tab.canvas_widget._zoom_job

    @_map_zoom_job.setter
    def _map_zoom_job(self, val: str | None) -> None:
        self.map_tab.canvas_widget._zoom_job = val

    def _map_zoom(self, e: Any) -> None:
        self.map_tab.canvas_widget._on_zoom(e)

    def _map_flush_view(self) -> None:
        self.map_tab.canvas_widget._flush_view()

    def _routes_zoom(self, e: Any) -> None:
        self.routes_tab.canvas_widget._on_zoom(e)

    @property
    def _route_pts(self) -> list[list[float]]:
        return self.routes_tab.route_pts

    @_route_pts.setter
    def _route_pts(self, val: list[list[float]]) -> None:
        self.routes_tab.route_pts = val

    @property
    def _driver(self) -> Any:
        return self.routes_tab.driver

    @_driver.setter
    def _driver(self, val: Any) -> None:
        self.routes_tab.driver = val


def main(cfg: dict[str, Any] | AppConfig | None = None) -> None:
    """Launch the WARDOGS minimap studio GUI."""
    if cfg is None:
        cfg = AppConfig.load("config.json")
    app = App(cfg)
    app.mainloop()
