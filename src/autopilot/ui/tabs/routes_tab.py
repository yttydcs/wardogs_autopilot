"""Route planning tab with interactive waypoint editing, presets, and autopilot controls."""

from __future__ import annotations

import math
import os
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

from ...common.config import AppConfig
from ...navigation.follow import FollowDriver
from ...vision import locator
from ..map_canvas import InteractiveMapCanvas
from ..presets import PresetManager


class RoutesTab(ttk.Frame):
    """Tab frame for plotting waypoints, managing presets, and running the autopilot."""

    def __init__(
        self,
        master: tk.Widget,
        app_cfg: AppConfig,
        loc_thread_supplier: Callable[[], Any],
        map_name_supplier: Callable[[], str],
        map_store_supplier: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.app_cfg = app_cfg
        self.get_loc = loc_thread_supplier
        self.get_map_name = map_name_supplier
        self.get_store = map_store_supplier

        self.preset_mgr = PresetManager()
        self.route_pts: list[list[float]] = []
        self.driver: FollowDriver | None = None
        self._drag_idx: int | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Top preset toolbar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(bar, text="Preset:").pack(side="left")
        self.p_name = tk.StringVar()
        ttk.Entry(bar, textvariable=self.p_name, width=16).pack(side="left", padx=4)
        ttk.Button(bar, text="Save", command=self.preset_save).pack(side="left", padx=2)
        ttk.Button(bar, text="Update", command=self.preset_overwrite).pack(side="left", padx=2)
        self.p_sel = ttk.Combobox(bar, state="readonly", width=16)
        self.p_sel.pack(side="left", padx=(10, 2))
        ttk.Button(bar, text="Load", command=self.preset_load_sel).pack(side="left", padx=2)
        ttk.Button(bar, text="Delete", command=self.preset_delete).pack(side="left", padx=2)

        # Hint toolbar
        hint = ttk.Frame(self)
        hint.pack(fill="x", padx=6)
        ttk.Label(
            hint,
            text=("LMB — add a point, drag — move, RMB — delete nearest, wheel — zoom, MMB — pan"),
        ).pack(side="left")
        ttk.Button(hint, text="Clear all", command=self.routes_clear).pack(side="right")

        # Interactive Map Canvas
        self.canvas_widget = InteractiveMapCanvas(
            self,
            on_overlay=self._draw_overlay,
        )
        self.canvas_widget.pack(fill="both", expand=True, padx=6, pady=3)

        # Canvas mouse bindings for route editing
        c = self.canvas_widget.canvas
        c.bind("<Button-1>", self._on_click, add="+")
        c.bind("<B1-Motion>", self._on_drag, add="+")
        c.bind("<ButtonRelease-1>", self._on_release, add="+")
        c.bind("<Button-3>", self._on_right, add="+")

        # Bottom control bar
        ctl = ttk.Frame(self)
        ctl.pack(fill="x", padx=6, pady=(2, 6))
        self.follow_btn = ttk.Button(ctl, text="Follow route", command=self.follow_toggle)
        self.follow_btn.pack(side="left")
        self.invert_var = tk.BooleanVar(value=False)
        self.invert_ck = ttk.Checkbutton(
            ctl, text="Invert", variable=self.invert_var, command=self.routes_invert
        )
        self.invert_ck.pack(side="left", padx=6)
        self.dbg_var = tk.BooleanVar(value=bool(self.app_cfg.navigator.debug))
        self.dbg_ck = ttk.Checkbutton(ctl, text="Nav log", variable=self.dbg_var)
        self.dbg_ck.pack(side="left", padx=2)
        self.routes_status = ttk.Label(ctl, text="", foreground="#7cc4ff")
        self.routes_status.pack(side="left", padx=10, fill="x", expand=True)

        self.preset_reload()

    # Preset management
    def preset_reload(self) -> None:
        """Reload list of presets for the current map."""
        self.preset_mgr = PresetManager(subdir=f"data/presets/{self.get_map_name()}")
        names = self.preset_mgr.list_presets()
        self.p_sel["values"] = names
        if names:
            last = self.app_cfg.navigator.last_preset
            self.p_sel.set(last if last in names else names[0])

    def preset_save(self) -> None:
        name = self.p_name.get().strip()
        if not name:
            messagebox.showerror("Presets", "Enter a preset name")
            return
        path = self.preset_mgr.preset_path(name)
        if os.path.exists(path):
            messagebox.showinfo(
                "Presets", f'Preset "{name}" already exists. Use "Update" to overwrite.'
            )
            return
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            messagebox.showerror("Presets", f"Failed to save preset: {exc}")
            return
        self.app_cfg.navigator.last_preset = name
        self.preset_reload()
        self.p_sel.set(name)

    def preset_overwrite(self) -> None:
        name = self.p_name.get().strip() or self.p_sel.get().strip()
        if not name:
            messagebox.showerror("Presets", "Select or enter a preset name to overwrite")
            return
        try:
            self.preset_mgr.save_preset(name, self.route_pts)
        except Exception as exc:
            messagebox.showerror("Presets", f"Failed to update preset: {exc}")
            return
        self.app_cfg.navigator.last_preset = name
        self.preset_reload()
        self.p_sel.set(name)

    def preset_load_sel(self) -> None:
        name = self.p_sel.get().strip()
        if not name:
            return
        try:
            pts = self.preset_mgr.load_preset(name)
        except Exception as exc:
            messagebox.showerror("Presets", f'Failed to read preset "{name}": {exc}')
            return
        self.route_pts = pts
        self.p_name.set(name)
        self.app_cfg.navigator.last_preset = name
        self.routes_refresh()

    def preset_delete(self) -> None:
        name = self.p_sel.get().strip()
        if not name:
            return
        if not messagebox.askyesno("Presets", f'Delete preset "{name}"?'):
            return
        self.preset_mgr.delete_preset(name)
        self.preset_reload()

    # Route editing
    def _routes_hit(self, x: float, y: float, r: float = 12.0) -> int | None:
        best, bd = None, float(r)
        for i, (nx, ny) in enumerate(self.route_pts):
            cx, cy = self.canvas_widget.to_canvas(nx, ny)
            d = math.hypot(cx - x, cy - y)
            if d < bd:
                best, bd = i, d
        return best

    def _on_click(self, e: Any) -> None:
        if self.canvas_widget.disp is None:
            return
        idx = self._routes_hit(e.x, e.y)
        if idx is not None:
            self._drag_idx = idx
            return
        nx, ny = self.canvas_widget.to_native(e.x, e.y)
        max_coord = float(self.canvas_widget._map_size)
        self.route_pts.append([max(0.0, min(max_coord, nx)), max(0.0, min(max_coord, ny))])
        self.routes_refresh()

    def _on_drag(self, e: Any) -> None:
        if self._drag_idx is None or self.canvas_widget.disp is None:
            return
        nx, ny = self.canvas_widget.to_native(e.x, e.y)
        max_coord = float(self.canvas_widget._map_size)
        self.route_pts[self._drag_idx] = [
            max(0.0, min(max_coord, nx)),
            max(0.0, min(max_coord, ny)),
        ]
        self.routes_refresh()

    def _on_release(self, e: Any) -> None:
        self._drag_idx = None

    def _on_right(self, e: Any) -> None:
        if self.canvas_widget.disp is None:
            return
        idx = self._routes_hit(e.x, e.y)
        if idx is not None:
            del self.route_pts[idx]
            self.routes_refresh()

    def routes_clear(self) -> None:
        self.route_pts = []
        self.routes_refresh()

    def routes_invert(self) -> None:
        if self.driver is not None:
            messagebox.showinfo("Routes", "Stop the autopilot, then invert")
            self.invert_var.set(False)
            return
        self.route_pts.reverse()
        self.routes_refresh()

    def routes_refresh(self) -> None:
        if self.canvas_widget.disp is not None:
            self._draw_overlay(self.canvas_widget.canvas, self.canvas_widget.disp)
        length_m = 0.0
        for i in range(1, len(self.route_pts)):
            p0, p1 = self.route_pts[i - 1], self.route_pts[i]
            d_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            length_m += d_px / 1.7
        self.routes_status.config(text=f"route: {len(self.route_pts)} pts, ~{int(length_m)} m")

    # Autopilot control
    def follow_toggle(self) -> None:
        """Start or stop the background FollowDriver thread."""
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
            self.follow_btn.config(text="Follow route")
            self.routes_status.config(text="autopilot stopped", foreground="#7cc4ff")
            return
        if len(self.route_pts) < 2:
            messagebox.showerror("Routes", "Route not set (at least 2 points)")
            return

        loc_thread = self.get_loc()
        nav_cfg = self.app_cfg.navigator
        try:
            kb = self._make_kb(nav_cfg.port)
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("Autopilot", f"Failed to create key driver:\n{exc}")
            return

        self.driver = FollowDriver(
            loc=loc_thread,
            pts=[(p[0], p[1]) for p in self.route_pts],
            nav_cfg=nav_cfg,
            kb=kb,
            debug=self.dbg_var.get(),
        )
        self.driver.start()
        self.follow_btn.config(text="Stop")
        self.routes_status.config(text="autopilot enabled", foreground="#8ae234")

    def _make_kb(self, port: str) -> Any:
        if self.app_cfg.navigator.key_source == "software":
            from ...hardware.software_keyboard import SoftwareKeyDriver

            return SoftwareKeyDriver()

        from ...hardware.arduino_keyboard import ArduinoKeyDriver

        try:
            return ArduinoKeyDriver(port)
        except Exception as exc:
            raise RuntimeError(f"Arduino ({port}) unavailable: {exc}") from exc

    def emergency_stop(self) -> None:
        """Emergency stop handler invoked via global hotkey."""
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
            self.follow_btn.config(text="Follow route")
            self.routes_status.config(text="EMERGENCY STOP", foreground="#ff3b3b")

    def sync_driver_state(self) -> None:
        """Main-thread poll: finalize the UI when the driver finished the route itself."""
        d = self.driver
        if d is not None and d.state == "finished":
            self.driver = None
            self.follow_btn.config(text="Follow route")
            self.routes_status.config(text="route finished — autopilot off", foreground="#7cc4ff")

    def _draw_overlay(self, canvas: tk.Canvas, disp: tuple[float, float, float] | None) -> None:
        if disp is None:
            return

        # Draw waypoints
        canvas.delete("route")
        pts = [self.canvas_widget.to_canvas(nx, ny) for nx, ny in self.route_pts]
        if len(pts) > 1:
            flat = [v for p in pts for v in p]
            canvas.create_line(*flat, fill="#7ce06a", width=3, tags="route")
        for i, (cx, cy) in enumerate(pts):
            canvas.create_oval(
                cx - 7,
                cy - 7,
                cx + 7,
                cy + 7,
                outline="#7ce06a",
                width=2,
                fill="#242424",
                tags="route",
            )
            canvas.create_text(
                cx, cy, text=str(i + 1), fill="#ffffff", font=("Segoe UI", 9, "bold"), tags="route"
            )

        # Draw vehicle position marker
        loc = self.get_loc()
        item = loc.latest if loc else None
        canvas.delete("rvmarker")
        if item is None:
            return

        pose = item.get("pose")
        mp = item.get("map_px_disp") or item.get("map_px")
        if pose is None or mp is None:
            return

        x, y = self.canvas_widget.to_canvas(mp[0], mp[1])
        heading = locator.heading_deg(pose)
        rad = math.radians(heading)
        alen = 28.0
        canvas.create_line(
            x,
            y,
            x + alen * math.sin(rad),
            y - alen * math.cos(rad),
            fill="#ff3b3b",
            width=3,
            arrow="last",
            arrowshape=(7, 9, 3),
            tags="rvmarker",
        )
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#ffdd00", width=2, tags="rvmarker")
