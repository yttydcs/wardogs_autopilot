"""Map diagnostics tab with interactive canvas, tuning sliders, and snapshot tools."""

from __future__ import annotations

import json
import math
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any

from ...common.log import get_logger
from ...vision import locator
from ..debug_collage import save_debug_snapshot
from ..map_canvas import InteractiveMapCanvas

logger = get_logger("map_tab")


class MapTab(ttk.Frame):
    """Map view tab providing interactive navigation diagnostics and locator tuning."""

    def __init__(
        self,
        master: tk.Widget,
        cfg: dict[str, Any],
        save_cfg_fn: Callable[[], None],
        loc_thread_supplier: Callable[[], Any],
        on_pick_roi: Callable[[], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.cfg = cfg
        self.save_cfg = save_cfg_fn
        self.get_loc = loc_thread_supplier
        self.on_pick_roi = on_pick_roi

        self.tune_vars: dict[str, tk.StringVar] = {}
        self.tune_status: ttk.Label
        self.map_status: ttk.Label
        self.map_name = "zestafona"
        self._disp_th: float | None = None
        self._last_loc: dict[str, Any] | None = None

        self._snap_busy = False
        self._build_ui()

    def _build_ui(self) -> None:
        # Top action toolbar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(top, text="Pick minimap zone", command=self.on_pick_roi).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(top, text="Save frame", command=self.save_debug_frame).pack(
            side="left", padx=(0, 8)
        )
        self.map_status = ttk.Label(top, text="", foreground="#7cc4ff")
        self.map_status.pack(side="left", fill="x", expand=True)
        ttk.Label(top, text="wheel - zoom, MMB - pan", foreground="#8a8a8a").pack(
            side="right", padx=(8, 0)
        )

        # Locator tuning bar
        tune = ttk.Frame(self)
        tune.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(tune, text="locator tune:", foreground="#8a8a8a").pack(side="left")

        trk = ttk.Frame(tune)
        trk.pack(side="left", padx=(4, 0))
        ttk.Label(trk, text="TRACK", foreground="#7cc4ff").pack(side="left", padx=(0, 4))
        ttk.Label(trk, text="ratio", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["ratio_local"] = tk.StringVar(
            value=str(self._loc_tune_cur("ratio_local", 0.85))
        )
        ttk.Entry(trk, width=5, textvariable=self.tune_vars["ratio_local"]).pack(
            side="left", padx=2
        )
        ttk.Label(trk, text="min_inl", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["min_inl_local"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("min_inl_local", 3)))
        )
        ttk.Entry(trk, width=3, textvariable=self.tune_vars["min_inl_local"]).pack(
            side="left", padx=2
        )
        ttk.Label(trk, text="inl%", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["min_inl_rate_local"] = tk.StringVar(
            value=str(self._loc_tune_cur("min_inl_rate_local", 0.0))
        )
        ttk.Entry(trk, width=4, textvariable=self.tune_vars["min_inl_rate_local"]).pack(
            side="left", padx=2
        )
        ttk.Label(trk, text="rad", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["track_radius"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("track_radius", 900)))
        )
        ttk.Entry(trk, width=5, textvariable=self.tune_vars["track_radius"]).pack(
            side="left", padx=2
        )

        acq = ttk.Frame(tune)
        acq.pack(side="left", padx=(10, 0))
        ttk.Label(acq, text="RE-ACQ", foreground="#ff7c7c").pack(side="left", padx=(0, 4))
        ttk.Label(acq, text="ratio", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["ratio_global"] = tk.StringVar(
            value=str(self._loc_tune_cur("ratio_global", 0.72))
        )
        ttk.Entry(acq, width=5, textvariable=self.tune_vars["ratio_global"]).pack(
            side="left", padx=2
        )
        ttk.Label(acq, text="min_inl", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["min_inl_global"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("min_inl_global", 15)))
        )
        ttk.Entry(acq, width=3, textvariable=self.tune_vars["min_inl_global"]).pack(
            side="left", padx=2
        )
        ttk.Label(acq, text="inl%", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["min_inl_rate_global"] = tk.StringVar(
            value=str(self._loc_tune_cur("min_inl_rate_global", 0.45))
        )
        ttk.Entry(acq, width=4, textvariable=self.tune_vars["min_inl_rate_global"]).pack(
            side="left", padx=2
        )

        misc = ttk.Frame(tune)
        misc.pack(side="left", padx=(10, 0))
        ttk.Label(misc, text="vote", foreground="#8a8a8a").pack(side="left")
        self.tune_vars["vote_need"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("vote_need", 3)))
        )
        ttk.Entry(misc, width=3, textvariable=self.tune_vars["vote_need"]).pack(side="left", padx=2)
        ttk.Label(misc, text="head", foreground="#8a8a8a").pack(side="left", padx=(6, 0))
        self.tune_vars["heading_gate_deg"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("heading_gate_deg", 0)))
        )
        ttk.Entry(misc, width=4, textvariable=self.tune_vars["heading_gate_deg"]).pack(
            side="left", padx=2
        )
        ttk.Label(misc, text="skip", foreground="#8a8a8a").pack(side="left", padx=(6, 0))
        self.tune_vars["vote_inl_skip"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("vote_inl_skip", 40)))
        )
        ttk.Entry(misc, width=3, textvariable=self.tune_vars["vote_inl_skip"]).pack(
            side="left", padx=2
        )
        ttk.Label(misc, text="hold", foreground="#8a8a8a").pack(side="left", padx=(6, 0))
        self.tune_vars["hold_frames"] = tk.StringVar(
            value=str(int(self._loc_tune_cur("hold_frames", 5)))
        )
        ttk.Entry(misc, width=3, textvariable=self.tune_vars["hold_frames"]).pack(
            side="left", padx=2
        )

        ttk.Button(tune, text="Apply", command=self.apply_tune).pack(side="left", padx=(10, 0))
        ttk.Button(tune, text="Reset", command=self.reset_tune).pack(side="left", padx=(4, 0))
        self.tune_status = ttk.Label(tune, text="", foreground="#8ae234")
        self.tune_status.pack(side="left", padx=(8, 0))

        # Diagnostics bar
        dbg_bar = ttk.Frame(self)
        dbg_bar.pack(fill="x", padx=8, pady=(0, 2))
        collect = bool(self.cfg.setdefault("debug", {}).get("collect_fail_logs", True))
        self._collect_ck = tk.BooleanVar(value=collect)
        ttk.Checkbutton(
            dbg_bar,
            text="collect fail logs",
            variable=self._collect_ck,
            command=self.apply_collect_logs,
        ).pack(side="left")
        self.dbg_text = tk.Text(
            dbg_bar,
            height=2,
            wrap="word",
            state="disabled",
            bg="#2b2b2b",
            fg="#ffcf6a",
            insertbackground="#ffcf6a",
            font=("Consolas", 9),
            relief="flat",
            highlightthickness=0,
        )
        self.dbg_text.pack(side="left", fill="x", expand=True)
        ttk.Button(dbg_bar, text="Copy", command=self.copy_debug).pack(side="left", padx=(6, 0))

        # Interactive Map Canvas
        self.canvas_widget = InteractiveMapCanvas(
            self,
            on_overlay=self._draw_overlay,
            on_status=self._on_canvas_status,
        )
        self.canvas_widget.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas_widget.canvas.bind("<Double-Button-1>", self._on_double_click)

    def _on_canvas_status(self, zoom_txt: str) -> None:
        self.map_status.config(text=f"map: {self.map_name}  {zoom_txt}")

    def _on_double_click(self, e: Any = None) -> None:
        loc = self._last_loc
        mp = (loc.get("map_px_disp") or loc.get("map_px")) if loc else None
        if mp is not None:
            self.canvas_widget.center_on(mp[0], mp[1])
        else:
            self.canvas_widget.redraw()

    def _loc_tune_cur(self, name: str, default: Any) -> Any:
        block = self.cfg.setdefault("locator", {})
        return block.get(name, default)

    def apply_tune(self) -> None:
        """Validate tuning inputs, update configuration, and notify LiveLocator."""
        ranges = {
            "ratio_local": (0.1, 1.0, float, "TRACK ratio in [0.1 .. 1.0]"),
            "min_inl_local": (1, 50, int, "TRACK min_inl in [1 .. 50]"),
            "min_inl_rate_local": (0.0, 1.0, float, "TRACK inl% in [0.0 .. 1.0]"),
            "track_radius": (50, 4000, int, "TRACK rad in [50 .. 4000] px"),
            "ratio_global": (0.1, 1.0, float, "RE-ACQ ratio in [0.1 .. 1.0]"),
            "min_inl_global": (1, 50, int, "RE-ACQ min_inl in [1 .. 50]"),
            "min_inl_rate_global": (0.0, 1.0, float, "RE-ACQ inl% in [0.0 .. 1.0]"),
            "vote_need": (1, 10, int, "vote in [1 .. 10]"),
            "heading_gate_deg": (0, 180, int, "head gate in [0 .. 180] deg"),
            "vote_inl_skip": (1, 200, int, "skip in [1 .. 200] inl"),
            "hold_frames": (0, 30, int, "hold in [0 .. 30] frames"),
        }
        parsed = {}
        for name, (lo, hi, typ, desc) in ranges.items():
            s = self.tune_vars[name].get().strip()
            try:
                v = typ(float(s) if typ is float else int(s))
            except ValueError:
                self.tune_status.config(text=f"Invalid {desc}", foreground="#ff7c7c")
                return
            if not (lo <= v <= hi):
                self.tune_status.config(text=f"Invalid {desc}", foreground="#ff7c7c")
                return
            parsed[name] = v

        block = self.cfg.setdefault("locator", {})
        block.update(parsed)
        self.save_cfg()
        loc = self.get_loc()
        if loc is not None:
            if hasattr(loc, "apply_tune"):
                loc.apply_tune(block)
            elif hasattr(loc, "cfg") and isinstance(loc.cfg, dict):
                loc.cfg.setdefault("locator", {}).update(block)
        self.tune_status.config(text="applied", foreground="#8ae234")

    def reset_tune(self) -> None:
        """Reset tuning parameters to schema defaults."""
        from ...common.config import LocatorConfig

        defaults = LocatorConfig().model_dump()
        for k, v in defaults.items():
            if k in self.tune_vars:
                self.tune_vars[k].set(str(v))
        self.apply_tune()

    def apply_collect_logs(self) -> None:
        enabled = bool(self._collect_ck.get())
        self.cfg.setdefault("debug", {})["collect_fail_logs"] = enabled
        self.save_cfg()
        loc = self.get_loc()
        if loc is not None and hasattr(loc, "set_collect_fail_logs"):
            loc.set_collect_fail_logs(enabled)

    def copy_debug(self) -> None:
        txt = self._diagnostic_text()
        self.clipboard_clear()
        self.clipboard_append(txt)

    def _diagnostic_text(self) -> str:
        """Report failures and startup state even before a map canvas is ready."""
        loc = self.get_loc()
        latest = getattr(loc, "latest", None) if loc is not None else None
        item = latest if latest is not None else self._last_loc
        item = item or {}
        payload = {
            "map": self.map_name,
            "capture": self.cfg.get("capture", {}),
            "phase": getattr(loc, "phase", "") or "waiting for localization",
            "error": getattr(loc, "error", None),
            "attempt": getattr(loc, "attempt", 0),
            "ts": item.get("ts"),
            "elapsed": item.get("elapsed"),
            "map_px": item.get("map_px"),
            "pose": item.get("pose"),
            "diag": item.get("diag") or {},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def _update_diagnostics(self) -> None:
        txt = self._diagnostic_text()
        if txt == getattr(self, "_last_debug_text", None):
            return
        self.dbg_text.configure(state="normal")
        self.dbg_text.delete("1.0", "end")
        self.dbg_text.insert("1.0", txt)
        self.dbg_text.configure(state="disabled")
        self._last_debug_text = txt

    def save_debug_frame(self) -> None:
        """Capture live diagnostic snapshot asynchronously."""
        if self._snap_busy:
            return
        loc = self.get_loc()
        if loc is None:
            return
        self._snap_busy = True

        def worker() -> None:
            try:
                mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
                if mm_gray is not None:
                    pose = latest.get("pose") if isinstance(latest, dict) else None
                    raw_diag = latest.get("diag") if isinstance(latest, dict) else None
                    diag: dict[str, Any] = dict(raw_diag) if isinstance(raw_diag, dict) else {}
                    save_debug_snapshot("output", mm_gray, mm_bgr, mask, pose, diag, latest)
            except Exception as exc:
                logger.error("Failed to save debug snapshot: %s", exc)
            finally:
                self._snap_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def update_loc(self, last_loc: dict[str, Any] | None) -> None:
        """Receive latest localization pose item from main event loop."""
        self._last_loc = last_loc
        self._update_diagnostics()
        if self.canvas_widget.disp is not None:
            self._draw_overlay(self.canvas_widget.canvas, self.canvas_widget.disp)

    def _draw_overlay(self, canvas: tk.Canvas, disp: tuple[float, float, float] | None) -> None:
        item = self._last_loc
        if disp is None or item is None:
            return

        pose = item.get("pose")
        mp = item.get("map_px_disp") or item.get("map_px")
        elapsed = item.get("elapsed")
        delay_txt = "" if elapsed is None else f"  proc: {int(elapsed * 1000)} ms"

        win = (item.get("diag") or {}).get("win_map")
        if win:
            x0, y0 = self.canvas_widget.to_canvas(win[0], win[1])
            x1, y1 = self.canvas_widget.to_canvas(win[2], win[3])
            canvas.delete("chunkbox")
            canvas.create_rectangle(x0, y0, x1, y1, outline="#2e8bff", dash=(4, 3), tags="chunkbox")

        if pose is None or mp is None:
            loc = self.get_loc()
            phase = getattr(loc, "phase", "") if loc else ""
            self.map_status.config(text=(phase if phase else "searching...") + delay_txt)
            for t in ("marker", "arrow", "chunkbox"):
                canvas.delete(t)
            self._search_zone_draw(canvas, getattr(loc, "search_now", None), "searchzone")
            return

        x, y = self.canvas_widget.to_canvas(mp[0], mp[1])
        heading = locator.heading_deg(pose)
        if self._disp_th is None:
            self._disp_th = heading
        else:
            dth = (heading - self._disp_th + 540.0) % 360.0 - 180.0
            self._disp_th = (self._disp_th + dth * 0.4) % 360.0
        heading = self._disp_th
        rad = math.radians(heading)
        alen = 30.0
        ax = x + alen * math.sin(rad)
        ay = y - alen * math.cos(rad)

        for t in ("marker", "arrow", "searchzone"):
            canvas.delete(t)

        canvas.create_line(
            x, y, ax, ay, fill="#ff3b3b", width=3, arrow="last", arrowshape=(8, 10, 3), tags="arrow"
        )
        r = 6
        canvas.create_oval(x - r, y - r, x + r, y + r, outline="#ffdd00", width=2, tags="marker")
        canvas.create_line(x - r - 4, y, x + r + 4, y, fill="#ffdd00", width=1, tags="marker")
        canvas.create_line(x, y - r - 4, x, y + r + 4, fill="#ffdd00", width=1, tags="marker")

        self.map_status.config(
            text=f"map px: x={mp[0]:.0f} y={mp[1]:.0f}   heading: {heading:.1f}°   "
            f"s={pose['s']:.3f} inl={pose.get('inl', 0)}{delay_txt}"
        )

    def _search_zone_draw(self, canvas: tk.Canvas, region: Any, tags: str) -> None:
        canvas.delete(tags)
        if region is None or self.canvas_widget.disp is None:
            return
        ms = locator._mini_scale()
        s = self.canvas_widget.disp[0]
        if region[0] == "global":
            x0, y0 = self.canvas_widget.to_canvas(0, 0)
            x1, y1 = self.canvas_widget.to_canvas(
                self.canvas_widget._map_size, self.canvas_widget._map_size
            )
            canvas.create_rectangle(
                x0, y0, x1, y1, outline="#ff9500", dash=(4, 3), width=2, tags=tags
            )
        else:
            _, cxm, cym, radm = region
            cx, cy = self.canvas_widget.to_canvas(cxm * ms, cym * ms)
            r = radm * ms / self.canvas_widget._thumb * s
            canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r, outline="#ff9500", dash=(4, 3), width=2, tags=tags
            )
