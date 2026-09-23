"""Capture zone (ROI) configuration, map cache management, and live capture preview."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np

from ... import PROJECT_ROOT, crashlog
from ...vision import locator
from ..imaging import to_photo
from ..roi_selector import RoiSelector

DARK_CANVAS = "#1e1e1e"
MAX_KP_DRAW = 300


class RoiTab(ttk.Frame):
    """Tab frame for selecting and fine-tuning minimap screen capture ROI."""

    def __init__(
        self,
        master: tk.Widget,
        cfg: dict[str, Any],
        save_cfg_fn: Callable[[], None],
        screen_cap_supplier: Callable[[], Any],
        loc_thread_supplier: Callable[[], Any],
        on_map_rebuilt: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self.cfg = cfg
        self.save_cfg = save_cfg_fn
        self.get_cap = screen_cap_supplier
        self.get_loc = loc_thread_supplier
        self.on_map_rebuilt = on_map_rebuilt

        self.roi_vars: dict[str, tk.StringVar] = {}
        self._roi_pick_busy = False
        self._cache_rebuild_busy = False
        self._roi_photo: Any = None

        self._build_ui()

    def _build_ui(self) -> None:
        ttk.Label(
            self,
            text=(
                "1. Open the minimap in-game (M key).\n"
                '2. Press "Pick zone" - the whole screen appears, drag a rectangle over the minimap.\n'
                "3. Enter to apply, Esc to cancel.\n\n"
                "Fine tuning (absolute screen pixels):"
            ),
        ).pack(anchor="w", padx=8, pady=4)

        row = ttk.Frame(self)
        row.pack(anchor="w", padx=8)
        current_roi = self.cfg.get("capture", {}).get("mmap_roi", [0, 0, 0, 0])
        for i, name in enumerate(("x", "y", "w", "h")):
            ttk.Label(row, text=name).pack(side="left", padx=(6 if i else 0, 0))
            val = str(current_roi[i]) if i < len(current_roi) else "0"
            v = tk.StringVar(value=val)
            ttk.Entry(row, width=6, textvariable=v).pack(side="left", padx=2)
            self.roi_vars[name] = v

        ttk.Button(self, text="Pick zone on screen", command=self.pick_roi).pack(
            anchor="w", padx=8, pady=4
        )
        ttk.Button(self, text="Apply manually", command=self.apply_roi).pack(anchor="w", padx=8)
        self.status_lbl = ttk.Label(self, text="", foreground="green")
        self.status_lbl.pack(anchor="w", padx=8, pady=4)

        # Map Cache & SIFT Feature Index Management
        lf = ttk.LabelFrame(self, text="Map Cache & SIFT Feature Index")
        lf.pack(fill="x", padx=8, pady=(16, 8))

        row_map = ttk.Frame(lf)
        row_map.pack(fill="x", padx=8, pady=(6, 4))
        ttk.Label(row_map, text="Map:").pack(side="left")

        all_maps = locator.get_store().available_maps()
        if not all_maps:
            all_maps = ["zestafona", "bakurani", "ozeti"]

        self._cache_map_sel = ttk.Combobox(row_map, values=all_maps, state="readonly", width=12)
        cur_map = self.cfg.get("map", {}).get("name", "zestafona")
        self._cache_map_sel.set(cur_map if cur_map in all_maps else all_maps[0])
        self._cache_map_sel.pack(side="left", padx=6)
        self._cache_map_sel.bind("<<ComboboxSelected>>", lambda _e: self.cache_status_refresh())

        self._cache_rebuild_btn = ttk.Button(
            row_map,
            text="Rebuild Cache & SIFT Index",
            command=self._cache_rebuild_click,
        )
        self._cache_rebuild_btn.pack(side="left", padx=6)

        self._cache_status_lbl = ttk.Label(lf, text="", foreground="#7cc4ff")
        self._cache_status_lbl.pack(anchor="w", padx=8, pady=(2, 6))
        self.cache_status_refresh()

        # Live 3-Panel Capture Preview
        prev_lf = ttk.LabelFrame(
            self, text="Live Capture Preview (Raw / Mask / Recognition Markers)"
        )
        prev_lf.pack(fill="both", expand=True, padx=8, pady=(4, 6))

        self.roi_canvas = tk.Canvas(
            prev_lf,
            bg=DARK_CANVAS,
            highlightthickness=1,
            highlightbackground="#3f3f3f",
        )
        self.roi_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self._roi_img_id = self.roi_canvas.create_image(0, 0, anchor="center")

    def pick_roi(self) -> None:
        """Capture screen and launch interactive ROI selection window."""
        if self._roi_pick_busy:
            return
        self._roi_pick_busy = True
        self.status_lbl.config(text="grabbing the screen...", foreground="#8a8a8a")
        cap = self.get_cap()
        if cap is None:
            self._roi_pick_fail("screen capture unavailable")
            return

        mon = cap.monitors[1] if len(cap.monitors) > 1 else cap.monitors[0]

        def grab_and_open() -> None:
            try:
                raw = cap._sct.grab(mon)
                bgr = np.array(raw)[:, :, :3].copy()
            except Exception as exc:
                err_msg = str(exc)
                self.after(0, lambda: self._roi_pick_fail(err_msg))
                return
            self.after(0, lambda: self._roi_pick_open(bgr, mon))

        threading.Thread(target=grab_and_open, daemon=True).start()

    def _roi_pick_fail(self, exc: str) -> None:
        self._roi_pick_busy = False
        self.status_lbl.config(text=f"screen grab failed: {exc}", foreground="red")

    def _roi_pick_open(self, bgr: np.ndarray, mon: Any) -> None:
        self._roi_pick_busy = False
        sel = RoiSelector(self, bgr, mon, self._roi_done, lambda: None)
        sel.focus_force()
        self.wait_window(sel)

    def _roi_done(self, roi: list[int]) -> None:
        x, y, w, h = roi
        for name, val in zip(("x", "y", "w", "h"), (x, y, w, h), strict=True):
            self.roi_vars[name].set(str(val))

        self.cfg.setdefault("capture", {})["mmap_roi"] = [x, y, w, h]
        self.save_cfg()
        self.status_lbl.config(
            text=f"OK: [{x}, {y}, {w}, {h}] — saved to config.json",
            foreground="green",
        )

    def apply_roi(self) -> None:
        """Parse manual entry values, validate bounds, and update configuration."""
        try:
            roi = [int(self.roi_vars[n].get()) for n in ("x", "y", "w", "h")]
        except ValueError:
            self.status_lbl.config(text="Error: integers are required", foreground="red")
            return
        x, y, w, h = roi
        if x < 0 or y < 0 or w < 16 or h < 16:
            self.status_lbl.config(
                text="Error: x>=0, y>=0, w>=16, h>=16 required",
                foreground="red",
            )
            return
        self.cfg.setdefault("capture", {})["mmap_roi"] = roi
        self.save_cfg()
        self.status_lbl.config(text=f"OK: {roi}", foreground="green")

    def cache_status_refresh(self) -> None:
        """Update map cache readiness status label."""
        if not hasattr(self, "_cache_map_sel") or not hasattr(self, "_cache_status_lbl"):
            return
        name = self._cache_map_sel.get()
        txt, color = self._get_map_cache_status(name)
        self._cache_status_lbl.config(text=txt, foreground=color)

    def _get_map_cache_status(self, name: str) -> tuple[str, str]:
        if not name:
            return "No map selected", "#8a8a8a"
        data_dir = os.path.join(PROJECT_ROOT, "data", "maps")
        mu_path = os.path.join(data_dir, f"{name}_mu.npy")
        feat_path = os.path.join(data_dir, f"{name}_feat.npz")

        has_mu = os.path.exists(mu_path)
        has_feat = os.path.exists(feat_path)
        missing_previews = [
            sz
            for sz in locator.PREVIEW_SIZES
            if not os.path.exists(os.path.join(data_dir, f"{name}_preview_{sz}.npy"))
        ]

        png_path = os.path.join(data_dir, f"{name}_map.png")
        has_png = os.path.exists(png_path)

        if not has_png and not has_mu and not has_feat:
            return f"Not downloaded: run python tools/download_map.py {name}", "#ff7c7c"
        if not has_mu and not has_feat:
            return "Cache not built: mu.npy and SIFT index missing (press Rebuild)", "#ff7c7c"
        if not has_mu:
            return "Cache incomplete: mu.npy missing (press Rebuild)", "#ff7c7c"
        if not has_feat:
            return "Cache incomplete: SIFT feature index missing (press Rebuild)", "#ffaa00"
        if missing_previews:
            return (
                f"Cache incomplete: missing previews {missing_previews} (press Rebuild)",
                "#ffaa00",
            )

        try:
            with np.load(feat_path) as idx:
                from ...vision.featureindex import _INDEX_NORM

                if str(idx.get("norm", [""])[0]) != _INDEX_NORM:
                    return "SIFT index is outdated (press Rebuild)", "#ffaa00"
                sig = str(idx.get("gray_sig", [""])[0])
                n_tiles = int(idx.get("gw", 0)) * int(idx.get("gh", 0))
                return (
                    f"Ready: mu OK, SIFT index OK ({n_tiles} tiles, {sig}), mipmaps 512..16384 OK",
                    "#8ae234",
                )
        except Exception:
            return "Ready: mu OK, SIFT index OK, mipmaps OK", "#8ae234"

    def _cache_rebuild_click(self) -> None:
        name = self._cache_map_sel.get().strip()
        if not name or self._cache_rebuild_busy:
            return
        png_path = os.path.join(PROJECT_ROOT, "data", "maps", f"{name}_map.png")
        if not os.path.exists(png_path):
            messagebox.showerror(
                "Map Missing",
                f"Map file '{name}_map.png' not found.\n\nPlease download it with:\n  python tools/download_map.py {name}",
            )
            return
        if not messagebox.askyesno(
            "Map Cache",
            f'Rebuild map cache and SIFT feature index for "{name}"?\n\n'
            "This will regenerate mu.npy, preview mipmaps, and the SIFT descriptor index.\n"
            "This process runs in the background and may take several minutes.",
        ):
            return
        self._cache_rebuild_busy = True
        self._cache_rebuild_btn.config(state="disabled")
        self._cache_status_lbl.config(text="Starting rebuild...", foreground="#ffaa00")
        threading.Thread(target=self._cache_rebuild_worker, args=(name,), daemon=True).start()

    def _cache_rebuild_worker(self, name: str) -> None:
        def on_prog(msg: str) -> None:
            self.after(0, lambda: self._cache_status_lbl.config(text=msg, foreground="#ffaa00"))

        try:
            locator.rebuild_map_cache(name, progress_cb=on_prog)
            cb = self.on_map_rebuilt
            if cb is not None:
                self.after(0, lambda: cb(name))

            def done_ui() -> None:
                self._cache_rebuild_busy = False
                self._cache_rebuild_btn.config(state="normal")
                self.cache_status_refresh()
                messagebox.showinfo(
                    "Map Cache",
                    f'Map cache and SIFT index successfully rebuilt for "{name}"!',
                )

            self.after(0, done_ui)
        except Exception as exc:
            crashlog.log(f"rebuild map cache {name}", exc)
            err_msg = str(exc)

            def err_ui() -> None:
                self._cache_rebuild_busy = False
                self._cache_rebuild_btn.config(state="normal")
                self._cache_status_lbl.config(text=f"Rebuild error: {err_msg}", foreground="red")
                messagebox.showerror("Map Cache", f"Rebuild failed: {err_msg}")

            self.after(0, err_ui)

    def update_preview(self) -> None:
        """Poll latest captured frame and render 3-panel diagnostic preview."""
        loc = self.get_loc()
        if loc is None:
            return
        try:
            mm_gray, mm_bgr, mask, latest = loc.snapshot_debug()
        except Exception:
            return
        if mm_gray is None:
            return

        frame = (
            mm_bgr.copy()
            if (mm_bgr is not None and mm_bgr.size)
            else cv2.cvtColor(mm_gray, cv2.COLOR_GRAY2BGR)
        )
        h, w = frame.shape[:2]
        if h < 4 or w < 4:
            return

        diag = (latest.get("diag") or {}) if latest else {}

        def label(img: np.ndarray, text: str, col: tuple[int, int, int] = (0, 255, 255)) -> None:
            cv2.putText(
                img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        # 1. Raw frame
        p1 = frame.copy()
        label(p1, f"1. RAW ({w}x{h})", (0, 255, 255))

        # 2. Mask overlay
        p2 = frame.copy()
        if mask is not None and mask.size:
            m = np.asarray(mask, bool)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            overlay = p2.copy()
            overlay[m] = (0, 30, 220)
            cv2.addWeighted(overlay, 0.45, p2, 0.55, 0, p2)
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(p2, cnts, -1, (0, 160, 255), 1)
            pct = (m.sum() / float(m.size)) * 100.0
            label(p2, f"2. MASK ({pct:.1f}%)", (0, 255, 255))
        else:
            label(p2, "2. MASK (none)", (0, 100, 255))

        # 3. Recognition markers
        p3 = frame.copy()
        kp_pts = diag.get("kp_pts") or []
        inlier_pts = diag.get("inlier_pts") or []
        if len(kp_pts) > MAX_KP_DRAW:
            step = int(np.ceil(len(kp_pts) / float(MAX_KP_DRAW)))
            kp_draw = kp_pts[::step]
        else:
            kp_draw = kp_pts
        for pt in kp_draw:
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 255), -1)
        for pt in inlier_pts:
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 255, 0), -1)
            cv2.circle(p3, (int(round(pt[0])), int(round(pt[1]))), 6, (0, 200, 0), 1)
        n_kp = len(kp_pts)
        n_inl = len(inlier_pts)
        col = (0, 255, 0) if n_inl >= 4 else ((0, 255, 255) if n_kp > 0 else (0, 80, 255))
        label(p3, f"3. SIFT: {n_kp} pts ({n_inl} inl)", col)

        sep = np.full((h, 3, 3), 45, dtype=np.uint8)
        combined = np.hstack([p1, sep, p2, sep, p3])

        cw = self.roi_canvas.winfo_width()
        ch = self.roi_canvas.winfo_height()
        if cw > 20 and ch > 20:
            scale = min((cw - 12) / float(combined.shape[1]), (ch - 12) / float(combined.shape[0]))
            scale = min(scale, 2.0)
            if scale > 0.05:
                nw = max(1, int(round(combined.shape[1] * scale)))
                nh = max(1, int(round(combined.shape[0] * scale)))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
                disp = cv2.resize(combined, (nw, nh), interpolation=interp)
            else:
                disp = combined
        else:
            disp = combined

        try:
            self._roi_photo = to_photo(disp)
            self.roi_canvas.coords(self._roi_img_id, max(cw // 2, 1), max(ch // 2, 1))
            self.roi_canvas.itemconfig(self._roi_img_id, image=self._roi_photo)
        except Exception:
            pass
