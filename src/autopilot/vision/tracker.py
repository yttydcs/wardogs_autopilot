"""Background live localization: region capture -> pose -> position on the map.

The capture (screen grab) runs in a producer thread at a fixed ~fps feeding a
single-slot queue (drop-old); the localization consumer pops the latest frame
and runs the pose search. Decoupling the two keeps the UI's "proc: N ms" from
growing when a coarse map search stalls: the producer keeps delivering fresh
frames at the capture rate no matter how long global_pose takes.
"""

import collections
import json
import math
import os
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from .. import PROJECT_ROOT, crashlog
from ..common.config import AppConfig, CaptureConfig, LocatorConfig
from ..common.log import get_logger
from ..hardware.screen_capture import ScreenCapture
from . import locator

logger = get_logger("tracker")

FAIL_SAVE_PERIOD_S = 1.0
FAIL_KEEP_FILES = 200
FAIL_REASONS = (
    "no_features_flat",
    "no_match_global",
    "budget_timeout",
    "no_index",
    "index_no_match",
    "no_features_frame",
    "vote_reject",
)


def _ang_diff(a, b):
    """Smallest angular difference in degrees between two headings/angles."""
    return abs((a - b + 540.0) % 360.0 - 180.0)


def _vote_decide(buf, need, radius, pos):
    """Append pos to the vote buffer; return the agreeing cluster if at least
    `need` buffered jump candidates lie pairwise within `radius`, else None.

    Used by the relocation vote gate: a single lone candidate (a wrong
    re-acquisition) is kept for `need` frames and only published when enough
    consensus frames agree on the same place. The buffer is a maxlen deque, so
    stale candidates expire by themselves.
    """
    buf.append((float(pos[0]), float(pos[1])))
    pts = list(buf)
    for anchor in pts:
        cluster = [p for p in pts if math.hypot(p[0] - anchor[0], p[1] - anchor[1]) <= radius]
        if len(cluster) >= need and all(
            math.hypot(p[0] - q[0], p[1] - q[1]) <= radius
            for i, p in enumerate(cluster)
            for q in cluster[i + 1 :]
        ):
            return cluster
    return None


class _CaptureProducer(threading.Thread):
    """Screen capture at a fixed rate into a single-slot drop-old queue.

    Each item: dict(ts=time.time(), gray=mm_grayscale, bgr=mm_color,
    roi=roi, mask=ui_bool) with the UI mask already resized to the frame.
    The color frame is kept for debug dumps only; the pipeline uses gray.
    Errors land in crash.log and self.error (the consumer surfaces it).
    """

    def __init__(
        self,
        cfg: CaptureConfig | AppConfig | dict,
        mask,
        frame_source,
        stop,
        out_q,
    ) -> None:
        super().__init__(daemon=True)
        if isinstance(cfg, CaptureConfig):
            self.capture_cfg = cfg
            self.cfg = {"capture": cfg.model_dump()}
        elif isinstance(cfg, AppConfig):
            self.capture_cfg = cfg.capture
            self.cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            self.cfg = cfg
            raw_cap = cfg.get("capture")
            cap_dict = raw_cap if isinstance(raw_cap, dict) else cfg
            try:
                self.capture_cfg = CaptureConfig(**cap_dict)
            except Exception:
                self.capture_cfg = CaptureConfig()
        else:
            self.capture_cfg = CaptureConfig()
            self.cfg = {"capture": self.capture_cfg.model_dump()}

        self.mask = np.asarray(mask, bool)
        self.frame_source = frame_source
        self._stop = stop
        self._queue = out_q
        self.error: str | None = None

    def run(self) -> None:
        cap = None
        last_sign = None
        fps = float(self.capture_cfg.fps)
        period = 1.0 / max(1.0, fps)
        try:
            while not self._stop.is_set():
                try:
                    cap_cfg = self.cfg.get("capture") if isinstance(self.cfg, dict) else None
                    if cap_cfg and isinstance(cap_cfg, dict):
                        roi = cap_cfg.get("mmap_roi")
                        mon = int(cap_cfg.get("monitor", 0) or 0)
                    else:
                        roi = self.capture_cfg.mmap_roi
                        mon = self.capture_cfg.monitor
                    if not roi:
                        self._stop.wait(period)
                        continue
                    roi = tuple(int(v) for v in roi)
                    if self.frame_source is not None:
                        frame = self.frame_source()
                    else:
                        sign = (mon, roi)
                        if cap is None or sign != last_sign:
                            if cap is not None and hasattr(cap, "close"):
                                cap.close()
                            cap = ScreenCapture(mon, roi)
                            last_sign = sign
                        frame = cap.grab()
                    mm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = mm.shape
                    ui = self.mask
                    if ui.shape[:2] != (h, w):
                        ui = (
                            cv2.resize(ui.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                            > 0
                        )
                    item = dict(ts=time.time(), gray=mm, bgr=frame, roi=roi, mask=ui)
                    try:
                        self._queue.get_nowait()  # drop the stale frame
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(item)
                except Exception as exc:  # noqa: BLE001
                    crashlog.log("capture producer error", exc)
                    self.error = str(exc)
                    break
                self._stop.wait(period)
        finally:
            if cap is not None and hasattr(cap, "close"):
                cap.close()


class LiveLocator(threading.Thread):
    """Background live localization: consumer of grabbed frames -> pose.

    The result of the last processed frame is always in self.latest; the
    processing delay of the last frame is in self.latest['elapsed'] (seconds).
    """

    def __init__(self, cfg: AppConfig | dict, mask, frame_source=None) -> None:
        super().__init__(daemon=True)
        if isinstance(cfg, AppConfig):
            self.app_cfg = cfg
            self.cfg = cfg.to_dict()
        elif isinstance(cfg, dict):
            self.cfg = cfg
            try:
                self.app_cfg = AppConfig(**cfg)
            except Exception:
                self.app_cfg = AppConfig()
        else:
            self.app_cfg = AppConfig()
            self.cfg = self.app_cfg.to_dict()

        self.cap_cfg = self.app_cfg.capture
        self.loc_cfg = self.app_cfg.locator

        self.mask = np.asarray(mask, bool)
        self.frame_source = frame_source  # callable -> BGR ROI (simulator)
        self.error: str | None = None
        self.latest: dict[str, Any] | None = None
        self.mm_gray: np.ndarray | None = None  # last raw ROI frame for debugging
        self.mm_bgr: np.ndarray | None = None  # last raw ROI frame in color (debug only)
        self.attempt = 0
        self.phase = "loading map..."
        self._prev_xy: tuple[float, float] | None = None
        self._prev_th: float = 0.0
        self._good_xy: tuple[float, float] | None = None  # last accurate (SIFT) position
        self._stop = threading.Event()
        self._poll = 0.03
        # guards the {mm_gray, mm_bgr, mask} trio: the producer thread swaps
        # them per frame, and a UI reader must never see a mismatched combo
        # (e.g. new frame + old mask — that was an IndexError in the collage)
        self._snap_lock = threading.Lock()
        self._vote_buf: collections.deque[tuple[Any, ...]] = collections.deque(
            maxlen=self._loc_vote_frames()
        )
        self._last_accepted: tuple[float, float] | None = None
        self._disp_hist: collections.deque[tuple[float, float]] = collections.deque(maxlen=5)
        self._good_pose: dict[str, Any] | None = (
            None  # last accepted pose dict (for black-frame hold)
        )
        self._hold_left: int | None = None  # frames of black-frame hold still left
        self._last_fail_save: float = 0.0  # time of the last fail-frame dump (throttle)
        self._counts: collections.Counter[str] = collections.Counter()  # reject reason tally
        self.search_now: tuple[Any, ...] | str | None = (
            None  # sector being searched right now ('disc'/'global')
        )

    @property
    def locator_cfg(self) -> LocatorConfig:
        if (
            isinstance(self.cfg, dict)
            and "locator" in self.cfg
            and isinstance(self.cfg["locator"], dict)
        ):
            try:
                return LocatorConfig(**self.cfg["locator"])
            except Exception:
                pass
        return self.app_cfg.locator

    def stop(self) -> None:
        self._stop.set()

    def set_collect_fail_logs(self, enabled: bool) -> None:
        """Live toggle of the fail-frame collector (Map tab checkbox)."""
        enabled = bool(enabled)
        self.app_cfg.debug.collect_fail_logs = enabled
        if isinstance(self.cfg, dict):
            self.cfg.setdefault("debug", {})["collect_fail_logs"] = enabled
        logger.info("[tracker] fail-frame collection %s", "enabled" if enabled else "disabled")

    def _search_progress(self, region):
        """Live callback from the locator: the sector searched at this instant.

        Called on the localization thread while a frame search runs; read by
        the UI every poll. Nothing heavier than a reference assignment.
        """
        self.search_now = region

    def snapshot_debug(self):
        """(mm_gray, mm_bgr, mask, latest) captured atomically for debug dumps.

        mm_gray and mask always come from the same captured frame; mixing them
        across frames (when the ROI changes shape) previously produced a
        boolean-index shape mismatch in the debug collage.
        """
        with self._snap_lock:
            return (self.mm_gray, self.mm_bgr, self.mask, self.latest)

    # --- config knobs for the relocation vote gate (locator block) ---
    def _loc_keys(self):
        lc = self.locator_cfg
        return (
            int(lc.vote_need),
            float(lc.vote_radius_px),
            float(lc.jump_gate_px),
            float(lc.heading_gate_deg),
            int(lc.vote_inl_skip),
        )

    def _loc_vote_frames(self):
        return int(self.locator_cfg.vote_frames)

    def _loc_hold_frames(self):
        """Black/empty-frame hold: frames to reuse the last good pose."""
        return int(self.locator_cfg.hold_frames)

    def _count_reject(self, diag) -> None:
        """Tally the reject reason and expose the top counts on the diag."""
        r = diag.get("reject")
        if r:
            self._counts[r] += 1
        diag["reject_tally"] = dict(self._counts.most_common(5))

    def run(self) -> None:
        try:
            locator.load_global_map()
            idx = locator.get_store().get_index()
            if idx is not None:
                self.phase = "preparing full-map search index..."
                idx.prepare_global_matcher()
            self.phase = "searching pose..."
            q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            prod = _CaptureProducer(self.cfg, self.mask, self.frame_source, self._stop, q)
            prod.start()
            while not self._stop.is_set():
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    self._stop.wait(self._poll)
                    continue
                mm = item["gray"]
                with self._snap_lock:
                    self.mm_gray = mm
                    self.mm_bgr = item.get("bgr")
                    self.mask = item["mask"]
                roi = item["roi"]
                t0 = item["ts"]
                if prod.error:
                    self.error = prod.error
                    break
                # frame budget: coarse map search can stall for tens of
                # seconds (low-texture areas); with a budget the
                # localization returns within max 3 s and the thread does
                # not hang along with it (nor did steer and UI)
                pose, diag = locator.global_pose(
                    mm,
                    self.mask,
                    prev_xy=self._prev_xy,
                    prev_th=self._prev_th,
                    debug=True,
                    budget=3.0,
                    progress=self._search_progress,
                )
                self.search_now = None  # the frame's search is done
                self._count_reject(diag)
                slow = time.time() - t0
                if slow > 3.0 and self.attempt % 20 == 0:
                    logger.warning("[locator] frame took %.1f s (budget 3 s) — map search", slow)

                # autosave of a "failed" frame (black/empty or no pose)
                # for post-run analysis; enabled by the "collect fail logs"
                # checkbox in the app (config 'debug.collect_fail_logs')
                collect = bool(self.app_cfg.debug.collect_fail_logs)
                if collect and pose is None and diag.get("reject") in FAIL_REASONS:
                    now = time.time()
                    if now - self._last_fail_save >= FAIL_SAVE_PERIOD_S:
                        self._last_fail_save = now
                        self._save_fail_frame(diag, mm, item.get("bgr"), item.get("mask"))
                mp = None
                good = False
                cand = None  # pending vote approval: (px, py, heading, inl)
                if pose is not None:
                    px = pose["map_x"]
                    py = pose["map_y"]
                    good = True
                    cand = (px, py, pose["th"], int(pose.get("inl", 0)))
                elapsed = time.time() - t0
                diag["roi"] = list(roi)
                if cand is not None:
                    # relocation vote gate: a pose that jumped far away from
                    # the last accepted position (or flipped its heading beyond
                    # heading_gate_deg) must be confirmed by a few frames
                    # agreeing on the same place before it is published — UNLESS
                    # the pose is very strong (inl >= vote_inl_skip), which a
                    # wrong re-acquisition practically never is.
                    need, rad, gate, hgate, inl_skip = self._loc_keys()
                    if self._last_accepted is not None:
                        d_jump = math.hypot(
                            cand[0] - self._last_accepted[0], cand[1] - self._last_accepted[1]
                        )
                        d_th = _ang_diff(cand[2], self._prev_th)
                        why = None
                        if d_jump > gate:
                            why = "jump %.0f px" % d_jump
                        elif hgate > 0 and d_th > hgate:
                            why = "heading %.0f deg" % d_th
                        if why is not None and cand[3] < inl_skip:
                            vote = _vote_decide(self._vote_buf, need, rad, (cand[0], cand[1]))
                            if vote is None:
                                diag["reject"] = "vote_reject"
                                diag["detail"] = "%s needs %d agreeing frames (have %d)" % (
                                    why,
                                    need,
                                    len(self._vote_buf),
                                )
                                diag["vote"] = dict(
                                    count=len(self._vote_buf),
                                    need=need,
                                    dist=d_jump,
                                    th=d_th,
                                    accepted=False,
                                    inl=cand[3],
                                )
                                self.latest = dict(
                                    ts=t0,
                                    pose=None,
                                    map_px=None,
                                    good=False,
                                    elapsed=elapsed,
                                    diag=diag,
                                )
                                self.attempt += 1
                                continue
                            self._vote_buf.clear()
                            diag["vote"] = dict(
                                count=len(vote),
                                need=need,
                                dist=d_jump,
                                th=d_th,
                                accepted=True,
                                inl=cand[3],
                            )
                        elif why is not None:
                            self._vote_buf.clear()
                    else:
                        self._vote_buf.clear()
                    # approved: commit the state that _prev_xy feeds the next
                    # global_pose call with, and the position to publish
                    if good:
                        self._prev_xy = (cand[0], cand[1])
                        self._prev_th = cand[2]
                        self._good_xy = (cand[0], cand[1])
                        self._good_pose = pose
                        mp = (cand[0], cand[1])
                    else:
                        self._prev_xy = (cand[0], cand[1])  # bridge only
                        mp = (cand[0], cand[1])
                    self._last_accepted = mp
                    self._disp_hist.append(mp)
                disp = mp
                if len(self._disp_hist) >= 3:
                    xs = np.median([p[0] for p in self._disp_hist])
                    ys = np.median([p[1] for p in self._disp_hist])
                    disp = (float(xs), float(ys))
                # black/empty-frame hold: a flat capture (minimap briefly not
                # drawn) while a good pose exists is NOT a real localization
                # loss — reuse the last accepted pose for a few frames so the
                # marker and the navigator do not drop out on every capture void.
                if pose is None:
                    hf = self._loc_hold_frames()
                    if hf > 0 and self._good_xy is not None:
                        if self._hold_left is None:
                            self._hold_left = hf
                        if self._hold_left > 0:
                            self._hold_left -= 1
                            d2 = dict(diag)
                            d2["reject"] = "hold"
                            d2["detail"] = "holding last pose (%d left)" % self._hold_left
                            self.latest = dict(
                                ts=t0,
                                pose=self._good_pose,
                                map_px=self._good_xy,
                                map_px_disp=self._good_xy,
                                good=False,
                                elapsed=elapsed,
                                diag=d2,
                            )
                            self.attempt += 1
                            continue
                    self._hold_left = None
                else:
                    self._hold_left = None
                self.latest = dict(
                    ts=t0,
                    pose=pose,
                    map_px=mp,
                    map_px_disp=disp,
                    good=good,
                    elapsed=elapsed,
                    diag=diag,
                )
                self.attempt += 1
        except Exception as exc:  # noqa: BLE001
            crashlog.log("locator thread exited with an error", exc)
            self.error = str(exc)

    def _fail_payload(self, diag: dict, mm, mask) -> dict[str, Any]:
        """Structured context of a failed frame for offline replay/triage."""
        loc = self.locator_cfg.model_dump()
        prev = self._prev_xy if self._prev_xy is not None else diag.get("prev")
        return {
            "ts": time.time(),
            "reject": diag.get("reject"),
            "detail": diag.get("detail"),
            "mode": diag.get("mode"),
            "prev": list(prev) if prev is not None else None,
            "roi": diag.get("roi"),
            "attempt": self.attempt,
            "frame_shape": list(mm.shape),
            "mm_mean": diag.get("mm_mean"),
            "mm_std": diag.get("mm_std"),
            "mm_mask_frac": diag.get("mm_mask_frac"),
            "kp_mm": diag.get("kp_mm"),
            "kp_chunk": diag.get("kp_chunk"),
            "good": diag.get("good1"),
            "inl": diag.get("inl1"),
            "s": diag.get("s1"),
            "th": diag.get("th1"),
            "t": diag.get("t1"),
            "search_discs": diag.get("search_discs"),
            "search_global": diag.get("search_global"),
            "vote": diag.get("vote"),
            "reject_tally": diag.get("reject_tally"),
            "mask_px": int(np.asarray(mask).sum())
            if mask is not None and getattr(mask, "size", 0)
            else None,
            "thr": {
                k: loc.get(k)
                for k in (
                    "max_kp_frame",
                    "track_radius",
                    "ratio_local",
                    "min_inl_local",
                    "min_inl_rate_local",
                    "ratio_global",
                    "min_inl_global",
                    "min_inl_rate_global",
                    "vote_need",
                    "vote_inl_skip",
                    "jump_gate_px",
                    "heading_gate_deg",
                )
                if loc.get(k) is not None
            },
        }

    def _save_fail_frame(self, diag: dict, mm, bgr=None, mask=None) -> None:
        """Autosave a failed frame (gray + color + context) for post-run analysis."""
        try:
            out_dir = os.path.join(PROJECT_ROOT, "output")
            os.makedirs(out_dir, exist_ok=True)
            ms = int((time.time() % 1.0) * 1000)
            base = "debug_fail_%s_%03d" % (time.strftime("%Y%m%d_%H%M%S"), ms)
            cv2.imwrite(os.path.join(out_dir, base + ".png"), mm)
            if bgr is not None and getattr(bgr, "size", 0):
                cv2.imwrite(os.path.join(out_dir, base + "_rgb.png"), bgr)
            payload = self._fail_payload(diag, mm, mask)
            with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write("reject=%s\n" % diag.get("reject"))
                f.write("detail=%s\n" % diag.get("detail"))
                f.write("prev=%s\n" % (payload["prev"],))
                f.write("roi=%s\n" % (diag.get("roi"),))
                f.write("kp_mm=%d kp_chunk=%d\n" % (diag.get("kp_mm", 0), diag.get("kp_chunk", 0)))
                f.write(
                    "mm_std=%.1f mask=%d%%\n"
                    % (diag.get("mm_std") or 0, int(100 * (diag.get("mm_mask_frac") or 0)))
                )
                tally = diag.get("reject_tally") or {}
                if tally:
                    f.write("tally=%s\n" % " ".join("%s=%d" % (k, v) for k, v in tally.items()))
                f.write("thr=%s\n" % json.dumps(payload["thr"], sort_keys=True))
            self._prune_fail_files(out_dir)
            logger.info(
                "[tracker] saved fail frame %s (reject=%s kp=%s)",
                base,
                diag.get("reject"),
                diag.get("kp_mm"),
            )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _prune_fail_files(out_dir: str, keep: int = FAIL_KEEP_FILES) -> None:
        """Keep only the newest `keep` debug_fail_* files (names sort by time)."""
        files = sorted(p for p in os.listdir(out_dir) if p.startswith("debug_fail_"))
        while len(files) > keep:
            try:
                os.remove(os.path.join(out_dir, files[0]))
            except OSError:
                pass
            files = files[1:]
