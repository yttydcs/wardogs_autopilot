"""Full-map matcher and player localization.

Estimates player position on the full map from minimap captures using SIFT
descriptors matched against an offline spatial FeatureIndex (featureindex.py).
Delegates map file and cache management to MapStore and image filtering to
preprocessing.py.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, Literal, overload

import cv2
import numpy as np

from ..common.log import get_logger
from .map_store import (
    COLOR_PREVIEW_SIZE,
    DATA_MAPS,
    FULL_DIR,
    PREVIEW_SIZES,
    ROOT,
    MapStore,
)
from .preprocessing import (
    MASK_PATH,
    bgr_to_gray,
    crop_win,
    fill_norm,
    make_mask,
    norm8,
    shadow_fill_norm,
)

__all__ = [
    "COLOR_PREVIEW_SIZE",
    "DATA_MAPS",
    "FULL_DIR",
    "MASK_PATH",
    "PREVIEW_SIZES",
    "ROOT",
    "MapLocator",
    "MapStore",
    "available_maps",
    "bgr_to_gray",
    "build_previews",
    "color_map",
    "crop_win",
    "ensure_previews",
    "fill_norm",
    "full_map_size",
    "global_pose",
    "heading_deg",
    "load_chunk2map",
    "load_global_map",
    "load_previews",
    "make_mask",
    "map_name",
    "norm8",
    "rebuild_map_cache",
    "set_map",
    "shadow_fill_norm",
]

logger = get_logger("locator")

DEFAULT_MAX_KP = 1200
FAST_BUDGET_FRAC = 0.4


def _over(t0: float, budget: float | None) -> bool:
    """Check whether frame time budget has elapsed."""
    return budget is not None and (time.time() - t0) > budget


def _mark_search(
    diag: dict[str, Any], discs: list[tuple[float, float, float]], global_pass: bool
) -> None:
    """Publish the search region(s) of this frame into the diagnostics."""
    diag["search_discs"] = [[float(v) for v in d] for d in discs]
    diag["search_global"] = bool(global_pass)


class MapLocator:
    """SIFT-based localization engine."""

    def __init__(self, store: MapStore | None = None) -> None:
        self.store = store or MapStore()
        self.sift = cv2.SIFT.create(nfeatures=6000, contrastThreshold=0.05, edgeThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = 0.80
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def _clahe_preprocess(self, mm: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
        """Fill UI pixels with the background median, then local-contrast (CLAHE).

        Yields fewer, more structural keypoints than the percentile stretch, so
        it is ~2x faster to match; used as an optional first pass with a
        fallback to `shadow_fill_norm` when it finds no pose.
        """
        m = mm.copy()
        if ui_mask is not None and ui_mask.size:
            m[ui_mask] = int(np.median(m[~ui_mask]))
        return self.clahe.apply(m)

    def heading_deg(self, pose: dict[str, Any]) -> float:
        """Player's heading on the map: 0 deg = north, 90 deg = east (clockwise)."""
        return float(pose["th"])

    def _detect(self, mmf: np.ndarray, max_kp: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """SIFT keypoints/descriptors of one frame, capped by descending response.

        Tree canopy and other repetitive texture can yield thousands of weak,
        non-distinctive keypoints (measured 1400+ on a forest frame) that inflate
        the BF.knnMatch cost and dilute RANSAC without adding real inliers.
        Ranking by response and keeping `max_kp` retains the structural points
        while bounding the per-frame match cost.
        """
        kp, desc = self.sift.detectAndCompute(mmf, None)
        if desc is None or max_kp <= 0 or len(kp) <= max_kp:
            return kp, desc
        order = np.argsort([k.response for k in kp])[::-1][:max_kp]
        return [kp[int(i)] for i in order], desc[order]

    def _mm_center_to_map(self, r: dict[str, Any], mm: np.ndarray) -> tuple[float, float]:
        """Minimap center in map (mu) coords from a pose with map-space translation."""
        px, py = mm.shape[1] / 2.0, mm.shape[0] / 2.0
        th_r = np.radians(r["th"])
        a, b = r["s"] * np.cos(th_r), r["s"] * np.sin(th_r)
        return (r["t"][0] + a * px - b * py, r["t"][1] + b * px + a * py)

    def _pose_via_index(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        idx: Any,
        cx: float,
        cy: float,
        r: float | None,
        thr: dict[str, Any] | None = None,
        budget: float | None = None,
        t0: float | None = None,
        feats: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """SIFT match of the minimap against index descriptors within radius r (or globally)."""
        start_t = t0 if t0 is not None else time.time()
        if feats is not None:
            kp1, d1 = feats
        else:
            mmf = shadow_fill_norm(mm, ui_mask)
            max_kp = int(self.store.loc_cfg().get("max_kp_frame", DEFAULT_MAX_KP))
            kp1, d1 = self._detect(mmf, max_kp)
        diag: dict[str, Any] = dict(
            mmi_shape=tuple(mm.shape),
            roi=None,
            kp_mm=0 if d1 is None else len(d1),
            kp_chunk=0,
            good1=0,
            inl1=0,
            good2=0,
            inl2=0,
            s1=None,
            th1=None,
            t1=None,
            s2=None,
            th2=None,
            t2=None,
            reject=None,
            detail="",
        )
        if d1 is None or len(d1) < 4:
            diag["reject"] = "no_features_frame"
            diag["detail"] = "index path: frame without SIFT features"
            return None, diag

        if thr is None:
            cfg = self.store.loc_cfg()
            thr = dict(
                ratio=float(cfg.get("ratio", self.ratio)),
                min_inl=int(cfg.get("min_inl", 4)),
                min_inl_rate=float(cfg.get("min_inl_rate", 0.0)),
            )
        min_inl = int(thr["min_inl"])
        ratio = float(thr["ratio"])
        min_rate = float(thr["min_inl_rate"])

        global_kn = None
        if r is None:
            pts, desc, global_kn = idx.global_matches(d1)
            cands = [(0, pts, desc)]
            scope = "global"
        else:
            cands = idx.radius_candidates(cx, cy, r)
            scope = f"local(r={r:.0f})"

        best: dict[str, Any] | None = None
        best_len: int | None = None
        tried = []
        for _li, pts, d2 in cands:
            if _over(start_t, budget):
                break
            if d2 is None or len(d2) < 2:
                continue
            diag["kp_chunk"] = max(diag["kp_chunk"], len(d2))
            kn = global_kn if global_kn is not None else self.bf.knnMatch(d1, d2, k=2)
            good = [
                pair[0]
                for pair in kn
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
            ]
            # Reject many-to-one matches before RANSAC. Repeated tree texture
            # otherwise produces a degenerate, near-zero-scale transform.
            if good:
                reverse = self.bf.match(np.asarray([d2[g.trainIdx] for g in good]), d1)
                good = [g for i, g in enumerate(good) if reverse[i].trainIdx == g.queryIdx]
            diag["good1"] = max(diag["good1"], len(good))
            if len(good) < 4:
                tried.append((len(good), 0, "too few mutual matches"))
                continue
            src = np.array([kp1[g.queryIdx].pt for g in good], dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array([pts[g.trainIdx] for g in good], dtype=np.float32).reshape(-1, 1, 2)
            m3, inl_mask = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=6.0,
                maxIters=10000,
                confidence=0.999,
            )
            if m3 is None or inl_mask is None:
                tried.append((len(good), 0, "no geometric model"))
                continue
            inl = int(inl_mask.sum())
            diag["inl1"] = max(diag["inl1"], inl)
            if inl < min_inl:
                tried.append((len(good), inl, "too few inliers"))
                continue
            if min_rate > 0.0 and inl < min_rate * len(good):
                tried.append((len(good), inl, f"inl_rate {inl / max(len(good), 1):.2f}"))
                continue
            s = float(np.hypot(m3[0, 0], m3[0, 1]))
            if not float(self.store.loc_cfg().get("min_pose_scale", 0.25)) <= s <= 2.6:
                tried.append((len(good), inl, f"scale {s:.2f}"))
                continue
            inliers = [good[i] for i, v in enumerate(inl_mask.flatten()) if v]
            inlier_pts = [kp1[g.queryIdx].pt for g in inliers]
            cand_pose: dict[str, Any] = dict(
                s=s,
                th=float(np.degrees(np.arctan2(m3[1, 0], m3[0, 0]))),
                t=(float(m3[0, 2]), float(m3[1, 2])),
                inl=inl,
                n_match=len(good),
                inlier_pts=inlier_pts,
            )
            if best is None or inl > int(best["inl"]):
                best = cand_pose
                best_len = len(d2)

        diag["kp_pts"] = [kp.pt for kp in kp1]
        if best is None:
            diag["inlier_pts"] = []
            diag["reject"] = "budget_timeout" if _over(start_t, budget) else "index_no_match"
            diag["detail"] = f"index ({scope}) search found no pose (tried={tried or '-'})"
            return None, diag

        diag.update(
            kp_mm=len(d1),
            kp_chunk=best_len,
            good1=best["n_match"],
            inl1=best["inl"],
            s1=best["s"],
            th1=best["th"],
            t1=best["t"],
            good2=best["n_match"],
            inl2=best["inl"],
            s2=best["s"],
            th2=best["th"],
            t2=best["t"],
            inlier_pts=best.get("inlier_pts", []),
            reject=None,
            detail="OK (index)",
        )
        return best, diag

    def _index_find(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        idx: Any,
        cx: float | None,
        cy: float | None,
        min_inl: int = 4,
        budget: float | None = None,
        t0: float | None = None,
        progress: Callable[[tuple[Any, ...]], None] | None = None,
        feats: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
    ) -> (
        tuple[dict[str, Any] | None, dict[str, Any], float, float, float]
        | tuple[None, dict[str, Any]]
    ):
        """Index-based pose: growing radius around (cx, cy), then whole map."""
        start_t = t0 if t0 is not None else time.time()
        cfg = self.store.loc_cfg()
        rad = float(cfg["local_radius"])
        growth = float(cfg["radius_growth"])
        track_radius = float(cfg.get("track_radius", rad * 2.0))

        base = dict(
            ratio=float(cfg.get("ratio", self.ratio)),
            min_inl=int(cfg.get("min_inl", min_inl)),
            min_inl_rate=float(cfg.get("min_inl_rate", 0.0)),
        )
        local_thr = dict(
            ratio=float(cfg.get("ratio_local", base["ratio"])),
            min_inl=int(cfg.get("min_inl_local", base["min_inl"])),
            min_inl_rate=float(cfg.get("min_inl_rate_local", base["min_inl_rate"])),
        )
        global_thr = dict(
            ratio=float(cfg.get("ratio_global", base["ratio"])),
            min_inl=int(cfg.get("min_inl_global", base["min_inl"])),
            min_inl_rate=float(cfg.get("min_inl_rate_global", base["min_inl_rate"])),
        )
        total = idx.total_features()
        last_diag: dict[str, Any] | None = None
        discs: list[tuple[float, float, float]] = []
        global_pass = False

        if cx is not None and cy is not None:
            qx, qy = cx, cy
            for _ in range(10):
                if _over(start_t, budget):
                    break
                if progress is not None:
                    progress(("disc", qx, qy, rad))
                discs.append((qx, qy, rad))
                cands = idx.radius_candidates(qx, qy, rad)
                thr = global_thr if rad > track_radius else local_thr
                res, dd = self._pose_via_index(
                    mm,
                    ui_mask,
                    idx,
                    qx,
                    qy,
                    rad,
                    thr=thr,
                    budget=budget,
                    t0=start_t,
                    feats=feats,
                )
                last_diag = dd
                if res is not None:
                    mx, my = self._mm_center_to_map(res, mm)
                    if abs(mx - qx) <= rad + 2.0 and abs(my - qy) <= rad + 2.0:
                        _mark_search(dd, discs, False)
                        return res, dd, mx - rad, my - rad, rad
                cov = sum(len(p) for _, p, _ in cands)
                if total and cov >= 0.30 * total:
                    break
                rad *= growth

        if not _over(start_t, budget):
            global_pass = True
            if progress is not None:
                progress(("global",))
            res, dd = self._pose_via_index(
                mm,
                ui_mask,
                idx,
                0.0,
                0.0,
                None,
                thr=global_thr,
                budget=budget,
                t0=start_t,
                feats=feats,
            )
            last_diag = dd
            if res is not None:
                mx, my = self._mm_center_to_map(res, mm)
                # A cold-start hypothesis is only accepted after an independent
                # exact local match with more supporting points.
                refine_radius = max(200.0, max(mm.shape) * res["s"] * 1.5)
                refined, rd = self._pose_via_index(
                    mm,
                    ui_mask,
                    idx,
                    mx,
                    my,
                    refine_radius,
                    thr={**local_thr, "min_inl": max(8, global_thr["min_inl"])},
                    budget=budget,
                    t0=start_t,
                    feats=feats,
                )
                if refined is None:
                    dd["reject"] = "index_no_match"
                    dd["detail"] = "global candidate failed local verification: " + rd["detail"]
                    dd["refine"] = {
                        k: v for k, v in rd.items() if k not in ("kp_pts", "inlier_pts")
                    }
                    _mark_search(dd, discs, True)
                    return None, dd
                rx, ry = self._mm_center_to_map(refined, mm)
                if math.hypot(rx - mx, ry - my) > refine_radius / 2:
                    dd["reject"] = "index_no_match"
                    dd["detail"] = "global candidate and local verification disagree"
                    _mark_search(dd, discs, True)
                    return None, dd
                res, dd, mx, my = refined, rd, rx, ry
                rad = max(float(cfg["local_radius"]), 350.0)
                _mark_search(dd, discs, True)
                return res, dd, mx - rad, my - rad, rad

        if last_diag is not None:
            _mark_search(last_diag, discs, global_pass)
        if _over(start_t, budget):
            if last_diag is None:
                last_diag = dict(reject="budget_timeout", detail="frame budget exhausted")
                _mark_search(last_diag, discs, global_pass)
            elif last_diag.get("reject") is None:
                last_diag["reject"] = "budget_timeout"
                last_diag["detail"] = "frame budget exhausted"
        return None, last_diag or {}

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float = ...,
        debug: Literal[True] = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]: ...

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float = ...,
        debug: Literal[False] = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
    ) -> dict[str, Any] | None: ...

    @overload
    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = ...,
        min_inl: int = ...,
        prev_th: float = ...,
        debug: bool = ...,
        budget: float | None = ...,
        progress: Callable[[tuple[Any, ...]], None] | None = ...,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None: ...

    def global_pose(
        self,
        mm: np.ndarray,
        ui_mask: np.ndarray | None,
        prev_xy: tuple[float, float] | None = None,
        min_inl: int = 4,
        prev_th: float = 0.0,
        debug: bool = False,
        budget: float | None = None,
        progress: Callable[[tuple[Any, ...]], None] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None:
        """Estimate player position on WHOLE map in native map px via feature index."""
        t0 = time.time()
        diag: dict[str, Any] = dict(
            mmi_shape=tuple(mm.shape),
            roi=None,
            win_map=None,
            kp_mm=0,
            kp_chunk=0,
            good1=0,
            inl1=0,
            good2=0,
            inl2=0,
            s1=None,
            th1=None,
            t1=None,
            s2=None,
            th2=None,
            t2=None,
            reject=None,
            detail="",
            mode="index",
            mm_mean=None,
            mm_std=None,
            mm_mask_frac=None,
            prev=None,
            search_discs=None,
            search_global=False,
        )

        diag["mm_mean"] = float(mm.mean())
        diag["mm_std"] = float(mm.std())
        if ui_mask is not None and ui_mask.size:
            diag["mm_mask_frac"] = float(ui_mask.mean())
        max_kp = int(self.store.loc_cfg().get("max_kp_frame", DEFAULT_MAX_KP))
        fast = bool(self.store.loc_cfg().get("fast_clahe", False))
        passes: list[Callable[[np.ndarray, np.ndarray | None], np.ndarray]] = (
            [self._clahe_preprocess, shadow_fill_norm] if fast else [shadow_fill_norm]
        )

        _ = self.store.load_global_map()
        ms = self.store.mini_scale()

        cx = cy = None
        if prev_xy is not None:
            diag["prev"] = (float(prev_xy[0]), float(prev_xy[1]))
            cx, cy = prev_xy[0] / ms, prev_xy[1] / ms

        idx = self.store.get_index()
        if idx is None:
            active_name = self.store.map_name()
            diag["reject"] = "no_index"
            diag["detail"] = (
                f"feature index not built for {active_name} — run "
                f"python -m autopilot.vision.featureindex --build {active_name}"
            )
            return (None, diag) if debug else None

        _kf: list[cv2.KeyPoint] = []
        _df: np.ndarray | None = None
        nfeat = 0
        fp: Any = None
        for i, prep in enumerate(passes):
            # the fast pass gets only a slice of the budget so a fallback can run
            sub_budget = budget
            if budget is not None and len(passes) > 1 and i < len(passes) - 1:
                sub_budget = budget * FAST_BUDGET_FRAC
            mmf = prep(mm, ui_mask)
            _kf, _df = self._detect(mmf, max_kp)
            nfeat = 0 if _df is None else len(_df)
            if nfeat < 4:
                continue
            fp = self._index_find(
                mm,
                ui_mask,
                idx,
                cx,
                cy,
                min_inl=min_inl,
                budget=sub_budget,
                t0=t0,
                progress=progress,
                feats=(_kf, _df),
            )
            if fp is not None and len(fp) == 5 and fp[0] is not None:
                break

        diag["kp_mm"] = nfeat
        diag["kp_pts"] = [kp.pt for kp in _kf] if _kf else []
        diag["inlier_pts"] = []
        if nfeat < 4:
            diag["reject"] = "no_features_flat"
            mask_pct = int(100 * (diag["mm_mask_frac"] or 0))
            diag["detail"] = (
                f"frame without texture (feat={nfeat}, std={diag['mm_std']:.1f}, "
                f"mask={mask_pct}%) — map search impossible"
            )
            return (None, diag) if debug else None
        r = d = None
        win = orig = None
        if fp is not None and len(fp) == 5 and fp[0] is not None:
            ri = fp[0]
            di = fp[1]
            wx0 = float(fp[2])
            wy0 = float(fp[3])
            rad = float(fp[4])
            r = dict(
                s=ri["s"],
                th=ri["th"],
                t=(ri["t"][0] - wx0, ri["t"][1] - wy0),
                inl=ri["inl"],
                n_match=ri["n_match"],
            )
            d = di
            win = np.zeros((int(2 * rad), int(2 * rad)), np.uint8)
            orig = (wy0, wx0)
        elif fp is not None and len(fp) >= 2 and fp[1] is not None:
            d = fp[1]

        if r is None:
            if d is not None:
                for k in ("kp_chunk", "good1", "inl1", "refine"):
                    if k in d:
                        diag[k] = d[k]
                diag["reject"] = d.get("reject") or "index_no_match"
                diag["detail"] = d.get("detail") or "index search found no pose"
                for k in ("search_discs", "search_global"):
                    if k in d:
                        diag[k] = d[k]
            return (None, diag) if debug else None

        if d is not None:
            for k in (
                "mmi_shape",
                "kp_mm",
                "kp_chunk",
                "good1",
                "inl1",
                "good2",
                "inl2",
                "s1",
                "th1",
                "t1",
                "s2",
                "th2",
                "t2",
                "reject",
                "detail",
                "kp_pts",
                "inlier_pts",
            ):
                if k in d:
                    diag[k] = d[k]
            for k in ("search_discs", "search_global"):
                if k in d:
                    diag[k] = d[k]

        wy0, wx0 = orig  # type: ignore[misc]
        diag["win_map"] = [
            wx0 * ms,
            wy0 * ms,
            (wx0 + win.shape[1]) * ms,  # type: ignore[union-attr]
            (wy0 + win.shape[0]) * ms,  # type: ignore[union-attr]
        ]

        pc = (mm.shape[1] / 2.0, mm.shape[0] / 2.0)
        th_r = np.radians(r["th"])
        wx = r["t"][0] + r["s"] * (np.cos(th_r) * pc[0] - np.sin(th_r) * pc[1])
        wy = r["t"][1] + r["s"] * (np.sin(th_r) * pc[0] + np.cos(th_r) * pc[1])
        pose = dict(
            s=r["s"],
            th=r["th"],
            t=r["t"],
            inl=r["inl"],
            n_match=r["n_match"],
            map_x=(wx0 + wx) * ms,
            map_y=(wy0 + wy) * ms,
        )
        return (pose, diag) if debug else pose


# ==============================================================================
# Backward Compatibility Module Facade
# ==============================================================================

_DEFAULT_STORE = MapStore()
_DEFAULT_LOCATOR = MapLocator(_DEFAULT_STORE)

# Legacy shared dict access (e.g. tools/map_match_debug.py -> locator._G["mu"])
_G = _DEFAULT_STORE._g

# Expose algorithms on the module level
SIFT = _DEFAULT_LOCATOR.sift
BF = _DEFAULT_LOCATOR.bf
RATIO = _DEFAULT_LOCATOR.ratio


def get_store() -> MapStore:
    return _DEFAULT_STORE


def set_map(name: str) -> None:
    _DEFAULT_STORE.set_map(name)


def map_name() -> str:
    return _DEFAULT_STORE.map_name()


def available_maps() -> list[str]:
    return _DEFAULT_STORE.available_maps()


def full_map_size(name: str | None = None) -> tuple[int, int] | None:
    return _DEFAULT_STORE.full_map_size(name)


def load_chunk2map(name: str | None = None) -> dict[str, float] | None:
    return _DEFAULT_STORE.load_chunk2map(name)


def load_global_map() -> np.ndarray:
    return _DEFAULT_STORE.load_global_map()


def load_previews() -> dict[int, np.ndarray] | None:
    return _DEFAULT_STORE.load_previews()


def ensure_previews() -> bool:
    return _DEFAULT_STORE.ensure_previews()


def build_previews(full: np.ndarray) -> None:
    _DEFAULT_STORE.build_previews(full)


def color_map() -> np.ndarray:
    return _DEFAULT_STORE.color_map()


def rebuild_map_cache(name: str, progress_cb: Callable[[str], None] | None = None) -> bool:
    return _DEFAULT_STORE.rebuild_map_cache(name, progress_cb)


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float = ...,
    debug: Literal[True] = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
) -> tuple[dict[str, Any] | None, dict[str, Any]]: ...


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float = ...,
    debug: Literal[False] = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
) -> dict[str, Any] | None: ...


@overload
def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = ...,
    min_inl: int = ...,
    prev_th: float = ...,
    debug: bool = ...,
    budget: float | None = ...,
    progress: Callable[[tuple[Any, ...]], None] | None = ...,
) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None: ...


def global_pose(
    mm: np.ndarray,
    ui_mask: np.ndarray | None,
    prev_xy: tuple[float, float] | None = None,
    min_inl: int = 4,
    prev_th: float = 0.0,
    debug: bool = False,
    budget: float | None = None,
    progress: Callable[[tuple[Any, ...]], None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]] | dict[str, Any] | None:
    return _DEFAULT_LOCATOR.global_pose(
        mm,
        ui_mask,
        prev_xy=prev_xy,
        min_inl=min_inl,
        prev_th=prev_th,
        debug=debug,
        budget=budget,
        progress=progress,
    )


def heading_deg(pose: dict[str, Any]) -> float:
    return _DEFAULT_LOCATOR.heading_deg(pose)


# Internal legacy helpers exposed for tests and tools
def _mini_scale(name: str | None = None) -> float:
    return _DEFAULT_STORE.mini_scale(name)


def _loc_cfg() -> dict[str, Any]:
    return _DEFAULT_STORE.loc_cfg()


def _map_cfg() -> dict[str, Any]:
    return _DEFAULT_STORE.map_cfg()


def _gray_sig() -> str:
    return _DEFAULT_STORE.gray_sig()


def _gray_sig_on_disk(name: str | None = None) -> str | None:
    return _DEFAULT_STORE.gray_sig_on_disk(name)


def _get_index() -> Any:
    return _DEFAULT_STORE.get_index()


def _full_path(name: str | None = None) -> str:
    return _DEFAULT_STORE.full_path(name)


def _cache_path(name: str, suffix: str) -> str:
    return _DEFAULT_STORE.cache_path(name, suffix)


def _preview_path(name: str, n: int) -> str:
    return _DEFAULT_STORE.preview_path(name, n)


def _norm8(g: np.ndarray) -> np.ndarray:
    return norm8(g)


def _bgr_to_gray(bgr: np.ndarray, conv: str = "luma", gamma: float = 1.0) -> np.ndarray:
    return bgr_to_gray(bgr, conv=conv, gamma=gamma)


def _fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None, per: tuple[float, float]) -> np.ndarray:
    return fill_norm(mm, ui_mask, per)


def _shadow_fill_norm(mm: np.ndarray, ui_mask: np.ndarray | None) -> np.ndarray:
    return shadow_fill_norm(mm, ui_mask)


def _crop_win(
    mu: np.ndarray, cy: float, cx: float, rh: int, rw: int
) -> tuple[np.ndarray, tuple[int, int]]:
    return crop_win(mu, cy, cx, rh, rw)


if __name__ == "__main__":
    logger.info("loading the global map cache...")
    t0_main = time.time()
    mu_main = load_global_map()
    logger.info("ok: mu=%s (%.1fs)", tuple(mu_main.shape), time.time() - t0_main)
