"""Path geometry, waypoint tracking, and trajectory calculations."""

from __future__ import annotations

import math

from .steering_controller import wrap180


class PathTracker:
    """Tracks position relative to waypoints, cross-track error, and lookahead angles."""

    def __init__(
        self,
        pts: list[tuple[float, float]],
        arrive_r: float = 55.0,
        xte_m: float = 6.0,
        max_extra: float = 0.12,
        mh_dt: float = 0.5,
        mh_min_d: float = 10.0,
        mh_max_d: float = 30.0,
    ) -> None:
        self.pts = list(pts)
        self.idx = 0
        self.arrive_r = float(arrive_r)
        self.xte_m = float(xte_m)
        self.max_extra = float(max_extra)

        self._samples: list[tuple[float, float, float]] = []  # (ts, x, y)
        self._mh: float | None = None
        self._mh_t = 0.0
        self._mv = 0.0
        self._mh_dt = float(mh_dt)
        self._mh_min_d = float(mh_min_d)
        self._mh_max_d = float(mh_max_d)

    @property
    def mv(self) -> float:
        return self._mv

    @property
    def mh(self) -> float | None:
        return self._mh

    @property
    def mh_t(self) -> float:
        return self._mh_t

    @property
    def samples(self) -> list[tuple[float, float, float]]:
        return self._samples

    def clean_stale_samples(self, now: float, max_age: float = 1.5) -> bool:
        """Prune samples older than max_age. Returns True if any sample remains."""
        if not self._samples:
            return False
        if now - self._samples[-1][0] > max_age:
            self._samples = [s for s in self._samples if now - s[0] <= max_age]
        if not self._samples:
            self._mh = None
            self._mh_t = 0.0
            self._mv = 0.0
        return bool(self._samples)

    def lost_limit(self) -> float:
        """Max plausible speed for ghost-pose detection (px/s)."""
        return max(self._mv * 2.0 + 8.0, 50.0)

    def push_pose(self, ts: float, x: float, y: float) -> bool:
        """Append a fresh pose sample and update motion-derived course (M-heading)."""
        if self._samples and ts <= self._samples[-1][0]:
            return False

        if self._samples and ts - self._samples[-1][0] > 1.5:
            self._samples.clear()
            self._mh = None
            self._mh_t = 0.0
            self._mv = 0.0

        if self._samples:
            base = ts - self._mh_dt
            bx, by, bt = self._samples[0][1], self._samples[0][2], self._samples[0][0]
            for s in self._samples:
                if s[0] <= base:
                    bt, bx, by = s[0], s[1], s[2]
                else:
                    break
            dts = ts - bt
            d = math.hypot(x - bx, y - by)
            if dts > 1e-3:
                self._mv = d / dts
                if d >= self._mh_min_d:
                    self._mh_t = ts
                    raw = math.degrees(math.atan2(x - bx, -(y - by))) % 360.0
                    if self._mh is None:
                        self._mh = raw
                    else:
                        dd = wrap180(raw - self._mh)
                        dd = max(-self._mh_max_d, min(self._mh_max_d, dd))
                        self._mh = (self._mh + dd * 0.5) % 360.0

        self._samples.append((ts, x, y))
        if len(self._samples) > 16:
            self._samples = self._samples[-16:]
        return True

    def motion_heading(self, now: float) -> float | None:
        """Return course only while supported by recent measurable movement."""
        if self._mh is None or not 0.0 <= now - self._mh_t <= 0.75 or self._mv < 3.0:
            return None
        return self._mh

    def pose_at(self, now: float) -> tuple[float, float] | None:
        """Estimate current position via linear interpolation/extrapolation."""
        n = len(self._samples)
        if n == 0:
            return None
        if n == 1:
            return (self._samples[0][1], self._samples[0][2])

        t0, x0, y0 = self._samples[-2]
        t1, x1, y1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 1e-4:
            return (x1, y1)

        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        d = now - t1
        if d < 0:
            k = (now - t0) / dt
            return (x0 + (x1 - x0) * k, y0 + (y1 - y0) * k)
        if d <= self.max_extra:
            return (x1 + vx * d, y1 + vy * d)
        return (x1, y1)

    def advance_waypoint(self, mp: tuple[float, float]) -> tuple[float, float, float, bool]:
        """Find the nearest unpassed point and check arrival radius.

        Returns (target_x, target_y, distance, arrived_at_end).
        """
        if self.idx >= len(self.pts):
            return 0.0, 0.0, 0.0, True

        bi = self.idx
        bd = math.hypot(self.pts[bi][0] - mp[0], self.pts[bi][1] - mp[1])
        for i in range(self.idx + 1, len(self.pts)):
            di = math.hypot(self.pts[i][0] - mp[0], self.pts[i][1] - mp[1])
            if di < bd:
                bi, bd = i, di
        self.idx = bi
        tx, ty = self.pts[bi]
        dist = bd

        if dist < self.arrive_r:
            self.idx += 1
            if self.idx >= len(self.pts):
                return tx, ty, dist, True
            tx, ty = self.pts[self.idx]
            dist = math.hypot(tx - mp[0], ty - mp[1])

        return tx, ty, dist, False

    def calc_xte_and_bearing(
        self,
        mp: tuple[float, float],
        px_per_m: float,
    ) -> tuple[float, float, float]:
        """Calculate cross-track error, corridor limit, and pure pursuit target bearing."""
        look = max(90.0, min(240.0, 60.0 + self._mv * 2.6))
        si = self.idx if self.idx >= 1 else 1
        ax3, ay3 = self.pts[si - 1]
        bx3, by3 = self.pts[si]
        sx3 = bx3 - ax3
        sy3 = by3 - ay3
        l2 = sx3 * sx3 + sy3 * sy3

        xte = 0.0
        if l2 > 1e-6:
            xte = ((mp[0] - ax3) * sy3 - (mp[1] - ay3) * sx3) / math.sqrt(l2)

        xte_lim = self.xte_m * px_per_m

        if abs(xte) > xte_lim:
            foot_t = ((mp[0] - ax3) * sx3 + (mp[1] - ay3) * sy3) / l2
            foot_t = min(max(foot_t, 0.0), 1.0)
            ax2 = ax3 + foot_t * sx3
            ay2 = ay3 + foot_t * sy3
        elif self.idx + 1 < len(self.pts):
            ax, ay = self.pts[self.idx]
            bx, by = self.pts[self.idx + 1]
            sx = bx - ax
            sy = by - ay
            seg2 = sx * sx + sy * sy
            if seg2 > 1e-6:
                t = ((mp[0] - ax) * sx + (mp[1] - ay) * sy) / seg2
                t = min(max(t, 0.0), 1.0)
                tl = min(t + look / math.sqrt(seg2), 0.95)
                ax2 = ax + tl * sx
                ay2 = ay + tl * sy
            else:
                ax2, ay2 = bx, by
        else:
            tx, ty = self.pts[self.idx] if self.idx < len(self.pts) else self.pts[-1]
            ax2, ay2 = tx, ty

        bearing = math.degrees(math.atan2(ax2 - mp[0], -(ay2 - mp[1]))) % 360.0
        return xte, xte_lim, bearing

    def calc_road_turn(self, heading: float) -> tuple[float, float]:
        """Compute immediate turn angle and lookahead road turn curvature."""
        if self.idx + 1 < len(self.pts):
            out_h = (
                math.degrees(
                    math.atan2(
                        self.pts[self.idx + 1][0] - self.pts[self.idx][0],
                        -(self.pts[self.idx + 1][1] - self.pts[self.idx][1]),
                    )
                )
                % 360.0
            )
            if self.idx > 0:
                in_h = (
                    math.degrees(
                        math.atan2(
                            self.pts[self.idx][0] - self.pts[self.idx - 1][0],
                            -(self.pts[self.idx][1] - self.pts[self.idx - 1][1]),
                        )
                    )
                    % 360.0
                )
            else:
                in_h = heading
            turn_angle = abs(wrap180(out_h - in_h))
        else:
            in_h = heading
            turn_angle = 0.0

        ahead_t = 0.0
        acc = 0.0
        h_prev = in_h
        for i in range(self.idx + 1, len(self.pts)):
            h_next = (
                math.degrees(
                    math.atan2(
                        self.pts[i][0] - self.pts[i - 1][0],
                        -(self.pts[i][1] - self.pts[i - 1][1]),
                    )
                )
                % 360.0
            )
            ahead_t = max(ahead_t, abs(wrap180(h_next - h_prev)))
            h_prev = h_next
            acc += math.hypot(
                self.pts[i][0] - self.pts[i - 1][0],
                self.pts[i][1] - self.pts[i - 1][1],
            )
            if acc >= 200.0 or i >= self.idx + 6:
                break

        road_turn = max(turn_angle, ahead_t)
        return turn_angle, road_turn
