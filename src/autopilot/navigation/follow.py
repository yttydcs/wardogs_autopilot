"""Autopilot route follower thread for WARDOGS.

Coordinates localization poses, route tracking, speed control, steering impulses,
and Arduino keyboard driver key injection.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Sequence
from typing import Any

from ..common.config import NavigatorConfig
from ..common.log import get_logger
from .path_tracker import PathTracker
from .speed_controller import SpeedController
from .steering_controller import SteeringController, wrap180
from .telemetry import NavTelemetryLogger

logger = get_logger("follow")


class FollowDriver(threading.Thread):
    """Route-following background driver thread.

    Takes a LiveLocator and a list of route points, computes target bearing
    and speed, steers via pulse impulses, and injects keys into the key driver.
    """

    def __init__(
        self,
        loc: Any,
        pts: Sequence[tuple[float, float]],
        arrive_r: float | None = None,
        slow_r: float | None = None,
        dead: float | None = None,
        poll: float | None = None,
        kb: Any = None,
        brake_d: float | None = None,
        dead_off: float | None = None,
        speed_cap_kmh: float | None = None,
        xte_m: float | None = None,
        debug: bool | None = None,
        turn_deg: float | None = None,
        hold_max: float | None = None,
        stop_speed_kmh: float | None = None,
        stop_hold: float | None = None,
        stop_timeout: float | None = None,
        nav_cfg: NavigatorConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(daemon=True)
        if isinstance(nav_cfg, dict):
            try:
                self.nav_cfg = NavigatorConfig(**nav_cfg)
            except Exception:
                self.nav_cfg = NavigatorConfig()
        elif isinstance(nav_cfg, NavigatorConfig):
            self.nav_cfg = nav_cfg
        else:
            self.nav_cfg = NavigatorConfig()

        self.loc = loc
        self.pts = list(pts)
        self.arrive_r = float(arrive_r if arrive_r is not None else self.nav_cfg.arrive_r)
        self.slow_r = float(slow_r if slow_r is not None else self.nav_cfg.slow_r)
        base_dead = dead if dead is not None else self.nav_cfg.dead
        self.dead = max(float(base_dead), 6.0)
        self.poll = float(poll if poll is not None else self.nav_cfg.poll)
        self.brake_d = float(brake_d if brake_d is not None else self.nav_cfg.brake_d)
        cfg_dead_off = dead_off if dead_off is not None else self.nav_cfg.dead_off
        self.dead_off = max(
            float(cfg_dead_off if cfg_dead_off is not None else self.dead * 0.3), 2.0
        )
        self.speed_cap_kmh = float(
            speed_cap_kmh if speed_cap_kmh is not None else self.nav_cfg.speed_cap_kmh
        )
        self.xte_m = float(xte_m if xte_m is not None else self.nav_cfg.xte_m)
        self.dbg = bool(debug if debug is not None else self.nav_cfg.debug)
        self.stop_speed_kmh = float(
            stop_speed_kmh if stop_speed_kmh is not None else self.nav_cfg.stop_speed_kmh
        )
        self.stop_hold = float(stop_hold if stop_hold is not None else self.nav_cfg.stop_hold)
        self.stop_timeout = float(
            stop_timeout if stop_timeout is not None else self.nav_cfg.stop_timeout
        )
        self.kb = kb

        # Sub-controllers
        self.path = PathTracker(self.pts, arrive_r=self.arrive_r, xte_m=self.xte_m)
        self.speed_ctrl = SpeedController(speed_cap_kmh=self.speed_cap_kmh, brake_d=self.brake_d)
        self.steer_ctrl = SteeringController(
            dead=self.dead,
            dead_off=self.dead_off,
            turn_deg=(turn_deg if turn_deg is not None else self.nav_cfg.turn_deg),
            hold_max=(hold_max if hold_max is not None else self.nav_cfg.hold_max),
        )
        self.telemetry = NavTelemetryLogger(dbg_target=self.dbg)

        # Public state for UI polling
        self.state = "idle"
        self.last: dict[str, Any] = dict(
            idx=0, dist=0.0, bearing=0.0, err=0.0, heading=0.0, turn=0.0, speed=0.0
        )
        self.err: Exception | None = None
        self.speed = 0.0

        self._last_mp: tuple[float, float] | None = None
        self._last_t = 0.0
        self._heading: float | None = None
        self._lost = False
        self._dbg_n = 0
        self._stop_ev = threading.Event()
        self._final_stop = False
        self._stop_s_t: float | None = None
        self._final_t0: float | None = None

    @property
    def idx(self) -> int:
        return self.path.idx

    @idx.setter
    def idx(self, val: int) -> None:
        self.path.idx = val

    # Backward-compatible property facades
    @property
    def _samples(self) -> list[tuple[float, float, float]]:
        return self.path.samples

    @_samples.setter
    def _samples(self, val: list[tuple[float, float, float]]) -> None:
        self.path._samples = val

    @property
    def _mv(self) -> float:
        return self.path.mv

    @property
    def _mh(self) -> float | None:
        return self.path.mh

    @property
    def _mh_t(self) -> float:
        return self.path.mh_t

    @property
    def _ang(self) -> float:
        return self.steer_ctrl.ang

    @property
    def _steer(self) -> int:
        return self.steer_ctrl.steer

    @property
    def _micro(self) -> bool:
        return self.steer_ctrl.micro

    @property
    def _px_per_m(self) -> float:
        return self.speed_ctrl._px_per_m

    @property
    def _vmax_px(self) -> float:
        return self.speed_ctrl._vmax_px

    @property
    def _runaway(self) -> bool:
        return self.speed_ctrl.runaway

    @property
    def _braking(self) -> bool:
        return self.speed_ctrl.is_braking

    def stop(self) -> None:
        """Signal thread to stop and release all held keyboard keys."""
        self._stop_ev.set()
        try:
            if self.kb is not None:
                close = getattr(self.kb, "close", self.kb.release_all)
                close()
        except OSError:
            pass
        self.telemetry.close()

    def _px_per_m_now(self) -> float:
        return self.speed_ctrl.px_per_m_now()

    def _kmh(self, px_s: float) -> float:
        return self.speed_ctrl.to_kmh(px_s)

    def _m(self, d_px: float) -> float:
        return self.speed_ctrl.to_meters(d_px)

    def _lost_limit(self) -> float:
        return self.path.lost_limit()

    def _push_pose(self, ts: float, x: float, y: float) -> bool:
        added = self.path.push_pose(ts, x, y)
        if added:
            self.speed_ctrl.update_scale(self.path.mv)
        return added

    def _pose_at(self, now: float) -> tuple[float, float] | None:
        return self.path.pose_at(now)

    def _impulse(self, err: float) -> float:
        return self.steer_ctrl.calc_impulse(err)

    def _dbg_open(self) -> None:
        self.telemetry.open(self.pts, self._get_telemetry_params())

    def _dbg_tick(self, row: dict[str, Any]) -> None:
        self.telemetry.tick(row, self.pts, self._get_telemetry_params())

    def _dbg_flush(self) -> None:
        self.telemetry.flush()

    def _dbg_close(self) -> None:
        self.telemetry.close()

    def _wait(self, base: float) -> None:
        self._stop_ev.wait(base * random.uniform(0.75, 1.35))

    def _rel(self, state: str) -> None:
        try:
            if self.kb is not None:
                self.kb.release_all()
        except OSError:
            pass
        self.state = state

    def _update_speed(self, now: float, mp: tuple[float, float]) -> None:
        """Smoothed instantaneous speed estimate from successive poses."""
        prev = self._last_mp
        self._last_mp = (float(mp[0]), float(mp[1]))
        if prev is not None and self._last_t:
            dtp = max(now - self._last_t, 1e-3)
            inst = math.hypot(mp[0] - prev[0], mp[1] - prev[1]) / dtp
            self.speed = min(400.0, self.speed * 0.6 + inst * 0.4)
        else:
            self.speed = 0.0
        self._last_t = now

    def _final_brake_dist(self) -> float:
        """Distance to the final waypoint at which full-stop braking begins."""
        mv = self.path.mv
        return (max(mv, 0.0) ** 2 / (2.0 * self.brake_d) + 12.0) * 1.5

    def _stop_thr(self) -> float:
        """Full-stop speed threshold in px/s (km/h knob, px floor applied)."""
        pm = self.speed_ctrl.px_per_m_now()
        if pm > 0:
            return max(self.stop_speed_kmh * pm / 3.6, 6.0)
        return 6.0

    def _fully_stopped(self, now: float) -> bool:
        """True once the vehicle stayed below the stop speed long enough (or timed out)."""
        thr = self._stop_thr()
        if max(self.speed, self.path.mv) <= thr:
            if self._stop_s_t is None:
                self._stop_s_t = now
            elif now - self._stop_s_t >= self.stop_hold:
                return True
        else:
            self._stop_s_t = None
        t0 = self._final_t0
        return t0 is not None and (now - t0) >= self.stop_timeout

    def _finish(self) -> None:
        """Full stop at the final waypoint: disable the autopilot (as with F7)."""
        logger.info("[nav] full stop at final waypoint — disabling autopilot")
        self.state = "finished"
        try:
            if self.kb is not None:
                self.kb.release_all()
        except OSError:
            pass
        self._stop_ev.set()

    def _get_telemetry_params(self) -> dict[str, Any]:
        return dict(
            arrive_r=self.arrive_r,
            slow_r=self.slow_r,
            dead=self.dead,
            dead_off=self.dead_off,
            brake_d=self.brake_d,
            lead_t=self.steer_ctrl.lead_t,
            settle_t=self.steer_ctrl.settle_t,
            imp_k=self.steer_ctrl.imp_k,
            w_est=self.steer_ctrl.w_est,
            t_min=self.steer_ctrl.t_min,
            t_max=self.steer_ctrl.t_max,
            turn_deg=self.steer_ctrl.turn_deg,
            hold_max=self.steer_ctrl.hold_max,
            speed_cap_kmh=self.speed_cap_kmh,
            v_cruise=self.speed_ctrl.v_cruise,
            xte_m=self.xte_m,
            stop_speed_kmh=self.stop_speed_kmh,
            stop_hold=self.stop_hold,
            stop_timeout=self.stop_timeout,
        )

    def run(self) -> None:
        """Main navigation control loop."""
        try:
            while not self._stop_ev.is_set():
                now = time.time()
                new_sample = False
                it = getattr(self.loc, "latest", None)
                if it is not None:
                    mpx = it.get("map_px")
                    ts = it.get("ts", now)
                    if mpx is not None:
                        x, y = float(mpx[0]), float(mpx[1])
                        # Ghost pose detection
                        if self.path.samples and ts > self.path.samples[-1][0]:
                            lt, lx, ly = self.path.samples[-1]
                            dts = ts - lt
                            d = math.hypot(x - lx, y - ly)
                            if dts > 1e-3 and d / dts > self.path.lost_limit():
                                self._lost = True
                                self._dbg_tick(
                                    dict(
                                        kind="lost",
                                        t=now,
                                        sa=round(now - lt, 2),
                                        d=round(self._m(d), 1),
                                        v_claim=round(d / dts, 1),
                                    )
                                )
                                self._rel("lost")
                                self._wait(0.25)
                                continue
                        new_sample = self._push_pose(ts, x, y)

                if new_sample:
                    self._lost = False

                if not self.path.clean_stale_samples(now, max_age=1.5):
                    self._rel("wait_pose")
                    self._wait(0.2)
                    continue

                mp = self.path.pose_at(now)
                if mp is None:
                    self._rel("wait_pose")
                    self._wait(0.2)
                    continue

                pose = it.get("pose") if it else None
                if pose is None:
                    self._rel("wait_pose")
                    self._wait(0.3)
                    continue

                # Advance waypoints
                final_seg = len(self.pts) > 1 and self.path.idx >= len(self.pts) - 1
                tx, ty, dist, arrived = self.path.advance_waypoint(mp)
                if arrived and not final_seg:
                    self._rel("arrived")
                    self._wait(0.3)
                    continue

                # The last waypoint is the active target: brake with SPACE down to a
                # full stop, then disable the autopilot (the same as pressing F7).
                if final_seg and not self._final_stop and dist < self._final_brake_dist():
                    self._final_stop = True
                    self._final_t0 = now
                    self._stop_s_t = None
                    logger.info(
                        "[nav] final waypoint %d in stop range (dist=%.0fm) "
                        "sv=%.0fpx/s mv=%.0fpx/s",
                        self.path.idx,
                        self._m(dist),
                        self.speed,
                        self.path.mv,
                    )

                if self._final_stop:
                    self._update_speed(now, mp)
                    if self._fully_stopped(now):
                        self._finish()
                        continue
                    self._rel("final_stop")
                    try:
                        if self.kb is not None:
                            self.kb.set_state({"SPACE": True})
                    except OSError as exc:
                        self.err = exc
                        self.state = "key_error"
                        self._wait(0.3)
                        continue
                    self.steer_ctrl.force_release(now, self.path.mh_t)
                    self.err = None
                    if self.dbg:
                        self._dbg_tick(
                            dict(
                                t=round(now, 4),
                                tick=self._dbg_n,
                                kind="final_stop",
                                pose_age=round(now - float(it["ts"]), 3) if it else 0.0,
                                sample_age=round(now - self.path.samples[-1][0], 3),
                                new_sample=bool(new_sample),
                                mode="S",
                                mv=round(self.path.mv, 1),
                                heading=round(self._heading or 0.0, 2),
                                speed=round(self.speed, 1),
                                dist=round(dist, 1),
                                keys="SPACE",
                                idx=self.path.idx,
                            )
                        )
                    if self._dbg_n % 5 == 0:
                        logger.info(
                            "[nav] final-stop braking sv=%.1fpx/s mv=%.1fpx/s dist=%.0fm",
                            self.speed,
                            self.path.mv,
                            self._m(dist),
                        )
                    self._dbg_n += 1
                    self.last = dict(
                        idx=self.path.idx,
                        dist=dist,
                        bearing=0.0,
                        err=0.0,
                        heading=self._heading or 0.0,
                        turn=0.0,
                        speed=self.speed,
                    )
                    self.state = "final_stop"
                    self._wait(self.poll)
                    continue

                # Speed estimation from successive positions
                self._update_speed(now, mp)

                # Cross-track error & pure pursuit bearing
                xte, xte_lim, bearing = self.path.calc_xte_and_bearing(mp, self._px_per_m_now())

                # Heading calculation & smoothing
                mh_age = now - self.path.mh_t if self.path.mh is not None else 1e9
                mh_on = self.path.mh is not None and self.path.mv > 10.0 and mh_age < 1.5
                if self.path.mh is not None and mh_age < 3.0:
                    heading_src = self.path.mh
                elif pose is not None:
                    heading_src = float(pose["th"]) % 360.0
                else:
                    heading_src = self.path.mh or 0.0

                if self._heading is None:
                    self._heading = heading_src
                else:
                    dd = wrap180(heading_src - self._heading)
                    self._heading = (self._heading + dd * 0.5) % 360.0
                heading = self._heading
                err = wrap180(bearing - heading)

                # Road geometry angles
                turn_angle, road_turn = self.path.calc_road_turn(heading)
                tgt_spd = self.speed_ctrl.calc_target_speed(road_turn, xte, xte_lim)

                # Steering state machine
                steer_action = self.steer_ctrl.step(now, err, heading, self.path.mh, self.path.mh_t)
                keys: dict[str, bool] = {}
                if steer_action == 1:
                    keys["D"] = True
                elif steer_action == -1:
                    keys["A"] = True

                # Throttle and braking evaluation
                gas_w, brake_space = self.speed_ctrl.decide_throttle_and_brake(
                    mv=self.path.mv,
                    tgt_spd=tgt_spd,
                    dist=dist,
                    arrive_r=self.arrive_r,
                    turn_angle=turn_angle,
                    steer=self.steer_ctrl.steer,
                    micro=self.steer_ctrl.micro,
                    road_turn=road_turn,
                    turn_min=10.0,
                    err=err,
                    xte=xte,
                    xte_lim=xte_lim,
                )

                if brake_space:
                    keys["SPACE"] = True
                    keys.pop("A", None)
                    keys.pop("D", None)
                    self.steer_ctrl.force_release(now, self.path.mh_t)
                elif gas_w:
                    keys["W"] = True

                if self._dbg_n % 5 == 0:
                    hmode = "M" if mh_on else "S"
                    logger.info(
                        "[nav] mp=%.0f,%.0f goal=%d (%.0f,%.0f) dist=%.0fm "
                        "bearing=%.1f heading=%.1f err=%.1f xte=%.1fm "
                        "turn=%.0f tgt=%.0fkm/h spar=%.0fkm/h(%.0f) "
                        "keys=%s th=%s",
                        mp[0],
                        mp[1],
                        self.path.idx,
                        tx,
                        ty,
                        self._m(dist),
                        bearing,
                        heading,
                        err,
                        self._m(abs(xte)),
                        turn_angle,
                        self._kmh(tgt_spd),
                        self._kmh(self.path.mv),
                        self.path.mv,
                        "".join(k for k in ("W", "A", "D", "SPACE") if keys.get(k)),
                        hmode,
                    )

                self._dbg_n += 1
                try:
                    if self.kb is not None:
                        self.kb.set_state(keys)
                except OSError as exc:
                    self.err = exc
                    self.state = "key_error"
                    self._wait(0.3)
                    continue

                self.err = None
                if self.dbg:
                    self._dbg_tick(
                        dict(
                            t=round(now, 4),
                            tick=self._dbg_n,
                            pose_age=round(now - float(it["ts"]), 3) if it else 0.0,
                            sample_age=round(now - self.path.samples[-1][0], 3),
                            new_sample=bool(new_sample),
                            mode="M" if mh_on else "S",
                            mv=round(self.path.mv, 1),
                            mh=round(self.path.mh, 1) if self.path.mh is not None else None,
                            heading=round(heading, 2),
                            bearing=round(bearing, 2),
                            err=round(err, 2),
                            turn=round(turn_angle, 1),
                            roadT=round(road_turn, 1),
                            tgt=round(tgt_spd, 1),
                            dist=round(dist, 1),
                            xte=round(abs(xte), 1),
                            xte_m=round(self._m(abs(xte)), 1),
                            speed=round(self.speed, 1),
                            v=round(self.path.mv, 1),
                            runaway=self.speed_ctrl.runaway,
                            ang=round(self.steer_ctrl.ang, 2),
                            steer=self.steer_ctrl.steer,
                            micro=self.steer_ctrl.micro,
                            braking=bool(brake_space),
                            keys="".join(k for k in ("W", "A", "D", "SPACE") if keys.get(k)),
                            idx=self.path.idx,
                        )
                    )

                self.last = dict(
                    idx=self.path.idx,
                    dist=dist,
                    bearing=bearing,
                    err=err,
                    heading=heading,
                    turn=turn_angle,
                    speed=self.speed,
                )
                self.state = "run"
                self._wait(self.poll)
        finally:
            self.telemetry.close()
            try:
                if self.kb is not None:
                    close = getattr(self.kb, "close", self.kb.release_all)
                    close()
            except OSError:
                pass
