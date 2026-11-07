"""Behaviour layer: decides what the sampling planner is asked to do.

States
  RACELINE  nothing to deal with; the planner follows (or returns to) the
            raceline, or recovers / falls back on its own
  AVOID     a static obstacle in the way: sample around it. The side chosen
            first (left or right of the obstacle) is kept until the
            obstacle is clear_dist behind the car's rear
  TRAIL     a moving opponent ahead and no safe pass: follow the raceline
            at a speed that holds trail_gap behind it (dropping back
            smoothly if it is alongside), still checking every obstacle,
            the opponent included, so an abandoned pass does not steer
            back into it
  OVERTAKE  a moving opponent ahead, we are faster by at least
            overtake_min_advantage, and there is room to sit beside it: a
            path keeps overtake_margin from the opponent taken as a parked
            box pass_length long (timing it at a modest closing speed alone
            says nothing about the lane beside it, so it could commit to a
            side too narrow to pass on). The side is kept until the opponent is
            clear_dist behind. The pass continues while a path on that side
            keeps overtake_keep_margin (hysteresis: a pass that only just
            qualified does not flip back and forth); if it stops being safe,
            back to TRAIL, and no new attempt for overtake_cooldown

Collisions with the opponent are costly, so OVERTAKE needs a bigger margin
than static avoidance and is abandoned rather than forced.
"""
from dataclasses import dataclass, replace

import numpy as np

from frenet_planner.frenet import s_diff
from frenet_planner.planner import AVOID, FALLBACK, plan

RACELINE, TRAIL, OVERTAKE = 'RACELINE', 'TRAIL', 'OVERTAKE'
LEFT, RIGHT = 1, -1


@dataclass
class Commitment:
    """A side of one obstacle: s_front is where its front edge was when
    committed (static obstacles do not move)."""
    side: int
    d_centre: float
    s_front: float


class Behavior:

    def __init__(self, params, track_length):
        self.params = params
        self.p = params.behavior
        # Overtake plans weight clearance far more: beside an opponent, the
        # widest line beats staying near the raceline
        pl = params.planner
        self.overtake_params = replace(params, planner=replace(
            pl, weights=replace(pl.weights, obstacle=params.behavior.overtake_obstacle_weight)))
        self.length = track_length
        self.state = RACELINE
        self.commit = None
        self.v_trail = None          # speed command while trailing [m/s]
        self.why = ''                # last overtake abort, for debugging
        self.cooldown = 0            # planning cycles before another overtake attempt
        self.opponent = None         # index into the obstacle set, for display

    # --- helpers -------------------------------------------------------------

    def _rel(self, s_obs, s_car):
        return s_diff(s_obs, s_car, self.length)

    def _behind(self, s_front, s_car):
        """True once an obstacle front edge at s_front is clear_dist behind
        the car's rear."""
        rear = -0.5 * self.params.vehicle.length
        return self._rel(s_front, s_car) < rear - self.p.clear_dist

    def _side_range(self, commit):
        return (commit.d_centre, np.inf) if commit.side == LEFT else (-np.inf, commit.d_centre)

    def _opponent_ahead(self, obs, s):
        """Index of the nearest moving obstacle whose rear is less than
        opponent_range ahead and whose front is not yet clear_dist behind
        the car (so alongside counts); None if there is none."""
        if len(obs) == 0:
            return None
        moving = obs.v_s != 0.0
        rel = self._rel(obs.s_min, s)
        ok = moving & (rel < self.p.opponent_range) & ~self._behind(obs.s_min + obs.s_len, s)
        if not ok.any():
            return None
        return int(np.flatnonzero(ok)[np.argmin(rel[ok])])

    def _blocking(self, obs, s):
        """Index of the nearest obstacle not yet behind the car's pose."""
        rel = self._rel(obs.s_min, s)
        ahead = np.flatnonzero(rel > -0.5 * self.params.vehicle.length)
        if len(ahead) == 0:
            return None
        return int(ahead[np.argmin(rel[ahead])])

    # --- one cycle -----------------------------------------------------------

    def step(self, state, prev_d_end, v_car, ref, obstacles, footprint):
        """Decide and plan. Returns the Plan to drive; self.state says why."""
        prm, p = self.params, self.p
        s = state.s
        opp = self._opponent_ahead(obstacles, s)
        self.opponent = opp

        # Release a static commitment once its obstacle is behind
        if self.commit is not None and self.state in (RACELINE, AVOID) \
                and self._behind(self.commit.s_front, s):
            self.commit = None

        if opp is None:
            if self.state == OVERTAKE:
                self.commit = None
            self.state = RACELINE
            return self._static(state, prev_d_end, v_car, ref, obstacles, footprint)

        o = obstacles
        v_opp = float(o.v_s[opp])
        d_opp = 0.5 * float(o.d_min[opp] + o.d_max[opp])
        v_ref = float(ref.speed(s))
        v_pass = max(v_car, v_ref)
        # Passes are timed at the speed we will realistically have over the
        # next metres: what we do now, and at least the advantage the pass
        # is premised on. Timing at full raceline speed from a trailing
        # speed would make every pull-out look like a lunge
        v_time = max(v_car, v_opp + p.overtake_min_advantage)

        half = 0.5 * prm.vehicle.length
        gap = float(self._rel(o.s_min[opp], s)) - half                      # our nose to its rear
        passed = float(self._rel(o.s_min[opp] + o.s_len[opp], s)) < -half   # its front behind our rear

        self.cooldown = max(self.cooldown - 1, 0)
        if self.state == OVERTAKE:
            pl = plan(state, prev_d_end, v_pass, ref, obstacles, footprint, self.overtake_params,
                      d_end_range=self._side_range(self.commit), margin=p.overtake_keep_margin, v_cap=v_ref,
                      v_time=v_time)
            # Anything but FALLBACK is fine: RACELINE means it is no longer
            # in the way. Stay committed until it is clear_dist behind
            if pl.mode != FALLBACK:
                return pl
            c = pl.cands
            self.why = (f'abort: {pl.mode}, of {len(c.d_end)} in_bounds {c.in_bounds.sum()} '
                        f'kappa_ok {c.kappa_ok.sum()} clear {c.collision_free.sum()} '
                        f'(best clearance {c.clearance.min(axis=1).max():.2f})')
            self.commit = None                 # pass no longer safe
            self.cooldown = int(round(p.overtake_cooldown * prm.planner.rate))
        if passed:
            # Ahead of it (or the pass just ended level with its front):
            # nothing to trail; drive on, still checking it as an obstacle
            self.state = RACELINE
            return self._static(state, prev_d_end, v_car, ref, obstacles, footprint)
        if self.state != OVERTAKE and self.cooldown == 0 and v_ref - v_opp >= p.overtake_min_advantage:
            lane = obstacles.subset(np.ones(len(obstacles), dtype=bool))
            lane.s_len[opp] = max(lane.s_len[opp], p.pass_length)
            lane.v_s[opp] = lane.v_d[opp] = 0.0
            pl = plan(state, prev_d_end, v_pass, ref, lane, footprint, self.overtake_params,
                      margin=p.overtake_margin, v_cap=v_ref, v_time=v_time)
            if pl.mode == AVOID and abs(pl.d_end - d_opp) >= p.pass_offset:
                self.state = OVERTAKE
                self.commit = Commitment(LEFT if pl.d_end > d_opp else RIGHT, d_opp, np.nan)
                return pl

        # TRAIL: gap-holding speed. A hard stop only for an opponent close
        # ahead; otherwise never slower than drop_back_speed below it, so
        # an abandoned pass drops back gradually instead of braking beside it
        self.state = TRAIL
        self.v_trail = 0.0 if 0.0 <= gap < p.trail_min_gap else \
            float(np.clip(v_opp + p.trail_gain * (gap - p.trail_gap), max(v_opp - p.drop_back_speed, 0.0), v_ref))
        # Timed at the speed we will hold, which the gap law keeps at or
        # below the opponent's once inside trail_gap
        return plan(state, prev_d_end, self.v_trail, ref, obstacles, footprint, prm, v_cap=self.v_trail,
                    v_time=min(self.v_trail, v_opp))

    def _static(self, state, prev_d_end, v_car, ref, obstacles, footprint):
        side = self._side_range(self.commit) if self.commit is not None else None
        pl = plan(state, prev_d_end, v_car, ref, obstacles, footprint, self.params, d_end_range=side)
        self.state = AVOID if pl.mode == AVOID else RACELINE
        if pl.mode == AVOID and self.commit is None:
            k = self._blocking(obstacles, state.s)
            if k is not None:
                d_c = 0.5 * float(obstacles.d_min[k] + obstacles.d_max[k])
                self.commit = Commitment(LEFT if pl.d_end > d_c else RIGHT, d_c,
                                         float(obstacles.s_min[k] + obstacles.s_len[k]))
        elif pl.mode == FALLBACK and side is not None:
            # Nothing valid on the committed side any more (the obstacle
            # moved or grew): free the side so next cycle can choose again
            self.commit = None
        return pl
