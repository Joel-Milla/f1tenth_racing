"""One cycle of the Frenet sampling planner (Werling et al. 2010, lateral
part, parameterised by arc length).

From the car's state (s, d, d', d'') it fits quintic d(s) candidates to a
grid of end offsets x horizon lengths, rejects those that leave the track,
need more curvature than the steering allows or come too close to an
obstacle, and picks the cheapest of the rest. The chosen path gets a speed
profile and is converted to map coordinates for the path follower.

All candidates share one arc-length grid u = 0 .. path_length ahead of the
car; past its horizon a candidate holds its end offset, so short and long
candidates are checked over the same stretch of track.

Modes
  RACELINE  no relevant obstacle and the "home" candidate (back to d = 0
            over the longest horizon; exactly the raceline once the car is
            on it) stays inside the bounds: drive it, no sampling. Its
            curvature is not checked: with nothing to avoid, the raceline is
            the plan even where it is tighter than the steering allows
  RECOVER   no relevant obstacle, but home leaves the bounds (e.g. the car
            is angled at a wall and the slow return would overshoot into
            it): full sampling, raceline speed. If no candidate also meets
            the curvature limit, the cheapest one inside the bounds
  AVOID     obstacles near the path: full sampling, speed capped at v_avoid
  FALLBACK  obstacles, and sampling found nothing valid: home at
            v_fallback, stopping stop_margin short of the first collision
"""
from dataclasses import dataclass

import numpy as np

from frenet_planner.collision import clearance_along
from frenet_planner.frenet import s_diff
from frenet_planner.quintic import quintic_coeffs, quintic_eval, quintic_jerk_integral

RACELINE, RECOVER, AVOID, FALLBACK = 'RACELINE', 'RECOVER', 'AVOID', 'FALLBACK'

# 1 / clearance cost: clearances below this count as this [m]
CLEARANCE_FLOOR = 0.01
# Curvature below this is treated as straight in the lateral-acceleration cap [1/m]
KAPPA_EPS = 1e-6
# End offsets closer than this are the same candidate [m]
D_END_TOL = 1e-3


@dataclass
class FrenetState:
    s: float
    d: float
    d1: float   # dd/ds
    d2: float   # d2d/ds2


@dataclass
class Candidates:
    """n candidates on the shared u grid of m samples."""
    d_end: np.ndarray        # (n,)
    horizon: np.ndarray      # (n,)
    d: np.ndarray            # (n, m)
    d1: np.ndarray
    d2: np.ndarray
    kappa: np.ndarray        # (n, m) curvature of the path in the map frame
    clearance: np.ndarray    # (n, m) to the nearest obstacle, inf if none
    in_bounds: np.ndarray    # (n,) bool
    kappa_ok: np.ndarray
    collision_free: np.ndarray
    cost: np.ndarray         # (n,) whether rejected or not

    @property
    def valid(self):
        return self.in_bounds & self.kappa_ok & self.collision_free


@dataclass
class Plan:
    mode: str
    s0: float                # car's s; sample k is at raceline s0 + u[k]
    u: np.ndarray            # (m,)
    cands: Candidates
    chosen: int
    v: np.ndarray            # (m,) speed along the chosen path
    x: np.ndarray            # (m,) chosen path in the map frame
    y: np.ndarray
    yaw: np.ndarray

    @property
    def d_end(self):
        return float(self.cands.d_end[self.chosen])

    @property
    def d(self):
        return self.cands.d[self.chosen]

    @property
    def d1(self):
        return self.cands.d1[self.chosen]

    @property
    def d2(self):
        return self.cands.d2[self.chosen]


def relevant_mask(obstacles, s0, path_length, behind, track_length):
    """Obstacles the path can meet: rear edge before the end of the path,
    front edge no more than behind metres behind the car."""
    rear = s_diff(obstacles.s_min, s0, track_length)
    return (rear < path_length) & (rear + obstacles.s_len > -behind)


def end_offsets(lo, hi, n, extra):
    """n offsets evenly across [lo, hi], plus the extra ones inside it."""
    grid = np.linspace(lo, hi, n)
    for e in extra:
        if lo <= e <= hi and np.abs(grid - e).min() > D_END_TOL:
            grid = np.append(grid, e)
    return np.sort(grid)


def evaluate(state, prev_d_end, d_end, horizon, u, ref, obstacles, footprint, t, p, vehicle, margin):
    """Fit, check and cost the candidates (d_end[i], horizon[i]).

    t: time the car reaches each u, for moving obstacles. p: PlannerParams.
    vehicle: VehicleParams. margin: clearance below which a path collides.
    """
    path, bounds = ref.path, ref.bounds
    s = state.s + u
    kr = path.eval(s)[3]
    coeffs = quintic_coeffs(state.d, state.d1, state.d2, d_end, horizon)
    d, d1, d2, _ = quintic_eval(coeffs, horizon, u)
    kappa = path.path_curvature(s, d, d1, d2)

    inset = 0.5 * vehicle.width + p.bound_margin
    lo, hi = inset - bounds.right(s), bounds.left(s) - inset
    # Bounds are checked from bounds_grace (before that is where the car
    # already is: checking it would reject everything near a wall) to the
    # candidate's own horizon. Past it the path only holds d_end as a
    # placeholder, re-planned long before the car gets there; checking that
    # against a track that narrows ahead would forbid holding a passing line
    checked = (u >= p.bounds_grace) & (u <= np.asarray(horizon)[:, None] + 1e-9)
    in_bounds = (((d >= lo) & (d <= hi)) | ~checked).all(axis=1)
    # Where the raceline itself is tighter than the steering allows, allow
    # it (and a little more) rather than reject everything there
    kappa_limit = np.maximum(vehicle.kappa_max, np.abs(kr) * (1.0 + p.kappa_ref_slack))
    kappa_ok = (np.abs(kappa) <= kappa_limit).all(axis=1)
    clearance = clearance_along(s, d, d1, kr, t, footprint, obstacles, path.length)
    min_clear = clearance.min(axis=1)
    collision_free = min_clear >= margin

    w = p.weights
    obs_term = np.where(np.isfinite(min_clear), 1.0 / np.maximum(min_clear, CLEARANCE_FLOOR), 0.0)
    cost = (w.offset * np.abs(d).mean(axis=1)
            + w.jerk * quintic_jerk_integral(coeffs, horizon)
            + w.curvature * np.abs(kappa).max(axis=1)
            + w.obstacle * obs_term
            + w.switch * np.abs(d_end - prev_d_end)
            + w.length * (horizon.max() - horizon))
    return Candidates(d_end=d_end, horizon=horizon, d=d, d1=d1, d2=d2, kappa=kappa, clearance=clearance,
                      in_bounds=in_bounds, kappa_ok=kappa_ok, collision_free=collision_free, cost=cost)


def cheapest(cands, ok):
    return int(np.argmin(np.where(ok, cands.cost, np.inf)))


def speed_profile(u, s, kappa, clearance, mode, ref, p, speed, margin, v_cap=None):
    """Speed along the chosen path: raceline speed (global scale applied),
    capped by lateral acceleration on the path's own curvature, by v_cap if
    given (else v_avoid in AVOID), by v_fallback in FALLBACK and there zero
    from stop_margin before the first collision on the path; then a backward
    pass so braking for any of these starts early enough. speed: SpeedParams."""
    v = ref.speed(s)
    v = np.minimum(v, np.sqrt(speed.a_lat_max / np.maximum(np.abs(kappa), KAPPA_EPS)))
    if v_cap is not None:
        v = np.minimum(v, v_cap)
    elif mode == AVOID:
        v = np.minimum(v, p.v_avoid)
    if mode == FALLBACK:
        v = np.minimum(v, p.v_fallback)
        blocked = np.flatnonzero(clearance < margin)
        if len(blocked):
            v[u >= u[blocked[0]] - p.stop_margin] = 0.0
    ds = np.diff(u)
    for k in range(len(v) - 2, -1, -1):
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2.0 * speed.a_decel_max * ds[k]))
    return v


def plan(state, prev_d_end, v_plan, ref, obstacles, footprint, params,
         d_end_range=None, margin=None, v_cap=None, v_time=None):
    """One planning cycle.

    state:       FrenetState of the car
    prev_d_end:  end offset chosen last cycle (0 if none), for hysteresis
    v_plan:      speed used to scale the horizons and to time the path
    obstacles:   ObstacleSet (static or moving)
    params:      Params (planner, vehicle, collision, speed sections)
    d_end_range: (lo, hi) to sample only end offsets in it, e.g. to stay on
                 the side of an obstacle already committed to
    margin:      clearance below which a path collides; default
                 collision.obstacle_margin
    v_cap:       speed cap replacing v_avoid (FALLBACK still caps at v_fallback)
    v_time:      speed the car will hold, to time moving obstacles; default
                 v_plan floored at v_plan_min (conservative: arrives sooner)
    """
    p, vehicle = params.planner, params.vehicle
    margin = params.collision.obstacle_margin if margin is None else margin
    path = ref.path
    u = np.arange(0.0, p.path_length + 0.5 * p.ds, p.ds)
    t = u / (max(v_plan, p.v_plan_min) if v_time is None else max(v_time, KAPPA_EPS))
    near = obstacles.subset(relevant_mask(obstacles, state.s, p.path_length,
                                          p.obstacle_behind, path.length))

    longest = np.array([min(p.horizon_max, p.path_length)])
    home = evaluate(state, prev_d_end, np.zeros(1), longest, u, ref, near, footprint, t, p, vehicle, margin)
    if len(near) == 0 and home.in_bounds[0]:
        mode, cands, chosen = RACELINE, home, 0
    else:
        horizon = np.clip(v_plan * np.asarray(p.horizon_times, dtype=float), p.horizon_min, longest[0])
        s = state.s + u
        inset = 0.5 * vehicle.width + p.bound_margin
        checked = (u >= p.bounds_grace) & (u <= horizon.max() + 1e-9)
        lo = (inset - ref.bounds.right(s))[checked].max()
        hi = (ref.bounds.left(s) - inset)[checked].min()
        if d_end_range is not None:
            lo, hi = max(lo, d_end_range[0]), min(hi, d_end_range[1])
        # No band at all when the track is narrower than the car plus margins
        cands = None
        if lo <= hi:
            d_grid = end_offsets(lo, hi, p.n_offsets, (0.0, prev_d_end))
            D, H = np.meshgrid(d_grid, np.unique(horizon), indexing='ij')
            cands = evaluate(state, prev_d_end, D.ravel(), H.ravel(), u, ref, near, footprint, t, p,
                             vehicle, margin)
        if cands is not None and cands.valid.any():
            mode, chosen = AVOID if len(near) else RECOVER, cheapest(cands, cands.valid)
        elif len(near) == 0:
            mode = RECOVER
            if cands is not None and cands.in_bounds.any():
                chosen = cheapest(cands, cands.in_bounds)
            else:
                cands = home if cands is None else _append(cands, home)
                chosen = len(cands.d_end) - 1
        else:
            # Drive the home candidate; keep the rejected ones for display
            mode = FALLBACK
            cands = home if cands is None else _append(cands, home)
            chosen = len(cands.d_end) - 1

    s = state.s + u
    v = speed_profile(u, s, cands.kappa[chosen], cands.clearance[chosen], mode, ref, p, params.speed, margin, v_cap)
    x, y, yaw = path.frenet_to_cartesian(s, cands.d[chosen], cands.d1[chosen])
    return Plan(mode=mode, s0=state.s, u=u, cands=cands, chosen=chosen, v=v, x=x, y=y, yaw=yaw)


def _append(a, b):
    """Concatenate two Candidates."""
    return Candidates(*(np.concatenate([getattr(a, f), getattr(b, f)]) for f in Candidates.__dataclass_fields__))
