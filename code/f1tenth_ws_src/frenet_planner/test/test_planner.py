"""Planning cycle on a ring track with known geometry.

Circle raceline r = 2 (CCW, kappa = 0.5, s = 2 * angle). Two rings: a
narrow one (walls at r = 1.5 / 2.8: 0.5 m left, 0.8 m right) and a wide,
symmetric one (r = 1.0 / 3.0: 1 m each side).
"""
import numpy as np
import pytest

from conftest import CONFIG_YAML
from frenet_planner.collision import ObstacleSet, car_footprint
from frenet_planner.params import Params
import dataclasses

from frenet_planner.planner import AVOID, FALLBACK, RACELINE, RECOVER, FrenetState, plan
from frenet_planner.reference import Reference
from frenet_planner.track_bounds import TrackBounds, compute_track_bounds
from test_track_bounds import circle_ref, ring_grid  # noqa: F401 (fixture)

V_REF = 2.0
V_PLAN = 2.0
BOX = 0.3
TOL = 1e-9


@pytest.fixture(scope='module')
def params():
    return Params.from_yaml(CONFIG_YAML)


@pytest.fixture(scope='module')
def footprint(params):
    v = params.vehicle
    return car_footprint(v.length, v.width, params.collision.n_circles, 0.0)


def make_ref(path, r_in, r_out):
    grid = ring_grid(r_in, r_out)
    bounds = compute_track_bounds(path, grid, 0.05, 3.0, 0.0)
    return Reference(path=path, grid=grid, bounds=bounds, v_ref=np.full(len(path.s_table), V_REF))


@pytest.fixture(scope='module')
def narrow(circle_ref):  # noqa: F811
    return make_ref(circle_ref, 1.5, 2.8)


@pytest.fixture(scope='module')
def wide(circle_ref):  # noqa: F811
    return make_ref(circle_ref, 1.0, 3.0)


def box(s_c, d_c, v_s=0.0):
    return ObstacleSet.from_boxes([(s_c - BOX / 2, BOX, d_c - BOX / 2, d_c + BOX / 2, v_s, 0.0)])


NONE = ObstacleSet.from_boxes([])


def run(ref, obstacles, footprint, params, s=1.0, d=0.0, prev_d_end=0.0, d1=0.0):
    return plan(FrenetState(s, d, d1, 0.0), prev_d_end, V_PLAN, ref, obstacles, footprint, params)


def chosen_ok(pl, params):
    """The driven path satisfies every check it was selected by."""
    c, k = pl.cands, pl.chosen
    assert c.valid[k]
    assert c.clearance[k].min() >= params.collision.obstacle_margin


def test_no_obstacle_follows_the_raceline(wide, footprint, params):
    pl = run(wide, NONE, footprint, params)
    assert pl.mode == RACELINE and len(pl.cands.d_end) == 1
    assert np.allclose(pl.d, 0.0)


def test_returns_to_the_raceline_smoothly(wide, footprint, params):
    pl = run(wide, NONE, footprint, params, d=0.4)
    assert pl.mode == RACELINE
    assert pl.d[0] == pytest.approx(0.4)
    assert pl.d[-1] == pytest.approx(0.0)
    assert np.all(np.diff(pl.d) <= TOL)          # monotonic, no overshoot


def test_box_on_the_line_is_avoided(wide, footprint, params):
    pl = run(wide, box(4.0, 0.0), footprint, params)
    assert pl.mode == AVOID
    chosen_ok(pl, params)
    assert abs(pl.d_end) > BOX / 2 + footprint.radius


@pytest.mark.parametrize('d_box,side', [(0.3, -1), (-0.3, 1)])
def test_passes_on_the_side_with_room(wide, footprint, params, d_box, side):
    pl = run(wide, box(4.0, d_box), footprint, params)
    assert pl.mode == AVOID
    chosen_ok(pl, params)
    assert np.sign(pl.d_end) == side


def test_only_gap_is_taken_on_a_narrow_track(narrow, footprint, params):
    """Box just right of the line. Passing left would be the smaller detour,
    but the 0.5 m to the left wall is too little; the 0.8 m right is enough."""
    pl = run(narrow, box(4.0, -0.05), footprint, params)
    assert pl.mode == AVOID and pl.d_end < 0
    chosen_ok(pl, params)


@pytest.mark.parametrize('prev', [0.6, -0.6])
def test_hysteresis_keeps_the_committed_side(wide, footprint, params, prev):
    """Box dead centre on a symmetric track: either side works, and the
    planner sticks with the side it chose last cycle."""
    pl = run(wide, box(4.0, 0.0), footprint, params, prev_d_end=prev)
    assert np.sign(pl.d_end) == np.sign(prev)


def test_blocked_track_falls_back_and_stops_short(narrow, footprint, params):
    wall = ObstacleSet.from_boxes([(3.0, BOX, -0.8, 0.5, 0.0, 0.0)])
    pl = run(narrow, wall, footprint, params)
    assert pl.mode == FALLBACK
    assert pl.d_end == 0.0
    assert pl.v[0] <= params.planner.v_fallback + TOL
    p = params.planner
    first_hit = pl.u[np.flatnonzero(pl.cands.clearance[pl.chosen] < params.collision.obstacle_margin)[0]]
    assert np.all(pl.v[pl.u >= first_hit - p.stop_margin] == 0.0)
    # Braking to that stop starts early enough
    dv2 = np.diff(pl.v ** 2) / np.diff(pl.u)
    assert np.all(dv2 >= -2 * params.speed.a_decel_max - 1e-6)


def test_speed_respects_lateral_and_braking_limits(wide, footprint, params):
    sp = params.speed
    pl = run(wide, box(3.0, 0.0), footprint, params)
    kappa = pl.cands.kappa[pl.chosen]
    assert np.all(pl.v ** 2 * np.abs(kappa) <= sp.a_lat_max + 1e-6)
    assert np.all(pl.v <= params.planner.v_avoid + TOL)
    dv2 = np.diff(pl.v ** 2) / np.diff(pl.u)
    assert np.all(dv2 >= -2 * sp.a_decel_max - 1e-6)


def test_lateral_acceleration_caps_a_fast_raceline(circle_ref, footprint, params):  # noqa: F811
    """Raceline speed 6 m/s on a 2 m radius: sqrt(a_lat_max * r) is the cap."""
    fast = make_ref(circle_ref, 1.0, 3.0)
    fast.v_ref = np.full_like(fast.v_ref, 6.0)
    pl = run(fast, NONE, footprint, params)
    assert pl.v.max() == pytest.approx(np.sqrt(params.speed.a_lat_max * 2.0), rel=0.02)


def test_obstacle_across_the_finish_line(wide, footprint, params, circle_ref):  # noqa: F811
    L = circle_ref.length
    pl = run(wide, box(0.1, 0.0), footprint, params, s=L - 3.0)
    assert pl.mode == AVOID
    chosen_ok(pl, params)


def test_obstacle_behind_is_ignored(wide, footprint, params):
    behind = params.planner.obstacle_behind
    pl = run(wide, box(4.0 - behind - BOX - 0.2, 0.0), footprint, params, s=4.0)
    assert pl.mode == RACELINE


def in_band(pl, ref, params):
    p, v = params.planner, params.vehicle
    s = pl.s0 + pl.u
    inset = 0.5 * v.width + p.bound_margin
    on = (pl.u >= p.bounds_grace) & (pl.u <= pl.cands.horizon[pl.chosen] + TOL)
    return (np.all(pl.d[on] <= ref.bounds.left(s[on]) - inset + TOL)
            and np.all(pl.d[on] >= inset - ref.bounds.right(s[on]) - TOL))


def test_heading_for_the_wall_recovers_inside_the_bounds(narrow, footprint, params):
    """Car 0.2 m left of the line, angled 11 deg further left, 0.3 m from
    the wall. The slow return to the raceline would overshoot into the wall;
    the planner samples a sharper way back."""
    pl = run(narrow, NONE, footprint, params, d=0.2, d1=0.2)
    assert pl.mode == RECOVER
    assert in_band(pl, narrow, params)
    assert pl.v[0] > params.planner.v_avoid      # no obstacle: no avoidance cap


def test_braking_for_a_stop_starts_early_enough(circle_ref, footprint, params):  # noqa: F811
    """Blocked track at 3 m/s: the speed profile brakes at a_decel_max into
    the stop, rather than stepping down to 0 at the stop point."""
    fast = make_ref(circle_ref, 1.0, 3.0)
    fast.v_ref = np.full_like(fast.v_ref, 6.0)
    quick = dataclasses.replace(params, planner=dataclasses.replace(params.planner, v_fallback=3.0))
    wall = ObstacleSet.from_boxes([(4.5, BOX, -1.0, 1.0, 0.0, 0.0)])
    pl = run(fast, wall, footprint, quick)
    assert pl.mode == FALLBACK
    assert pl.v.max() > 2.5
    dv2 = np.diff(pl.v ** 2) / np.diff(pl.u)
    assert np.all(dv2 >= -2 * params.speed.a_decel_max - 1e-6)


def test_chosen_path_stays_in_bounds(narrow, footprint, params):
    pl = run(narrow, box(3.0, -0.3), footprint, params)
    assert in_band(pl, narrow, params)


def test_track_narrowing_past_the_horizon_allows_holding_a_line(wide, footprint, params, circle_ref):  # noqa: F811
    """Car 0.6 m right of the line beside a long obstacle; the track's right
    side narrows only in the last half metre of the path, past every
    horizon. Holding the line is valid: that stretch is re-planned long
    before the car gets there."""
    b = wide.bounds
    s0 = 1.0
    d_right = np.where((b.s > s0 + params.planner.horizon_max + 0.3) & (b.s < s0 + 6.0), 0.5, b.d_right)
    narrowing = Reference(path=wide.path, grid=wide.grid, v_ref=wide.v_ref,
                          bounds=TrackBounds(s=b.s, d_left=b.d_left, d_right=d_right, length=b.length))
    long_box = ObstacleSet.from_boxes([(s0 + 0.5, 3.0, -0.15, 0.3, 0.0, 0.0)])
    pl = run(narrowing, long_box, footprint, params, s=s0, d=-0.6, prev_d_end=-0.6)
    assert pl.mode == AVOID
    assert pl.d_end < -0.5


def test_slower_car_ahead_is_timed_not_treated_as_parked(wide, footprint, params):
    """A car 2.5 m ahead doing our speed never gets closer, so the line
    stays valid; the same box parked forces a lane change."""
    moving = run(wide, box(3.5, 0.0, v_s=V_PLAN), footprint, params)
    parked = run(wide, box(3.5, 0.0), footprint, params)
    assert moving.mode == AVOID and abs(moving.d_end) < 0.05
    assert abs(parked.d_end) > 0.3
