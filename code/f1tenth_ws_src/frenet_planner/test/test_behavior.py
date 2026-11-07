"""Behaviour state machine on the ring tracks of test_planner (raceline speed
V_REF = 2 m/s everywhere)."""
import dataclasses

import numpy as np
import pytest

from conftest import CONFIG_YAML
from frenet_planner.behavior import LEFT, OVERTAKE, RACELINE, RIGHT, TRAIL, Behavior, Commitment
from frenet_planner.collision import ObstacleSet, car_footprint
from frenet_planner.params import Params
from frenet_planner.planner import AVOID, FrenetState
from test_planner import V_REF, make_ref
from test_track_bounds import circle_ref  # noqa: F401 (fixture)

CAR = 0.58
OPP_W = 0.31


@pytest.fixture(scope='module')
def params():
    return Params.from_yaml(CONFIG_YAML)


@pytest.fixture(scope='module')
def footprint(params):
    v = params.vehicle
    return car_footprint(v.length, v.width, params.collision.n_circles, 0.0)


@pytest.fixture(scope='module')
def wide(circle_ref):  # noqa: F811
    return make_ref(circle_ref, 1.0, 3.0)


@pytest.fixture(scope='module')
def narrow(circle_ref):  # noqa: F811
    """0.4 m left and 0.6 m right of the line: no lane beside a car on it."""
    return make_ref(circle_ref, 1.6, 2.6)


def car(rear, v, d=0.0):
    """A car-sized moving obstacle with its rear edge at s = rear."""
    return (rear, CAR, d - OPP_W / 2, d + OPP_W / 2, v, 0.0)


def step(b, ref, boxes, footprint, s=1.0, d=0.0, v_car=V_REF, prev=0.0):
    return b.step(FrenetState(s, d, 0.0, 0.0), prev, v_car, ref, ObstacleSet.from_boxes(boxes), footprint)


def new(params, ref):
    return Behavior(params, ref.path.length)


def test_nothing_around_is_raceline(params, wide, footprint):
    b = new(params, wide)
    step(b, wide, [], footprint)
    assert b.state == RACELINE


def test_static_box_commits_a_side_until_it_is_behind(params, wide, footprint):
    """Box dead centre on a symmetric track: the first choice (left, pulled
    by last cycle's offset) is kept even when last cycle's offset now pulls
    right; released once the box is clear_dist behind the car's rear."""
    b = new(params, wide)
    box = (3.85, 0.3, -0.15, 0.15, 0.0, 0.0)
    pl = step(b, wide, [box], footprint, prev=0.6)
    assert b.state == AVOID and b.commit.side == LEFT and pl.d_end > 0
    pl = step(b, wide, [box], footprint, s=1.5, d=0.2, prev=-0.6)
    assert pl.d_end > 0
    s_past = 4.15 + CAR / 2 + params.behavior.clear_dist + 0.1
    step(b, wide, [box], footprint, s=s_past, d=0.5, prev=0.5)
    assert b.commit is None and b.state == RACELINE


def test_slower_car_without_room_is_trailed(params, narrow, footprint):
    """No lane beside a car on the line: trail it, at its speed plus
    trail_gain x (gap - trail_gap)."""
    b = new(params, narrow)
    p = params.behavior
    gap = 1.8
    step(b, narrow, [car(1.0 + CAR / 2 + gap, 1.0)], footprint)
    assert b.state == TRAIL
    assert b.v_trail == pytest.approx(1.0 + p.trail_gain * (gap - p.trail_gap))


def test_trail_stops_only_for_a_car_close_ahead(params, narrow, footprint):
    b = new(params, narrow)
    step(b, narrow, [car(1.0 + CAR / 2 + 0.3, 1.0)], footprint)
    assert b.state == TRAIL and b.v_trail == 0.0


def test_slower_car_with_room_is_overtaken(params, wide, footprint):
    b = new(params, wide)
    pl = step(b, wide, [car(1.0 + CAR / 2 + 1.5, 1.0)], footprint, v_car=1.0)
    assert b.state == OVERTAKE
    assert b.commit.side in (LEFT, RIGHT)
    assert np.sign(pl.d_end) == b.commit.side
    assert abs(pl.d_end) >= params.behavior.pass_offset


def test_no_overtake_without_a_speed_advantage(params, wide, footprint):
    b = new(params, wide)
    v_opp = V_REF - params.behavior.overtake_min_advantage + 0.1
    step(b, wide, [car(1.0 + CAR / 2 + 1.5, v_opp)], footprint, v_car=v_opp)
    assert b.state == TRAIL


def test_passed_car_is_not_trailed(params, wide, footprint):
    """Regression: a slower car whose front is just behind our rear must not
    be trailed (the speed law would brake us in front of it) or overtaken
    again; we drive on."""
    b = new(params, wide)
    s_car = 5.0
    rear = s_car - CAR / 2 - CAR - 0.1          # its front 0.1 m behind our rear
    pl = step(b, wide, [car(rear, 1.0)], footprint, s=s_car, d=-0.6, prev=-0.6)
    # Still within the planner's look-behind, so it may avoid it, but
    # neither trail nor try to overtake it, and keep moving faster than it
    assert b.state in (RACELINE, AVOID)
    assert pl.v[0] > 1.0


def test_finished_pass_holds_its_side_until_clear(params, wide, footprint):
    """Opponent just behind us but not yet clear_dist behind: nothing is in
    the way (the planner returns RACELINE), which is not an abort. The pass
    stays committed, with no cooldown."""
    b = new(params, wide)
    b.state = OVERTAKE
    b.commit = Commitment(RIGHT, 0.0, np.nan)
    s_car = 5.0
    behind = params.planner.obstacle_behind + 0.05     # out of the planner's view ...
    assert behind - CAR / 2 < params.behavior.clear_dist  # ... not yet clear
    rear = s_car - behind - CAR
    step(b, wide, [car(rear, 1.0)], footprint, s=s_car, d=-0.6, prev=-0.6)
    assert b.state == OVERTAKE and b.cooldown == 0


def test_pass_continues_between_keep_and_entry_margins(params, wide, footprint):
    """Hysteresis: a pass under way is kept while clearance is above
    overtake_keep_margin, even where it would not be started."""
    p = params.behavior
    entry_only = dataclasses.replace(params, behavior=dataclasses.replace(
        p, overtake_keep_margin=p.overtake_margin + 1.0))
    keep_ok = dataclasses.replace(params, behavior=dataclasses.replace(
        p, overtake_margin=p.overtake_margin + 1.0))
    opp = [car(1.0 + CAR / 2 + 1.5, 1.0)]
    for prm, expected in ((entry_only, TRAIL), (keep_ok, OVERTAKE)):
        b = new(prm, wide)
        b.state = OVERTAKE
        b.commit = Commitment(RIGHT, 0.0, np.nan)
        step(b, wide, opp, footprint, v_car=1.0, d=-0.3, prev=-0.6)
        assert b.state == expected


def test_no_new_attempt_during_cooldown(params, wide, footprint):
    b = new(params, wide)
    b.cooldown = 3
    opp = [car(1.0 + CAR / 2 + 1.5, 1.0)]
    for _ in range(2):
        step(b, wide, opp, footprint, v_car=1.0)
        assert b.state == TRAIL
    step(b, wide, opp, footprint, v_car=1.0)
    step(b, wide, opp, footprint, v_car=1.0)
    assert b.state == OVERTAKE
