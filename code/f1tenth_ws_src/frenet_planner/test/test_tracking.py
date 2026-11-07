"""N-of-M persistence and the constant-velocity Kalman filter."""
import numpy as np
import pytest

from conftest import CONFIG_YAML
from frenet_planner.detection import Detection
from frenet_planner.frenet import s_diff
from frenet_planner.params import Params
from frenet_planner.tracking import ObstacleTracker

LOOP = 20.0
DT = 0.025          # 40 Hz scans
NOISE = 0.03        # detection noise, s and d [m]


@pytest.fixture
def tracker():
    return ObstacleTracker(LOOP, Params.from_yaml(CONFIG_YAML).tracking)


def det(s_min, d_c, width=0.3, s_len=0.3):
    return Detection(s_min=s_min % LOOP, s_len=s_len, d_min=d_c - width / 2, d_max=d_c + width / 2,
                     x=0.0, y=0.0, n_points=10)


def run(tracker, pattern, s=5.0, d=0.0):
    """Feed one obstacle seen (1) or not (0) per scan; return the number of
    confirmed tracks after each scan."""
    out = []
    for k, seen in enumerate(pattern):
        tracker.step([det(s, d)] if seen else [], k * DT)
        out.append(len(tracker.confirmed()))
    return out


def test_three_in_a_row_confirms_on_the_third(tracker):
    assert run(tracker, [1, 1, 1]) == [0, 0, 1]


def test_flicker_confirms_at_three_of_five(tracker):
    assert run(tracker, [1, 0, 1, 0, 1]) == [0, 0, 0, 0, 1]


def test_two_of_five_never_confirms_and_is_dropped(tracker):
    assert run(tracker, [1, 1, 0, 0, 0]) == [0, 0, 0, 0, 0]
    assert tracker.tracks == []


def test_confirmed_track_coasts_then_is_dropped(tracker):
    """Once confirmed, a track missed scan after scan stays reported,
    predicted forward, until coast_time after its last detection."""
    coast = tracker.p.coast_time
    run(tracker, [1] * 10)                      # detections at k = 0..9
    last = 9 * DT
    k = 10
    while (k * DT) - last <= coast - 1e-9:
        tracker.step([], k * DT)
        assert len(tracker.confirmed()) == 1
        k += 1
    tracker.step([], k * DT)
    assert tracker.tracks == []


def test_coasting_track_keeps_moving(tracker):
    simulate(tracker, 3.0, 1.5, 0.0, 0.0, 1.0)
    tr = tracker.confirmed()[0]
    s_last, t_last = tr.x[0], tr.t
    tracker.step([], t_last + 0.2)
    assert tracker.confirmed()[0].x[0] == pytest.approx(s_last + 1.5 * 0.2, abs=0.05)


def test_scattered_noise_never_confirms(tracker):
    rng = np.random.default_rng(1)
    for k in range(200):
        tracker.step([det(rng.uniform(0, LOOP), rng.uniform(-0.8, 0.8))], k * DT)
        assert tracker.confirmed() == []


def simulate(tracker, s0, v_s, d0, v_d, seconds, seed=0):
    rng = np.random.default_rng(seed)
    ids = set()
    for k in range(int(seconds / DT)):
        t = k * DT
        tracker.step([det(s0 + v_s * t + rng.normal(0, NOISE), d0 + v_d * t + rng.normal(0, NOISE))], t)
        ids |= {tr.id for tr in tracker.confirmed()}
    return ids


def test_moving_opponent_velocity(tracker):
    ids = simulate(tracker, 3.0, 1.5, 0.2, 0.0, 1.5)
    assert len(ids) == 1
    tr = tracker.confirmed()[0]
    assert tr.x[2] == pytest.approx(1.5, abs=0.15)
    assert tr.x[3] == pytest.approx(0.0, abs=0.15)
    obs = tracker.obstacle_set()
    assert obs.v_s[0] == pytest.approx(tr.x[2])
    assert obs.d_max[0] - obs.d_min[0] == pytest.approx(0.3)


def test_static_box_reported_still(tracker):
    simulate(tracker, 8.0, 0.0, -0.3, 0.0, 1.5)
    obs = tracker.obstacle_set()
    assert len(obs) == 1
    assert obs.v_s[0] == 0.0 and obs.v_d[0] == 0.0
    assert obs.s_min[0] == pytest.approx(8.0, abs=0.03)


def test_track_crosses_the_finish_line(tracker):
    # 2 m/s from s = L - 1.5 through s = 0, one track the whole way
    ids = simulate(tracker, LOOP - 1.5, 2.0, 0.0, 0.0, 1.5)
    assert len(ids) == 1
    tr = tracker.confirmed()[0]
    assert 0.0 <= tr.x[0] < LOOP
    assert tr.x[0] == pytest.approx((LOOP - 1.5 + 2.0 * (1.5 - DT)) % LOOP, abs=0.1)
    assert tr.x[2] == pytest.approx(2.0, abs=0.15)


def test_box_on_the_finish_line_stays_put(tracker):
    """Static box right on s = 0: noise puts detections alternately just
    before and just after the line. The update must use the short way
    round, not a ~L jump."""
    for k in range(40):
        s = -0.02 if k % 2 else 0.02
        tracker.step([det(s, 0.0)], k * DT)
        for tr in tracker.tracks:
            assert abs(s_diff(tr.x[0], 0.0, LOOP)) < 0.05
            assert abs(tr.x[2]) < 1.0
    assert len(tracker.confirmed()) == 1


def test_velocity_follows_a_braking_opponent(tracker):
    """2 m/s for 1 s, then brakes at 3 m/s^2 to 0.5 m/s and holds. Within
    0.5 s of reaching 0.5 m/s the estimate must have followed; a filter
    without process noise grows overconfident and lags far behind."""
    rng = np.random.default_rng(3)
    s, v, t_slow = 3.0, 2.0, None
    for k in range(int(3.0 / DT)):
        t = k * DT
        tracker.step([det(s + rng.normal(0, NOISE), rng.normal(0, NOISE))], t)
        if t >= 1.0:
            v = max(0.5, v - 3.0 * DT)
            if v == 0.5 and t_slow is None:
                t_slow = t
        s += v * DT
        if t_slow is not None and t > t_slow + 0.5:
            assert tracker.confirmed()[0].x[2] == pytest.approx(0.5, abs=0.25)


def test_two_cars_side_by_side_keep_their_ids(tracker):
    rng = np.random.default_rng(2)
    first = None
    for k in range(60):
        t = k * DT
        s = 4.0 + 1.0 * t
        tracker.step([det(s + rng.normal(0, NOISE), 0.35 + rng.normal(0, NOISE)),
                      det(s + 0.2 + rng.normal(0, NOISE), -0.35 + rng.normal(0, NOISE))], t)
        by_side = {tr.id: tr.x[1] > 0 for tr in tracker.confirmed()}
        if len(by_side) == 2:
            first = first or by_side
            assert by_side == first
    assert first is not None


def test_visible_face_change_does_not_make_a_parked_box_move(tracker):
    """Parked box at d = 0.10..0.40. Passing it, the LiDAR switches from its
    rear face to its near side face (d = 0.09..0.17): the centre jumps 0.12
    m sideways. It must stay static, with the d extent last seen, not a
    filtered centre that overshoots towards the car."""
    for k in range(20):
        tracker.step([Detection(5.0, 0.3, 0.10, 0.40, 0.0, 0.0, 10)], k * DT)
    for k in range(20, 26):
        tracker.step([Detection(5.0, 0.3, 0.09, 0.17, 0.0, 0.0, 10)], k * DT)
        obs = tracker.obstacle_set()
        assert obs.v_s[0] == 0.0 and obs.v_d[0] == 0.0
        assert obs.d_min[0] == pytest.approx(0.09) and obs.d_max[0] == pytest.approx(0.17)


def test_passing_a_moving_car_reports_no_lateral_velocity(tracker):
    """Opponent at 1.2 m/s on the line. As we pull alongside, the LiDAR
    switches from its rear face (d -0.15..0.15) to its near side face
    (d -0.15 only): the measured centre jumps towards us. The reported box
    must move along s only, or it would be projected into our lane."""
    for k in range(40):
        t = k * DT
        near_side = k >= 25
        d_min, d_max = (-0.15, -0.14) if near_side else (-0.15, 0.15)
        tracker.step([Detection((3.0 + 1.2 * t) % LOOP, 0.58, d_min, d_max, 0.0, 0.0, 10)], t)
    obs = tracker.obstacle_set()
    assert obs.v_s[0] == pytest.approx(1.2, abs=0.2)
    assert obs.v_d[0] == 0.0
    assert obs.d_min[0] == pytest.approx(-0.15)


def test_brief_track_does_not_coast(tracker):
    """Confirmed after three detections, then gone: too short a history to
    coast, so it is dropped by N-of-M like any phantom."""
    assert tracker.p.coast_min_hits > 3
    counts = run(tracker, [1, 1, 1, 0, 0, 0])
    assert counts == [0, 0, 1, 1, 1, 0]
    assert tracker.tracks == []
