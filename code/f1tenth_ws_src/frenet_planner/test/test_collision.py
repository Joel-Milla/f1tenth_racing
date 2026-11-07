"""Collision checks on hand-built cases with known answers."""
import numpy as np
import pytest

from frenet_planner.collision import ObstacleSet, car_footprint, circle_centers, path_clearance

# f110_gym car, pose at its centre
CAR_LENGTH, CAR_WIDTH = 0.58, 0.30
FP = car_footprint(CAR_LENGTH, CAR_WIDTH, 3, 0.0)
FRONT = FP.offsets[-1]
TRACK = 100.0                       # loop length for the straight-track cases [m]
U = np.arange(0.0, 5.0, 0.1)        # samples ahead of the car [m]
STRAIGHT = np.zeros_like(U)         # raceline curvature
TOL = 1e-9


def box(s_min, s_len, d_min, d_max, v_s=0.0, v_d=0.0):
    return ObstacleSet.from_boxes([(s_min, s_len, d_min, d_max, v_s, v_d)])


def straight_path(s0, d):
    """Path(s) parallel to a straight raceline at constant offsets d."""
    d = np.atleast_1d(np.asarray(d, dtype=float))[:, None] + 0.0 * U
    return s0 + U, d, np.zeros_like(d)


def test_footprint_covers_the_car():
    assert FP.offsets == pytest.approx([-CAR_LENGTH / 3, 0.0, CAR_LENGTH / 3])
    xs, ys = np.meshgrid(np.linspace(-CAR_LENGTH / 2, CAR_LENGTH / 2, 59),
                         np.linspace(-CAR_WIDTH / 2, CAR_WIDTH / 2, 31))
    dist = np.hypot(xs[..., None] - FP.offsets, ys[..., None]).min(axis=-1)
    assert dist.max() <= FP.radius + TOL


def test_footprint_center_offset_shifts_circles():
    rear_axle = car_footprint(CAR_LENGTH, CAR_WIDTH, 3, 0.165)
    assert rear_axle.offsets == pytest.approx(FP.offsets + 0.165)
    assert rear_axle.radius == pytest.approx(FP.radius)


def test_no_obstacles_is_infinitely_clear():
    s, d, d1 = straight_path(10.0, [0.0, 0.3])
    c = path_clearance(s, d, d1, STRAIGHT, 0.0, FP, ObstacleSet.from_boxes([]), TRACK)
    assert np.all(np.isinf(c))


def test_box_on_the_line_collides_and_side_paths_clear_it():
    obs = box(12.0, 0.3, -0.15, 0.15)          # car-wide box 2 m ahead
    s, d, d1 = straight_path(10.0, [0.0, 0.6, -0.6])
    c = path_clearance(s, d, d1, STRAIGHT, 0.0, FP, obs, TRACK)
    assert c[0] < 0.0
    # Parallel paths pass beside it: lateral gap minus circle radius
    assert c[1] == pytest.approx(0.6 - 0.15 - FP.radius)
    assert c[2] == pytest.approx(0.6 - 0.15 - FP.radius)


def test_box_beyond_the_horizon_is_clear_by_the_s_gap():
    obs = box(10.0 + U[-1] + 1.0, 0.3, -0.15, 0.15)
    s, d, d1 = straight_path(10.0, 0.0)
    c = path_clearance(s, d, d1, STRAIGHT, 0.0, FP, obs, TRACK)
    assert c[0] == pytest.approx(1.0 - FRONT - FP.radius)


def test_box_behind_the_car_is_clear():
    obs = box(8.0, 0.3, -0.15, 0.15)           # ends 1.7 m behind the pose
    s, d, d1 = straight_path(10.0, 0.0)
    c = path_clearance(s, d, d1, STRAIGHT, 0.0, FP, obs, TRACK)
    assert c[0] == pytest.approx(1.7 + FP.offsets[0] - FP.radius)


def test_box_straddling_the_finish_line():
    """Box from s = L - 0.1 to s = 0.2. Paths approaching from just before the
    line, or starting just after it, both hit it; one far away does not."""
    obs = box(TRACK - 0.1, 0.3, -0.15, 0.15)
    for s0 in (TRACK - 2.0, 0.05):
        s, d, d1 = straight_path(s0, 0.0)
        assert path_clearance(s, d, d1, STRAIGHT, 0.0, FP, obs, TRACK)[0] < 0.0
    s, d, d1 = straight_path(TRACK / 2, 0.0)
    assert path_clearance(s, d, d1, STRAIGHT, 0.0, FP, obs, TRACK)[0] > 10.0


def test_leading_car_at_same_speed_stays_clear():
    """Opponent 3 m ahead on the line at our speed: the static check says the
    path goes through it, the time-matched check says the gap holds."""
    v = 1.0
    s, d, d1 = straight_path(10.0, 0.0)
    t = U / v
    lead = box(13.0, 0.3, -0.06, 0.06, v_s=v)
    clear = path_clearance(s, d, d1, STRAIGHT, t, FP, lead, TRACK)[0]
    assert clear == pytest.approx(3.0 - FRONT - FP.radius)
    parked = box(13.0, 0.3, -0.06, 0.06)
    assert path_clearance(s, d, d1, STRAIGHT, t, FP, parked, TRACK)[0] < 0.0


def test_catching_a_slower_car_collides():
    s, d, d1 = straight_path(10.0, 0.0)
    t = U / 2.0                                   # we drive 2 m/s
    slow = box(11.5, 0.3, -0.06, 0.06, v_s=0.5)   # it drives 0.5 m/s
    assert path_clearance(s, d, d1, STRAIGHT, t, FP, slow, TRACK)[0] < 0.0


def test_car_drifting_into_our_lane():
    s, d, d1 = straight_path(10.0, 0.5)
    t = U / 1.0
    # Starts on the line 2 m ahead and slides left at 0.3 m/s: by the time we
    # reach it (t = 2 s) it is at d = 0.54..0.66, across our path at d = 0.5
    drifter = box(12.0, 0.3, -0.06, 0.06, v_d=0.3)
    assert path_clearance(s, d, d1, STRAIGHT, 0.0, FP, drifter, TRACK)[0] > 0.0
    assert path_clearance(s, d, d1, STRAIGHT, t, FP, drifter, TRACK)[0] < 0.0


def test_heading_rotates_the_footprint():
    """Path crossing the raceline at 45 deg: circles fan out diagonally."""
    s = np.array([5.0])
    c_s, c_d = circle_centers(s, np.array([[0.2]]), np.array([[1.0]]), np.array([0.0]), FP.offsets)
    k = np.sqrt(0.5)
    assert c_s[0, 0] == pytest.approx(5.0 + FP.offsets * k)
    assert c_d[0, 0] == pytest.approx(0.2 + FP.offsets * k)


def test_s_gap_shrinks_on_the_inside_of_a_corner():
    """On a left turn of radius 2 m (kappa = 0.5), 1 m of s at d = +1 is
    0.5 m of real distance; at d = -1 it is 1.5 m."""
    kappa = np.full_like(U, 0.5)
    for d_off, metric in ((1.0, 0.5), (-1.0, 1.5)):
        s, d, d1 = straight_path(10.0, d_off)
        obs = box(10.0 + U[-1] + 1.0, 0.3, d_off - 0.1, d_off + 0.1)
        c = path_clearance(s, d, d1, kappa, 0.0, FP, obs, TRACK)[0]
        # Front circle at the last sample is FRONT / metric of s ahead of it
        gap_s = (U[-1] + 1.0 - (U[-1] + FRONT / metric)) * metric
        assert c == pytest.approx(gap_s - FP.radius)
