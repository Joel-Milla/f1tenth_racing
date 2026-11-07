"""Obstacle detection on synthetic scans with known answers.

Scene: the ring track of test_track_bounds (walls at r = 1.5 and 2.8, CCW
raceline circle r = 2, so d_left = 0.5 and d_right = 0.8, s = 2 * angle).
Scans are ray-cast exactly against the analytic walls and box polygons.
"""
import dataclasses

import numpy as np
import pytest

from conftest import CONFIG_YAML
from frenet_planner.detection import Box, Detector, breakpoint_clusters, merge_boxes, scan_to_map
from frenet_planner.params import Params
from frenet_planner.track_bounds import TrackBounds, compute_track_bounds
from test_track_bounds import circle_ref, ring_grid  # noqa: F401 (fixture)

R_LINE, R_IN, R_OUT = 2.0, 1.5, 2.8
N_BEAMS, FOV = 1080, 4.7
ANGLE_MIN = -FOV / 2
ANGLE_INC = FOV / (N_BEAMS - 1)
LASER_OFFSET = 0.275
BOX = 0.30
TOL = 0.05   # map cells are 5 cm
MAX_BEAM_GAP = 2


@pytest.fixture(scope='module')
def params():
    p = Params.from_yaml(CONFIG_YAML).detection
    return dataclasses.replace(p, laser_offset_x=LASER_OFFSET)


@pytest.fixture(scope='module')
def detector(circle_ref, params):  # noqa: F811
    grid = ring_grid(R_IN, R_OUT)
    bounds = compute_track_bounds(circle_ref, grid, 0.05, 3.0, 0.0)
    return Detector(circle_ref, bounds, grid, params)


def car_pose(s):
    """Car on the raceline at arc length s, heading along it (CCW)."""
    th = s / R_LINE
    return R_LINE * np.cos(th), R_LINE * np.sin(th), th + np.pi / 2


def box_polygon(s, d, length=BOX, width=BOX):
    """Box centred on (s, d), aligned with the raceline there. d > 0 is
    towards the ring centre."""
    th = s / R_LINE
    c = (R_LINE - d) * np.array([np.cos(th), np.sin(th)])
    t = np.array([-np.sin(th), np.cos(th)])   # along the track
    n = -np.array([np.cos(th), np.sin(th)])   # left, towards the centre
    hl, hw = 0.5 * length, 0.5 * width
    return np.array([c - hl * t - hw * n, c + hl * t - hw * n, c + hl * t + hw * n, c - hl * t + hw * n])


def ray_cast(ox, oy, angles, polygons, max_range=30.0):
    """Exact ranges from (ox, oy) to the two ring walls and the polygons."""
    ux, uy = np.cos(angles), np.sin(angles)
    r = np.full(len(angles), max_range)
    b = ox * ux + oy * uy
    for radius in (R_IN, R_OUT):
        disc = b * b - (ox * ox + oy * oy - radius * radius)
        sq = np.sqrt(np.maximum(disc, 0.0))
        for t in (-b - sq, -b + sq):
            hit = (disc >= 0) & (t > 1e-9)
            r = np.where(hit, np.minimum(r, t), r)
    for poly in polygons:
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            ex, ey = q - p
            den = ux * ey - uy * ex
            with np.errstate(divide='ignore', invalid='ignore'):
                t = ((p[0] - ox) * ey - (p[1] - oy) * ex) / den
                v = ((p[0] - ox) * uy - (p[1] - oy) * ux) / den
            hit = (np.abs(den) > 1e-12) & (t > 1e-9) & (v >= 0) & (v <= 1)
            r = np.where(hit, np.minimum(r, t), r)
    return r


def scan(pose, polygons):
    x, y, yaw = pose
    lx, ly = x + LASER_OFFSET * np.cos(yaw), y + LASER_OFFSET * np.sin(yaw)
    return ray_cast(lx, ly, yaw + ANGLE_MIN + ANGLE_INC * np.arange(N_BEAMS), polygons)


def detect(detector, s_car, polygons, ranges=None):
    pose = car_pose(s_car)
    if ranges is None:
        ranges = scan(pose, polygons)
    return detector.detect(ranges, ANGLE_MIN, ANGLE_INC, 0.0, pose, s_car)


# --- pieces ------------------------------------------------------------------

def test_scan_to_map_transform():
    # Car at (1, 2) facing +y; beams straight ahead and to the left
    ranges = np.array([2.0, np.inf, 1.0, 0.01, 50.0])
    beam, r, x, y = scan_to_map(ranges, 0.0, np.pi / 4, 0.05, 30.0, (1.0, 2.0, np.pi / 2), 0.275)
    assert list(beam) == [0, 2]                  # inf, too short, too long dropped
    assert x == pytest.approx([1.0, 0.0])        # beam 2 points 90 deg left = -x
    assert y == pytest.approx([2.0 + 0.275 + 2.0, 2.0 + 0.275])


def test_breakpoint_keeps_a_surface_and_splits_two():
    # A straight wall 1 m away, seen across 40 beams, then a second object
    # 0.5 m further back. One break, between them.
    a = ANGLE_INC * np.arange(60)
    r = np.r_[1.0 / np.cos(a[:40] - a[20]), np.full(20, 1.5)]
    beam = np.arange(60)
    x, y = r * np.cos(a), r * np.sin(a)
    clusters = breakpoint_clusters(beam, r, x, y, ANGLE_INC, 0.17, 0.01, MAX_BEAM_GAP)
    assert [len(c) for c in clusters] == [40, 20]


def clusters_of(beams, r_of_beam):
    """Cluster sizes for the given kept beams of a surface r(beam)."""
    beams = np.asarray(beams)
    a = ANGLE_INC * beams
    r = r_of_beam(beams)
    return [len(c) for c in breakpoint_clusters(beams, r, r * np.cos(a), r * np.sin(a),
                                                ANGLE_INC, 0.17, 0.01, MAX_BEAM_GAP)]


def test_breakpoint_bridges_a_few_lost_returns():
    def arc(b):
        return np.ones(len(b))
    assert clusters_of(np.r_[0:50, 51:100], arc) == [99]        # 1 lost
    assert clusters_of(np.r_[0:50, 52:100], arc) == [98]        # 2 lost
    assert clusters_of(np.r_[0:50, 53:100], arc) == [50, 47]    # 3 lost: split


def test_breakpoint_splits_across_dropped_beams_even_when_close():
    """Two objects 1 m away with 10 filtered-out beams between them are only
    ~5 cm apart, under D_max for neighbouring beams, but still separate."""
    def arc(b):
        return np.ones(len(b))
    assert clusters_of(np.r_[0:20, 30:50], arc) == [20, 20]


def test_breakpoint_does_not_merge_far_objects_over_a_long_dropped_run():
    """Surface 4 m away, 20 beams filtered out between two parts 0.37 m
    apart. Measuring D_max over the whole run would allow ~4.6 m and merge
    them."""
    def far(b):
        return np.full(len(b), 4.0)
    assert clusters_of(np.r_[0:30, 50:80], far) == [30, 30]


# --- whole scans ---------------------------------------------------------------

def test_empty_track_has_no_detections(detector):
    assert detect(detector, 0.0, []).detections == []


@pytest.mark.parametrize('d_box', [0.0, 0.15, -0.4])
def test_box_ahead_is_found_where_it_is(detector, d_box):
    s_box = 2.0
    res = detect(detector, 0.0, [box_polygon(s_box, d_box)])
    assert len(res.detections) == 1
    det = res.detections[0]
    # Rear face seen; box extended forward to at least min_s_extent
    assert det.s_min == pytest.approx(s_box - BOX / 2, abs=TOL)
    assert det.s_len >= detector.p.min_s_extent - 1e-9
    assert det.d_min == pytest.approx(d_box - BOX / 2, abs=TOL)
    assert det.d_max == pytest.approx(d_box + BOX / 2, abs=TOL)


def test_two_boxes_two_detections(detector):
    # Near box on the outside of the curve, far one on the inside, so the
    # near one does not hide the far one
    res = detect(detector, 0.0, [box_polygon(1.5, -0.4), box_polygon(2.5, 0.15)])
    s_min = sorted(d.s_min for d in res.detections)
    assert s_min == pytest.approx([1.5 - BOX / 2, 2.5 - BOX / 2], abs=TOL)


def test_box_near_a_wall_is_clipped_by_the_wall_margin(detector):
    """Box at d = 0.10..0.40 with the left wall at d = 0.5: its points within
    wall_margin of the wall are taken for wall, so only the rest is seen."""
    res = detect(detector, 0.0, [box_polygon(2.0, 0.25)])
    assert len(res.detections) == 1
    det = res.detections[0]
    assert det.d_min == pytest.approx(0.10, abs=TOL)
    assert det.d_max <= R_LINE - R_IN - detector.p.wall_margin + TOL


def test_box_across_the_finish_line(detector, circle_ref):  # noqa: F811
    L = circle_ref.length
    res = detect(detector, L - 1.5, [box_polygon(0.0, 0.0)])
    assert len(res.detections) == 1
    det = res.detections[0]
    assert circle_ref.s_diff(det.s_min, L - BOX / 2) == pytest.approx(0.0, abs=TOL)
    assert det.s_len < 1.0


def test_oversized_and_single_point_clusters_dropped(detector):
    plank = box_polygon(2.0, 0.0, length=1.2, width=0.1)  # 1.2 m along the track
    assert detect(detector, 0.0, [plank]).detections == []
    pose = car_pose(0.0)
    ranges = scan(pose, [])
    ranges[N_BEAMS // 2] = 1.0                          # one stray return on the raceline
    assert detect(detector, 0.0, [], ranges=ranges).detections == []


def test_points_outside_the_bounds_dropped(circle_ref, params):  # noqa: F811
    """Bounds narrower than the walls: a box between bound and wall is
    ignored, the same box with the true bounds is found."""
    grid = ring_grid(R_IN, R_OUT)
    s = np.arange(0.0, circle_ref.length, 0.05)
    narrow = TrackBounds(s=s, d_left=np.full_like(s, 0.1), d_right=np.full_like(s, 0.8), length=circle_ref.length)
    box = [box_polygon(2.0, 0.25, width=0.1)]           # spans d = 0.2..0.3
    assert detect(Detector(circle_ref, narrow, grid, params), 0.0, box).detections == []
    true_bounds = compute_track_bounds(circle_ref, grid, 0.05, 3.0, 0.0)
    assert len(detect(Detector(circle_ref, true_bounds, grid, params), 0.0, box).detections) == 1


# --- merging and box extension ----------------------------------------------

def box(s_lo, s_hi, d_lo, d_hi, n=10):
    return Box(s_lo, s_hi, d_lo, d_hi, 0.0, 0.0, n)


def merged(boxes, params):
    return merge_boxes(boxes, params.merge_gap, params.max_cluster_size, params.min_s_extent)


def test_side_face_sliver_merges_into_its_box(params):
    """Rear face seen flat at s = 2.0, side face starting 0.15 m further on:
    beyond merge_gap as seen, inside it once the rear face is extended."""
    out = merged([box(2.0, 2.01, 0.10, 0.40), box(2.15, 2.28, 0.09, 0.10)], params)
    assert len(out) == 1
    assert (out[0].s_lo, out[0].s_hi, out[0].d_lo, out[0].d_hi, out[0].n) == (2.0, 2.28, 0.09, 0.40, 20)


def test_object_behind_another_stays_separate(params):
    out = merged([box(2.0, 2.01, -0.15, 0.15), box(2.5, 2.51, -0.15, 0.15)], params)
    assert len(out) == 2


def test_two_cars_side_by_side_stay_separate(params):
    """Within merge_gap in d, but together 0.65 m wide: more than max_cluster_size."""
    out = merged([box(2.0, 2.01, 0.05, 0.35), box(2.0, 2.01, -0.30, 0.00)], params)
    assert len(out) == 2


def test_box_behind_the_car_extends_backwards(detector):
    """Once passed, the LiDAR sees only the box's front face. Extending that
    backwards keeps the rear edge where it was, so a tracked static box
    does not appear to jump forward (and gain a velocity)."""
    s_box = 2.0
    ahead = detect(detector, 0.0, [box_polygon(s_box, 0.0)]).detections
    # 0.7 m past the box's front face. Turned round to face it, so the scan
    # sees just that face (facing forward it would be in the rear blind cone)
    s_car = s_box + BOX / 2 + 0.7
    x, y, yaw = car_pose(s_car)
    pose = (x, y, yaw + np.pi)
    behind = detector.detect(scan(pose, [box_polygon(s_box, 0.0)]), ANGLE_MIN, ANGLE_INC, 0.0,
                             pose, s_car).detections
    assert len(ahead) == 1 and len(behind) == 1
    for det in (ahead[0], behind[0]):
        assert det.s_min == pytest.approx(s_box - BOX / 2, abs=TOL)
