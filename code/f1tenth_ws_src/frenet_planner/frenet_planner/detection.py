"""Obstacle detection from one LiDAR scan.

Pipeline: beams -> map-frame points -> drop points near static walls ->
Frenet (s, d) -> drop points outside the track -> adaptive breakpoint
clustering -> size filter -> one Frenet box per cluster.

A box only covers the surface the LiDAR saw: for an obstacle ahead its rear
face, for one behind its front face, both almost flat in s. Boxes are
therefore given a minimum length, extended away from the car (forward for
obstacles ahead, backward for ones entirely behind the car), so the tracked
rear edge does not jump when the visible face changes.
"""
from dataclasses import dataclass, field

import numpy as np

from frenet_planner.frenet import s_diff

# Pairwise distances are computed only for clusters up to this many points;
# bigger ones are measured by their bounding-box diagonal, an upper bound.
MAX_PAIRWISE_POINTS = 400


@dataclass
class Detection:
    """One cluster as a Frenet box: s in [s_min, s_min + s_len] (wrapped),
    d in [d_min, d_max]. x, y is the centroid of its points (for markers)."""
    s_min: float
    s_len: float
    d_min: float
    d_max: float
    x: float
    y: float
    n_points: int


@dataclass
class DetectionResult:
    """x, y: points that survived the wall and track filters. clusters:
    index arrays into them, before the size filters. detections: clusters
    that passed the size filters."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    clusters: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    # Of all valid points within max_range (before any filter): how many,
    # and the fraction farther than mismatch_dist from every map wall
    n_points: int = 0
    mismatch: float = 0.0


def scan_to_map(ranges, angle_min, angle_increment, range_min, max_range, pose, laser_offset):
    """Valid beams of a scan as map-frame points.

    pose:         (x, y, yaw) of the car in the map frame
    laser_offset: laser position ahead of the pose point along the heading
    Beams that are non-finite, shorter than range_min or longer than
    max_range are dropped. Returns beam index, range, x, y of the kept beams.
    """
    r = np.asarray(ranges, dtype=float)
    beam = np.arange(len(r))
    with np.errstate(invalid='ignore'):
        ok = np.isfinite(r) & (r >= range_min) & (r <= max_range)
    beam, r = beam[ok], r[ok]
    a = angle_min + beam * angle_increment
    lx, ly = laser_offset + r * np.cos(a), r * np.sin(a)
    px, py, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return beam, r, px + c * lx - s * ly, py + s * lx + c * ly


def breakpoint_clusters(beam, r, x, y, angle_increment, lam, sigma_r, max_beam_gap):
    """Adaptive breakpoint detection (Borges & Aldon 2004).

    Consecutive kept points split when their distance exceeds
        D_max = r_prev * sin(dphi) / sin(lam - dphi) + 3 sigma_r,
    the gap a surface at grazing angle lam would leave between beams dphi
    apart. They also split when more than max_beam_gap consecutive beams
    between them were dropped: a run of filtered-out wall or off-track points separates
    two objects even if they are close, while one or two lost returns inside
    one object do not. (Letting dphi grow over long dropped runs instead
    makes D_max blow up as dphi nears lam and merges distant objects.)
    Returns a list of index arrays, in beam order.
    """
    n = len(beam)
    if n == 0:
        return []
    dbeam = np.diff(beam)
    dphi = dbeam * angle_increment
    with np.errstate(divide='ignore', invalid='ignore'):
        d_max = r[:-1] * np.sin(dphi) / np.sin(lam - dphi) + 3.0 * sigma_r
    gap = np.hypot(np.diff(x), np.diff(y))
    split = (dbeam - 1 > max_beam_gap) | (dphi >= lam) | (gap > d_max)
    return np.split(np.arange(n), np.flatnonzero(split) + 1)


def cluster_diameter(x, y):
    """Largest distance between two points of a cluster [m]."""
    if len(x) > MAX_PAIRWISE_POINTS:
        return float(np.hypot(np.ptp(x), np.ptp(y)))
    return float(np.sqrt(((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2).max()))


@dataclass
class Box:
    """A cluster's observed extent, s relative to the car, plus what is
    needed to merge centroids."""
    s_lo: float
    s_hi: float
    d_lo: float
    d_hi: float
    sum_x: float
    sum_y: float
    n: int


def extended_s(b, min_s_extent):
    """A box's s interval once lengthened to min_s_extent away from the car
    (s is relative to the car: forward if any of it is ahead, else back)."""
    if b.s_hi < 0.0:
        return min(b.s_lo, b.s_hi - min_s_extent), b.s_hi
    return b.s_lo, max(b.s_hi, b.s_lo + min_s_extent)


def merge_boxes(boxes, gap, max_size, min_s_extent):
    """Merge boxes within gap of each other in d and, once extended to
    min_s_extent, in s, as long as what was actually seen of the merged box
    stays within max_size across (hypot of its s and d extents). Repairs an
    object split into several clusters, e.g. a box's side face seen at a
    grazing angle, without fusing two nearby objects into one too big to be
    either."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                (a_lo, a_hi), (b_lo, b_hi) = extended_s(a, min_s_extent), extended_s(b, min_s_extent)
                if (max(a_lo, b_lo) - min(a_hi, b_hi) > gap
                        or max(a.d_lo, b.d_lo) - min(a.d_hi, b.d_hi) > gap):
                    continue
                m = Box(min(a.s_lo, b.s_lo), max(a.s_hi, b.s_hi), min(a.d_lo, b.d_lo), max(a.d_hi, b.d_hi),
                        a.sum_x + b.sum_x, a.sum_y + b.sum_y, a.n + b.n)
                if np.hypot(m.s_hi - m.s_lo, m.d_hi - m.d_lo) > max_size:
                    continue
                boxes[i] = m
                del boxes[j]
                merged = True
                break
            if merged:
                break
    return boxes


class Detector:
    """Holds what one scan's detection needs from the static world: the
    reference path, track bounds and the map's wall-distance field."""

    def __init__(self, path, bounds, grid, params):
        """path: ReferencePath, bounds: TrackBounds, grid: OccupancyGrid,
        params: DetectionParams."""
        self.path = path
        self.bounds = bounds
        self.grid = grid
        self.p = params
        self.wall_dist = grid.wall_distance()

    def detect(self, ranges, angle_min, angle_increment, range_min, pose, s_hint, mismatch_dist=None):
        """pose: (x, y, yaw) of the car; s_hint: the car's s, which seeds
        the windowed Frenet projection of the scan points. mismatch_dist:
        if given, also report the scan-to-map mismatch fraction."""
        p, path = self.p, self.path
        beam, r, x, y = scan_to_map(ranges, angle_min, angle_increment, range_min,
                                    p.max_range, pose, p.laser_offset_x)
        wall = self.grid.sample(self.wall_dist, x, y, 0.0)
        n_all = len(x)
        mismatch = float(np.mean(wall > mismatch_dist)) if mismatch_dist is not None and n_all else 0.0
        keep = wall >= p.wall_margin
        beam, r, x, y = beam[keep], r[keep], x[keep], y[keep]
        if len(x) == 0:
            return DetectionResult(n_points=n_all, mismatch=mismatch)

        s, d = path.cartesian_to_frenet(x, y, s_hint=s_hint,
                                        window=(p.frenet_window_back, p.frenet_window_fwd))
        tol = p.track_bound_tolerance
        inside = (d <= self.bounds.left(s) + tol) & (d >= -self.bounds.right(s) - tol)
        beam, r, x, y, s, d = beam[inside], r[inside], x[inside], y[inside], s[inside], d[inside]

        clusters = breakpoint_clusters(beam, r, x, y, angle_increment, p.breakpoint_lambda,
                                       p.range_sigma, p.max_beam_gap)
        # s relative to the car, so boxes near s = 0 compare without wrapping
        rel = s_diff(s, s_hint, path.length)
        boxes = [Box(rel[idx].min(), rel[idx].max(), d[idx].min(), d[idx].max(),
                     x[idx].sum(), y[idx].sum(), len(idx))
                 for idx in clusters if cluster_diameter(x[idx], y[idx]) <= p.max_cluster_size]
        detections = []
        for b in merge_boxes(boxes, p.merge_gap, p.max_cluster_size, p.min_s_extent):
            if b.n < p.min_cluster_points:
                continue
            s_lo, s_hi = extended_s(b, p.min_s_extent)
            detections.append(Detection(
                s_min=float(path.wrap(s_hint + s_lo)), s_len=float(s_hi - s_lo),
                d_min=float(b.d_lo), d_max=float(b.d_hi),
                x=b.sum_x / b.n, y=b.sum_y / b.n, n_points=b.n))
        return DetectionResult(x=x, y=y, clusters=clusters, detections=detections, n_points=n_all,
                               mismatch=mismatch)
