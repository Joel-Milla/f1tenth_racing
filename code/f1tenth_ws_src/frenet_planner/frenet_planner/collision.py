"""Collision checking of candidate paths against obstacle boxes, all in the
Frenet frame.

The car is a row of circles along its centreline. An obstacle is an
axis-aligned box in (s, d) moving at constant (v_s, v_d). A path sample at
time t is checked against every box moved to where it will be at t, so a
static obstacle is just one with zero velocity.

Clearance is the distance from the edge of the nearest circle to the nearest
box edge: negative means overlap. Callers compare it with their safety margin,
which is the same as inflating every box by that margin.
"""
from dataclasses import dataclass, fields

import numpy as np

from frenet_planner.frenet import s_diff

# 1 - kappa * d converts an s difference into metres at lateral offset d. It
# reaches 0 at the raceline's centre of curvature, where the Frenet frame is
# singular. Paths that close to it are rejected by the curvature check; this
# floor only keeps the arithmetic finite until then.
METRIC_FLOOR = 0.1


@dataclass
class Footprint:
    """Circle centres along the car's heading, measured from the pose point,
    and the radius that makes them cover the car's rectangle."""
    offsets: np.ndarray
    radius: float


def car_footprint(length, width, n_circles, center_offset):
    """n_circles equal circles covering a length x width rectangle whose
    centre is center_offset ahead of the pose point (0 when the pose is the
    centre of the car, as in f1tenth_gym; ~wheelbase/2 for a rear-axle pose)."""
    half_seg = 0.5 * length / n_circles
    offsets = center_offset - 0.5 * length + half_seg * (2.0 * np.arange(n_circles) + 1.0)
    return Footprint(offsets=offsets, radius=float(np.hypot(half_seg, 0.5 * width)))


@dataclass
class ObstacleSet:
    """K obstacle boxes, struct-of-arrays. A box covers s in
    [s_min, s_min + s_len] (wrapped) and d in [d_min, d_max] at t = 0, and
    moves at (v_s, v_d)."""
    s_min: np.ndarray
    s_len: np.ndarray
    d_min: np.ndarray
    d_max: np.ndarray
    v_s: np.ndarray
    v_d: np.ndarray

    @classmethod
    def from_boxes(cls, boxes):
        """boxes: iterable of (s_min, s_len, d_min, d_max, v_s, v_d)."""
        a = np.asarray(list(boxes), dtype=float).reshape(-1, 6)
        return cls(*(a[:, k].copy() for k in range(6)))

    def __len__(self):
        return len(self.s_min)

    @classmethod
    def concat(cls, *sets):
        return cls(*(np.concatenate([getattr(o, f.name) for o in sets]) for f in fields(cls)))

    def subset(self, mask):
        return ObstacleSet(*(getattr(self, f.name)[mask] for f in fields(self)))


def circle_centers(s, d, d1, kappa_ref, offsets):
    """Frenet coordinates of the footprint circles along a path.

    s, kappa_ref: (m,) raceline arc length and curvature at each sample
    d, d1:        (..., m) lateral offset and dd/ds of each path
    offsets:      (c,) circle offsets along the car's heading
    Returns s_c, d_c of shape (..., m, c). s_c is not wrapped.
    """
    metric = np.maximum(1.0 - kappa_ref * d, METRIC_FLOOR)
    dtheta = np.arctan2(d1, metric)  # heading relative to the raceline
    off = np.asarray(offsets, dtype=float)
    s_c = np.broadcast_to(s, d.shape)[..., None] + off * (np.cos(dtheta) / metric)[..., None]
    d_c = d[..., None] + off * np.sin(dtheta)[..., None]
    return s_c, d_c


def path_clearance(s, d, d1, kappa_ref, t, footprint, obstacles, track_length):
    """Smallest clearance between each path and any obstacle.

    s, kappa_ref: (m,) raceline arc length and curvature at each sample
    d, d1:        (n, m) lateral offset and dd/ds of each of n paths
    t:            time the car reaches each sample, broadcastable to (n, m)
    Returns (n,) clearances [m]; +inf when there are no obstacles.
    """
    return clearance_along(s, d, d1, kappa_ref, t, footprint, obstacles, track_length).min(axis=1)


def clearance_along(s, d, d1, kappa_ref, t, footprint, obstacles, track_length):
    """Clearance to the nearest obstacle at every sample of every path,
    shape (n, m); +inf when there are no obstacles. Arguments as for
    path_clearance."""
    d = np.atleast_2d(d)
    if len(obstacles) == 0:
        return np.full(d.shape, np.inf)
    s_c, d_c = circle_centers(s, d, np.atleast_2d(d1), kappa_ref, footprint.offsets)
    metric = np.maximum(1.0 - kappa_ref[:, None] * d_c, METRIC_FLOOR)  # (n, m, c)

    t = np.broadcast_to(t, d.shape)[..., None, None]  # (n, m, 1, 1)
    o = obstacles
    s_lo = o.s_min + o.v_s * t                        # (n, m, 1, k)
    d_lo = o.d_min + o.v_d * t
    d_hi = o.d_max + o.v_d * t

    ds = s_diff(s_c[..., None], s_lo, track_length)  # (n, m, c, k)
    gap_s = np.maximum(np.maximum(-ds, ds - o.s_len), 0.0) * metric[..., None]
    dc = d_c[..., None]
    gap_d = np.maximum(np.maximum(d_lo - dc, dc - d_hi), 0.0)
    dist = np.hypot(gap_s, gap_d) - footprint.radius
    return dist.min(axis=(2, 3))
