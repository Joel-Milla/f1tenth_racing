"""Track boundaries d_left(s), d_right(s), measured by ray-casting the static
map perpendicular to the raceline."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d


@dataclass
class TrackBounds:
    """d_left and d_right are distances (both positive) from the raceline to
    the wall on each side, so the free band at s is [-d_right(s), d_left(s)]."""
    s: np.ndarray
    d_left: np.ndarray
    d_right: np.ndarray
    length: float

    def _lookup(self, table, s):
        s = np.mod(s, self.length)
        return np.interp(s, np.r_[self.s, self.length], np.r_[table, table[0]])

    def left(self, s):
        return self._lookup(self.d_left, s)

    def right(self, s):
        return self._lookup(self.d_right, s)


def compute_track_bounds(ref, grid, ds, max_range, min_filter):
    """
    ref:        ReferencePath
    grid:       OccupancyGrid
    ds:         spacing of the output tables along s [m]
    max_range:  rays stop here if they hit nothing [m]
    min_filter: window over which each bound is replaced by its running
                minimum [m]; 0 disables
    """
    n = max(int(round(ref.length / ds)), 1)
    s = np.arange(n) * (ref.length / n)
    rx, ry, yaw, _, _ = ref.eval(s)
    nx, ny = -np.sin(yaw), np.cos(yaw)  # left normal

    # Half-cell steps so a ray cannot skip a one-cell wall
    step = 0.5 * grid.resolution
    r = np.arange(0.0, max_range + step, step)

    bounds = []
    for side in (1.0, -1.0):
        px = rx[:, None] + side * nx[:, None] * r[None, :]
        py = ry[:, None] + side * ny[:, None] * r[None, :]
        hit = grid.is_occupied(px, py)
        first = np.argmax(hit, axis=1)
        # Distance of the last free sample before the wall: conservative by < step
        dist = np.where(hit.any(axis=1), r[np.maximum(first - 1, 0)], max_range)
        dist[hit[:, 0]] = 0.0  # raceline itself inside a wall
        bounds.append(dist)
    d_left, d_right = bounds

    if min_filter > 0.0:
        width = 2 * int(round(0.5 * min_filter / (ref.length / n))) + 1
        d_left = minimum_filter1d(d_left, width, mode='wrap')
        d_right = minimum_filter1d(d_right, width, mode='wrap')
    return TrackBounds(s=s, d_left=d_left, d_right=d_right, length=ref.length)
