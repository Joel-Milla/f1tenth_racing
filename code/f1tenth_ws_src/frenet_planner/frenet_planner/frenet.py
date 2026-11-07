"""Reference path (the raceline) and Frenet <-> Cartesian conversions.

The raceline is a closed loop of length L. s is arc length in [0, L) and
wraps at the start/finish line; d is the signed lateral offset, positive to
the left of the driving direction.
"""
import numpy as np
from scipy.interpolate import CubicSpline, splev, splprep

# Dense samples per raceline point when measuring the arc length of the
# smoothing spline. Arc-length error falls with the square of this.
ARC_LENGTH_OVERSAMPLE = 20
# Newton steps refining the foot point after the coarse table search. The
# table point is within ds/2 of the answer, so 3 steps reach sub-millimetre.
NEWTON_ITERS = 3
# Points closer than this are treated as duplicates when loading [m].
DUPLICATE_TOL = 1e-3


def load_raceline(path):
    """Read an x,y CSV with a header row. Returns (N, 2) with no repeated
    closing point and no zero-length segments."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    xy = np.column_stack((data['x'], data['y']))
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > DUPLICATE_TOL]
    xy = xy[keep]
    if np.hypot(*(xy[-1] - xy[0])) < DUPLICATE_TOL:
        xy = xy[:-1]
    return xy


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def s_diff(a, b, length):
    """Signed shortest arc-length difference a - b on a loop, in [-L/2, L/2)."""
    return (np.asarray(a) - b + 0.5 * length) % length - 0.5 * length


class ReferencePath:
    """Periodic arc-length spline through a closed raceline."""

    def __init__(self, xy, smoothing, ds, search_back, search_fwd):
        """
        xy:          (N, 2) closed-loop points, first point not repeated
        smoothing:   RMS deviation allowed from the points [m]; 0 interpolates
        ds:          spacing of the lookup tables [m] (rounded so it divides L)
        search_back, search_fwd: default window around s_hint [m]
        """
        xy = np.asarray(xy, dtype=float)
        closed = np.vstack([xy, xy[:1]])  # splprep(per=1) ignores the last point's value
        tck, _ = splprep(closed.T, s=len(xy) * smoothing ** 2, per=1)

        u = np.linspace(0.0, 1.0, ARC_LENGTH_OVERSAMPLE * len(xy) + 1)
        px, py = splev(u, tck)
        s_dense = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        self.length = float(s_dense[-1])

        n = max(int(round(self.length / ds)), 4)
        self.ds = self.length / n
        self.s_table = np.arange(n) * self.ds
        gx, gy = splev(np.interp(self.s_table, s_dense, u), tck)
        knots = np.r_[self.s_table, self.length]
        pts = np.column_stack([np.r_[gx, gx[0]], np.r_[gy, gy[0]]])
        self._spline = CubicSpline(knots, pts, bc_type='periodic')

        self.x_table, self.y_table = np.asarray(gx), np.asarray(gy)
        _, _, self.yaw_table, self.kappa_table, self.dkappa_table = self.eval(self.s_table)
        self.search_back = search_back
        self.search_fwd = search_fwd

    # --- s bookkeeping -------------------------------------------------------

    def wrap(self, s):
        return np.mod(s, self.length)

    def s_diff(self, a, b):
        """Signed shortest arc-length difference a - b, in [-L/2, L/2)."""
        return s_diff(a, b, self.length)

    # --- evaluation ----------------------------------------------------------

    def eval(self, s):
        """x, y, heading, curvature, d(curvature)/ds at arc length s (wrapped)."""
        s = self.wrap(np.asarray(s, dtype=float))
        p, d1, d2, d3 = (self._spline(s, k) for k in range(4))
        x1, y1, x2, y2, x3, y3 = d1[..., 0], d1[..., 1], d2[..., 0], d2[..., 1], d3[..., 0], d3[..., 1]
        speed2 = x1 * x1 + y1 * y1
        num = x1 * y2 - y1 * x2
        den = speed2 ** 1.5
        kappa = num / den
        dnum = x1 * y3 - y1 * x3
        dden = 3.0 * np.sqrt(speed2) * (x1 * x2 + y1 * y2)
        # d/ds with s = spline parameter; |r'| = 1 to ~1e-5, so no chain-rule factor
        dkappa = (dnum * den - num * dden) / (den * den)
        return p[..., 0], p[..., 1], np.arctan2(y1, x1), kappa, dkappa

    # --- conversions ---------------------------------------------------------

    def frenet_to_cartesian(self, s, d, d_prime=None):
        """(s, d) -> (x, y, yaw). Without d_prime, yaw is the raceline heading
        at s. With d_prime = dd/ds, yaw is the heading of the curve d(s)."""
        rx, ry, ryaw, kappa, _ = self.eval(s)
        d = np.asarray(d, dtype=float)
        x = rx - d * np.sin(ryaw)
        y = ry + d * np.cos(ryaw)
        yaw = ryaw if d_prime is None else ryaw + np.arctan2(d_prime, 1.0 - kappa * d)
        return x, y, wrap_angle(yaw)

    def cartesian_to_frenet(self, x, y, s_hint=None, window=None, reset_dist=None):
        """(x, y) -> (s, d), s wrapped into [0, L).

        s_hint:     search only the table points in a window around it
                    (fast); None searches the whole loop
        window:     (back, fwd) override of the default window [m]
        reset_dist: points whose |d| exceeds this after a windowed search are
                    searched again over the whole loop. Use it for the car's own
                    pose, not for scan points (those are legitimately far).
        Scalars in give scalars out.
        """
        scalar = np.ndim(x) == 0
        px, py = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
        s, d = self._project(px, py, s_hint, window)
        if s_hint is not None and reset_dist is not None:
            redo = np.abs(d) > reset_dist
            if redo.any():
                s[redo], d[redo] = self._project(px[redo], py[redo], None, None)
        if scalar:
            return float(s[0]), float(d[0])
        return s, d

    def _project(self, px, py, s_hint, window):
        n = len(self.s_table)
        if s_hint is None:
            idx = np.arange(n)
        else:
            back, fwd = window if window is not None else (self.search_back, self.search_fwd)
            nb = min(int(np.ceil(back / self.ds)), n // 2)
            nf = min(int(np.ceil(fwd / self.ds)), n - nb - 1)
            i0 = int(round(float(self.wrap(s_hint)) / self.ds))
            idx = (i0 + np.arange(-nb, nf + 1)) % n
        d2 = (px[:, None] - self.x_table[idx]) ** 2 + (py[:, None] - self.y_table[idx]) ** 2
        s = self.s_table[idx[np.argmin(d2, axis=1)]]

        # Newton on f(s) = (r(s) - p) . r'(s) = 0, steps clipped to one table cell
        for _ in range(NEWTON_ITERS):
            r, r1, r2 = (self._spline(s, k) for k in range(3))
            ex, ey = r[:, 0] - px, r[:, 1] - py
            f = ex * r1[:, 0] + ey * r1[:, 1]
            fp = r1[:, 0] ** 2 + r1[:, 1] ** 2 + ex * r2[:, 0] + ey * r2[:, 1]
            step = np.where(fp > 0.0, f / np.where(fp > 0.0, fp, 1.0), 0.0)
            s = self.wrap(s - np.clip(step, -self.ds, self.ds))

        r, r1 = self._spline(s), self._spline(s, 1)
        tn = np.hypot(r1[:, 0], r1[:, 1])
        d = (r1[:, 0] * (py - r[:, 1]) - r1[:, 1] * (px - r[:, 0])) / tn
        return s, d

    def path_curvature(self, s, d, d1, d2):
        """Curvature of the Cartesian curve traced by d(s), given d, d' = dd/ds
        and d'' = d^2d/ds^2 at s. Standard Frenet relation (Werling 2010,
        low-speed s-parameterised form)."""
        _, _, _, kr, dkr = self.eval(s)
        one_m = 1.0 - kr * d
        dtheta = np.arctan2(d1, one_m)
        cos_t, tan_t = np.cos(dtheta), np.tan(dtheta)
        return ((d2 + (dkr * d + kr * d1) * tan_t) * cos_t ** 2 / one_m + kr) * cos_t / one_m
