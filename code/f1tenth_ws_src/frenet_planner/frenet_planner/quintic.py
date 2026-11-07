"""Quintic lateral profiles d(u), u = s - s0 the arc length travelled along the
raceline from the car. Every function broadcasts, so a whole grid of
candidates (d_end x horizon length) is fitted and evaluated in one call.

Past the end of its horizon a profile holds d = d_end with zero slope, so
candidates of different lengths share one u grid and are checked over the
same stretch of track.
"""
import numpy as np

N_COEFFS = 6


def quintic_coeffs(d0, d1, d2, d_end, length):
    """Coefficients a0..a5 of d(u) = sum_k a_k u^k on [0, length] with
    d(0) = d0, d'(0) = d1, d''(0) = d2 and d(length) = d_end, d'(length) =
    d''(length) = 0 (primes are d/du).

    d0, d1, d2 are the car's state (scalars); d_end and length broadcast
    against each other. Returns shape broadcast(d_end, length) + (6,).
    """
    d_end, T = np.broadcast_arrays(np.asarray(d_end, dtype=float), np.asarray(length, dtype=float))
    h = d_end - d0
    T2 = T * T
    T3 = T2 * T
    a = np.empty(d_end.shape + (N_COEFFS,))
    a[..., 0] = d0
    a[..., 1] = d1
    a[..., 2] = 0.5 * d2
    a[..., 3] = (20.0 * h - 12.0 * d1 * T - 3.0 * d2 * T2) / (2.0 * T3)
    a[..., 4] = (-30.0 * h + 16.0 * d1 * T + 3.0 * d2 * T2) / (2.0 * T3 * T)
    a[..., 5] = (12.0 * h - 6.0 * d1 * T - d2 * T2) / (2.0 * T3 * T2)
    return a


def quintic_eval(coeffs, length, u):
    """d, d', d'', d''' of each profile at the arc lengths u.

    coeffs: (..., 6) from quintic_coeffs
    length: horizon of each profile, broadcastable to coeffs.shape[:-1]
    u:      (m,) arc lengths >= 0, shared by all profiles
    Returns four arrays of shape coeffs.shape[:-1] + (m,). For u > length
    the profile holds its end state: d = d_end and all derivatives 0.
    """
    a = np.asarray(coeffs, dtype=float)[..., None, :]
    T = np.asarray(length, dtype=float)[..., None]
    u = np.asarray(u, dtype=float)
    past = u > T
    x = np.minimum(u, T)
    a0, a1, a2, a3, a4, a5 = (a[..., k] for k in range(N_COEFFS))
    d = a0 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * a5))))
    d1 = a1 + x * (2.0 * a2 + x * (3.0 * a3 + x * (4.0 * a4 + x * 5.0 * a5)))
    d2 = 2.0 * a2 + x * (6.0 * a3 + x * (12.0 * a4 + x * 20.0 * a5))
    d3 = 6.0 * a3 + x * (24.0 * a4 + x * 60.0 * a5)
    d1, d2, d3 = (np.where(past, 0.0, v) for v in (d1, d2, d3))
    return d, d1, d2, d3


def quintic_jerk_integral(coeffs, length):
    """Exact integral of d'''(u)^2 over [0, length], the smoothness cost.
    d''' = 6 a3 + 24 a4 u + 60 a5 u^2, so the integral is a polynomial in
    length. Broadcasts like quintic_eval; returns coeffs.shape[:-1]."""
    a = np.asarray(coeffs, dtype=float)
    T = np.asarray(length, dtype=float)
    p, q, r = 6.0 * a[..., 3], 24.0 * a[..., 4], 60.0 * a[..., 5]
    # (p + q u + r u^2)^2 = p^2 + 2pq u + (q^2 + 2pr) u^2 + 2qr u^3 + r^2 u^4
    return (p * p * T + p * q * T ** 2 + (q * q + 2.0 * p * r) * T ** 3 / 3.0
            + 0.5 * q * r * T ** 4 + r * r * T ** 5 / 5.0)
