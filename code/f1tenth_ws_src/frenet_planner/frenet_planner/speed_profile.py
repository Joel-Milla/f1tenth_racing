"""Raceline speed profile from curvature (the raceline CSV has no speeds)."""
import numpy as np

# Curvature below this is treated as straight, avoiding division by zero [1/m].
KAPPA_EPS = 1e-6


def curvature_speed_profile(kappa, ds, v_max, a_lat_max, a_accel_max, a_decel_max):
    """Closed-loop speed profile on a uniform s grid.

    v = min(v_max, sqrt(a_lat_max / |kappa|)), then a forward pass limits
    acceleration and a backward pass limits braking. Each pass goes around the
    loop twice so the limits carry across the start/finish line.
    """
    kappa = np.asarray(kappa, dtype=float)
    n = len(kappa)
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), KAPPA_EPS)))
    for i in range(1, 2 * n):
        k, prev = i % n, (i - 1) % n
        v[k] = min(v[k], np.sqrt(v[prev] ** 2 + 2.0 * a_accel_max * ds))
    for i in range(2 * n - 2, -1, -1):
        k, nxt = i % n, (i + 1) % n
        v[k] = min(v[k], np.sqrt(v[nxt] ** 2 + 2.0 * a_decel_max * ds))
    return v
