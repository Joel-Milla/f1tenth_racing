"""Quintic lateral profiles: boundary conditions, hold past the horizon,
derivatives and the jerk integral."""
import numpy as np
import pytest

from frenet_planner.quintic import quintic_coeffs, quintic_eval, quintic_jerk_integral

TOL = 1e-9
# Car state at the start of a lane change: offset, slope, d'' (curvature-like)
D0, D1, D2 = 0.12, -0.3, 0.8
D_END = np.linspace(-0.6, 0.6, 9)[:, None]   # (9, 1)
LENGTH = np.array([1.5, 2.5, 4.0])[None, :]  # (1, 3)


def test_boundary_conditions():
    a = quintic_coeffs(D0, D1, D2, D_END, LENGTH)
    assert a.shape == (9, 3, 6)
    start = quintic_eval(a, LENGTH, np.array([0.0]))
    for got, want in zip(start[:3], (D0, D1, D2)):
        assert np.allclose(got[..., 0], want, atol=TOL)
    # Evaluate exactly at each candidate's own horizon
    for j, T in enumerate(LENGTH[0]):
        d, d1, d2, _ = quintic_eval(a[:, j], LENGTH[0, j], np.array([T]))
        assert np.allclose(d[:, 0], D_END[:, 0], atol=TOL)
        assert np.allclose(d1, 0.0, atol=TOL)
        assert np.allclose(d2, 0.0, atol=TOL)


def test_holds_end_state_past_horizon():
    a = quintic_coeffs(D0, D1, D2, D_END, LENGTH)
    u = np.arange(0.0, 6.0, 0.1)
    d, d1, d2, d3 = quintic_eval(a, LENGTH, u)
    assert d.shape == (9, 3, len(u))
    past = u[None, None, :] > LENGTH[..., None]
    assert np.allclose(np.where(past, d - D_END[..., None], 0.0), 0.0, atol=TOL)
    for v in (d1, d2, d3):
        assert np.all(v[np.broadcast_to(past, v.shape)] == 0.0)


def test_derivatives_match_finite_differences():
    a = quintic_coeffs(D0, D1, D2, 0.5, 3.0)
    h = 1e-5
    u = np.linspace(0.1, 2.9, 15)
    d, d1, d2, d3 = quintic_eval(a, 3.0, u)
    for f, df in ((0, d1), (1, d2), (2, d3)):
        lo = quintic_eval(a, 3.0, u - h)[f]
        hi = quintic_eval(a, 3.0, u + h)[f]
        assert np.allclose((hi - lo) / (2 * h), df, atol=1e-5)


def test_jerk_integral_matches_quadrature():
    a = quintic_coeffs(D0, D1, D2, D_END, LENGTH)
    exact = quintic_jerk_integral(a, LENGTH)
    for j, T in enumerate(LENGTH[0]):
        u = np.linspace(0.0, T, 20001)
        d3 = quintic_eval(a[:, j], T, u)[3]
        numeric = np.trapz(d3 ** 2, u, axis=-1)
        assert np.allclose(exact[:, j], numeric, rtol=1e-6)


def test_already_at_target_is_flat():
    a = quintic_coeffs(0.3, 0.0, 0.0, 0.3, 2.0)
    d, d1, d2, d3 = quintic_eval(a, 2.0, np.linspace(0.0, 3.0, 31))
    assert np.allclose(d, 0.3) and np.allclose(d1, 0.0) and np.allclose(d3, 0.0)
    assert quintic_jerk_integral(a, 2.0) == pytest.approx(0.0, abs=TOL)


def test_longer_horizon_is_smoother():
    """Same lane change spread over more track needs less jerk."""
    a = quintic_coeffs(0.0, 0.0, 0.0, 0.5, LENGTH[0])
    jerk = quintic_jerk_integral(a, LENGTH[0])
    assert np.all(np.diff(jerk) < 0)
