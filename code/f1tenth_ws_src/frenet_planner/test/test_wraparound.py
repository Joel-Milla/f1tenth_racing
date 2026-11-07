"""s wraparound at the start/finish line."""
import numpy as np
import pytest

from frenet_planner.speed_profile import curvature_speed_profile
from frenet_planner.track_bounds import TrackBounds

TOL = 0.01


def test_wrap_and_s_diff(levine):
    L = levine.length
    assert levine.wrap(L + 0.3) == pytest.approx(0.3)
    assert levine.wrap(-0.3) == pytest.approx(L - 0.3)
    assert levine.s_diff(0.1, L - 0.1) == pytest.approx(0.2)
    assert levine.s_diff(L - 0.1, 0.1) == pytest.approx(-0.2)
    assert levine.s_diff(5.0, 3.0) == pytest.approx(2.0)


def test_eval_is_periodic(levine):
    L = levine.length
    s = np.array([0.0, 0.02, 0.3, 1.7])
    for a, b in zip(levine.frenet_to_cartesian(s, 0.2), levine.frenet_to_cartesian(s + L, 0.2)):
        assert np.allclose(a, b)
    for a, b in zip(levine.frenet_to_cartesian(s, 0.2), levine.frenet_to_cartesian(s - L, 0.2)):
        assert np.allclose(a, b)


@pytest.mark.parametrize('s_true,hint_offset', [(-0.02, 0.07), (0.03, -0.13), (-0.4, 0.5), (0.4, -0.5)])
def test_windowed_search_across_the_line(levine, s_true, hint_offset):
    """Point on one side of s = 0, hint on the other side."""
    L = levine.length
    x, y, _ = levine.frenet_to_cartesian(s_true, -0.25)
    s, d = levine.cartesian_to_frenet(float(x), float(y), s_hint=levine.wrap(s_true + hint_offset))
    assert 0.0 <= s < L
    assert abs(levine.s_diff(s, levine.wrap(s_true))) < TOL
    assert d == pytest.approx(-0.25, abs=TOL)


def test_tracking_through_the_line(levine):
    """Drive a point through the finish line, seeding each search with the
    previous result, as the node does. Progress must stay positive and smooth."""
    L = levine.length
    s_path = np.arange(L - 2.0, L + 2.0, 0.05)
    x, y, _ = levine.frenet_to_cartesian(s_path, 0.1 * np.sin(s_path))
    s_prev = levine.wrap(s_path[0])
    steps = []
    for xi, yi in zip(x, y):
        s, _ = levine.cartesian_to_frenet(float(xi), float(yi), s_hint=s_prev)
        steps.append(levine.s_diff(s, s_prev))
        s_prev = s
    steps = np.array(steps[1:])
    assert (steps > 0).all()
    assert np.abs(steps - 0.05).max() < 0.02


def test_bounds_lookup_is_periodic():
    s = np.linspace(0.0, 10.0, 100, endpoint=False)
    b = TrackBounds(s=s, d_left=1.0 + 0.1 * np.sin(2 * np.pi * s / 10.0), d_right=np.full(100, 0.5), length=10.0)
    q = np.array([0.0, 0.05, 3.3, 9.97])
    assert np.allclose(b.left(q), b.left(q + 10.0))
    # Interpolation across the seam uses the first table value, not an extrapolation
    assert b.left(9.95) == pytest.approx(0.5 * (b.d_left[-1] + b.d_left[0]))


def test_speed_profile_continuous_across_the_line():
    # One tight corner straddling the seam: braking for it must start before
    # the end of the array, i.e. the backward pass wraps
    n, ds = 400, 0.1
    kappa = np.zeros(n)
    kappa[:10] = 1.0
    kappa[-10:] = 1.0
    a_decel = 4.0
    v = curvature_speed_profile(kappa, ds, 6.0, 4.0, 3.0, a_decel)
    assert v[-10] == pytest.approx(2.0)
    # Braking limit holds across the seam and everywhere else
    dv2 = np.r_[v[1:], v[0]] ** 2 - v ** 2
    assert (dv2 >= -2 * a_decel * ds - 1e-9).all()
    assert v[-11] == pytest.approx(np.sqrt(2.0 ** 2 + 2 * a_decel * ds))
