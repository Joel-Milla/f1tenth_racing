import numpy as np
import pytest

from conftest import TABLE_DS, ellipse_points
from frenet_planner.frenet import ReferencePath, load_raceline

ROUND_TRIP_TOL = 0.01      # requirement: < 1 cm
N_SAMPLES = 3000
HINT_NOISE = 0.3           # s_hint is this far off the truth, at most [m]
MAX_D = 0.6                # lateral offsets tested [m]
# Stay well inside the centre of curvature on the inside of corners,
# where the foot point stops being unique
MAX_KAPPA_D = 0.7


def sample_frenet(ref, rng):
    s = rng.uniform(0.0, ref.length, N_SAMPLES)
    kappa = ref.eval(s)[3]
    # d > 0 is the inside of a left (kappa > 0) turn
    d = rng.uniform(-MAX_D, MAX_D, N_SAMPLES)
    inside = d * kappa > 0
    d[inside] = np.clip(d[inside], -MAX_KAPPA_D / np.abs(kappa[inside]), MAX_KAPPA_D / np.abs(kappa[inside]))
    return s, d


@pytest.mark.parametrize('name', ['ellipse', 'levine'])
@pytest.mark.parametrize('use_hint', [True, False])
def test_round_trip_under_1cm(name, use_hint, request):
    ref = request.getfixturevalue(name)
    rng = np.random.default_rng(0)
    s, d = sample_frenet(ref, rng)
    x, y, _ = ref.frenet_to_cartesian(s, d)

    s2 = np.empty_like(s)
    d2 = np.empty_like(d)
    for i in range(N_SAMPLES):
        hint = s[i] + rng.uniform(-HINT_NOISE, HINT_NOISE) if use_hint else None
        s2[i], d2[i] = ref.cartesian_to_frenet(x[i], y[i], s_hint=hint)
    x2, y2, _ = ref.frenet_to_cartesian(s2, d2)

    assert np.abs(ref.s_diff(s2, s)).max() < ROUND_TRIP_TOL
    assert np.abs(d2 - d).max() < ROUND_TRIP_TOL
    assert np.hypot(x2 - x, y2 - y).max() < ROUND_TRIP_TOL


def test_vectorised_matches_scalar(levine):
    rng = np.random.default_rng(1)
    s, d = sample_frenet(levine, rng)
    s, d = s[:200], d[:200]
    x, y, _ = levine.frenet_to_cartesian(s, d)
    sv, dv = levine.cartesian_to_frenet(x, y)
    ss = [levine.cartesian_to_frenet(xi, yi)[0] for xi, yi in zip(x, y)]
    assert np.allclose(sv, ss)
    assert np.abs(dv - d).max() < ROUND_TRIP_TOL


def test_left_is_positive_d(ellipse):
    # CCW ellipse: at t = 0 the point (4, 0) heads +y, so left is -x
    s0, _ = ellipse.cartesian_to_frenet(4.0, 0.0)
    _, d_in = ellipse.cartesian_to_frenet(3.5, 0.0, s_hint=s0)
    _, d_out = ellipse.cartesian_to_frenet(4.5, 0.0, s_hint=s0)
    assert d_in == pytest.approx(0.5, abs=1e-3)
    assert d_out == pytest.approx(-0.5, abs=1e-3)


def test_arc_length_and_curvature_of_circle():
    r = 2.0
    ref = ReferencePath(ellipse_points(r, r, 126), 0.0, TABLE_DS, 1.0, 3.0)
    assert ref.length == pytest.approx(2 * np.pi * r, rel=1e-4)
    assert np.allclose(ref.kappa_table, 1.0 / r, rtol=1e-3)
    assert np.abs(ref.dkappa_table).max() < 1e-2


def test_ellipse_curvature_matches_analytic(ellipse):
    a, b = 4.0, 2.5
    x, y = ellipse.x_table, ellipse.y_table
    t = np.arctan2(y / b, x / a)
    k_true = a * b / (a ** 2 * np.sin(t) ** 2 + b ** 2 * np.cos(t) ** 2) ** 1.5
    assert np.abs(ellipse.kappa_table - k_true).max() < 0.01 * k_true.max()


def test_heading_with_d_prime(ellipse):
    """yaw from frenet_to_cartesian(d_prime=...) matches the finite-difference
    heading of the curve d(s)."""
    s = np.linspace(0.0, ellipse.length, 4000, endpoint=False)
    amp, wavelength = 0.3, 5.0
    d = amp * np.sin(2 * np.pi * s / wavelength)
    dp = amp * 2 * np.pi / wavelength * np.cos(2 * np.pi * s / wavelength)
    x, y, yaw = ellipse.frenet_to_cartesian(s, d, d_prime=dp)
    fd = np.arctan2(np.gradient(y), np.gradient(x))
    err = np.angle(np.exp(1j * (yaw - fd)))[5:-5]
    assert np.abs(err).max() < 1e-2


def test_path_curvature_matches_finite_difference(ellipse):
    s = np.linspace(0.0, ellipse.length, 8000, endpoint=False)
    w = 2 * np.pi / 5.0
    amp = 0.3
    d = amp * np.sin(w * s)
    d1 = amp * w * np.cos(w * s)
    d2 = -amp * w * w * np.sin(w * s)
    k = ellipse.path_curvature(s, d, d1, d2)

    x, y, _ = ellipse.frenet_to_cartesian(s, d)
    xd, yd = np.gradient(x), np.gradient(y)
    xdd, ydd = np.gradient(xd), np.gradient(yd)
    k_fd = (xd * ydd - yd * xdd) / (xd ** 2 + yd ** 2) ** 1.5
    assert np.abs(k - k_fd)[5:-5].max() < 0.01 * np.abs(k).max()


def test_load_raceline_drops_repeated_closing_point(tmp_path):
    p = tmp_path / 'wp.csv'
    p.write_text('x,y\n0,0\n1,0\n1,1\n1,1\n0,1\n0,0\n')
    xy = load_raceline(str(p))
    assert xy.tolist() == [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_reset_dist_recovers_from_bad_hint(levine):
    s_true, d_true = 30.0, 0.2
    x, y, _ = levine.frenet_to_cartesian(s_true, d_true)
    bad_hint = s_true + 15.0
    s, d = levine.cartesian_to_frenet(float(x), float(y), s_hint=bad_hint, reset_dist=1.0)
    assert abs(levine.s_diff(s, s_true)) < ROUND_TRIP_TOL
    assert d == pytest.approx(d_true, abs=ROUND_TRIP_TOL)
