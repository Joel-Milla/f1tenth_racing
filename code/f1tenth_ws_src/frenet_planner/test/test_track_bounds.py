import numpy as np
import pytest
from PIL import Image

from conftest import TABLE_DS, ellipse_points
from frenet_planner.frenet import ReferencePath
from frenet_planner.occupancy import OccupancyGrid, load_map
from frenet_planner.track_bounds import compute_track_bounds

RES = 0.05
MAX_RANGE = 3.0


def ring_grid(r_in, r_out, half_size=4.0):
    n = int(2 * half_size / RES)
    c = (np.arange(n) + 0.5) * RES - half_size
    gx, gy = np.meshgrid(c, c)  # occupied[iy, ix]
    r = np.hypot(gx, gy)
    return OccupancyGrid(occupied=(r < r_in) | (r > r_out), resolution=RES,
                         origin_x=-half_size, origin_y=-half_size)


@pytest.fixture(scope='module')
def circle_ref():
    return ReferencePath(ellipse_points(2.0, 2.0, 126), 0.0, TABLE_DS, 1.0, 3.0)


def test_ring_asymmetric(circle_ref):
    # CCW circle r = 2: left is towards the centre
    b = compute_track_bounds(circle_ref, ring_grid(1.5, 2.8), 0.05, MAX_RANGE, 0.0)
    # The analytic circle is drawn in RES-sized cells, so the pixel wall sits
    # up to one cell either side of it
    assert np.abs(b.d_left - 0.5).max() <= RES + 1e-9
    assert np.abs(b.d_right - 0.8).max() <= RES + 1e-9


def corridor_grid(left_wall, right_wall, gap=None):
    """Straight corridor along +x, walls at y = left_wall and y = -right_wall.
    gap = (x0, x1) removes a stretch of the left wall (a doorway)."""
    nx, ny = 400, 200
    x = (np.arange(nx) + 0.5) * RES - 10.0
    y = (np.arange(ny) + 0.5) * RES - 5.0
    gx, gy = np.meshgrid(x, y)
    occ = (gy > left_wall) | (gy < -right_wall)
    if gap is not None:
        occ &= ~((gx > gap[0]) & (gx < gap[1]) & (gy > left_wall) & (gy < left_wall + 0.5))
    return OccupancyGrid(occupied=occ, resolution=RES, origin_x=-10.0, origin_y=-5.0)


@pytest.fixture(scope='module')
def stadium_ref():
    """Long oval whose bottom straight runs along y = 0 from x = -5 to 5."""
    pts = []
    for x in np.arange(-5.0, 5.0, 0.1):
        pts.append((x, 0.0))
    for t in np.arange(-np.pi / 2, np.pi / 2, 0.1 / 2.0):
        pts.append((5.0 + 2.0 * np.cos(t), 2.0 + 2.0 * np.sin(t)))
    for x in np.arange(5.0, -5.0, -0.1):
        pts.append((x, 4.0))
    for t in np.arange(np.pi / 2, 3 * np.pi / 2, 0.1 / 2.0):
        pts.append((-5.0 + 2.0 * np.cos(t), 2.0 + 2.0 * np.sin(t)))
    return ReferencePath(np.array(pts), 0.0, TABLE_DS, 1.0, 3.0)


def straight_bounds(ref, b):
    s0, _ = ref.cartesian_to_frenet(-3.0, 0.0)
    s1, _ = ref.cartesian_to_frenet(3.0, 0.0)
    s = np.arange(s0, s1, 0.05)
    return b.left(s), b.right(s)


def test_doorway_hidden_by_min_filter(stadium_ref):
    # Only the bottom straight is checked; the rest of the oval lies inside
    # the corridor walls and its bounds are meaningless here
    grid = corridor_grid(0.5, 0.7, gap=(0.0, 0.4))
    raw = compute_track_bounds(stadium_ref, grid, 0.05, MAX_RANGE, 0.0)
    filt = compute_track_bounds(stadium_ref, grid, 0.05, MAX_RANGE, 0.6)
    l_raw, r_raw = straight_bounds(stadium_ref, raw)
    l_f, r_f = straight_bounds(stadium_ref, filt)
    assert l_raw.max() > 0.9                 # rays escape through the doorway
    assert l_f.max() <= 0.5 + 1e-9           # filter closes it
    assert np.allclose(r_f, 0.7 - RES / 2, atol=RES)


def test_load_map_orientation_and_thresholds(tmp_path):
    # 4x3 image. Top-left pixel black (occupied) must land at the max-y row.
    img = np.full((3, 4), 254, dtype=np.uint8)
    img[0, 0] = 0      # occupied, top-left
    img[2, 3] = 205    # unknown (p_occ ~ 0.2), bottom-right
    Image.fromarray(img).save(tmp_path / 'm.pgm')
    (tmp_path / 'm.yaml').write_text('image: m.pgm\nresolution: 0.5\norigin: [-1.0, -1.0, 0.0]\n'
                                     'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    g = load_map(str(tmp_path / 'm.yaml'), unknown_is_occupied=False)
    assert g.is_occupied(-0.75, 0.25)        # top-left cell centre
    assert not g.is_occupied(0.75, -0.75)    # unknown treated as free
    assert not g.is_occupied(0.25, -0.25)
    assert g.is_occupied(5.0, 5.0)           # outside the map
    g2 = load_map(str(tmp_path / 'm.yaml'), unknown_is_occupied=True)
    assert g2.is_occupied(0.75, -0.75)
