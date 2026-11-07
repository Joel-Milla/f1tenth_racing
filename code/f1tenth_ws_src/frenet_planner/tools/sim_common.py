"""Helpers shared by the offline simulation tools: test-map painting and
geometry. Not part of the planner."""
import os

import numpy as np
import yaml
from PIL import Image, ImageDraw

from f110_gym.envs.collision_models import get_vertices

# |kappa| below this counts as straight [1/m]
KAPPA_STRAIGHT = 0.05
# Test box, about one car wide [m]
BOX_SIZE = 0.30


def longest_straight(path):
    """(s at the middle, length) of the longest stretch of the raceline with
    |kappa| < KAPPA_STRAIGHT. Runs through s = 0 count."""
    straight = np.abs(path.kappa_table) < KAPPA_STRAIGHT
    n = len(straight)
    best, best_len, run = 0, 0, 0
    for i in range(2 * n):
        run = run + 1 if straight[i % n] else 0
        if best_len < run <= n:
            best, best_len = i, run
    return float(path.wrap((best - best_len / 2) * path.ds)), best_len * path.ds


def frenet_box_polygon(path, s_c, d_c, length, width):
    """Corners, in loop order, of a rectangle centred at (s_c, d_c) and
    aligned with the raceline there."""
    x, y, yaw = path.frenet_to_cartesian(s_c, d_c)
    return get_vertices(np.array([float(x), float(y), float(yaw)]), length, width)


def paint_map(clean_yaml, out_base, polygons):
    """Write out_base.png / out_base.yaml: the map with polygons (map-frame
    (k, 2) arrays) drawn in as walls. Returns the yaml path."""
    with open(clean_yaml) as f:
        meta = yaml.safe_load(f)
    img = Image.open(os.path.join(os.path.dirname(clean_yaml), meta['image'])).convert('L')
    draw = ImageDraw.Draw(img)
    res, (ox, oy) = meta['resolution'], meta['origin'][:2]
    h = img.size[1]
    for poly in polygons:
        draw.polygon([((x - ox) / res, h - (y - oy) / res) for x, y in poly], fill=0)
    img.save(out_base + '.png')
    meta['image'] = os.path.basename(out_base) + '.png'
    with open(out_base + '.yaml', 'w') as f:
        yaml.safe_dump(meta, f)
    return out_base + '.yaml'


def _separated(a, b):
    """Separating-axis test for two convex polygons."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for nx, ny in np.column_stack([-edges[:, 1], edges[:, 0]]):
            pa, pb = a @ (nx, ny), b @ (nx, ny)
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def _point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.hypot(*(a + t * ab - p))


def polygon_distance(a, b):
    """Distance between two convex polygons ((k, 2) corners in order); 0 if
    they overlap."""
    if not _separated(a, b):
        return 0.0
    best = np.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(len(q)):
                best = min(best, _point_segment(v, q[i], q[(i + 1) % len(q)]))
    return float(best)
