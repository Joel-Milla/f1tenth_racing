import os
import sys

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PKG_ROOT)

from frenet_planner.frenet import ReferencePath, load_raceline  # noqa: E402

LEVINE_CSV = os.path.join(PKG_ROOT, '..', 'pure_pursuit', 'waypoints', 'levine.csv')
CONFIG_YAML = os.path.join(PKG_ROOT, 'config', 'frenet_planner.yaml')

# Values matching config/frenet_planner.yaml, used to build test references
SMOOTHING = 0.02
TABLE_DS = 0.05
SEARCH_BACK = 1.0
SEARCH_FWD = 3.0


def ellipse_points(a, b, n):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t), b * np.sin(t)])


@pytest.fixture(scope='session')
def ellipse():
    """Smooth CCW loop with known geometry: a=4, b=2.5, ~21 m, 0.1 m spacing."""
    return ReferencePath(ellipse_points(4.0, 2.5, 210), 0.0, TABLE_DS, SEARCH_BACK, SEARCH_FWD)


@pytest.fixture(scope='session')
def levine():
    if not os.path.exists(LEVINE_CSV):
        pytest.skip(f'{LEVINE_CSV} not found')
    return ReferencePath(load_raceline(LEVINE_CSV), SMOOTHING, TABLE_DS, SEARCH_BACK, SEARCH_FWD)
