"""Builds everything that depends only on the raceline and the static map."""
from dataclasses import dataclass

import numpy as np

from frenet_planner.frenet import ReferencePath, load_raceline
from frenet_planner.occupancy import OccupancyGrid, load_map
from frenet_planner.speed_profile import curvature_speed_profile
from frenet_planner.track_bounds import TrackBounds, compute_track_bounds


@dataclass
class Reference:
    path: ReferencePath
    grid: OccupancyGrid
    bounds: TrackBounds
    v_ref: np.ndarray  # raceline speed on path.s_table, global_scale applied

    def speed(self, s):
        p = self.path
        return np.interp(p.wrap(s), np.r_[p.s_table, p.length], np.r_[self.v_ref, self.v_ref[0]])


def build_reference(params):
    rp = params.raceline
    path = ReferencePath(load_raceline(rp.file), rp.smoothing, rp.ds, rp.search_back, rp.search_fwd)
    grid = load_map(params.map.yaml_file, params.map.unknown_is_occupied)
    tp = params.track
    bounds = compute_track_bounds(path, grid, tp.ds, tp.max_range, tp.min_filter)
    sp = params.speed
    v = curvature_speed_profile(path.kappa_table, path.ds, sp.v_max, sp.a_lat_max,
                                sp.a_accel_max, sp.a_decel_max)
    return Reference(path=path, grid=grid, bounds=bounds, v_ref=sp.global_scale * v)
