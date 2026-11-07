"""Static occupancy map, read straight from a ROS map_server yaml + image."""
import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

# A map origin yaw below this is treated as zero [rad].
ORIGIN_YAW_TOL = 1e-6


@dataclass
class OccupancyGrid:
    """occupied[iy, ix] is True for walls. Row iy = 0 is at origin_y, so the
    array is the image flipped vertically."""
    occupied: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    def world_to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin_x) / self.resolution).astype(int)
        iy = np.floor((np.asarray(y) - self.origin_y) / self.resolution).astype(int)
        return ix, iy

    def is_occupied(self, x, y):
        """Vectorised lookup. Points outside the map count as occupied."""
        return self.sample(self.occupied, x, y, True)

    def sample(self, table, x, y, outside):
        """Vectorised lookup of a per-cell table shaped like occupied;
        points outside the map get the value outside."""
        ix, iy = self.world_to_cell(x, y)
        ny, nx = self.occupied.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.full(np.shape(ix), outside, dtype=table.dtype)
        out[inside] = table[iy[inside], ix[inside]]
        return out

    def wall_distance(self):
        """Per-cell distance to the nearest occupied cell centre [m]; 0 in
        occupied cells."""
        return distance_transform_edt(~self.occupied) * self.resolution


def load_map(yaml_path, unknown_is_occupied):
    """Same thresholds as nav2_map_server in trinary mode."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    origin = meta['origin']
    if abs(origin[2]) > ORIGIN_YAW_TOL:
        raise ValueError(f'{yaml_path}: rotated map origins are not supported')

    image_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), meta['image'])
    pixels = np.asarray(Image.open(image_path).convert('L'), dtype=float) / 255.0
    p_occ = pixels if meta.get('negate', 0) else 1.0 - pixels
    occupied = p_occ > meta['occupied_thresh']
    if unknown_is_occupied:
        occupied |= p_occ >= meta['free_thresh']
    return OccupancyGrid(occupied=np.flipud(occupied), resolution=float(meta['resolution']),
                         origin_x=float(origin[0]), origin_y=float(origin[1]))
