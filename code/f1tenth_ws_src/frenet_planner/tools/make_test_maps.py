#!/usr/bin/env python3
"""Write the simulator test maps: the clean sim map with a ~car-wide box
painted mid-way along the longest straight, once on the left of the
raceline and once on the right. Also writes maps/test_boxes.yaml with each
box's Frenet position, which tools/sim_eval.py reads.

These maps are for the SIMULATOR only (gym_bridge map_path). The planner
keeps the clean map: anything not in it is an obstacle.

    /usr/bin/python3 tools/make_test_maps.py [--offset 0.25]
"""
import argparse
import os
import sys

import yaml

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frenet_planner.params import Params  # noqa: E402
from frenet_planner.reference import build_reference  # noqa: E402
from sim_common import BOX_SIZE, frenet_box_polygon, longest_straight, paint_map  # noqa: E402

MAP_DIR = os.path.join(PKG_ROOT, 'maps')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_ROOT, 'config', 'frenet_planner.yaml'))
    ap.add_argument('--offset', type=float, default=0.25,
                    help='|d| of the box centre from the raceline [m]')
    args = ap.parse_args()

    params = Params.from_yaml(args.config)
    ref = build_reference(params)
    s_mid, length = longest_straight(ref.path)
    os.makedirs(MAP_DIR, exist_ok=True)
    clean = os.path.splitext(os.path.basename(params.map.yaml_file))[0]
    boxes = {}
    for side, d in (('left', args.offset), ('right', -args.offset)):
        name = f'{clean}_box_{side}'
        poly = frenet_box_polygon(ref.path, s_mid, d, BOX_SIZE, BOX_SIZE)
        paint_map(params.map.yaml_file, os.path.join(MAP_DIR, name), [poly])
        boxes[name] = dict(s=round(s_mid, 3), d=d, size=BOX_SIZE, corners=poly.round(4).tolist())
        print(f'{name}: box centre s {s_mid:.2f} (middle of the {length:.1f} m straight), d {d:+.2f}, '
              f'half-widths left {float(ref.bounds.left(s_mid)):.2f} right {float(ref.bounds.right(s_mid)):.2f}')
    with open(os.path.join(MAP_DIR, 'test_boxes.yaml'), 'w') as f:
        f.write('# Written by tools/make_test_maps.py. Box centre in Frenet (s, d) of the\n'
                '# planner raceline, side length, and map-frame corners.\n')
        yaml.safe_dump(boxes, f, default_flow_style=None)


if __name__ == '__main__':
    main()
