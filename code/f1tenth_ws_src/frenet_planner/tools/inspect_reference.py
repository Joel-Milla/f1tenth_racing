#!/usr/bin/env python3
"""Build the reference from the params YAML, print stats and timings, and
optionally save a plot.

    python3 tools/inspect_reference.py [--params config/frenet_planner.yaml] [--plot out.png]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from frenet_planner.params import Params  # noqa: E402
from frenet_planner.reference import build_reference  # noqa: E402

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'frenet_planner.yaml')
TIMING_REPEATS = 200
SCAN_POINTS = 1080             # f1tenth LiDAR beams
SCAN_WINDOW = (2.0, 10.0)      # scan points lie up to ~10 m ahead
PLAN_POINTS = 15 * 3 * 40      # ~15 d_end x 3 horizons x 4 m / 0.1 m


def timeit(fn):
    fn()
    t = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        fn()
    return (time.perf_counter() - t) / TIMING_REPEATS * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=DEFAULT_PARAMS)
    ap.add_argument('--plot')
    args = ap.parse_args()

    p = Params.from_yaml(args.params)
    t = time.perf_counter()
    ref = build_reference(p)
    t_build = (time.perf_counter() - t) * 1e3
    path, b = ref.path, ref.bounds

    k = np.abs(path.kappa_table)
    km = p.vehicle.kappa_max
    width = b.d_left + b.d_right
    half = 0.5 * p.vehicle.width
    print(f'raceline       {p.raceline.file}')
    print(f'length         {path.length:.2f} m, table ds {path.ds:.4f} m, build {t_build:.0f} ms')
    print(f'|kappa|        max {k.max():.2f} 1/m, kappa_max (steering) {km:.2f} 1/m, '
          f'{(k > km).sum() * path.ds:.2f} m of the loop above it')
    for side, arr in (('d_left', b.d_left), ('d_right', b.d_right)):
        print(f'{side:<14} min {arr.min():.2f}  median {np.median(arr):.2f}  max {arr.max():.2f} m')
    print(f'track width    min {width.min():.2f}  median {np.median(width):.2f} m '
          f'(= {np.median(width) / p.vehicle.width:.1f} car widths)')
    tight = min(b.d_left.min(), b.d_right.min())
    print(f'closest wall   {tight:.2f} m from raceline '
          f'(car half-width {half:.2f} m -> {tight - half:.2f} m clearance on the raceline)')
    v = ref.v_ref
    print(f'raceline speed min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f} m/s '
          f'(global_scale {p.speed.global_scale}), lap at that speed {np.sum(path.ds / v):.1f} s')

    rng = np.random.default_rng(0)
    s0 = 20.0
    x0, y0, _ = path.frenet_to_cartesian(s0 + 0.3, 0.2)
    sx, sy, _ = path.frenet_to_cartesian(s0 + rng.uniform(0, 8, SCAN_POINTS), rng.uniform(-0.8, 0.8, SCAN_POINTS))
    ps, pd = rng.uniform(s0, s0 + 4, PLAN_POINTS), rng.uniform(-0.6, 0.6, PLAN_POINTS)
    print('timings (this machine):')
    print(f'  cartesian_to_frenet, car pose, windowed   '
          f'{timeit(lambda: path.cartesian_to_frenet(float(x0), float(y0), s_hint=s0, reset_dist=1.0)):.3f} ms')
    print(f'  cartesian_to_frenet, car pose, global     '
          f'{timeit(lambda: path.cartesian_to_frenet(float(x0), float(y0))):.3f} ms')
    print(f'  cartesian_to_frenet, {SCAN_POINTS} scan pts, windowed '
          f'{timeit(lambda: path.cartesian_to_frenet(sx, sy, s_hint=s0, window=SCAN_WINDOW)):.3f} ms')
    print(f'  frenet_to_cartesian, {PLAN_POINTS} pts             '
          f'{timeit(lambda: path.frenet_to_cartesian(ps, pd, d_prime=pd)):.3f} ms')
    print(f'  bounds lookup, {PLAN_POINTS} pts                   '
          f'{timeit(lambda: (b.left(ps), b.right(ps))):.3f} ms')

    if args.plot:
        plot(p, ref, args.plot)
        print(f'plot written to {args.plot}')


def plot(p, ref, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path, b, g = ref.path, ref.bounds, ref.grid
    s = b.s
    lx, ly, _ = path.frenet_to_cartesian(s, b.left(s))
    rx, ry, _ = path.frenet_to_cartesian(s, -b.right(s))
    cx, cy, _ = path.frenet_to_cartesian(s, 0.0)
    km = p.vehicle.kappa_max
    over = np.abs(path.eval(s)[3]) > km

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(1, 2, 1)
    ny, nx = g.occupied.shape
    ext = [g.origin_x, g.origin_x + nx * g.resolution, g.origin_y, g.origin_y + ny * g.resolution]
    ax.imshow(~g.occupied, cmap='gray', origin='lower', extent=ext, vmin=-1, vmax=1)
    ax.plot(cx, cy, 'b-', lw=1, label='raceline (smoothed)')
    ax.plot(lx, ly, 'g-', lw=1, label='d_left bound')
    ax.plot(rx, ry, 'r-', lw=1, label='d_right bound')
    ax.plot(cx[over], cy[over], 'm.', ms=6, label=f'|kappa| > kappa_max ({km:.2f})')
    ax.plot(cx[0], cy[0], 'k*', ms=14, label='s = 0')
    pad = 1.5
    ax.set_xlim(min(lx.min(), rx.min()) - pad, max(lx.max(), rx.max()) + pad)
    ax.set_ylim(min(ly.min(), ry.min()) - pad, max(ly.max(), ry.max()) + pad)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Levine: raceline and ray-cast track bounds')

    ax1 = fig.add_subplot(3, 2, 2)
    ax1.plot(path.s_table, path.kappa_table, 'b-', lw=1)
    ax1.axhline(km, color='m', ls='--', lw=1)
    ax1.axhline(-km, color='m', ls='--', lw=1)
    ax1.set_ylabel('kappa [1/m]')
    ax2 = fig.add_subplot(3, 2, 4, sharex=ax1)
    ax2.plot(s, b.d_left, 'g-', lw=1, label='d_left')
    ax2.plot(s, -b.d_right, 'r-', lw=1, label='-d_right')
    ax2.axhline(0, color='b', lw=0.5)
    ax2.set_ylabel('bound [m]')
    ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(3, 2, 6, sharex=ax1)
    ax3.plot(path.s_table, ref.v_ref, 'k-', lw=1)
    ax3.set_ylabel(f'v_ref [m/s] (x{p.speed.global_scale})')
    ax3.set_xlabel('s [m]')
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == '__main__':
    main()
