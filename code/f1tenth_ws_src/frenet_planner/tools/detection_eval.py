#!/usr/bin/env python3
"""Offline check of obstacle detection and tracking on f1tenth_gym scans.

No ROS: scans come straight from f110_gym's ScanSimulator2D on the sim map,
with test boxes painted into a copy of it, and opponents ray-cast the same
way the gym does (from the ego's base_link, not its laser; see --no-gym-quirk).
The ego car is moved along the raceline at a set speed and offset, and every
scan goes through the planner's Detector and ObstacleTracker exactly as the
node runs them, at the bridge's reported angle increment.

Reports per scenario: how often a truly present obstacle is reported,
position and velocity error, time to confirm, false positives, and the
detect + track time per scan.

Run with the system python (ROS Humble's), not conda:
    /usr/bin/python3 tools/detection_eval.py
"""
import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f110_gym.envs.collision_models import get_vertices  # noqa: E402
from f110_gym.envs.laser_models import ScanSimulator2D, ray_cast  # noqa: E402

from frenet_planner.detection import Detector  # noqa: E402
from frenet_planner.frenet import s_diff  # noqa: E402
from frenet_planner.params import Params  # noqa: E402
from frenet_planner.reference import build_reference  # noqa: E402
from frenet_planner.tracking import ObstacleTracker  # noqa: E402
from sim_common import BOX_SIZE, frenet_box_polygon, longest_straight, paint_map  # noqa: E402

# f1tenth_gym defaults (f110_env.py, sim.yaml)
N_BEAMS = 1080
FOV = 4.7
SCAN_NOISE = 0.01
CAR_LENGTH, CAR_WIDTH = 0.58, 0.31
SCAN_RATE = 40.0                       # [Hz]

# Evaluation windows
EVAL_GAP = (0.5, 4.5)                  # obstacle rear face this far ahead of the ego's nose [m]
MATCH_S = 0.35                         # a track within this of the truth in s, ...
MATCH_D = 0.05                         # ... and overlapping it in d to within this (a map cell), matches [m]
VEL_SETTLE = 0.5                       # velocity error measured from this long after confirmation [s]


@dataclass
class Truth:
    """An obstacle's rear face s(t), d range, and speed along s."""
    s0: float
    d_min: float
    d_max: float
    v_s: float = 0.0

    def s_at(self, t, length):
        return (self.s0 + self.v_s * t) % length


@dataclass
class Scenario:
    name: str
    boxes: list = field(default_factory=list)       # (s_centre, d_centre)
    opponent: tuple = None                           # (s0 of its centre, speed)
    ego_s0: float = 0.0
    ego_d: float = 0.0
    ego_speed: float = 2.0
    duration: float = 6.0
    follow_gap: float = None                         # ego slows to the opponent's speed at this gap [m]
    pose_noise: tuple = (0.0, 0.0)                   # sigma of (xy [m], yaw [rad])


def paint_temp_map(clean_yaml, out_dir, polygons):
    return paint_map(clean_yaml, os.path.join(out_dir, f'map_{len(os.listdir(out_dir))}'), polygons)


def run(scenario, ref, params, map_dir, gym_quirk, seed, verbose=False):
    path = ref.path
    L = path.length
    rng = np.random.default_rng(seed)

    truths, polygons = [], []
    for s_c, d_c in scenario.boxes:
        polygons.append(frenet_box_polygon(path, s_c, d_c, BOX_SIZE, BOX_SIZE))
        truths.append(Truth(s_c - BOX_SIZE / 2, d_c - BOX_SIZE / 2, d_c + BOX_SIZE / 2))
    if scenario.opponent is not None:
        s_opp, v_opp = scenario.opponent
        truths.append(Truth(s_opp - CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_WIDTH / 2, v_opp))

    sim = ScanSimulator2D(N_BEAMS, FOV)
    sim.set_map(paint_temp_map(params.map.yaml_file, map_dir, polygons) if polygons else params.map.yaml_file,
                '.png')
    scan_angles = -FOV / 2 + np.arange(N_BEAMS) * sim.get_increment()
    reported_inc = FOV / N_BEAMS              # what gym_bridge puts in LaserScan

    detector = Detector(path, ref.bounds, ref.grid, params.detection)
    tracker = ObstacleTracker(L, params.tracking)
    laser = params.detection.laser_offset_x

    in_range = matched = 0
    s_err, d_err, v_err = [], [], []
    fp_ids, fp_frames = set(), 0
    first_in_range, confirm_delay = {}, {}
    cpu = []
    s_ego, v_ego = scenario.ego_s0, scenario.ego_speed
    n_scans = int(scenario.duration * SCAN_RATE)
    dt = 1.0 / SCAN_RATE
    for k in range(n_scans):
        t = k * dt
        x, y, yaw = (float(v) for v in path.frenet_to_cartesian(s_ego, scenario.ego_d))
        laser_pose = np.array([x + laser * np.cos(yaw), y + laser * np.sin(yaw), yaw])
        ranges = sim.scan(laser_pose, rng, std_dev=SCAN_NOISE)
        if scenario.opponent is not None:
            tr = truths[-1]
            ox, oy, oyaw = path.frenet_to_cartesian(tr.s_at(t, L) + CAR_LENGTH / 2, 0.0)
            verts = get_vertices(np.array([float(ox), float(oy), float(oyaw)]), CAR_LENGTH, CAR_WIDTH)
            origin = np.array([x, y, yaw]) if gym_quirk else laser_pose
            ranges = ray_cast(origin, ranges, scan_angles, verts)

        sxy, syaw = scenario.pose_noise
        pose = (x + rng.normal(0, sxy), y + rng.normal(0, sxy), yaw + rng.normal(0, syaw))
        t0 = time.perf_counter()
        res = detector.detect(ranges, -FOV / 2, reported_inc, 0.0, pose, s_ego)
        tracker.step(res.detections, t)
        cpu.append(time.perf_counter() - t0)

        confirmed = tracker.confirmed()
        nose = s_ego + CAR_LENGTH / 2
        used = set()
        for j, tru in enumerate(truths):
            s_true = tru.s_at(t, L)
            gap = s_diff(s_true, nose, L)
            best = None
            for trk in confirmed:
                if (abs(s_diff(trk.x[0], s_true, L)) < MATCH_S
                        and trk.d_max > tru.d_min - MATCH_D and trk.d_min < tru.d_max + MATCH_D):
                    best = trk
            if best is not None:
                used.add(best.id)
            if not EVAL_GAP[0] <= gap <= EVAL_GAP[1]:
                continue
            in_range += 1
            first_in_range.setdefault(j, t)
            if best is None:
                continue
            matched += 1
            confirm_delay.setdefault(j, t - first_in_range[j])
            s_err.append(s_diff(best.x[0], s_true, L))
            d_err.append(0.5 * (best.d_min + best.d_max) - 0.5 * (tru.d_min + tru.d_max))
            if t - first_in_range[j] - confirm_delay[j] >= VEL_SETTLE:
                v_err.append(best.x[2] - tru.v_s)
        fps = [trk.id for trk in confirmed if trk.id not in used]
        if verbose and fps:
            for trk in confirmed:
                if trk.id in fps:
                    near = min(truths, key=lambda tr: abs(s_diff(trk.x[0], tr.s_at(t, L), L)), default=None)
                    rel = f', s - nearest truth {s_diff(trk.x[0], near.s_at(t, L), L):+.2f}' if near else ''
                    print(f'    t {t:5.2f} unmatched #{trk.id}: {s_diff(trk.x[0], nose, L):+.2f} m ahead of nose, '
                          f'd {trk.d_min:+.2f}..{trk.d_max:+.2f}, v_s {trk.x[2]:+.2f}{rel}')
        fp_ids.update(fps)
        fp_frames += bool(fps)

        if scenario.follow_gap is not None:
            gap = s_diff(truths[-1].s_at(t, L), nose, L)
            v_ego = truths[-1].v_s if gap < scenario.follow_gap else scenario.ego_speed
        s_ego = path.wrap(s_ego + v_ego * dt)

    def stats(a):
        a = np.abs(np.asarray(a))
        return f'mean {a.mean():.3f} max {a.max():.3f}' if len(a) else 'n/a'

    cpu = np.array(cpu) * 1e3
    print(f'\n== {scenario.name}')
    if truths:
        print(f'  reported while {EVAL_GAP[0]}-{EVAL_GAP[1]} m ahead: {matched}/{in_range} scans '
              f'({100.0 * matched / max(in_range, 1):.1f}%)')
        print('  time to confirm: ' + ', '.join(f'{v * 1e3:.0f} ms' for v in confirm_delay.values()))
        print(f'  |s error| {stats(s_err)}   |d error| {stats(d_err)}   [m]')
        if any(tr.v_s for tr in truths):
            print(f'  |v_s error| after {VEL_SETTLE} s: {stats(v_err)} [m/s]')
    print(f'  false positives: {len(fp_ids)} tracks, in {fp_frames}/{n_scans} scans')
    print(f'  detect + track: mean {cpu.mean():.2f} ms, p95 {np.percentile(cpu, 95):.2f} ms, max {cpu.max():.2f} ms')
    return dict(matched=matched, in_range=in_range, fp_tracks=len(fp_ids), fp_frames=fp_frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_ROOT, 'config', 'frenet_planner.yaml'))
    ap.add_argument('--no-gym-quirk', action='store_true',
                    help='ray-cast the opponent from the laser, as a real LiDAR would')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--only', help='run only scenarios whose name contains this')
    ap.add_argument('--verbose', action='store_true', help='print every unmatched (false positive) track')
    args = ap.parse_args()

    params = Params.from_yaml(args.config)
    ref = build_reference(params)
    path = ref.path
    s_mid, straight_len = longest_straight(path)
    print(f'track {path.length:.1f} m; longest straight {straight_len:.1f} m, middle at s = {s_mid:.1f}')
    left, right = float(ref.bounds.left(s_mid)), float(ref.bounds.right(s_mid))
    print(f'  half-widths there: left {left:.2f} m, right {right:.2f} m')
    gym = not args.no_gym_quirk
    print(f'opponent ray-cast from {"base_link (gym behaviour)" if gym else "the laser"}')

    approach = s_mid - 6.0
    scenarios = [
        Scenario('(a) box mid-straight, left; ego passes on the right', boxes=[(s_mid, 0.25)],
                 ego_s0=approach, ego_d=-0.30, duration=4.5),
        Scenario('(b) box mid-straight, right; ego passes on the left', boxes=[(s_mid, -0.25)],
                 ego_s0=approach, ego_d=0.30, duration=4.5),
        Scenario('(c) opponent on the raceline at 1.0 m/s; ego closes at 2.0 then trails at 1.2 m',
                 opponent=(approach + 5.0, 1.0), ego_s0=approach, duration=8.0, follow_gap=1.2),
        Scenario('(d) clean lap, perfect localization', ego_s0=0.0, duration=path.length / 2.0),
        Scenario('(e) clean lap, pose noise 3 cm / 0.6 deg', ego_s0=0.0, duration=path.length / 2.0,
                 pose_noise=(0.03, 0.01)),
    ]
    with tempfile.TemporaryDirectory() as map_dir:
        for sc in scenarios:
            if args.only is None or args.only in sc.name:
                run(sc, ref, params, map_dir, gym, args.seed, args.verbose)


if __name__ == '__main__':
    main()
