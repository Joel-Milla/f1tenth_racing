#!/usr/bin/env python3
"""Closed-loop laps in f1tenth_gym without ROS.

The car is driven by PlannerCore (the same code the ROS node runs), the
team's PurePursuitController and, when localization is unhealthy, the team's
follow-the-gap, in f110_gym with the same laser offset as gym_bridge.

Scenarios
  boxes     box mid-straight, on the left one lap and the right the next (the
            simulator map is switched half a lap away from the box)
  clean     no obstacles: raceline laps for reference
  opponent  a second car on the raceline, slower (--opp-scale x our raceline
            speed), starting --opp-gap ahead; it does not react to us
  locfail   clean map; for --fail-time s the planner's pose is offset by
            --fail-offset (x, y, yaw), as a localization failure would. The
            simulator keeps the true pose, so collisions are real

Reports per lap: time, collisions, minimum distance between the car's body
and the box or opponent, states and modes used; overall the planning time.

config/sim_overrides.yaml is applied on top of --config, as the launch file
does with use_sim:=true. Needs ROS sourced (the team's nodes import rclpy)
and the system python:
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 tools/sim_eval.py --scenario boxes --laps 4
Run tools/make_test_maps.py first for 'boxes'.
"""
import argparse
import os
import sys
from collections import Counter

import gym
import numpy as np
import yaml

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.abspath(os.path.join(PKG_ROOT, '..'))
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SRC, 'pure_pursuit', 'scripts'))
sys.path.insert(0, os.path.join(SRC, 'gap_follow', 'scripts'))

try:
    import reactive_node  # noqa: E402
    from pure_pursuit_node import PurePursuitController  # noqa: E402
except ImportError as e:
    sys.exit(f'cannot import the team nodes ({e}); source /opt/ros/humble/setup.bash first')

from f110_gym.envs.collision_models import get_vertices  # noqa: E402

from frenet_planner.core import PlannerCore  # noqa: E402
from frenet_planner.health import GapFollowAdapter  # noqa: E402
from frenet_planner.params import Params  # noqa: E402
from sim_common import polygon_distance  # noqa: E402

SIM_YAML = os.path.join(SRC, 'f1tenth_gym_ros', 'config', 'sim.yaml')
MAP_DIR = os.path.join(PKG_ROOT, 'maps')
DT = 0.01                    # f110_gym physics step, as gym_bridge [s]
FOV, N_BEAMS = 4.7, 1080
CAR_LENGTH, CAR_WIDTH = 0.58, 0.31
STALL_SPEED = 0.05           # slower than this ... [m/s]
STALL_TIME = 3.0             # ... for this long is a DNF [s]
LAP_TIME_LIMIT = 120.0       # a lap slower than this is a DNF [s]
OPP_LOOKAHEAD = (0.3, 0.6)   # opponent pure pursuit L = a v + b [s, m]
SCENARIOS = ('boxes', 'clean', 'opponent', 'locfail')


class Opponent:
    """Second car: pure pursuit on the raceline at a fraction of its speed."""

    def __init__(self, core, scale, controller):
        self.path, self.ref = core.ref.path, core.ref
        self.xy = np.column_stack([self.path.x_table, self.path.y_table])
        self.scale = scale
        self.pp = controller
        self.s = None
        self.progress = 0.0

    def command(self, x, y, yaw, v):
        s, _ = self.path.cartesian_to_frenet(x, y, s_hint=self.s)
        if self.s is not None:
            self.progress += float(self.path.s_diff(s, self.s))
        self.s = s
        L = OPP_LOOKAHEAD[0] * v + OPP_LOOKAHEAD[1]
        target = self.pp.lookahead_point(self.xy, (x, y, yaw), L, True)
        return self.pp.steering_angle((x, y, yaw), target), self.scale * float(self.ref.speed(s))


def merge(base, over):
    """Recursively overwrite base with over (dicts), like layered ROS params."""
    for k, v in over.items():
        if isinstance(v, dict):
            merge(base[k], v)
        else:
            base[k] = v


def new_stat():
    return dict(min_obs=np.inf, collided=False, states=Counter(), max_abs_d=0.0, overtakes=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_ROOT, 'config', 'frenet_planner.yaml'))
    ap.add_argument('--scenario', choices=SCENARIOS, default='boxes')
    ap.add_argument('--laps', type=int, default=4)
    ap.add_argument('--scan-rate', type=float, default=40.0, help='scans fed to the planner [Hz]')
    ap.add_argument('--opp-scale', type=float, default=0.6, help='opponent speed / our raceline speed')
    ap.add_argument('--opp-gap', type=float, default=4.0, help='opponent start, ahead of us along s [m]')
    ap.add_argument('--fail-start', type=float, default=8.0, help='localization failure starts [s]')
    ap.add_argument('--fail-time', type=float, default=3.0, help='localization failure lasts [s]')
    ap.add_argument('--fail-offset', type=float, nargs=3, default=(0.0, 0.5, 0.15),
                    help='pose error during the failure: x [m], y [m], yaw [rad]')
    ap.add_argument('--trace', type=float, nargs=2, metavar=('T0', 'T1'),
                    help='print every planning cycle between these sim times [s]')
    ap.add_argument('--set', nargs=2, action='append', default=[], metavar=('SECTION.KEY', 'VALUE'),
                    help='override a parameter, e.g. --set detection.max_cluster_size 0.7')
    args = ap.parse_args()

    with open(args.config) as f:
        doc = yaml.safe_load(f)
    flat = doc['frenet_planner']['ros__parameters']
    with open(os.path.join(PKG_ROOT, 'config', 'sim_overrides.yaml')) as f:
        merge(flat, yaml.safe_load(f)['frenet_planner']['ros__parameters'])
    for key, value in args.set:
        *parents, leaf = key.split('.')
        d = flat
        for p in parents:
            d = d[p]
        d[leaf] = type(d[leaf])(yaml.safe_load(value))
    params = Params.from_dict(flat)
    with open(SIM_YAML) as f:
        sim = yaml.safe_load(f)['bridge']['ros__parameters']
    laser = sim['scan_distance_to_base_link']
    assert abs(laser - params.detection.laser_offset_x) < 1e-9, 'laser offset differs from sim.yaml'

    clean_map = os.path.splitext(params.map.yaml_file)[0]
    maps, boxes = [clean_map], [None]
    if args.scenario == 'boxes':
        with open(os.path.join(MAP_DIR, 'test_boxes.yaml')) as f:
            meta = yaml.safe_load(f)
        names = sorted(meta)                      # *_box_left, *_box_right
        maps = [os.path.join(MAP_DIR, n) for n in names]
        boxes = [np.array(meta[n]['corners']) for n in names]

    vp = params.vehicle
    ftg = GapFollowAdapter(reactive_node, params.gap_follow)
    core = PlannerCore(params, PurePursuitController(vp.wheelbase, vp.max_steer), ftg)
    path = core.ref.path
    L = path.length
    start = np.array([sim['sx'], sim['sy'], sim['stheta']])
    opponent = None
    if args.scenario == 'opponent':
        opponent = Opponent(core, args.opp_scale, PurePursuitController(vp.wheelbase, vp.max_steer))
        s0, _ = path.cartesian_to_frenet(start[0], start[1])
        ox, oy, oyaw = (float(v) for v in path.frenet_to_cartesian(s0 + args.opp_gap, 0.0))
        poses = np.array([start, [ox, oy, oyaw]])
    else:
        poses = start[None, :]
    env = gym.make('f110_gym:f110-v0', map=maps[0], map_ext='.png', num_agents=len(poses), timestep=DT,
                   lidar_dist=laser)
    obs, _, done, _ = env.reset(poses)

    def every(hz):
        return max(1, int(round(1.0 / (hz * DT))))
    scan_every, plan_every, control_every = every(args.scan_rate), every(params.planner.rate), \
        every(params.control.rate)

    opp_info = f'; opponent at {args.opp_scale:.2f} x raceline speed, {args.opp_gap:.1f} m ahead' if opponent else ''
    print(f'scenario {args.scenario}: {args.laps} laps on {", ".join(os.path.basename(m) for m in maps)}; '
          f'scan {1 / (scan_every * DT):.0f} Hz, plan {1 / (plan_every * DT):.0f} Hz, '
          f'control {1 / (control_every * DT):.0f} Hz{opp_info}')
    lap, lap_start_t, active, switched = 0, 0.0, 0, False
    steer = speed = 0.0
    opp_cmd = (0.0, 0.0)
    laps, stat = [], new_stat()
    still_since, ahead_prev = None, None
    fail = dict(first_bad=None, recovered=None)
    k = 0
    while lap < args.laps:
        t = k * DT
        x, y, yaw = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        v = obs['linear_vels_x'][0]
        failing = args.scenario == 'locfail' and args.fail_start <= t < args.fail_start + args.fail_time
        if failing:
            fx, fy, fyaw = args.fail_offset
            core.update_pose(x + fx, y + fy, yaw + fyaw, v)
        else:
            core.update_pose(x, y, yaw, v)
        if k % scan_every == 0:
            core.update_scan(obs['scans'][0], -FOV / 2, FOV / N_BEAMS, 0.0, t)
            if args.scenario == 'locfail':
                if core.using_fallback and fail['first_bad'] is None:
                    fail['first_bad'] = t
                if fail['first_bad'] is not None and not core.using_fallback and fail['recovered'] is None:
                    fail['recovered'] = t
        if k % plan_every == 0:
            core.step_plan()
            stat['states']['FOLLOW_GAP' if core.using_fallback else
                           f'{core.behavior.state}/{core.plan.mode}'] += 1
            if args.trace and args.trace[0] <= t <= args.trace[1]:
                trace(t, core, obs, opponent)
        if k % control_every == 0:
            out = core.control()
            if out is not None:
                steer, speed, _ = out
            if opponent is not None:
                opp_cmd = opponent.command(obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1],
                                           obs['linear_vels_x'][1])

        body = get_vertices(np.array([x, y, yaw]), CAR_LENGTH, CAR_WIDTH)
        if boxes[active] is not None:
            stat['min_obs'] = min(stat['min_obs'], polygon_distance(body, boxes[active]))
        if opponent is not None:
            opp_body = get_vertices(np.array([obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]]),
                                    CAR_LENGTH, CAR_WIDTH)
            stat['min_obs'] = min(stat['min_obs'], polygon_distance(body, opp_body))
            # Overtaken once our rear is past its front
            ahead = core.progress - CAR_LENGTH > opponent.progress + args.opp_gap
            if ahead_prev is not None and ahead and not ahead_prev:
                stat['overtakes'] += 1
            ahead_prev = ahead
        stat['max_abs_d'] = max(stat['max_abs_d'], abs(path.cartesian_to_frenet(x, y, s_hint=core.s)[1]))
        stat['collided'] |= bool(obs['collisions'][0])
        still_since = (still_since if still_since is not None else t) if abs(v) < STALL_SPEED and k > 100 else None

        done_lap = core.progress >= (lap + 1) * L
        dnf = stat['collided'] or (still_since is not None and t - still_since > STALL_TIME) \
            or t - lap_start_t > LAP_TIME_LIMIT
        if done_lap or dnf:
            laps.append(dict(lap=lap + 1, map=os.path.basename(maps[active]), time=t - lap_start_t,
                             clean=done_lap and not stat['collided'], **stat))
            print_lap(laps[-1], dnf and not done_lap)
            if dnf and not done_lap:
                break
            lap, lap_start_t, switched, stat = lap + 1, t, False, new_stat()
        if len(maps) > 1 and not switched and core.progress >= lap * L + 0.5 * L:
            active = lap % len(maps)
            env.update_map(maps[active] + '.yaml', '.png')
            switched = True

        action = [[steer, speed]] + ([list(opp_cmd)] if opponent is not None else [])
        obs, _, done, _ = env.step(np.array(action))
        k += 1

    times = np.array(core.plan_times) * 1e3
    clean = [lp['clean'] for lp in laps]
    best_run = max((len(r) for r in ''.join('1' if c else '0' for c in clean).split('0')), default=0)
    print(f'\nclean laps {sum(clean)}/{len(laps)}, longest run of consecutive clean laps: {best_run}')
    if args.scenario == 'locfail':
        fb, rec = fail['first_bad'], fail['recovered']
        print(f'localization failure {args.fail_start:.1f}-{args.fail_start + args.fail_time:.1f} s: '
              + (f'follow-the-gap from {fb:.2f} s ({1e3 * (fb - args.fail_start):.0f} ms after it began), '
                 if fb is not None else 'never detected, ')
              + (f'planner back at {rec:.2f} s' if rec is not None else 'never recovered'))
    print(f'planning cycle (last {len(times)}): mean {times.mean():.2f} ms, p95 {np.percentile(times, 95):.2f} ms, '
          f'max {times.max():.2f} ms')


def trace(t, core, obs, opponent):
    pl, b = core.plan, core.behavior
    line = (f'  t {t:6.2f} {b.state:8s}/{pl.mode:8s} ego s {core.s:6.2f} d {core.d:+.2f} v {core.speed:.2f} '
            f'| d_end {pl.d_end:+.2f} v_cmd {pl.v[0]:.2f}')
    if opponent is not None:
        path = core.ref.path
        os_, od = path.cartesian_to_frenet(obs['poses_x'][1], obs['poses_y'][1], s_hint=opponent.s)
        line += (f' | opp true s {os_:6.2f} d {od:+.2f} v {obs["linear_vels_x"][1]:.2f} '
                 f'gap {float(path.s_diff(os_, core.s)) - CAR_LENGTH:+.2f}')
    o = core.obstacles()
    line += ' | tracks ' + ' '.join(f'[s {a:.2f} len {ln:.2f} d {lo:+.2f}..{hi:+.2f} v {vs:+.2f}]'
                                    for a, ln, lo, hi, vs in zip(o.s_min, o.s_len, o.d_min, o.d_max, o.v_s))
    if b.why:
        line += ' | ' + b.why
        b.why = ''
    print(line)


def print_lap(lp, dnf):
    states = ', '.join(f'{m} {n}' for m, n in sorted(lp['states'].items()))
    status = 'DNF' if dnf else ('clean' if lp['clean'] else 'COLLISION')
    obs = f"min distance to box/opponent {lp['min_obs']:.3f} m, " if np.isfinite(lp['min_obs']) else ''
    ot = f"overtakes {lp['overtakes']}, " if lp['overtakes'] else ''
    print(f"lap {lp['lap']} [{lp['map']}]: {status}, {lp['time']:.2f} s, {obs}{ot}"
          f"max |d| {lp['max_abs_d']:.2f} m\n    plan cycles by state/mode: {states}")


if __name__ == '__main__':
    main()
