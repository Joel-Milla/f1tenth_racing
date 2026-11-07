"""The planner's full cycle without ROS: pose -> scan -> plan -> control.

The ROS node and the offline closed-loop evaluation (tools/sim_eval.py) both
drive this class, so what is evaluated offline is what runs on the car.

The path follower is passed in: any object with lookahead_point(path, pose,
L, loop), closest_index(path, pose) and steering_angle(pose, target), i.e.
the team's PurePursuitController. So is the fallback driver: any object with
drive(ranges, angle_min, angle_increment) -> (steer, speed) or None, i.e.
health.GapFollowAdapter around the team's follow-the-gap. Without one, an
unhealthy localization stops the car.
"""
import math
import time
from collections import deque

import numpy as np

from frenet_planner.behavior import Behavior
from frenet_planner.collision import ObstacleSet, car_footprint
from frenet_planner.detection import Detector
from frenet_planner.frenet import wrap_angle
from frenet_planner.health import LocalizationHealth
from frenet_planner.planner import FrenetState
from frenet_planner.reference import build_reference
from frenet_planner.tracking import ObstacleTracker

# Planning times kept for the timing summary
TIMING_WINDOW = 500


class PlannerCore:

    def __init__(self, params, controller, fallback_driver=None):
        self.params = params
        self.ref = build_reference(params)
        path = self.ref.path
        vp, cp = params.vehicle, params.collision
        self.footprint = car_footprint(vp.length, vp.width, cp.n_circles, cp.footprint_center_offset)
        self.detector = Detector(path, self.ref.bounds, self.ref.grid, params.detection)
        self.tracker = ObstacleTracker(path.length, params.tracking)
        self.controller = controller
        self.fallback_driver = fallback_driver
        self.behavior = Behavior(params, path.length)
        self.health = LocalizationHealth(params.health)
        self.fallback_cmd = None     # follow-the-gap command for the latest scan
        self.fallback_active = False

        self.pose = None             # (x, y, yaw)
        self.speed = 0.0
        self.s = None
        self.d = None
        self.heading_err = 0.0
        self.progress = 0.0          # arc length driven, not wrapped [m]
        self.test_obstacles = ObstacleSet.from_boxes([])
        self.last_detection = None
        self.plan = None
        self.plan_times = deque(maxlen=TIMING_WINDOW)

    # --- inputs --------------------------------------------------------------

    def update_pose(self, x, y, yaw, speed, pose_std=None):
        """pose_std: (position std, yaw std) of the localization, or None."""
        self.health.set_pose_std(pose_std)
        path = self.ref.path
        s, d = path.cartesian_to_frenet(x, y, s_hint=self.s, reset_dist=self.params.raceline.reset_dist)
        if self.s is not None:
            self.progress += float(path.s_diff(s, self.s))
        self.pose, self.speed, self.s, self.d = (x, y, yaw), speed, s, d
        self.heading_err = float(wrap_angle(yaw - path.eval(s)[2]))

    def update_scan(self, ranges, angle_min, angle_increment, range_min, t):
        if self.pose is None:
            return None
        res = self.detector.detect(ranges, angle_min, angle_increment, range_min, self.pose, self.s,
                                   self.params.health.mismatch_dist)
        self.tracker.step(res.detections, t)
        self.last_detection = res
        healthy = self.health.update(res.mismatch, res.n_points) or not self.params.health.fallback_enabled
        if not healthy:
            self.fallback_active = True
        elif self.fallback_active and self.can_hand_back():
            self.fallback_active = False
            self.reset_planning()
        if self.fallback_active and self.fallback_driver is not None:
            self.fallback_cmd = self.fallback_driver.drive(ranges, angle_min, angle_increment)
        return res

    def can_hand_back(self):
        """The planner takes back over only from a state it can plan from:
        inside the track bounds and roughly along the raceline. Follow-the-
        gap may leave the car angled at a wall, and a sampling planner
        restarted there has no feasible path back."""
        b = self.ref.bounds
        return (-float(b.right(self.s)) <= self.d <= float(b.left(self.s))
                and abs(self.heading_err) <= self.params.health.handback_max_heading)

    def reset_planning(self):
        """Forget everything built on the pose while it was bad: tracks
        (walls seen through a wrong pose look like obstacles), the last
        path (d' and d'' come from it) and the behaviour state."""
        p = self.params
        self.tracker = ObstacleTracker(self.ref.path.length, p.tracking)
        self.behavior = Behavior(p, self.ref.path.length)
        self.plan = None

    def add_test_obstacle(self, x, y):
        """Square test box centred on (x, y), for RViz clicks."""
        tp, path = self.params.test_obstacles, self.ref.path
        s, d = path.cartesian_to_frenet(x, y)
        half = 0.5 * tp.size
        box = (float(path.wrap(s - half)), tp.size, d - half, d + half, 0.0, 0.0)
        o = ObstacleSet.concat(self.test_obstacles, ObstacleSet.from_boxes([box]))
        self.test_obstacles = o.subset(np.arange(len(o)) >= len(o) - tp.max_count)
        return s, d

    def obstacles(self):
        return ObstacleSet.concat(self.test_obstacles, self.tracker.obstacle_set())

    # --- planning ------------------------------------------------------------

    def frenet_state(self):
        """Measured s and d; d' and d'' from last cycle's chosen path at the
        car's s, unless the car has strayed from it (or there is none), in
        which case d' comes from the heading and d'' = 0."""
        path, prev = self.ref.path, self.plan
        if prev is not None:
            u = float(path.s_diff(self.s, prev.s0))
            if prev.u[0] <= u <= prev.u[-1] and \
                    abs(np.interp(u, prev.u, prev.d) - self.d) <= self.params.planner.state_reset_dist:
                return FrenetState(self.s, self.d, float(np.interp(u, prev.u, prev.d1)),
                                   float(np.interp(u, prev.u, prev.d2)))
        kappa = float(path.eval(self.s)[3])
        return FrenetState(self.s, self.d, (1.0 - kappa * self.d) * math.tan(self.heading_err), 0.0)

    def step_plan(self):
        if self.pose is None:
            return None
        t0 = time.perf_counter()
        prev_d_end = self.plan.d_end if self.plan is not None else 0.0
        self.plan = self.behavior.step(self.frenet_state(), prev_d_end, self.speed, self.ref,
                                       self.obstacles(), self.footprint)
        self.plan_times.append(time.perf_counter() - t0)
        return self.plan

    # --- control -------------------------------------------------------------

    def lookahead(self):
        c = self.params.control
        return float(np.clip(c.lookahead_gain * self.speed + c.lookahead_base, c.lookahead_min, c.lookahead_max))

    @property
    def using_fallback(self):
        return self.fallback_active

    def control(self):
        """(steering angle, speed, lookahead point) on the current plan, or
        None before the first plan. While localization is unhealthy:
        follow-the-gap's command (lookahead point None), or a stop if there
        is no fallback driver or it gave no command."""
        if self.fallback_active:
            if self.fallback_driver is None or self.fallback_cmd is None:
                return 0.0, 0.0, None
            return self.fallback_cmd[0], self.fallback_cmd[1], None
        if self.plan is None or self.pose is None:
            return None
        p = self.plan
        xy = np.column_stack([p.x, p.y])
        target = self.controller.lookahead_point(xy, self.pose, self.lookahead(), False)
        steer = self.controller.steering_angle(self.pose, target)
        k = self.controller.closest_index(xy, self.pose)
        return steer, float(p.v[k]), target
