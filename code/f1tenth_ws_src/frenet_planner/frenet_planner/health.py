"""Localization health and the follow-the-gap fallback.

The planner is only as good as the pose: every obstacle, bound and path is
placed with it. LocalizationHealth watches the pose covariance and how well
the scan lines up with the static map, with hysteresis, and while it is
unhealthy the car is driven by the team's reactive follow-the-gap, which
needs only the scan.
"""
from types import SimpleNamespace

import numpy as np

PF3X3, ROS6X6 = 'pf3x3', 'ros6x6'
# Indices of var_x, var_y, var_yaw in the flat covariance
COV_INDEX = {PF3X3: (0, 4, 8), ROS6X6: (0, 7, 35)}


def pose_std(covariance, layout):
    """(position std, yaw std) from a flat pose covariance; None if it is all
    zero (no covariance given, as in the sim)."""
    c = np.asarray(covariance, dtype=float)
    if not np.any(c):
        return None
    ix, iy, iyaw = COV_INDEX[layout]
    return float(np.sqrt(max(c[ix] + c[iy], 0.0))), float(np.sqrt(max(c[iyaw], 0.0)))


class LocalizationHealth:

    def __init__(self, params):
        """params: HealthParams."""
        self.p = params
        self.healthy = True
        self.bad_run = 0
        self.good_run = 0
        self.reason = ''
        self.std = None
        self.mismatch = 0.0

    def set_pose_std(self, std):
        self.std = std

    def update(self, mismatch, n_points):
        """Fold in one scan's mismatch fraction; returns whether healthy."""
        p = self.p
        if n_points < p.min_scan_points:
            return self.healthy
        self.mismatch = mismatch
        reasons = []
        if self.std is not None and self.std[0] > p.max_position_std:
            reasons.append(f'position std {self.std[0]:.2f} m')
        if self.std is not None and self.std[1] > p.max_yaw_std:
            reasons.append(f'yaw std {self.std[1]:.2f} rad')
        if mismatch > p.max_mismatch_fraction:
            reasons.append(f'scan-map mismatch {100 * mismatch:.0f}%')
        if reasons:
            self.bad_run, self.good_run = self.bad_run + 1, 0
            self.reason = ', '.join(reasons)
            if self.bad_run >= p.unhealthy_after:
                self.healthy = False
        else:
            self.good_run, self.bad_run = self.good_run + 1, 0
            if self.good_run >= p.healthy_after:
                self.healthy = True
        return self.healthy


class _Capture:
    """Stands in for the node's drive publisher and keeps the last message."""

    def __init__(self):
        self.msg = None

    def publish(self, msg):
        self.msg = msg


class GapFollowAdapter:
    """Runs the team's ReactiveFollowGap.lidar_callback unchanged, without
    starting a ROS node: the object is created without __init__ (which only
    makes the publisher and subscriber), given the tuning attributes its
    __init__ would set, and a publisher stand-in that captures the command.

    reactive_module: the imported gap_follow reactive_node module.
    """

    def __init__(self, reactive_module, params):
        cls = reactive_module.ReactiveFollowGap
        self.ftg = cls.__new__(cls)
        self.ftg.range_max = params.range_max
        self.ftg.bubble_radius = params.bubble_radius
        self.ftg.corner_safe_dist = params.corner_safe_dist
        self.ftg.fov_half = np.radians(params.fov_half_deg)
        self.ftg.drivexpub = _Capture()

    def drive(self, ranges, angle_min, angle_increment):
        """(steering angle, speed), or None if it produced no command."""
        self.ftg.drivexpub.msg = None
        self.ftg.lidar_callback(SimpleNamespace(ranges=ranges, angle_min=angle_min,
                                                angle_increment=angle_increment))
        m = self.ftg.drivexpub.msg
        return None if m is None else (float(m.drive.steering_angle), float(m.drive.speed))
