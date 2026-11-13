#!/usr/bin/env python3
'''
Smoother, more accurate pure pursuit. Builds on pure_pursuit_node.py (left unchanged).

Fixes over PurePursuitController:
    - geometry is computed from the rear axle: on the car the particle filter pose
      is the laser frame, 0.27 m ahead of the rear axle, which made the controller
      oversteer and cut corners;
    - the closest waypoint is tracked in a window around the previous one instead
      of a global search, so it cannot jump to another part of the track;
    - the lookahead point is searched forward from the closest point without
      reordering the path;
    - the steering command is low-pass filtered and rate limited.
'''
import math

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray

from pure_pursuit_node import (CAR_POSE_TOPIC, DEFAULT_WAYPOINTS, SIM_POSE_TOPIC,
                               PurePursuitController, delete_marker, drive_msg,
                               global_to_local, line_marker, local_to_global, make_marker,
                               pose_from_msg, resample_polyline, sphere_marker)

# distance from the laser (particle filter pose) back to the rear axle on the car
LASER_TO_REAR_AXLE = 0.27


class SmoothPurePursuit(PurePursuitController):

    def __init__(self, wheelbase=0.33, max_steer=0.36, rear_axle_offset=LASER_TO_REAR_AXLE,
                 steer_tau=0.05, max_steer_rate=3.0, track_window=2.0):
        super().__init__(wheelbase, max_steer)
        self.rear_axle_offset = rear_axle_offset
        self.steer_tau = steer_tau
        self.max_steer_rate = max_steer_rate
        self.track_window = track_window
        self.tracked = {}
        self.last_steer = 0.0
        self.last_time = None

    def rear_axle(self, pose):
        x, y, th = pose
        return (x - self.rear_axle_offset * math.cos(th),
                y - self.rear_axle_offset * math.sin(th), th)

    def track_index(self, path, pose, loop, key):
        '''Closest path index to the rear axle, searched near the previous one.'''
        x, y, _ = self.rear_axle(pose)
        n = len(path)
        prev = self.tracked.get(key)
        if prev is not None and prev[0] is path:
            ds = max(math.hypot(*(path[min(1, n - 1)] - path[0])), 1e-3)
            w = max(int(self.track_window / ds), 1)
            idx = np.arange(prev[1] - w // 4, prev[1] + w + 1)
            idx = idx % n if loop else idx[(idx >= 0) & (idx < n)]
            d2 = (path[idx, 0] - x) ** 2 + (path[idx, 1] - y) ** 2
            k = int(np.argmin(d2))
            # lost (pose jump or relocalization): fall back to a global search
            if d2[k] < (self.track_window / 2) ** 2:
                i = int(idx[k])
                self.tracked[key] = (path, i)
                return i
        i = self.closest_index(path, (x, y))
        self.tracked[key] = (path, i)
        return i

    def lookahead_point(self, path, pose, L, loop=False, start=None):
        '''First point at distance L from the rear axle, walking forward from start.'''
        px, py, _ = self.rear_axle(pose)
        n = len(path)
        if start is None:
            start = self.closest_index(path, (px, py))
        d = np.hypot(path[:, 0] - px, path[:, 1] - py)
        beyond = np.flatnonzero(d[start:] >= L)
        if len(beyond):
            k = int(beyond[0])
        elif loop and start > 0:
            beyond = np.flatnonzero(d[:start] >= L)
            if len(beyond) == 0:
                return path[start - 1]
            k = n - start + int(beyond[0])
        else:
            return path[start - 1] if loop else path[n - 1]
        if k == 0:
            return path[start]
        p0, p1 = path[(start + k - 1) % n], path[(start + k) % n]
        v = p1 - p0
        w = p0 - np.array([px, py])
        a, b, c = v @ v, 2.0 * (v @ w), w @ w - L * L
        t = (-b + math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a) if a > 1e-12 else 1.0
        return p0 + min(max(t, 0.0), 1.0) * v

    def steering_angle(self, pose, target):
        return super().steering_angle(self.rear_axle(pose), target)

    def smooth(self, steer, now):
        '''Low-pass filter (time constant steer_tau) and rate limit the steering command.'''
        dt = None if self.last_time is None else now - self.last_time
        self.last_time = now
        if dt is None or dt <= 0.0 or dt > 0.5:
            self.last_steer = steer
            return steer
        s = self.last_steer + dt / (self.steer_tau + dt) * (steer - self.last_steer)
        step = self.max_steer_rate * dt
        s = min(max(s, self.last_steer - step), self.last_steer + step)
        self.last_steer = s
        return s

    def reset(self):
        self.last_steer = 0.0
        self.last_time = None


class PurePursuit(Node):

    def __init__(self):
        super().__init__("pure_pursuit_smooth_node")

        defaults = {
            "use_sim": False,
            "waypoints_file": DEFAULT_WAYPOINTS,
            "loop": True,
            "drive_topic": "/drive",
            "lookahead": 1.0,
            "speed": 2.5,
            "wheelbase": 0.33,
            "max_steer": 0.36,
            "rear_axle_offset": LASER_TO_REAR_AXLE,
            "steer_tau": 0.05,
            "max_steer_rate": 3.0,
            "track_window": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if self.use_sim:
            # sim odom pose is already the rear axle (base_link)
            self.rear_axle_offset = 0.0

        self.pp = SmoothPurePursuit(self.wheelbase, self.max_steer, self.rear_axle_offset,
                                    self.steer_tau, self.max_steer_rate, self.track_window)
        self.waypoints = PurePursuitController.load_waypoints(self.waypoints_file)
        self.get_logger().info(f"loaded {len(self.waypoints)} waypoints from {self.waypoints_file}")

        if self.use_sim:
            self.create_subscription(Odometry, SIM_POSE_TOPIC, self.pose_callback, 1)
        else:
            self.create_subscription(PoseStamped, CAR_POSE_TOPIC, self.pose_callback, 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/pure_pursuit_markers", 10)
        self.create_timer(1.0, self.publish_waypoints)

    def publish_waypoints(self):
        xs, ys = self.waypoints[:, 0], self.waypoints[:, 1]
        if self.loop:
            xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
        ma = MarkerArray()
        ma.markers.append(line_marker("map", self.get_clock().now().to_msg(), "waypoints", 0, xs, ys,
                                      ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8), width=0.08))
        self.marker_pub.publish(ma)

    def pose_callback(self, msg):
        pose = pose_from_msg(msg)
        start = self.pp.track_index(self.waypoints, pose, self.loop, "waypoints")
        target = self.pp.lookahead_point(self.waypoints, pose, self.lookahead, self.loop, start)
        now = self.get_clock().now()
        steer = self.pp.smooth(self.pp.steering_angle(pose, target), now.nanoseconds * 1e-9)

        stamp = now.to_msg()
        self.drive_pub.publish(drive_msg(stamp, self.speed, steer))

        ma = MarkerArray()
        ma.markers.append(sphere_marker("map", stamp, "tracked_point", 1, target[0], target[1],
                                        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), size=0.3))
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    print("SmoothPurePursuit Initialized")
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
