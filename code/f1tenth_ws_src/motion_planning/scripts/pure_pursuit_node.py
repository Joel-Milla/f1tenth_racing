#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_WAYPOINTS = ("src/pure_pursuit/waypoints/waypoints.csv")

SIM_POSE_TOPIC = "/ego_racecar/odom"
CAR_POSE_TOPIC = "/pf/viz/inferred_pose"


def pose_from_msg(msg):
    p = msg.pose.pose if isinstance(msg, Odometry) else msg.pose
    q = p.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return p.position.x, p.position.y, yaw


def global_to_local(px, py, pose):
    x, y, th = pose
    dx, dy = np.asarray(px) - x, np.asarray(py) - y
    c, s = math.cos(th), math.sin(th)
    return c * dx + s * dy, -s * dx + c * dy


def local_to_global(lx, ly, pose):
    x, y, th = pose
    lx, ly = np.asarray(lx), np.asarray(ly)
    c, s = math.cos(th), math.sin(th)
    return x + c * lx - s * ly, y + s * lx + c * ly


def resample_polyline(xs, ys, spacing):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return xs, ys
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if s[-1] < 1e-9:
        return xs[:1], ys[:1]
    t = np.append(np.arange(0.0, s[-1], spacing), s[-1])
    return np.interp(t, s, xs), np.interp(t, s, ys)


def drive_msg(stamp, speed, steer):
    msg = AckermannDriveStamped()
    msg.header.stamp = stamp
    msg.drive.speed = float(speed)
    msg.drive.steering_angle = float(steer)
    return msg


def make_marker(frame, stamp, ns, mid, mtype, color, size):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = size
    m.scale.y = size
    m.scale.z = size
    m.color = color
    return m


def line_marker(frame, stamp, ns, mid, xs, ys, color, width=0.05, z=0.05, strip=True):
    m = make_marker(frame, stamp, ns, mid, Marker.LINE_STRIP if strip else Marker.LINE_LIST, color, width)
    m.points = [Point(x=float(x), y=float(y), z=z) for x, y in zip(xs, ys)]
    return m


def sphere_marker(frame, stamp, ns, mid, x, y, color, size=0.25, z=0.1):
    m = make_marker(frame, stamp, ns, mid, Marker.SPHERE, color, size)
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = z
    return m


def delete_marker(frame, ns, mid):
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.action = Marker.DELETE
    return m


class PurePursuitController:

    def __init__(self, wheelbase=0.33, max_steer=0.36):
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    @staticmethod
    def load_waypoints(path):
        wp = np.atleast_2d(np.genfromtxt(path, delimiter=","))[:, :2]
        return wp[np.all(np.isfinite(wp), axis=1)]

    @staticmethod
    def closest_index(path, pos):
        return int(np.argmin((path[:, 0] - pos[0]) ** 2 + (path[:, 1] - pos[1]) ** 2))

    @staticmethod
    def forward_indices(n, start, loop):
        return (start + np.arange(n)) % n if loop else np.arange(start, n)

    def lookahead_point(self, path, pos, L, loop=False):
        px, py = pos[0], pos[1]
        order = self.forward_indices(len(path), self.closest_index(path, pos), loop)
        pts = path[order]
        d = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
        beyond = np.flatnonzero(d >= L)
        if len(beyond) == 0:
            return pts[-1]
        k = beyond[0]
        if k == 0:
            return pts[0]
        p0, p1 = pts[k - 1], pts[k]
        v = p1 - p0
        w = p0 - np.array([px, py])
        a, b, c = v @ v, 2.0 * (v @ w), w @ w - L * L
        t = (-b + math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a) if a > 1e-12 else 1.0
        return p0 + min(max(t, 0.0), 1.0) * v

    def steering_angle(self, pose, target):
        lx, ly = global_to_local(target[0], target[1], pose)
        d2 = float(lx * lx + ly * ly)
        if d2 < 1e-6:
            return 0.0
        curvature = 2.0 * float(ly) / d2
        return float(np.clip(math.atan(self.wheelbase * curvature), -self.max_steer, self.max_steer))


class PurePursuit(Node):

    def __init__(self):
        super().__init__("pure_pursuit_node")

        defaults = {
            "use_sim": False,
            "waypoints_file": DEFAULT_WAYPOINTS,
            "loop": True,
            "drive_topic": "/drive",
            "lookahead": 1.0,
            "speed": 2.5,
            "wheelbase": 0.33,
            "max_steer": 0.36,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)

        self.pp = PurePursuitController(self.wheelbase, self.max_steer)
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
        target = self.pp.lookahead_point(self.waypoints, pose, self.lookahead, self.loop)
        steer = self.pp.steering_angle(pose, target)

        stamp = self.get_clock().now().to_msg()
        self.drive_pub.publish(drive_msg(stamp, self.speed, steer))

        ma = MarkerArray()
        ma.markers.append(sphere_marker("map", stamp, "tracked_point", 1, target[0], target[1],
                                        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), size=0.3))
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()