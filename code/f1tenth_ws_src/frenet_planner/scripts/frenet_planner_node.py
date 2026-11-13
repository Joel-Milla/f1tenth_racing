#!/usr/bin/env python3
"""Frenet planner ROS node: a thin wrapper around frenet_planner.core.PlannerCore.

Each scan updates the obstacle tracks and the localization health; a timer
plans at planner.rate (behaviour: RACELINE / AVOID / TRAIL / OVERTAKE); pose
messages drive the team's PurePursuitController on the chosen path at
control.rate, with lookahead L = gain x speed + base. While localization is
unhealthy, the team's follow-the-gap drives instead. Unless node.start_armed,
it starts disarmed: it plans and publishes everything but commands speed 0
until armed.

Topics
  sub  node.pose_topic                  nav_msgs/Odometry
  sub  node.scan_topic                  sensor_msgs/LaserScan
  sub  /clicked_point                   geometry_msgs/PointStamped (planner-only test obstacles)
  sub  /goal_pose                       geometry_msgs/PoseStamped (RViz "2D Goal Pose": arm)
  sub  /initialpose                     geometry_msgs/PoseWithCovarianceStamped (RViz "2D Pose Estimate": arm)
  srv  /frenet_planner/arm              std_srvs/SetBool (arm / disarm)
  pub  node.drive_topic                 ackermann_msgs/AckermannDriveStamped
  pub  /frenet_planner/path             nav_msgs/Path (chosen path, map frame)
  pub  /frenet_planner/path_speed       std_msgs/Float32MultiArray (speed at each path pose)
  pub  /frenet_planner/frenet_state     std_msgs/Float64MultiArray (FRENET_STATE_FIELDS)
  pub  /frenet_planner/reference        visualization_msgs/MarkerArray (latched)
  pub  /frenet_planner/viz              visualization_msgs/MarkerArray (car state, lookahead point)
  pub  /frenet_planner/candidates       visualization_msgs/MarkerArray (green valid, red rejected, thick chosen)
  pub  /frenet_planner/obstacles        visualization_msgs/MarkerArray
"""
import math
import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Float32MultiArray, Float64MultiArray, MultiArrayDimension
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

from frenet_planner.behavior import OVERTAKE, TRAIL
from frenet_planner.core import PlannerCore
from frenet_planner.health import GapFollowAdapter, pose_std
from frenet_planner.params import NODE_NAME, Params
from frenet_planner.planner import AVOID, FALLBACK, RECOVER

# The team's pure pursuit and follow-the-gap are scripts in their packages.
# On the car PurePursuitController is in motion_planning (pure_pursuit's
# pure_pursuit_node.py there is a different script), so it is searched first.
for _pkg in ('motion_planning', 'pure_pursuit', 'gap_follow'):
    try:
        sys.path.append(os.path.join(get_package_prefix(_pkg), 'lib', _pkg))
    except Exception:
        pass
from pure_pursuit_node import PurePursuitController  # noqa: E402
try:
    import reactive_node  # noqa: E402
except ImportError:
    # No ReactiveFollowGap (the car): on a localization failure the car stops instead
    reactive_node = None

FRENET_STATE_FIELDS = ['s', 'd', 'heading_err', 'd_left', 'd_right', 'v_ref', 'progress']

BLUE = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
GREEN = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.6)
RED = ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.4)
MAGENTA = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=0.85, b=0.0, a=1.0)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
ORANGE = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
GREY = ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.8)
MODE_COLOR = {AVOID: ORANGE, RECOVER: YELLOW, FALLBACK: RED}
STATE_COLOR = {TRAIL: YELLOW, OVERTAKE: MAGENTA}

LINE_WIDTH = 0.04
POINT_SIZE = 0.12
TEXT_HEIGHT = 0.25
TEXT_OFFSET = 0.8      # text floats this far above the car [m]
CANDIDATE_WIDTH = 0.02
CHOSEN_WIDTH = 0.08
BOX_EDGE_POINTS = 6    # points per box edge, so edges follow the raceline's curve
SCAN_POINT_SIZE = 0.04
TRACK_WIDTH = 0.06
STATS_PERIOD = 5.0     # planning-time summary in the log every this many seconds


def nest(flat):
    """{'a.b': 1} -> {'a': {'b': 1}}"""
    out = {}
    for key, value in flat.items():
        *parents, leaf = key.split('.')
        d = out
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = value
    return out


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def make_marker(frame, stamp, ns, mid, mtype, color, size):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = mid
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.color = color
    return m


def points_marker(frame, stamp, ns, mid, mtype, xs, ys, color, size, z=0.02):
    m = make_marker(frame, stamp, ns, mid, mtype, color, size)
    m.points = [Point(x=float(x), y=float(y), z=z) for x, y in zip(xs, ys)]
    return m


def wipe_marker():
    m = Marker()
    m.action = Marker.DELETEALL
    return m


def closed(a):
    return np.r_[a, a[:1]]


def frenet_box_xy(path, s_min, s_len, d_min, d_max):
    """Closed outline of an (s, d) box in map coordinates."""
    e = np.linspace(0.0, 1.0, BOX_EDGE_POINTS)
    bs = s_min + s_len * np.r_[e, np.ones_like(e), e[::-1], np.zeros_like(e)]
    bd = np.r_[np.full_like(e, d_min), d_min + (d_max - d_min) * e,
               np.full_like(e, d_max), d_max + (d_min - d_max) * e]
    x, y, _ = path.frenet_to_cartesian(bs, bd)
    return x, y


class FrenetPlanner(Node):

    def __init__(self):
        super().__init__(NODE_NAME, automatically_declare_parameters_from_overrides=True)
        flat = {name: p.value for name, p in self.get_parameters_by_prefix('').items()
                if name != 'use_sim_time'}
        self.params = Params.from_dict(nest(flat))
        np_, vp = self.params.node, self.params.vehicle

        fallback = None if reactive_node is None else GapFollowAdapter(reactive_node, self.params.gap_follow)
        if fallback is None:
            self.get_logger().warn('gap_follow reactive_node not found: no follow-the-gap fallback, '
                                   'the car stops if localization fails')
        self.core = PlannerCore(self.params, PurePursuitController(vp.wheelbase, vp.max_steer), fallback)
        path = self.core.ref.path
        self.get_logger().info(
            f'reference: {path.length:.2f} m loop from {self.params.raceline.file}, '
            f'map {self.params.map.yaml_file}')
        over = np.abs(path.kappa_table) > vp.kappa_max
        if over.any():
            self.get_logger().warn(
                f'raceline curvature exceeds kappa_max={vp.kappa_max:.2f} '
                f'over {over.sum() * path.ds:.2f} m (max {np.abs(path.kappa_table).max():.2f})')

        self.last_viz = -math.inf
        self.last_obstacle_viz = -math.inf
        self.last_control = -math.inf
        self.last_mode = None
        self.last_fallback = False
        self.target = None
        self.armed = np_.start_armed

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, np_.drive_topic, 10)
        self.path_pub = self.create_publisher(Path, '/frenet_planner/path', 10)
        self.speed_pub = self.create_publisher(Float32MultiArray, '/frenet_planner/path_speed', 10)
        self.state_pub = self.create_publisher(Float64MultiArray, '/frenet_planner/frenet_state', 10)
        self.ref_pub = self.create_publisher(MarkerArray, '/frenet_planner/reference', latched)
        self.viz_pub = self.create_publisher(MarkerArray, '/frenet_planner/viz', 10)
        self.cand_pub = self.create_publisher(MarkerArray, '/frenet_planner/candidates', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, '/frenet_planner/obstacles', 10)
        self.create_subscription(Odometry, np_.pose_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, np_.scan_topic, self.scan_callback, 1)
        self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)
        # Arming: RViz "2D Pose Estimate" or "2D Goal Pose" arms; the service arms or disarms
        self.create_subscription(PoseStamped, '/goal_pose', lambda _: self.set_armed(True, 'goal pose clicked'), 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 lambda _: self.set_armed(True, 'initial pose set'), 10)
        self.create_service(SetBool, '/frenet_planner/arm', self.arm_service)
        if not self.armed:
            self.get_logger().warn('DISARMED: the car is held still. Set its pose with "2D Pose Estimate" '
                                   'to drive (or call /frenet_planner/arm).')

        self.reference_markers = self.build_reference_markers()
        self.publish_reference()
        self.create_timer(np_.static_viz_period, self.publish_reference)
        self.create_timer(1.0 / self.params.planner.rate, self.plan_callback)
        self.create_timer(STATS_PERIOD, self.log_stats)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def due(self, attr, rate):
        """True at most rate times per second for the timestamp attribute attr."""
        now = self.now()
        if now - getattr(self, attr) < 1.0 / rate:
            return False
        setattr(self, attr, now)
        return True

    # --- callbacks -----------------------------------------------------------

    def odom_callback(self, msg):
        p = msg.pose.pose
        x, y, yaw = p.position.x, p.position.y, yaw_from_quat(p.orientation)
        core = self.core
        core.update_pose(x, y, yaw, msg.twist.twist.linear.x,
                         pose_std(msg.pose.covariance, self.params.health.covariance_layout))
        self.publish_state()

        if self.due('last_control', self.params.control.rate):
            out = core.control()
            if out is not None:
                steer, speed, self.target = out
                if not self.armed:
                    steer, speed = 0.0, 0.0      # planning and showing, but holding still
                drive = AckermannDriveStamped()
                drive.header.stamp = self.get_clock().now().to_msg()
                drive.drive.steering_angle = float(steer)
                drive.drive.speed = float(speed)
                self.drive_pub.publish(drive)
        if self.due('last_viz', self.params.node.viz_rate):
            self.publish_state_markers(x, y)

    def set_armed(self, armed, why):
        if armed and not self.armed:
            # The pose may have been reset while disarmed: drop tracks and
            # the last path built on the old one
            self.core.reset_planning()
            self.get_logger().warn(f'ARMED ({why}): driving')
        elif not armed and self.armed:
            self.get_logger().warn(f'DISARMED ({why}): stopping; set "2D Pose Estimate" to drive again')
        self.armed = armed

    def arm_service(self, request, response):
        self.set_armed(request.data, 'service call')
        response.success = True
        response.message = 'armed' if self.armed else 'disarmed'
        return response

    def scan_callback(self, msg):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        res = self.core.update_scan(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, t)
        if self.core.using_fallback != self.last_fallback:
            self.last_fallback = self.core.using_fallback
            if self.last_fallback:
                self.get_logger().error(f'localization unhealthy ({self.core.health.reason}): follow-the-gap drives')
            else:
                self.get_logger().warn('localization healthy again: planner drives')
        if res is not None and self.due('last_obstacle_viz', self.params.node.viz_rate):
            self.publish_obstacle_markers(res)

    def plan_callback(self):
        pl = self.core.step_plan()
        if pl is None:
            return
        mode = f'{self.core.behavior.state}/{pl.mode}'
        if mode != self.last_mode:
            # rclpy forbids changing severity at one call site: separate calls
            if pl.mode == FALLBACK:
                self.get_logger().warn(f'{mode} at s {pl.s0:.2f}: no valid candidate, raceline at reduced speed')
            else:
                self.get_logger().info(f'{mode} at s {pl.s0:.2f}')
            self.last_mode = mode
        self.publish_path(pl)
        self.publish_candidates(pl)

    def clicked_callback(self, msg):
        s, d = self.core.add_test_obstacle(msg.point.x, msg.point.y)
        self.get_logger().info(f'test obstacle at s {s:.2f} d {d:+.2f} '
                               f'({len(self.core.test_obstacles)} total; the simulator does not see it)')

    def log_stats(self):
        t = np.array(self.core.plan_times) * 1e3
        if len(t):
            self.get_logger().info(f'planning cycle (last {len(t)}): mean {t.mean():.2f} ms, '
                                   f'p95 {np.percentile(t, 95):.2f} ms, max {t.max():.2f} ms')

    # --- outputs -------------------------------------------------------------

    def publish_state(self):
        core = self.core
        st = dict(s=core.s, d=core.d, heading_err=core.heading_err,
                  d_left=float(core.ref.bounds.left(core.s)), d_right=float(core.ref.bounds.right(core.s)),
                  v_ref=float(core.ref.speed(core.s)), progress=core.progress)
        out = Float64MultiArray()
        out.layout.dim = [MultiArrayDimension(label=','.join(FRENET_STATE_FIELDS),
                                              size=len(FRENET_STATE_FIELDS), stride=1)]
        out.data = [float(st[k]) for k in FRENET_STATE_FIELDS]
        self.state_pub.publish(out)

    def publish_path(self, pl):
        stamp = self.get_clock().now().to_msg()
        msg = Path()
        msg.header.frame_id = self.params.node.frame_id
        msg.header.stamp = stamp
        for x, y, yaw in zip(pl.x, pl.y, pl.yaw):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.speed_pub.publish(Float32MultiArray(data=[float(v) for v in pl.v]))

    def publish_candidates(self, pl):
        frame = self.params.node.frame_id
        stamp = self.get_clock().now().to_msg()
        path, c = self.core.ref.path, pl.cands
        s = pl.s0 + pl.u
        ma = MarkerArray()
        ma.markers.append(wipe_marker())
        valid = c.valid
        for i in range(len(c.d_end)):
            if i == pl.chosen:
                continue
            x, y, _ = path.frenet_to_cartesian(s, c.d[i])
            ma.markers.append(points_marker(frame, stamp, 'candidates', i, Marker.LINE_STRIP, x, y,
                                            GREEN if valid[i] else RED, CANDIDATE_WIDTH, z=0.03))
        color = STATE_COLOR.get(self.core.behavior.state, MODE_COLOR.get(pl.mode, BLUE))
        ma.markers.append(points_marker(frame, stamp, 'chosen', 0, Marker.LINE_STRIP, pl.x, pl.y,
                                        color, CHOSEN_WIDTH, z=0.06))
        o = self.core.test_obstacles
        for k in range(len(o)):
            x, y = frenet_box_xy(path, o.s_min[k], o.s_len[k], o.d_min[k], o.d_max[k])
            ma.markers.append(points_marker(frame, stamp, 'test_obstacles', k, Marker.LINE_STRIP,
                                            x, y, MAGENTA, LINE_WIDTH, z=0.05))
        self.cand_pub.publish(ma)

    def publish_obstacle_markers(self, res):
        """Grey: scan points left after the wall and track filters. Yellow:
        this scan's detections. Orange: confirmed tracks, with id and speed."""
        frame = self.params.node.frame_id
        stamp = self.get_clock().now().to_msg()
        path, tracker = self.core.ref.path, self.core.tracker
        ma = MarkerArray()
        ma.markers.append(wipe_marker())
        ma.markers.append(points_marker(frame, stamp, 'scan_points', 0, Marker.POINTS,
                                        res.x, res.y, GREY, SCAN_POINT_SIZE, z=0.05))
        for k, det in enumerate(res.detections):
            x, y = frenet_box_xy(path, det.s_min, det.s_len, det.d_min, det.d_max)
            ma.markers.append(points_marker(frame, stamp, 'detections', k, Marker.LINE_STRIP,
                                            x, y, YELLOW, LINE_WIDTH, z=0.06))
        for tr in tracker.confirmed():
            x, y = frenet_box_xy(path, tr.x[0], tr.s_len, tr.d_min, tr.d_max)
            ma.markers.append(points_marker(frame, stamp, 'tracks', tr.id, Marker.LINE_STRIP,
                                            x, y, ORANGE, TRACK_WIDTH, z=0.08))
            text = make_marker(frame, stamp, 'track_text', tr.id, Marker.TEXT_VIEW_FACING, ORANGE, TEXT_HEIGHT)
            text.pose.position.x, text.pose.position.y = float(x.mean()), float(y.mean())
            text.pose.position.z = TEXT_OFFSET
            moving = 'moving' if tracker.is_moving(tr) else 'static'
            text.text = f'#{tr.id} {moving} v_s {tr.x[2]:+.1f}'
            ma.markers.append(text)
        self.obstacle_pub.publish(ma)

    def build_reference_markers(self):
        frame = self.params.node.frame_id
        stamp = self.get_clock().now().to_msg()
        path, b = self.core.ref.path, self.core.ref.bounds
        cx, cy, _ = path.frenet_to_cartesian(b.s, 0.0)
        lx, ly, _ = path.frenet_to_cartesian(b.s, b.d_left)
        rx, ry, _ = path.frenet_to_cartesian(b.s, -b.d_right)
        ma = MarkerArray()
        ma.markers.append(points_marker(frame, stamp, 'raceline', 0, Marker.LINE_STRIP,
                                        closed(cx), closed(cy), BLUE, LINE_WIDTH))
        ma.markers.append(points_marker(frame, stamp, 'bound_left', 0, Marker.LINE_STRIP,
                                        closed(lx), closed(ly), GREEN, LINE_WIDTH))
        ma.markers.append(points_marker(frame, stamp, 'bound_right', 0, Marker.LINE_STRIP,
                                        closed(rx), closed(ry), RED, LINE_WIDTH))
        sx, sy, _ = path.frenet_to_cartesian(0.0, np.array([-b.right(0.0), b.left(0.0)]))
        ma.markers.append(points_marker(frame, stamp, 'start_line', 0, Marker.LINE_LIST,
                                        sx, sy, WHITE, 2 * LINE_WIDTH))
        over = np.abs(path.kappa_table) > self.params.vehicle.kappa_max
        if over.any():
            ma.markers.append(points_marker(frame, stamp, 'over_curvature', 0, Marker.SPHERE_LIST,
                                            path.x_table[over], path.y_table[over], MAGENTA, POINT_SIZE))
        return ma

    def publish_reference(self):
        self.ref_pub.publish(self.reference_markers)

    def publish_state_markers(self, x, y):
        frame = self.params.node.frame_id
        stamp = self.get_clock().now().to_msg()
        core = self.core
        path = core.ref.path
        dl, dr = float(core.ref.bounds.left(core.s)), float(core.ref.bounds.right(core.s))
        fx, fy, _ = path.frenet_to_cartesian(core.s, 0.0)
        bx, by, _ = path.frenet_to_cartesian(core.s, np.array([dl, -dr]))
        ma = MarkerArray()
        ma.markers.append(points_marker(frame, stamp, 'car_to_foot', 0, Marker.LINE_LIST,
                                        [x, fx], [y, fy], YELLOW, LINE_WIDTH, z=0.05))
        ma.markers.append(points_marker(frame, stamp, 'bounds_at_car', 0, Marker.SPHERE_LIST,
                                        bx, by, YELLOW, POINT_SIZE, z=0.05))
        if self.target is not None:
            ma.markers.append(points_marker(frame, stamp, 'lookahead', 0, Marker.SPHERE_LIST,
                                            [self.target[0]], [self.target[1]], WHITE, POINT_SIZE, z=0.1))
        mode = 'FOLLOW_GAP' if core.using_fallback else \
            (f'{core.behavior.state}/{core.plan.mode}' if core.plan is not None else '-')
        if not self.armed:
            mode = f'DISARMED ({mode})'
        text = make_marker(frame, stamp, 'frenet_text', 0, Marker.TEXT_VIEW_FACING, WHITE, TEXT_HEIGHT)
        text.pose.position.x, text.pose.position.y, text.pose.position.z = float(x), float(y), TEXT_OFFSET
        text.text = (f'{mode}  v {core.speed:.1f}  L {core.lookahead():.2f}\n'
                     f's {core.s:.2f}  d {core.d:+.2f}  dpsi {math.degrees(core.heading_err):+.0f} deg\n'
                     f'laps {core.progress / path.length:.2f}  scan-map mismatch '
                     f'{100 * core.health.mismatch:.0f}%')
        ma.markers.append(text)
        self.viz_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    print("FrenetPlanner Initialized")
    node = FrenetPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
