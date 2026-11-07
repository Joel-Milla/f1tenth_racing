#!/usr/bin/env python3
"""Sim opponent for head-to-head tests: drives the second gym_bridge car
(num_agent: 2) along the raceline with the team's PurePursuitController at
a fraction of the raceline speed profile. It does not react to the ego car.

    ros2 run f1tenth_gym_ros gym_bridge --ros-args -r __node:=bridge \\
        --params-file <sim.yaml> -p num_agent:=2
    /usr/bin/python3 tools/opponent_node.py --scale 0.6
"""
import argparse
import math
import os
import sys

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, os.path.join(PKG_ROOT, '..', 'pure_pursuit', 'scripts'))

from pure_pursuit_node import PurePursuitController  # noqa: E402

from frenet_planner.params import Params  # noqa: E402
from frenet_planner.reference import build_reference  # noqa: E402

LOOKAHEAD = (0.3, 0.6)       # L = a v + b [s, m]


class Opponent(Node):

    def __init__(self, params, scale, odom_topic, drive_topic):
        super().__init__('sim_opponent')
        self.ref = build_reference(params)
        path = self.ref.path
        self.xy = np.column_stack([path.x_table, path.y_table])
        self.pp = PurePursuitController(params.vehicle.wheelbase, params.vehicle.max_steer)
        self.scale = scale
        self.s = None
        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_subscription(Odometry, odom_topic, self.odom, 10)
        self.get_logger().info(f'opponent at {scale:.2f} x raceline speed on {drive_topic}')

    def odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        pose = (p.x, p.y, yaw)
        self.s, _ = self.ref.path.cartesian_to_frenet(p.x, p.y, s_hint=self.s)
        v = msg.twist.twist.linear.x
        target = self.pp.lookahead_point(self.xy, pose, LOOKAHEAD[0] * v + LOOKAHEAD[1], True)
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = float(self.pp.steering_angle(pose, target))
        out.drive.speed = float(self.scale * self.ref.speed(self.s))
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(PKG_ROOT, 'config', 'frenet_planner.yaml'))
    ap.add_argument('--scale', type=float, default=0.6, help='opponent speed / raceline speed')
    ap.add_argument('--odom-topic', default='/opp_racecar/odom')
    ap.add_argument('--drive-topic', default='/opp_drive')
    args = ap.parse_args()
    rclpy.init()
    node = Opponent(Params.from_yaml(args.config), args.scale, args.odom_topic, args.drive_topic)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
