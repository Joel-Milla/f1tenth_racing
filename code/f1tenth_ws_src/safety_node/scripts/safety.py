#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        # TODO: create ROS subscribers and publishers.
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        self.publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)

    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        # TODO: calculate TTC
        self.angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        self.ranges = np.array(scan_msg.ranges)

        # self.get_logger().info(f'len_angles: {len(self.angles)} len_ranges: {len(self.ranges)} angle_min: {scan_msg.angle_min} angle_max: {scan_msg.angle_max}')
        self.range_rate = np.maximum(0.0, (self.speed * np.cos(self.angles)))

        
        # TODO: publish command to brake
        pass

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()