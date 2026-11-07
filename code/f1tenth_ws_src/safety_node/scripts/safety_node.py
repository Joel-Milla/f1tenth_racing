#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from std_msgs.msg import Bool


import numpy as np

import matplotlib.pyplot as plt


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
        self.speed = 0.0001

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, subplot_kw={'projection': 'polar'}, figsize=(5, 15))
        plt.ion()
        plt.show()

        self.flag = 0

        
        # TODO: create ROS subscribers and publishers.
        self.odom_subscription = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        self.publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.range_pub = self.create_publisher(Float32MultiArray, "/range_pub", 10)
        self.range_rate_pub = self.create_publisher(LaserScan, "/range_rate_pub", 10)
        self.ttc_pub = self.create_publisher(LaserScan, "/ttc_pub", 10)
        self.testpublisher = self.create_publisher(AckermannDriveStamped, "/test", 10)


        self.brake_lock_pub = self.create_publisher(Bool, 'brake', 1)


    def odom_callback(self, odom_msg:Odometry):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg:LaserScan):
        # self.get_logger().info(str(len(scan_msg.ranges)))
        self.ranges = np.array(scan_msg.ranges)
        self.angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)

        arr_len = len(self.ranges)
        self.ranges = self.ranges
        self.angles = self.angles
        # self.get_logger().info("lens" + str(len(self.ranges)) + "...." + str(len(self.angles)))

        # self.speed = 1.0

        # TODO: calculate TTC
        self.range_rate = np.maximum(0.0001, (self.speed * np.cos(self.angles)))


        # self.get_logger().info(str(max(self.range_rate)))
        self.iTTC = self.ranges / self.range_rate


        # ax = plt.subplot(111, projection='polar')
        
        # ax1.scatter(self.angles, self.range_rate, c='g', s=10)
        self.ax1.cla()
        self.ax1.scatter(self.angles, self.ranges, c='g', s=10)
        self.ax1.set_title('Ranges')
        self.ax2.cla()
        self.ax2.scatter(self.angles, self.range_rate, c='g', s=10)
        self.ax2.set_title('Range Rate')
        self.ax3.cla()
        self.ax3.scatter(self.angles, np.clip(self.iTTC, 0, 3), c='b', s=10)
        self.ax3.set_title('iTTC')
        
        # self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        '''
        ax = plt.subplot(111, projection='polar')
        ax.scatter(self.angles, self.range_rate, c='g', s=10)
        plt.show(block = False)
        plt.pause(0.3)
        plt.clf()



        fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 5))
        axs[0].scatter(self.angles, self.ranges, c='r', s=10)
        axs[0].set_title('Ranges')
        axs[1].scatter(self.angles, self.range_rate, c='g', s=10)
        axs[1].set_title('Range Rate')
        axs[2].scatter(self.angles, self.iTTC, c='b', s=10)
        axs[2].set_title('iTTC')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)
        plt.clf()
        '''

        self.min_iTTC = np.min(self.iTTC)
        min_ind = np.argmin(self.iTTC)
        # self.get_logger().info(str(self.min_iTTC))

        test = AckermannDriveStamped()
        test.drive.speed = np.clip(self.min_iTTC, -5, 5)
        self.testpublisher.publish(test)

        test = Float32MultiArray()
        test.data = self.ranges.tolist()
        self.range_pub.publish(test)

        # test = LaserScan()
        # test.ranges = range_rate
        # self.range_rate_pub.publish(test)

        # test = LaserScan()
        # test.ranges = TTC
        # self.ttc_pub.publish(test)
        
        # TODO: publish command to brake
        if(self.min_iTTC < 1):
            self.get_logger().info("Stopping car" + str(self.min_iTTC) + " angle: " + str(np.rad2deg(self.angles[min_ind])))
            self.flag = 1

        lock_msg = Bool()
        lock_msg.data = True
        self.brake_lock_pub.publish(lock_msg)

        if(self.flag == 1):
            command = AckermannDriveStamped()
            command.drive.speed = 0.0
            self.publisher.publish(command)
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