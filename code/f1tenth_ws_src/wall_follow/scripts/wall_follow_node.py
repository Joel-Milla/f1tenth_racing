#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers        
        self.scan_subscription = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)

        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.testpublisher = self.create_publisher(AckermannDriveStamped, "/test", 10)

        self.theta = -60
        self.L = 0.1 #lookahead distance
        self.target_dist = 0.75


        # TODO: set PID gains
        self.kp = 0.15#0.15
        self.kd = 0.1
        # self.ki = 

        # TODO: store history
        # self.integral = 0
        self.prev_error = 0
        self.error = 0

        # TODO: store any necessary values you think you'll need

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """
        fov =  270
        beams = 1080

        angle = int(np.interp(angle, [-fov/2, fov/2], [0, 1080]))

        return range_data[angle]


        #TODO: implement
        
        return 0.0

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        b = self.get_range(range_data, -90)
        a = self.get_range(range_data, self.theta)
        self.get_logger().info(f"{a} {b}")

        # alpha = np.arctan((a*np.cos(np.deg2rad(self.theta)) - b), (a*np.sin(np.deg2rad(self.theta))))
        alpha = np.arctan((a*np.cos(np.deg2rad(self.theta)) - b) / (a*np.sin(np.deg2rad(self.theta))))
        D_t = b * np.cos(alpha) #current distance from wall
        new_D_t = D_t + (self.L * np.sin(alpha))

        return dist - new_D_t




        #TODO:implement
        return 0.0
    
    def get_velocity(self, angle):
        # return 2.0
        match angle:
            case num if 0 <= num < 10:
                return 2.5
            case num if 10 <= num < 20:
                return 2.25
            case _:
                return 1.50

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        # TODO: Use kp, ki & kd to implement a PID controller
        self.prev_error = self.error
        self.error = error

        proportional = self.kp * self.error
        derivative = self.kd * (self.prev_error - self.error)
        signal = np.clip(proportional + derivative, -30, 30)


        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = signal


        
        test = AckermannDriveStamped()
        test.drive.acceleration = self.error
        test.drive.jerk = derivative
        test.drive.speed = proportional
        test.drive.steering_angle_velocity = signal
        self.testpublisher.publish(test)

        self.publisher.publish(drive_msg)
        # TODO: fill in drive message and publish

    def scan_callback(self, msg:LaserScan):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        # self.get_logger().info(f'{msg.angle_min} {msg.angle_max} {len(msg.ranges)} {msg.angle_increment}')
        error = self.get_error(msg.ranges, self.target_dist) # TODO: replace with error calculated by get_error()

        correction_angle = np.interp(error, [-2, 2], [-30, 30])
        # self.get_logger().info(f'{error} {correction_angle}')

        # test = AckermannDriveStamped()
        # test.drive.acceleration = error
        # test.drive.jerk = correction_angle
        # self.testpublisher.publish(test)

        velocity = self.get_velocity(correction_angle) # TODO: calculate desired car velocity based on error
        self.pid_control(correction_angle, velocity) # TODO: actuate the car with PID


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()