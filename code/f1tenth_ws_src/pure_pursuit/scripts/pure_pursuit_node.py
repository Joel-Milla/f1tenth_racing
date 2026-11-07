#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import tf_transformations
# TODO CHECK: include needed ROS msg type headers and libraries

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        #odom_topic = '/ego_racecar/odom'

        self.L = 1.0
        self.waypoints = np.genfromtxt('/home/f1tenth/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv', delimiter=',', skip_header=1)
        #self.waypoints = np.genfromtxt('/home/f1tenth/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv', delimiter=',')
        # self.get_logger().info(f'{self.waypoints}')



        # TODO: create ROS subscribers and publishers
        # TODO: Subscribe to LIDAR
        self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        #self.odom_subscription = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        self.odom_subscription = self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 10)
        # TODO: Publish to drive
        self.publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "visualization_merker_array", 10)

    # def odom_callback(self, odom_msg: Odometry):
    #     pass

    def find_closest_idx(self, target):
        distances = np.linalg.norm(self.waypoints - target, axis=1)
        idx = np.argmin(distances)
        return idx, self.waypoints[idx]

    def global_to_local(self, target_global, car_global):
        x_c, y_c, theta_c = car_global
        x_t, y_t = target_global

        x_l = (x_t-x_c) * np.cos(theta_c) + (y_t-y_c)*np.sin(theta_c)
        y_l = -(x_t-x_c)*np.sin(theta_c) + (y_t-y_c)*np.cos(theta_c)

        return(x_l, y_l)

    def publish_waypoints(self, goal):
        marker_array = MarkerArray()

        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "waypoint_line"
        line_marker.id = 999  # Unique ID for the line
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.08  # Line width
        line_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)

        for (i, wp) in enumerate(self.waypoints):
            # self.get_logger().info(f'{wp}')
            p = Point()
            p.x, p.y = wp
            p.z = 0.1
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)


        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"  # Change to your fixed frame (e.g., "odom")
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "next_waypoint"
        sphere_marker.id = 99
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # Set position
        x, y = goal
        sphere_marker.pose.position.x = x
        sphere_marker.pose.position.y = y
        sphere_marker.pose.position.z = 0.1
        sphere_marker.pose.orientation.w = 1.0
        
        # Set scale (size of the sphere)
        sphere_marker.scale.x = 0.3
        sphere_marker.scale.y = 0.3
        sphere_marker.scale.z = 0.3
        
        # Set color (Green, fully opaque)
        sphere_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        

        marker_array.markers.append(sphere_marker)

        self.marker_pub.publish(marker_array)

    def lidar_callback(self, scan_msg: LaserScan):
        pass

    def pose_callback(self, odom_msg: PoseStamped):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        x = odom_msg.pose.position.x
        y = odom_msg.pose.position.y
        orientation = odom_msg.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)
        theta = yaw
        lookahead_x = x + self.L * np.cos(theta)
        lookahead_y = y + self.L * np.sin(theta)

        goal_idx, goal_point = self.find_closest_idx((lookahead_x, lookahead_y))
        


        # TODO: transform goal point to vehicle frame of reference
        local_goal_x, local_goal_y = self.global_to_local(goal_point, (lookahead_x, lookahead_y, theta))

        distance = np.linalg.norm((local_goal_x, local_goal_y))


        # TODO: calculate curvature/steering angle
        curvature = (2.0 * local_goal_y) / (distance ** 2)
        angle = np.arctan(self.L * curvature * 0.07)
        speed = 1.5 #0.5 * distance

        if(distance < 0.1):
            angle = 0.0
            curvature = 0.0

        # TODO: publish drive message, don't forget to limit the steering angle.
        # self.get_logger().info(f'{x} {y} {theta} {lookahead_x} {lookahead_y} {goal_point} {goal_idx} {local_goal_x} {local_goal_y} {distance} {speed} {np.rad2deg(angle)}')
        self.get_logger().info(f'{speed:.2f} {np.rad2deg(angle):.2f} {local_goal_y:.2f} {distance:.2f}')
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = angle
        self.publish_waypoints(goal_point)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
