#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import csv
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.in_simulation = False
        
        if self.in_simulation:
            self.subscription_odom = self.create_subscription(
                Odometry,
                'ego_racecar/odom',
                self.pose_callback,
                10)
        else:
            self.subscription_odom = self.create_subscription(
                PoseStamped,
                '/pf/viz/inferred_pose',
                self.pose_callback,
                10)
        self.publisher_vel = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.publisher_marker = self.create_publisher(Marker, 'waypoints', 10)
        self.publisher_marker_goal = self.create_publisher(Marker, 'goal', 10)
        
        # Parameters to tune
        # fast params
        # self.l_min = 0.8 # min possible l
        # self.C = 0.6
        # self.l = 1.0
        # self.dq = 0.17
        
        # slow params
        self.l_min = 0.8 # min possible l
        self.C = 0.6
        self.l = 1.2
        self.dq = 0.2
        
        waypoints_path = "/home/f1tenth/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv"
        self.waypoints = self.load_waypoints(waypoints_path)
        
        self.pose = Pose()    

    def pose_callback(self, pose_msg):
        self.publish_waypoints()
        if self.in_simulation:
            self.pose = pose_msg.pose.pose      # Odometry
        else:
            self.pose = pose_msg.pose           # PoseStamped
        
        curr_position = self.pose.position
        
        closest_point_indx = self.closest_point_index(curr_position)
        curr_goal_point = self.find_goal_point(curr_position, closest_point_indx)
        self.publish_goal(curr_goal_point)
        
        curr_goal_point = self.world_to_car_frame(curr_goal_point[0], curr_goal_point[1]) # convert goal point into the car frame

        x = curr_goal_point[0] 
        y = curr_goal_point[1]
        
        curvature_arc = (2 * y) / (x*x + y*y) # divide by distance between current point and goal
        speed = self.get_optimal_linear_speed(curvature_arc)
        self.l = max(self.l_min, self.C * speed) #update l
        
        steering_angle = curvature_arc * self.dq # multiply by dq to make small step
        
        self.get_logger().info(f"steering_angle: {steering_angle} -- speed {speed}")
        
        self.publish_velocity(speed, steering_angle)
    
    def world_to_car_frame(self, gx, gy) -> tuple[float, float]:
        p = self.pose.position
        q = self.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))   # quaternion → heading

        # 1) translate: offset from the car to the goal, still in world axes
        dx, dy = gx - p.x, gy - p.y

        # 2) rotate by -yaw: express that offset in the car's axes (x forward, y left)
        x_car =  math.cos(yaw) * dx + math.sin(yaw) * dy
        y_car = -math.sin(yaw) * dx + math.cos(yaw) * dy

        return (x_car, y_car)
    
    def publish_goal(self, goal):
        x, y = goal
        marker = Marker()
        
        marker = Marker()
        
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "goal"
        marker.id = 200 # unique id of the marker
        
        marker.type = Marker.SPHERE
            
        # set pose of marker
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.2 # width of the waypoints. defaults to zero (invisible) if not set
        marker.scale.y = 0.2 # width of the waypoints. defaults to zero (invisible) if not set
        marker.scale.z = 0.2 # width of the waypoints. defaults to zero (invisible) if not set
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0 # opacity
        
        marker.lifetime.sec = 0 # marker will last forever until overwritten
        
        self.publisher_marker_goal.publish(marker)
    
    def find_goal_point(self, pose, closest_point_indx):
        
        curr_indx = closest_point_indx
        old_distance, new_distance = np.inf, np.inf
        iters = 0
        
        while iters < len(self.waypoints):
            waypoint = self.waypoints[curr_indx]
            diff = (waypoint[0] - pose.x)**2 + (waypoint[1] - pose.y)**2
            old_distance = new_distance
            new_distance = abs(math.sqrt(diff) - self.l)
            
            if new_distance > old_distance: # means that previous one was closest (that is in front) to value of l
                return self.waypoints[curr_indx - 1]
            
            curr_indx = (curr_indx + 1) % len(self.waypoints) # wrap around
            iters += 1
        
        # if didnt found any waypoint, then log it and return nothing
        self.get_logger().info("No waypoint was found")
    
    def closest_point_index(self, pose):
        min_distance = np.inf
        min_indx = 0
        
        for curr_indx, waypoint in enumerate(self.waypoints):
            diff = (waypoint[0] - pose.x)**2 + (waypoint[1] - pose.y)**2
            diff = np.sqrt(diff)
            
            if diff < min_distance:
                min_indx = curr_indx
                min_distance = diff

        return min_indx
    
    def get_optimal_linear_speed(self, steering_angle):
        steering_angle_degrees = abs(np.rad2deg(steering_angle))
        # fast params
        # match steering_angle_degrees:
        #     case angle if abs(angle) <= 10:
        #         return 2.7
        #     case angle if abs(angle) <= 20:
        #         return 2.0
        # return 1.7
        
        # slow params
        match steering_angle_degrees:
            case angle if abs(angle) <= 10:
                return 1.7
            case angle if abs(angle) <= 20:
                return 1.5
        return 1.2
        
    def load_waypoints(self, path):
        '''
        Input: Path to the folder where waypoints are
        Output: (N,2) array where first index is x, and next index y
        '''
        
        points = []
        with open(path, "r") as f:
            for row in csv.reader(f):
                try:
                    x, y = float(row[0]), float(row[1])
                except (ValueError, IndexError):
                    continue  # skip header or empty lines
                points.append((x, y))
        return np.array(points)  # shape (N, 2)

    def publish_waypoints(self):
        marker = Marker()
        
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.ns = "waypoint"
        marker.id = 100 # unique id of the marker
        
        marker.type = Marker.LINE_STRIP
        
        for point in self.waypoints:
            x, y = point
            point_add = Point()
            point_add.x, point_add.y = x, y
            marker.points.append(point_add)
            
        # set pose of markers at zero
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05 # width of the waypoints. defaults to zero (invisible) if not set
        
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0 # opacity
        
        marker.lifetime.sec = 0 # marker will last forever until overwritten
        
        self.publisher_marker.publish(marker)
        
    def publish_velocity(self, speed, steering_angle):
        '''
        Publishes speed and angular velocity
        '''
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = speed
        ack_msg.drive.steering_angle = steering_angle
        
        self.publisher_vel.publish(ack_msg)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
