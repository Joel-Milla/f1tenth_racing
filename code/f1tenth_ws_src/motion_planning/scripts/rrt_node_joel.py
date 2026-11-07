#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math
from collections import deque
import random
from std_msgs.msg import ColorRGBA
import time
from scipy.ndimage import binary_dilation
import array


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class NodeTree(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.parent : NodeTree | None = None
        self.cost = None # only used in RRT*
        self.is_root = False
    
class Tree:
    def __init__(self, occupancy_grid, x_init, x_goal, k, curr_pos_grid, size_cell):
        self.count = 0
        self.occupancy_grid = occupancy_grid
        self.x_goal: np.ndarray = np.array(x_goal, dtype=float)          # shape (2,)        self.root = NodeTree()
        self.curr_pos_grid = curr_pos_grid
        self.size_cell = size_cell
        self.dq = 0.1
        self.threshold = 0.9 * self.dq # the threshold indicates difference needed to know if 'close' enough to the goal. close to dq
        
        self.positions = np.zeros((k+1, 2))
        self.node_list = []
        self.reached_goal = False
        self.goal_path : list[tuple[float, float]] = []
        
        self.add_node(x_init, None)
        self.create_tree_with_steps(k)
    
    def create_tree_with_steps(self, k):
        for _ in range(k):
            sample_point : tuple[float, float] = self.sample()
            nearest_node : NodeTree = self.nearest(sample_point)
            new_node : tuple[float, float] = self.steer(nearest_node, sample_point)
            new_node_collides = self.collides_with_obstacle(nearest_node, new_node)
            
            if not new_node_collides:
                self.add_node(new_node, nearest_node)
                
                if self.is_goal(): # check whether are close to goal
                    self.reached_goal = True
                    
                    for point in self.find_path(): # craete an array of xy positions of the path avoiding obstacles
                        self.goal_path.append((point.x, point.y))
                        
                    break # finish the loop

    def add_node(self, node : tuple[float, float], parent_node : NodeTree | None):
        """
        This adds a node to the data structure

        Args:
            node (float, float): a tuple representing the point to be added
            parent_node (NodeTree): the parent of the new node
        Returns:
            None
        """
        new_node = NodeTree()
        new_node.x, new_node.y = node
        new_node.parent = parent_node

        if self.count == 0:
            new_node.is_root = True
            self.root = new_node
        
        self.positions[self.count] = (new_node.x, new_node.y)
        self.node_list.append(new_node)

        self.count += 1
    
    def sample(self) -> tuple[float, float]:
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        
        if random.random() < 0.1:        # 10% of the time
            return (self.x_goal[0], self.x_goal[1])
        
        half_range_grid = self.size_cell * self.curr_pos_grid
        y = random.uniform(- half_range_grid, half_range_grid)            # random int, both ends included
        x = random.uniform(0, half_range_grid)  
        return (x, y)

    def nearest(self, sampled_point) -> NodeTree:
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        sx, sy = sampled_point
        diffs = self.positions[:self.count] - np.array([sx, sy])       # shape (N, 2), broadcasting subtracts from every row
        dist_sq = np.sum(diffs**2, axis=1)           # shape (N,), squared distance to each node
        nearest_idx = int(np.argmin(dist_sq))
        nearest_node = self.node_list[nearest_idx]
        return nearest_node
    
    def steer(self, nearest_node : NodeTree, sampled_point : tuple[float, float]) -> tuple[float, float]:
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        sx, sy = sampled_point
        dx, dy = sx - nearest_node.x, sy - nearest_node.y
        theta = np.arctan2(dy, dx)
        
        if (dx*dx + dy*dy) < (self.dq * self.dq):
            return (sx, sy)
        
        return (nearest_node.x + self.dq * np.cos(theta), nearest_node.y + self.dq * np.sin(theta))
            
    def collides_with_obstacle(self, nearest_node : NodeTree, new_node : tuple[float, float]):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): returns true if the path between the two nodes are in collision
                              with the occupancy grid
        """
        
        new_node_x, new_node_y = new_node
        dy, dx = nearest_node.y - new_node_y, nearest_node.x - new_node_x
        length = math.sqrt(dy*dy + dx*dx) # get distance between new and nearest node
        n = max(1, math.ceil(length / (self.size_cell / 2))) # number of steps to do in the direction of the nearest step
        
        near_node = np.array([ nearest_node.x, nearest_node.y])
        node = np.array(new_node)
        dist = node - near_node
        
        for iterator in range(n + 1): #iterate until reach the new_node
            new_coord = near_node + (iterator / n)*dist
            row, col = self.range_to_grid_frame((new_coord[0], new_coord[1]))
            if row is None:
                return True # if gone out of bounds, return true. so to avoid going towards outside grid
            
            if self.occupancy_grid[row][col] != 0: # occupied grid
                return True
        
        return False
    
    def range_to_grid_frame(self, position : tuple[float, float]):
        x, y = position

        col = self.curr_pos_grid + math.floor(x / self.size_cell)
        row = self.curr_pos_grid + math.floor(y / self.size_cell)
        
        if not self.within_grid_bounds(row, col):
            return (None, None)
        
        return (row, col)
    
    def within_grid_bounds(self, row, col):
        return 0 <= row < len(self.occupancy_grid) and 0 <= col < len(self.occupancy_grid)
    
    def is_goal(self):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        latest_node = self.positions[self.count - 1] # get (N,2) position of the latest node
        diff = latest_node - self.x_goal
        dist_sq = np.sum(diff**2)
        thrs_sq = self.threshold * self.threshold
        
        # this means taht are within distance
        if dist_sq < thrs_sq:
            return True
        
        return False    

    def find_path(self):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        curr_node = self.node_list[-1] # latest node
        parent_node = curr_node.parent
        
        while parent_node is not None:
            path.append(curr_node)
            
            curr_node = parent_node
            parent_node = curr_node.parent
        
        # Previous loop didnt add root
        path.append(curr_node)

        path.reverse() # invert the node to start in the root
        return path

# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        scan_topic = "/scan"
        self.in_simulation = False
        pose_topic = "ego_racecar/odom" if self.in_simulation else "/pf/viz/inferred_pose"

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        if self.in_simulation:
            self.pose_sub_ = self.create_subscription(
                Odometry,
                pose_topic,
                self.pose_callback,
                1)
        else:
            self.pose_sub_ = self.create_subscription(
                PoseStamped,
                pose_topic,
                self.pose_callback,
                1)
        self.pose_sub_

        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        self.scan_sub_
        
        self.pose = None
        
        # publishers
        self.publisher_vel = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.publisher_viz_occ_grid = self.create_publisher(OccupancyGrid, 'rrt/occupancy_grid', 10)
        self.publisher_viz_marker_waypoint = self.create_publisher(Marker, 'rrt/waypoints', 10)
        self.publisher_viz_marker_goal = self.create_publisher(Marker, 'rrt/goal', 10)
        self.publisher_viz_tree = self.create_publisher(MarkerArray, 'rrt/tree', 10)

        # load data
        self.waypoints = np.genfromtxt("/home/f1tenth/f1tenth_ws/src/pure_pursuit/waypoints/waypoints.csv", delimiter=',') # (N, 2) array size
        
        # set dimensions of the car
        total_width = 0.3
        half_width = total_width / 2.0 # divide by two to get the half of th diagonal
        
        total_length = 0.55
        half_length = total_length / 2.0 # divide by two to get the half of th diagonal
        self.r_rob = math.sqrt(half_width**2 + half_length**2) 
                
        # Occupancy grid params
        self.size_grid = 100
        self.size_cell = 0.1
        self.curr_pos_grid = int(self.size_grid // 2)
        self.occupied = 100
        self.occupancy_grid = np.zeros((self.size_grid, self.size_grid), dtype=np.int8)
        
        # params to tune
        self.k = 500 # iterations to do of the tree
        self.l_min = 0.5 # min possible l
        self.C = 0.8
        self.l = 3.0
        self.dq = 0.3
        
        self.rrt_l = 0.2
        self.scaling_rrob = 0.5
        
        # build the inflattion offsets
        self.inflation_offsets = self.build_inflation_offsets()
        k = max(abs(dr) for dr, _ in self.inflation_offsets)
        self.kernel = np.zeros((2 * k + 1, 2 * k + 1), dtype=bool)
        for dr, dc in self.inflation_offsets:
            self.kernel[dr + k, dc + k] = True
            
        self.create_timer(1.0, self.publish_waypoints)   # call publish_waypoints once per second
        
    def scan_callback(self, scan_msg : LaserScan):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        if self.pose is None:
            return
        
        self.latest_scan = scan_msg
        self.ranges = np.array(scan_msg.ranges, dtype=float) # get ranges of type float
        np.nan_to_num(self.ranges, False, np.inf, np.inf, np.inf) # in place, change nan/posinf/neginf to inf values
        self.angles = scan_msg.angle_min + np.arange(len(self.ranges)) * scan_msg.angle_increment
        
        ## Fill occupancy grid
        self.update_occupancy_grid() # create occupancy grid
           
    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        if self.in_simulation:
            self.pose = pose_msg.pose.pose      # Odometry
        else:
            self.pose = pose_msg.pose           # PoseStamped
        
        curr_position = self.pose.position
        
        closest_point_indx = self.get_closest_point_index(curr_position)
        curr_goal_point, indx = self.find_goal_point(curr_position, closest_point_indx)
        self.publish_goal(curr_goal_point)
        
        x_goal = self.world_to_car_frame(curr_goal_point[0], curr_goal_point[1])
        row, col = self.get_grid_position(x_goal)
        
        if row is None:
            self.get_logger().info('Waypoint not within bounds of grid.')
            return

        while self.occupancy_grid[row][col] == self.occupied: # iterate until your point is not in an occupied position
            indx = (indx + 1) % len(self.waypoints) # wrap around
            goal_point_x, goal_point_y = self.waypoints[indx]
            self.publish_goal((goal_point_x, goal_point_y))
            x_goal = self.world_to_car_frame(goal_point_x, goal_point_y)
            row, col = self.get_grid_position(x_goal)
            
            if row is None:
                self.get_logger().info('No free waypoint found within grid.')
                return
        
        if self.collides_with_obstacle(x_goal): # if collide with obstacle need to implement RRT
            tree = Tree(self.occupancy_grid, (0.0, 0.0), x_goal, self.k, self.curr_pos_grid, self.size_cell)
            
            self.publish_tree(tree)
            
            if not tree.reached_goal:
                # not able to reach goal forming tree. stop the car
                self.get_logger().info("Goal not reachable using the tree")
                self.publish_velocity(0.0, 0.0) # stop the car
                return
            
            self.pure_pursuit_for_rrt(tree.goal_path)
            
        else: # if no direct hit, follow pure pursuit
            self.publish_tree(None)
            
            # follow basic pure pursuit
            self.pure_pursuite_given_goal(x_goal)
        
    def pure_pursuite_given_goal(self, goal : tuple[float, float]):      
        # follow basic pure pursuit
        x, y = goal
        
        l = x*x + y*y
        curvature_arc = (2 * y) / l if l != 0 else 0 # curvature arc is 0 to avoid dividing by zero. keep straight
        speed = self.get_optimal_linear_speed(curvature_arc)
        # self.l = max(self.l_min, self.C * speed) #update l
        
        steering_angle = curvature_arc * self.dq
        # self.get_logger().info(f"steering_angle: {steering_angle} -- speed {speed}")
        
        self.publish_velocity(speed, steering_angle)
    
    def pure_pursuit_for_rrt(self, path : list[tuple[float, float]]):
        # find goal point which is closes point relative to length l
        goal_x, goal_y = path[-1] # if the next for loop doesn't find a close point with length l, then goal point will be the last point
        
        old_distance, new_distance = np.inf, np.inf
        
        for curr_indx, point in enumerate(path):
            x, y = point
            diff = (x)**2 + (y)**2 # the distance from current position to goal is just x^2 + y^2. as the goal_path is in cars coordiante frame and 
                                        # current position is at origin
            old_distance = new_distance
            new_distance = abs(math.sqrt(diff) - self.rrt_l) # use special rrt_l to hav a better steering
            
            if new_distance > old_distance: # means that previous one was closest (that is in front) to value of l
                goal_x, goal_y = path[curr_indx - 1]
                break
            
        # publish that goal
        self.publish_goal(self.car_to_world_frame(goal_x, goal_y))
        
        # intiate pure pursuit based on that goal
        distance_l = goal_x*goal_x + goal_y*goal_y
        self.pure_pursuite_given_goal((goal_x, goal_y)) # give the distance l to the target point, as that l will be use for the curvature
        
    def collides_with_obstacle(self, goal : tuple[float, float]):
        """
        This method should return whether the path between origin and goal is
        collision free.

        Args:
            goal (float, float): goal point
        Returns:
            collision (bool): returns true if the path between the two points are in collision
                              with the occupancy grid
        """
        
        new_node_x, new_node_y = goal
        dy, dx = 0.0 - new_node_y, 0.0 - new_node_x
        length = math.sqrt(dy*dy + dx*dx) # get distance between new and nearest node
        n = max(1, math.ceil(length / (self.size_cell / 2))) # number of steps to do in the direction of the nearest step
        
        goal_dir = np.array(goal)
        
        for iterator in range(n + 1): #iterate until reach the new_node
            new_coord = (iterator / n)*goal_dir # move towards the goal
            row, col = self.get_grid_position((new_coord[0], new_coord[1]))
            if row is None:
                return True # if gone out of bounds, return true. so to avoid going towards outside grid
            
            if self.occupancy_grid[row][col] != 0: # occupied grid
                return True
        
        return False
  
    def get_closest_point_index(self, pose):
        """
        Finds the waypoint closest to the car's current position

        Args:
            pose (geometry_msgs/Point): current position of the car in the map frame
        Returns:
            min_indx (int): index in self.waypoints of the closest waypoint
        """
        p = np.array([pose.x, pose.y])
        return int(np.argmin(np.sum((self.waypoints - p) ** 2, axis=1)))
      
    def find_goal_point(self, pose, closest_point_indx) -> tuple[np.ndarray,int]:
        """
        Finds the goal point to track: starting from the closest waypoint, the waypoint
        whose distance to the car is closest to the lookahead distance self.l

        Args:
            pose (geometry_msgs/Point): current position of the car in the map frame
            closest_point_indx (int): index of the waypoint closest to the car, where the search starts
        Returns:
            goal_point (np.ndarray): [x, y] of the goal waypoint in the map frame
            indx (int) index of the goal waypoint
        """
        p = np.array([pose.x, pose.y])
        n = len(self.waypoints)
        order = (closest_point_indx + np.arange(n)) % n            # walk forward from the closest waypoint, wrapping around
        dists = np.linalg.norm(self.waypoints[order] - p, axis=1)
        diffs = np.abs(dists - self.l)

        # same rule as your loop: stop at the first point where |dist - l| starts growing again
        rising = np.nonzero(diffs[1:] > diffs[:-1])[0]
        if len(rising) == 0:
            self.get_logger().info("No waypoint was found")
            return (np.array([]), -1)
        i = int(order[rising[0]])
        return (self.waypoints[i], i)

    def update_occupancy_grid(self):
        r, a = self.ranges, self.angles

        # keep only valid beams (all at once, no loop)
        valid = np.isfinite(r) & (r >= self.latest_scan.range_min) & (r <= self.latest_scan.range_max)
        r, a = r[valid], a[valid]

        # every beam -> (row, col) at once
        cols = self.curr_pos_grid + np.floor(r * np.cos(a) / self.size_cell).astype(int)
        rows = self.curr_pos_grid + np.floor(r * np.sin(a) / self.size_cell).astype(int)
        inside = (rows >= 0) & (rows < self.size_grid) & (cols >= 0) & (cols < self.size_grid)

        # mark hit cells (duplicates collapse automatically)
        hits = np.zeros((self.size_grid, self.size_grid), dtype=bool)
        hits[rows[inside], cols[inside]] = True

        # stamp your disk around every hit in one call
        inflated = binary_dilation(hits, structure=self.kernel)

        self.occupancy_grid.fill(0)                  # in place, so the Tree's reference stays valid
        self.occupancy_grid[inflated] = self.occupied
        self.publish_occupancy_grid()
    
    def build_inflation_offsets(self):
        '''
        Single opeartion ran at initialization, that marks on each obstacle how to fill a bubble around them in occupancy grid
        '''
        
        margin = math.sqrt(2) * self.size_cell # (meters) this is the diagonal of the square
        reach = margin + self.r_rob # (meters) radius + digonal of a single cell
        reach *= self.scaling_rrob # reduce reach to 80% of og value to have a less occupied occupancy grid
        k = math.ceil(reach / self.size_cell) # (cells) convert radius to cell units
        reach_cells_sq = (reach / self.size_cell) * (reach / self.size_cell)
        
        offsets = [] # array that will have shape to round up obstacles
        # can think that we are traversing a square that closes in the cricle. we iterate over all cells, and 
        # check if within circle (using radius in cell units) if yes, save the offset
        for drow in range(-k, k + 1): 
            for dcol in range(-k, k + 1):
                if (drow*drow + dcol*dcol) < reach_cells_sq:
                    offsets.append((drow, dcol))
    
        return offsets
    
    def mark_obstacle(self, row, col):
        '''
        Given a row and column, mark the obstacle as occupied and the neighbors around the bubble
        '''
        for drow, dcol in self.inflation_offsets:
            new_row = row + drow
            new_col = col + dcol
            if self.within_grid_bounds(new_row, new_col):
                self.occupancy_grid[new_row][new_col] = self.occupied           
    
    def range_to_grid_frame(self, curr_range, angle):
        x = curr_range * np.cos(angle)   # forward, meters
        y = curr_range * np.sin(angle)   # left, meters

        col = self.curr_pos_grid + math.floor(x / self.size_cell)
        row = self.curr_pos_grid + math.floor(y / self.size_cell)
        
        if not self.within_grid_bounds(row, col):
            return (None, None)
        
        return (row, col)
    
    def get_grid_position(self, position : tuple[float, float]):
        x, y = position

        col = self.curr_pos_grid + math.floor(x / self.size_cell)
        row = self.curr_pos_grid + math.floor(y / self.size_cell)
        
        if not self.within_grid_bounds(row, col):
            return (None, None)
        
        return (row, col)
    
    def within_grid_bounds(self, row, col):
        return 0 <= row < self.size_grid and 0 <= col < self.size_grid

    def publish_velocity(self, speed, steering_angle):
        '''
        Publishes speed and angular velocity
        '''
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = speed
        ack_msg.drive.steering_angle = steering_angle
        
        self.publisher_vel.publish(ack_msg)
        
    def publish_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.latest_scan.header.stamp
        msg.header.frame_id = self.latest_scan.header.frame_id  # the lidar's frame

        msg.info.resolution = self.size_cell   # meters per cell
        msg.info.width = self.size_grid        # number of columns (x, forward)
        msg.info.height = self.size_grid       # number of rows (y, left)

        # Where the corner of cell (0, 0) sits, relative to the car
        msg.info.origin.position.x = -self.curr_pos_grid * self.size_cell
        msg.info.origin.position.y = -self.curr_pos_grid * self.size_cell
        msg.info.origin.orientation.w = 1.0    # no rotation

        msg.data = array.array('b', self.occupancy_grid.ravel().tobytes())
        self.publisher_viz_occ_grid.publish(msg)

    def make_marker(self, marker_id, marker_type, scale, rgb) -> Marker:
        m = Marker()
        m.header.frame_id = self.latest_scan.header.frame_id   # same frame as the grid
        m.header.stamp = self.latest_scan.header.stamp
        m.ns = 'rrt'
        m.id = marker_id                  # unique id per marker, so each one gets replaced, not duplicated
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale   # line width for lines, diameter for spheres
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        return m

    def publish_tree(self, tree):
        markers = MarkerArray()
        
        if tree is None: # when the tree is empty, just publish empty so vis is reseted
            clear = Marker()                     # a Marker, which does have .action
            clear.action = Marker.DELETEALL
            markers.markers.append(clear)        # put it inside the MarkerArray
            self.publisher_viz_tree.publish(markers)
            return

        # Tree edges: one (parent, child) pair per node
        edges = self.make_marker(0, Marker.LINE_LIST, 0.01, (0.6, 0.6, 0.6))
        for node in tree.node_list:
            if node.parent is not None:
                edges.points.append(Point(x=float(node.parent.x), y=float(node.parent.y), z=0.0))
                edges.points.append(Point(x=float(node.x), y=float(node.y), z=0.0))
        markers.markers.append(edges)

        # Path: connected line from root to goal
        path = self.make_marker(1, Marker.LINE_STRIP, 0.04, (0.0, 1.0, 0.0))
        if tree.reached_goal:
            for x,y in tree.goal_path:
                path.points.append(Point(x=float(x), y=float(y), z=0.0))
        else:
            path.action = Marker.DELETE   # clear last cycle's path if none was found
        markers.markers.append(path)

        # Goal
        goal = self.make_marker(2, Marker.SPHERE, 0.15, (1.0, 0.0, 0.0))
        goal.pose.position.x = float(tree.x_goal[0])
        goal.pose.position.y = float(tree.x_goal[1])
        markers.markers.append(goal)

        self.publisher_viz_tree.publish(markers)

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
        
        self.publisher_viz_marker_waypoint.publish(marker)
        
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

    def car_to_world_frame(self, cx, cy) -> tuple[float, float]:
        p = self.pose.position
        q = self.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))   # quaternion → heading

        # 1) rotate by +yaw: express the car-frame offset in world axes
        dx = math.cos(yaw) * cx - math.sin(yaw) * cy
        dy = math.sin(yaw) * cx + math.cos(yaw) * cy

        # 2) translate: add the car's world position
        return (p.x + dx, p.y + dy)
    
    def publish_goal(self, goal : tuple[float, float]):
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
        
        self.publisher_viz_marker_goal.publish(marker)
        
    def get_optimal_linear_speed(self, steering_angle):
        steering_angle_degrees = abs(np.rad2deg(steering_angle))
        match steering_angle_degrees:
            case angle if abs(angle) <= 10:
                return 1.2
            case angle if abs(angle) <= 20:
                return 1.2
        return 1.1
  
    #######################################################
    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
