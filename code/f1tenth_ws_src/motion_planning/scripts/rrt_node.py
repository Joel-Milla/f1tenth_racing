#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pure_pursuit_node import (CAR_POSE_TOPIC, DEFAULT_WAYPOINTS, SIM_POSE_TOPIC,
                               PurePursuitController, delete_marker, drive_msg,
                               global_to_local, line_marker, local_to_global,
                               pose_from_msg, resample_polyline, sphere_marker)


class Tree:
    def __init__(self, capacity, root):
        '''
        Build an empty tree with a fixed capacity and insert the root node.

        Input:
            capacity: int, maximum number of nodes the preallocated arrays can hold.
            root: (x, y) tuple, coordinates of the root node (the car position).

        Output:
            None. Initializes the x/y/parent/cost arrays, the children lists and the
            node counter n, with the root stored at index 0.
        '''
        self.x = np.empty(capacity)
        self.y = np.empty(capacity)
        self.parent = np.full(capacity, -1, dtype=int)
        self.cost = np.zeros(capacity)
        self.children = [[] for _ in range(capacity)]
        self.n = 0
        self.add(root, -1, 0.0)

    def add(self, point, parent, cost):
        '''
        Append a new node to the tree and link it to its parent.

        Input:
            point: (x, y) tuple, coordinates of the new node.
            parent: int, index of the parent node (-1 for the root).
            cost: float, cost-to-come from the root to this node.

        Output:
            int, the index of the node that was just added.
        '''
        i = self.n
        self.x[i], self.y[i] = point
        self.parent[i] = parent
        self.cost[i] = cost
        if parent >= 0:
            self.children[parent].append(i)
        self.n += 1
        return i

    def point(self, i):
        '''
        Read the coordinates of one node.

        Input:
            i: int, node index.

        Output:
            (x, y) tuple of floats, the coordinates of node i.
        '''
        return float(self.x[i]), float(self.y[i])

    def xy(self):
        '''
        Expose the coordinates of every node currently in the tree.

        Input:
            None.

        Output:
            (xs, ys) tuple of numpy arrays, each of length n (views on the
            preallocated arrays, unused capacity excluded).
        '''
        return self.x[:self.n], self.y[:self.n]

    def reparent(self, i, new_parent, new_cost):
        '''
        Move a node under a new parent and propagate the cost change to its subtree.

        Input:
            i: int, index of the node to reparent.
            new_parent: int, index of the new parent node.
            new_cost: float, new cost-to-come of node i.

        Output:
            None. Updates the parent/children links and shifts the cost of node i and
            of every descendant by the same delta.
        '''
        self.children[self.parent[i]].remove(i)
        self.parent[i] = new_parent
        self.children[new_parent].append(i)
        delta = new_cost - self.cost[i]
        stack = [i]
        while stack:
            k = stack.pop()
            self.cost[k] += delta
            stack.extend(self.children[k])


class RRT(Node):
    def __init__(self):
        '''
        Start the RRT ROS node: read parameters, build the occupancy grid and set up I/O.

        Input:
            None (values come from ROS parameters, which fall back to the defaults
            declared in this method).

        Output:
            None. Loads the waypoints, allocates the local occupancy grid and its
            helper masks, resets the planner state, and creates the pose/scan
            subscriptions, the drive and marker publishers and the waypoint timer.
        '''
        super().__init__("rrt_node")

        defaults = {
            "use_sim": False,
            "waypoints_file": DEFAULT_WAYPOINTS,
            "loop": True,
            "scan_topic": "/scan",
            "drive_topic": "/drive",

            "lookahead": 1.0,
            "rrt_lookahead": 0.8,
            "speed": 2.0,
            "rrt_speed": 1.3,
            "wheelbase": 0.33,
            "max_steer": 0.36,
            "control_period": 0.02,
            "oc_resolution": 0.05,
            "oc_length": 7.0,
            "oc_width": 4.0,
            "oc_back_frac": 0.25,
            "inflation_radius": 0.18,
            "laser_offset_x": 0.0,
            "oc_viz_every": 4,

            "goal_dist": 3.0,
            "min_goal_dist": 1.0,
            "max_iter": 1500,
            "step": 0.4,
            "goal_bias": 0.15,
            "goal_tol": 0.5,
            "use_rrt_star": True,
            "rrt_star_radius": 0.6,
            "rrt_star_extra_iter": 150,
            "replan_period": 1.0,
            "clear_ticks_to_resume": 5
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)

        self.pp = PurePursuitController(self.wheelbase, self.max_steer)
        self.waypoints = PurePursuitController.load_waypoints(self.waypoints_file)
        self.get_logger().info(f"loaded {len(self.waypoints)} waypoints from {self.waypoints_file}")

        # oc[ix, iy]: 0 free, 1 occupied (inflated + everything the laser didn't see)
        nx = int(round(self.oc_length / self.oc_resolution))
        ny = int(round(self.oc_width / self.oc_resolution))
        self.oc = np.ones((nx, ny), dtype=np.uint8)
        self.oc_origin = (int(round(nx * self.oc_back_frac)), ny // 2)
        self.oc_frame = "ego_racecar/laser"
        self.oc_stamp = None
        self.inflation_offsets = self.build_inflation_offsets()
        self.car_local = (-self.laser_offset_x, 0.0)
        gx, gy = self.grid_to_world(*np.indices((nx, ny)))
        self.near_car = np.hypot(gx - self.car_local[0], gy - self.car_local[1]) < self.inflation_radius
        self.reach = np.zeros((nx, ny), dtype=bool)

        self.have_scan = False
        self.scan_count = 0
        self.rrt_path = None
        self.rrt_time = -math.inf
        self.clear_count = 0
        self.last_control = -math.inf
        self.free_samples = None
        self.rng = np.random.default_rng()

        if self.use_sim:
            self.create_subscription(Odometry, SIM_POSE_TOPIC, self.pose_callback, 1)
        else:
            self.create_subscription(PoseStamped, CAR_POSE_TOPIC, self.pose_callback, 1)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.oc_pub = self.create_publisher(MarkerArray, "/occupancy_markers", 10)
        self.rrt_pub = self.create_publisher(MarkerArray, "/rrt_markers", 10)
        self.wp_pub = self.create_publisher(MarkerArray, "/waypoint_markers", 10)
        self.create_timer(1.0, self.publish_waypoints)


    def now(self):
        '''
        Read the current ROS time.

        Input:
            None.

        Output:
            float, the node clock time in seconds.
        '''
        return self.get_clock().now().nanoseconds * 1e-9

    def world_to_grid(self, x, y):
        '''
        Convert laser-frame coordinates into occupancy grid cell indices.

        Input:
            x: float or array, laser-frame x coordinate(s) in metres.
            y: float or array, laser-frame y coordinate(s) in metres.

        Output:
            (ix, iy) tuple of int arrays, the grid indices (not clamped, so they can
            fall outside the grid).
        '''
        cx, cy = self.oc_origin
        ix = np.floor(np.asarray(x) / self.oc_resolution + cx).astype(int)
        iy = np.floor(np.asarray(y) / self.oc_resolution + cy).astype(int)
        return ix, iy

    def grid_to_world(self, ix, iy):
        '''
        Convert occupancy grid cell indices into laser-frame coordinates.

        Input:
            ix: int or array, grid index along x.
            iy: int or array, grid index along y.

        Output:
            (x, y) tuple of float arrays, the laser-frame coordinates of the cell
            centres in metres.
        '''
        cx, cy = self.oc_origin
        x = (np.asarray(ix) - cx + 0.5) * self.oc_resolution
        y = (np.asarray(iy) - cy + 0.5) * self.oc_resolution
        return x, y

    def in_grid(self, x, y):
        '''
        Test whether laser-frame points fall inside the occupancy grid.

        Input:
            x: float or array, laser-frame x coordinate(s) in metres.
            y: float or array, laser-frame y coordinate(s) in metres.

        Output:
            bool or bool array, True where the point maps to a cell inside the grid.
        '''
        ix, iy = self.world_to_grid(x, y)
        nx, ny = self.oc.shape
        return (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def map_to_laser(self, px, py, pose):
        '''
        Transform points from the global map frame into the laser frame.

        Input:
            px: float or array, map-frame x coordinate(s).
            py: float or array, map-frame y coordinate(s).
            pose: (x, y, yaw) tuple, current car pose in the map frame.

        Output:
            (lx, ly) tuple of floats or arrays, the same points in the laser frame
            (the laser mounting offset along x is removed).
        '''
        lx, ly = global_to_local(px, py, pose)
        return lx - self.laser_offset_x, ly

    def laser_to_map(self, lx, ly, pose):
        '''
        Transform points from the laser frame into the global map frame.

        Input:
            lx: float or array, laser-frame x coordinate(s).
            ly: float or array, laser-frame y coordinate(s).
            pose: (x, y, yaw) tuple, current car pose in the map frame.

        Output:
            (x, y) tuple of floats or arrays, the same points in the map frame.
        '''
        return local_to_global(np.asarray(lx) + self.laser_offset_x, ly, pose)

    def build_inflation_offsets(self):
        '''
        Precompute the disk of cell offsets used to inflate laser hits.

        Input:
            None (uses the inflation_radius and oc_resolution parameters).

        Output:
            numpy array of shape (k, 2) with the (dx, dy) cell offsets whose distance
            from the centre is within the inflation radius.
        '''
        r_cells = self.inflation_radius / self.oc_resolution
        r = int(np.ceil(r_cells))
        dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        disk = dx**2 + dy**2 <= r_cells**2
        return np.stack([dx[disk], dy[disk]], axis=1)

    def points_blocked(self, xs, ys):
        '''
        Check which points are unsafe to drive through.

        Input:
            xs: float or array, laser-frame x coordinate(s).
            ys: float or array, laser-frame y coordinate(s).

        Output:
            bool array, True where the point is occupied or outside the grid. Points
            within the inflation radius of the car itself are always reported free so
            the planner is not trapped by its own footprint.
        '''
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        ix, iy = self.world_to_grid(xs, ys)
        nx, ny = self.oc.shape
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        blocked = np.ones(len(xs), dtype=bool)
        blocked[inb] = self.oc[ix[inb], iy[inb]] == 1
        near_car = np.hypot(xs - self.car_local[0], ys - self.car_local[1]) < self.inflation_radius
        return blocked & ~near_car

    def polyline_blocked(self, xs, ys):
        '''
        Check whether a polyline crosses any occupied cell.

        Input:
            xs: array, laser-frame x coordinates of the polyline vertices.
            ys: array, laser-frame y coordinates of the polyline vertices.

        Output:
            bool, True if any point sampled along the polyline (at half the grid
            resolution) is blocked.
        '''
        rx, ry = resample_polyline(xs, ys, self.oc_resolution / 2)
        return bool(self.points_blocked(rx, ry).any())

    def build_occupancy_map(self, scan_msg):
        '''
        Rebuild the local occupancy grid from one laser scan.

        Input:
            scan_msg: sensor_msgs/LaserScan, the latest scan.

        Output:
            None. Fills self.oc with 1 (occupied/unknown) everywhere, inflates every
            laser hit by the inflation disk, then carves the rays back to 0 (free) up
            to each measured range, and records the scan frame and stamp.
        '''
        self.oc_frame = scan_msg.header.frame_id
        self.oc_stamp = scan_msg.header.stamp

        self.oc.fill(1)

        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        r_free = np.where(np.isinf(ranges), scan_msg.range_max, ranges)
        valid = np.isfinite(r_free) & (r_free >= scan_msg.range_min)
        r_free = np.where(valid, np.minimum(r_free, scan_msg.range_max), 0.0)

        nx, ny = self.oc.shape

        hit = valid & (ranges < scan_msg.range_max)
        hx, hy = self.world_to_grid(ranges[hit] * cos_a[hit], ranges[hit] * sin_a[hit])
        cells = (np.stack([hx, hy], axis=1)[:, None, :]
                 + self.inflation_offsets[None, :, :]).reshape(-1, 2)
        in_grid = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
        inflated = np.zeros((nx, ny), dtype=bool)
        inflated[cells[in_grid, 0], cells[in_grid, 1]] = True

        cx, cy = self.oc_origin
        max_d = np.hypot(max(cx, nx - cx), max(cy, ny - cy)) * self.oc_resolution
        d = np.arange(0.0, max_d, self.oc_resolution / 2)
        sx, sy = self.world_to_grid(d[None, :] * cos_a[:, None],
                                    d[None, :] * sin_a[:, None])
        free = ((sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
                & (d[None, :] < r_free[:, None]))
        fx, fy = sx[free], sy[free]
        keep = ~inflated[fx, fy]
        self.oc[fx[keep], fy[keep]] = 0

    def reachable_cells(self):
        '''
        Flood fill the free space that is actually connected to the car.

        Input:
            None (uses the current self.oc grid and the near_car mask).

        Output:
            bool array shaped like the grid, True for free cells reachable from the
            car through 4-connected free space.
        '''
        free = (self.oc == 0) | self.near_car
        reach = self.near_car.copy()
        count = int(reach.sum())
        while True:
            grown = reach.copy()
            grown[1:, :] |= reach[:-1, :]
            grown[:-1, :] |= reach[1:, :]
            grown[:, 1:] |= reach[:, :-1]
            grown[:, :-1] |= reach[:, 1:]
            grown &= free
            new_count = int(grown.sum())
            if new_count == count:
                return grown
            reach, count = grown, new_count

    def points_reachable(self, xs, ys):
        '''
        Check which points lie in the free space connected to the car.

        Input:
            xs: float or array, laser-frame x coordinate(s).
            ys: float or array, laser-frame y coordinate(s).

        Output:
            bool array, True where the point maps to a reachable cell (points outside
            the grid are False).
        '''
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        ix, iy = self.world_to_grid(xs, ys)
        nx, ny = self.oc.shape
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        out = np.zeros(len(xs), dtype=bool)
        out[inb] = self.reach[ix[inb], iy[inb]]
        return out

    def scan_callback(self, scan_msg):
        '''
        Handle an incoming laser scan.

        Input:
            scan_msg: sensor_msgs/LaserScan, the new scan.

        Output:
            None. Rebuilds the occupancy grid and the reachability mask, flags that a
            scan is available, and republishes the grid markers every oc_viz_every
            scans.
        '''
        self.build_occupancy_map(scan_msg)
        self.reach = self.reachable_cells()
        self.have_scan = True
        self.scan_count += 1
        if self.scan_count % self.oc_viz_every == 0:
            self.viz_oc()

    def pose_callback(self, pose_msg):
        '''
        Run one control step: pick a goal, choose waypoints or RRT, and publish a drive command.

        Input:
            pose_msg: nav_msgs/Odometry in simulation or geometry_msgs/PoseStamped on
            the car, the current pose.

        Output:
            None. Publishes an AckermannDriveStamped command (or a stop command when
            no goal or no path is available) plus the goal and tracked-point markers.
            Rate limited to one update per control_period.
        '''
        now = self.now()
        if not self.have_scan or now - self.last_control < self.control_period:
            return
        self.last_control = now
        pose = pose_from_msg(pose_msg)

        goal = self.select_goal(pose)
        if goal is None:
            self.get_logger().warn("no free goal on the waypoints ahead, stopping",
                                   throttle_duration_sec=1.0)
            self.stop()
            return
        goal_local, goal_map, direct_clear = goal

        self.clear_count = self.clear_count + 1 if direct_clear else 0
        use_waypoints = direct_clear and (self.rrt_path is None
                                          or self.clear_count >= self.clear_ticks_to_resume)

        if use_waypoints:
            if self.rrt_path is not None:
                self.rrt_path = None
                self.clear_rrt_markers()
            path, L, speed, loop = self.waypoints, self.lookahead, self.speed, self.loop
        else:
            if self.needs_replan(pose, now):
                new_path = self.plan(pose, goal_local)
                if new_path is not None:
                    self.rrt_path, self.rrt_time = new_path, now
                elif self.rrt_path is not None and self.path_blocked_ahead(self.rrt_path, pose):
                    self.rrt_path = None
            if self.rrt_path is None:
                self.get_logger().warn("RRT found no path, stopping", throttle_duration_sec=1.0)
                self.stop()
                return
            path, L, speed, loop = self.rrt_path, self.rrt_lookahead, self.rrt_speed, False

        target = self.pp.lookahead_point(path, pose, L, loop)
        steer = self.pp.steering_angle(pose, target)
        self.drive_pub.publish(drive_msg(self.get_clock().now().to_msg(), speed, steer))
        self.publish_targets(goal_map, target)

    def stop(self):
        '''
        Command the car to halt.

        Input:
            None.

        Output:
            None. Publishes a drive message with zero speed and zero steering angle.
        '''
        self.drive_pub.publish(drive_msg(self.get_clock().now().to_msg(), 0.0, 0.0))


    def select_goal(self, pose):
        '''
        Pick the goal point on the waypoint path ahead of the car.

        Input:
            pose: (x, y, yaw) tuple, current car pose in the map frame.

        Output:
            None if no usable goal exists, otherwise a tuple
            (goal_local, goal_map, direct_clear) where goal_local is the (x, y) goal
            in the laser frame, goal_map is the (x, y) goal in the map frame and
            direct_clear is a bool saying whether the waypoint path up to the goal is
            free of obstacles. If the waypoint goal is unreachable the nearest
            reachable free cell is used instead and direct_clear is False.
        '''
        wp = self.waypoints
        order = self.pp.forward_indices(len(wp), self.pp.closest_index(wp, pose), self.loop)
        pts = wp[order]
        lx, ly = self.map_to_laser(pts[:, 0], pts[:, 1], pose)

        ahead = np.flatnonzero(lx > self.car_local[0])
        if len(ahead) == 0:
            return None
        k0 = ahead[0]
        pts, lx, ly = pts[k0:], lx[k0:], ly[k0:]
        outside = np.flatnonzero(~self.in_grid(lx, ly))
        end = outside[0] if len(outside) else len(pts)
        if end == 0:
            return None
        pts, lx, ly = pts[:end], lx[:end], ly[:end]

        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))])
        s += math.hypot(lx[0] - self.car_local[0], ly[0] - self.car_local[1])
        kg = min(int(np.searchsorted(s, self.goal_dist)), len(s) - 1)

        ok = self.points_reachable(lx, ly)
        k = kg
        if not ok[kg]:
            ok_after = np.flatnonzero(ok[kg:])
            if len(ok_after):
                k = kg + int(ok_after[0])
            else:
                goal = self.nearest_reachable_cell(lx[kg], ly[kg])
                if goal is None:
                    return None
                gx, gy = self.laser_to_map(goal[0], goal[1], pose)
                return goal, (float(gx), float(gy)), False

        direct_clear = not self.polyline_blocked(np.r_[self.car_local[0], lx[:k + 1]],
                                                 np.r_[self.car_local[1], ly[:k + 1]])
        return (float(lx[k]), float(ly[k])), pts[k], direct_clear

    def nearest_reachable_cell(self, x, y):
        '''
        Find the reachable free cell closest to a requested point.

        Input:
            x: float, laser-frame x coordinate of the desired point.
            y: float, laser-frame y coordinate of the desired point.

        Output:
            (x, y) tuple of floats, the centre of the closest reachable free cell that
            is at least min_goal_dist ahead of the car, or None if no such cell exists.
        '''
        free = np.argwhere(self.reach & (self.oc == 0))
        fx, fy = self.grid_to_world(free[:, 0], free[:, 1])
        ok = (fx - self.car_local[0]) >= self.min_goal_dist
        if not ok.any():
            return None
        fx, fy = fx[ok], fy[ok]
        i = int(np.argmin((fx - x) ** 2 + (fy - y) ** 2))
        return float(fx[i]), float(fy[i])

    def path_blocked_ahead(self, path, pose):
        '''
        Check whether the remaining part of a path is obstructed.

        Input:
            path: numpy array of shape (n, 2), path points in the map frame.
            pose: (x, y, yaw) tuple, current car pose in the map frame.

        Output:
            bool, True if the path from the point closest to the car onwards crosses
            an occupied cell.
        '''
        i0 = self.pp.closest_index(path, pose)
        lx, ly = self.map_to_laser(path[i0:, 0], path[i0:, 1], pose)
        return self.polyline_blocked(np.r_[self.car_local[0], lx], np.r_[self.car_local[1], ly])

    def needs_replan(self, pose, now):
        '''
        Decide whether the stored RRT path has to be recomputed.

        Input:
            pose: (x, y, yaw) tuple, current car pose in the map frame.
            now: float, current time in seconds.

        Output:
            bool, True when there is no path, the path is older than replan_period,
            the car has almost reached its end, or the path ahead is blocked.
        '''
        if self.rrt_path is None or now - self.rrt_time > self.replan_period:
            return True
        end = self.rrt_path[-1]
        if math.hypot(end[0] - pose[0], end[1] - pose[1]) < self.rrt_lookahead:
            return True
        return self.path_blocked_ahead(self.rrt_path, pose)

    def plan(self, pose, goal):
        '''
        Grow an RRT (or RRT*) tree in the local free space and extract a path to the goal.

        Input:
            pose: (x, y, yaw) tuple, current car pose in the map frame.
            goal: (x, y) tuple, goal point in the laser frame.

        Output:
            numpy array of shape (n, 2) with the shortcut and resampled path in the
            map frame, or None if no collision-free path to the goal was found. Also
            publishes the tree and path markers.
        '''
        free = np.argwhere(self.oc == 0)
        fx, fy = self.grid_to_world(free[:, 0], free[:, 1])
        ahead = fx >= self.car_local[0]
        if not ahead.any():
            return None
        self.free_samples = (fx[ahead], fy[ahead])

        tree = Tree(self.max_iter + 1, self.car_local)
        goal_parents = []
        stop_at = self.max_iter
        for it in range(self.max_iter):
            if it >= stop_at:
                break
            sampled = self.sample(goal)
            nearest = self.nearest(tree, sampled)
            new = self.steer(tree, nearest, sampled)
            if self.check_collision(tree.point(nearest), new):
                continue

            if self.use_rrt_star:
                near = self.near(tree, new)
                parent, cost = self.choose_parent(tree, near, nearest, new)
                idx = tree.add(new, parent, cost)
                self.rewire(tree, near, idx)
            else:
                idx = tree.add(new, nearest, self.cost(tree, nearest) + self.line_cost(tree.point(nearest), new))

            if self.is_goal(tree, idx, goal):
                if not goal_parents:
                    stop_at = it + 1 + (self.rrt_star_extra_iter if self.use_rrt_star else 0)
                goal_parents.append(idx)

        if not goal_parents:
            self.publish_tree(tree, None)
            return None

        best = min(goal_parents, key=lambda i: self.cost(tree, i) + self.line_cost(tree.point(i), goal))
        path = np.vstack([self.find_path(tree, best), goal])
        path = self.shortcut(path)

        rx, ry = resample_polyline(path[:, 0], path[:, 1], 0.1)
        mx, my = self.laser_to_map(rx, ry, pose)
        path_map = np.column_stack([mx, my])
        self.publish_tree(tree, path_map)
        return path_map

    def sample(self, goal):
        '''
        Draw one random sample for the tree expansion.

        Input:
            goal: (x, y) tuple, goal point in the laser frame.

        Output:
            (x, y) tuple of floats: the goal itself with probability goal_bias,
            otherwise a random free cell centre jittered inside its cell.
        '''
        if self.rng.random() < self.goal_bias:
            return goal
        fx, fy = self.free_samples
        i = self.rng.integers(len(fx))
        jx, jy = (self.rng.random(2) - 0.5) * self.oc_resolution
        return float(fx[i] + jx), float(fy[i] + jy)

    def nearest(self, tree, sampled_point):
        '''
        Find the tree node closest to a sampled point.

        Input:
            tree: Tree, the current search tree.
            sampled_point: (x, y) tuple, the sampled point in the laser frame.

        Output:
            int, index of the nearest node in the tree (Euclidean distance).
        '''
        xs, ys = tree.xy()
        return int(np.argmin((xs - sampled_point[0]) ** 2 + (ys - sampled_point[1]) ** 2))

    def steer(self, tree, nearest_node, sampled_point):
        '''
        Take a bounded step from a tree node towards a sampled point.

        Input:
            tree: Tree, the current search tree.
            nearest_node: int, index of the node to expand from.
            sampled_point: (x, y) tuple, the sampled point in the laser frame.

        Output:
            (x, y) tuple of floats, the sampled point itself if it is within one step,
            otherwise the point exactly step metres along the direction to it.
        '''
        x0, y0 = tree.point(nearest_node)
        dx, dy = sampled_point[0] - x0, sampled_point[1] - y0
        d = math.hypot(dx, dy)
        if d <= self.step:
            return float(sampled_point[0]), float(sampled_point[1])
        return x0 + dx * self.step / d, y0 + dy * self.step / d

    def check_collision(self, p0, p1):
        '''
        Check whether the straight segment between two points hits an obstacle.

        Input:
            p0: (x, y) tuple, start point in the laser frame.
            p1: (x, y) tuple, end point in the laser frame.

        Output:
            bool, True if any point sampled along the segment (at half the grid
            resolution) is blocked.
        '''
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(math.ceil(dist / (self.oc_resolution / 2))) + 1)
        xs = np.linspace(p0[0], p1[0], n)
        ys = np.linspace(p0[1], p1[1], n)
        return bool(self.points_blocked(xs, ys).any())

    def segments_blocked(self, p, xs, ys):
        '''
        Collision-check a whole fan of segments that share one endpoint at once.

        Input:
            p: (x, y) tuple, the common endpoint in the laser frame.
            xs: array, x coordinates of the other endpoints.
            ys: array, y coordinates of the other endpoints.

        Output:
            bool array of length len(xs), True for each segment that crosses a blocked
            cell.
        '''
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if len(xs) == 0:
            return np.zeros(0, dtype=bool)
        longest = float(np.max(np.hypot(xs - p[0], ys - p[1])))
        n = max(2, int(math.ceil(longest / (self.oc_resolution / 2))) + 1)
        t = np.linspace(0.0, 1.0, n)[None, :]
        px = p[0] + (xs[:, None] - p[0]) * t
        py = p[1] + (ys[:, None] - p[1]) * t
        return self.points_blocked(px.ravel(), py.ravel()).reshape(len(xs), n).any(axis=1)

    def is_goal(self, tree, latest_added_node, goal):
        '''
        Test whether a newly added node can connect to the goal.

        Input:
            tree: Tree, the current search tree.
            latest_added_node: int, index of the node just added.
            goal: (x, y) tuple, goal point in the laser frame.

        Output:
            bool, True if the node is within goal_tol of the goal and the straight
            line to it is collision free.
        '''
        p = tree.point(latest_added_node)
        return self.line_cost(p, goal) < self.goal_tol and not self.check_collision(p, goal)

    def find_path(self, tree, latest_added_node):
        '''
        Walk the parent links back to the root to recover a path.

        Input:
            tree: Tree, the current search tree.
            latest_added_node: int, index of the node to start from.

        Output:
            numpy array of shape (n, 2) with the path points in the laser frame,
            ordered from the root to the given node.
        '''
        idx, path = latest_added_node, []
        while idx >= 0:
            path.append(tree.point(idx))
            idx = tree.parent[idx]
        return np.array(path[::-1])

    def shortcut(self, path):
        '''
        Smooth a path by skipping intermediate points whenever a direct line is free.

        Input:
            path: numpy array of shape (n, 2), path points in the laser frame.

        Output:
            numpy array of shape (m, 2) with m <= n, the shortcut path that keeps the
            same start and end points.
        '''
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and self.check_collision(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return np.array(out)

    def cost(self, tree, node):
        '''
        Read the cost-to-come stored for a node.

        Input:
            tree: Tree, the current search tree.
            node: int, node index.

        Output:
            float, the path cost from the root to that node.
        '''
        return float(tree.cost[node])

    def line_cost(self, n1, n2):
        '''
        Measure the cost of a straight connection between two points.

        Input:
            n1: (x, y) tuple, first point.
            n2: (x, y) tuple, second point.

        Output:
            float, the Euclidean distance between the two points.
        '''
        return math.hypot(n2[0] - n1[0], n2[1] - n1[1])

    def near(self, tree, point):
        '''
        Collect the tree nodes inside the RRT* neighbourhood of a point.

        Input:
            tree: Tree, the current search tree.
            point: (x, y) tuple, the query point in the laser frame.

        Output:
            int array with the indices of every node within rrt_star_radius of the
            point.
        '''
        xs, ys = tree.xy()
        d2 = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
        return np.flatnonzero(d2 <= self.rrt_star_radius ** 2)

    def choose_parent(self, tree, near, nearest, new):
        '''
        Pick the cheapest collision-free parent for a new RRT* node.

        Input:
            tree: Tree, the current search tree.
            near: int array, indices of the candidate neighbours.
            nearest: int, index of the nearest node (the fallback parent).
            new: (x, y) tuple, the new node position.

        Output:
            (parent, cost) tuple: the index of the chosen parent and the resulting
            cost-to-come of the new node.
        '''
        best, best_cost = nearest, self.cost(tree, nearest) + self.line_cost(tree.point(nearest), new)
        if len(near) == 0:
            return best, best_cost
        costs = tree.cost[near] + np.hypot(tree.x[near] - new[0], tree.y[near] - new[1])
        cheaper = costs < best_cost
        if cheaper.any():
            cand, cand_cost = near[cheaper], costs[cheaper]
            clear = ~self.segments_blocked(new, tree.x[cand], tree.y[cand])
            if clear.any():
                j = int(np.argmin(np.where(clear, cand_cost, np.inf)))
                best, best_cost = int(cand[j]), float(cand_cost[j])
        return best, best_cost

    def rewire(self, tree, near, new_idx):
        '''
        Rewire neighbouring nodes through a new node when that lowers their cost.

        Input:
            tree: Tree, the current search tree.
            near: int array, indices of the candidate neighbours.
            new_idx: int, index of the node just added.

        Output:
            None. Reparents every neighbour that is reachable more cheaply through the
            new node by a collision-free segment, updating the costs of their
            subtrees.
        '''
        near = near[near != tree.parent[new_idx]]
        if len(near) == 0:
            return
        p_new, c_new = tree.point(new_idx), self.cost(tree, new_idx)
        via_new = c_new + np.hypot(tree.x[near] - p_new[0], tree.y[near] - p_new[1])
        better = via_new < tree.cost[near] - 1e-9
        if not better.any():
            return
        cand, cand_cost = near[better], via_new[better]
        clear = ~self.segments_blocked(p_new, tree.x[cand], tree.y[cand])
        for j, c in zip(cand[clear], cand_cost[clear]):
            tree.reparent(int(j), new_idx, float(c))


    def _cube_list(self, ns, mid, color, value, stamp):
        '''
        Build one CUBE_LIST marker holding all grid cells with a given value.

        Input:
            ns: str, marker namespace.
            mid: int, marker id.
            color: std_msgs/ColorRGBA, the colour of the cubes.
            value: int, the grid value to draw (0 free, 1 occupied).
            stamp: builtin_interfaces/Time, the marker timestamp.

        Output:
            visualization_msgs/Marker, a cube list in the laser frame with one cube per
            matching cell.
        '''
        res = self.oc_resolution
        m = Marker()
        m.header.frame_id = self.oc_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = res
        m.scale.y = res
        m.scale.z = 0.01
        m.color = color
        cells = np.argwhere(self.oc == value)
        xs, ys = self.grid_to_world(cells[:, 0], cells[:, 1])
        m.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in zip(xs, ys)]
        return m

    def viz_oc(self):
        '''
        Publish the occupancy grid for RViz.

        Input:
            None.

        Output:
            None. Publishes a MarkerArray on /occupancy_markers with one cube list for
            the free cells and one for the occupied cells.
        '''
        stamp = self.oc_stamp if self.oc_stamp is not None else self.get_clock().now().to_msg()
        ma = MarkerArray()
        ma.markers.append(self._cube_list("free", 1001, ColorRGBA(r=0.1, g=0.8, b=0.8, a=0.2), 0, stamp))
        ma.markers.append(self._cube_list("occupancy", 1000, ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.3), 1, stamp))
        self.oc_pub.publish(ma)

    def publish_waypoints(self):
        '''
        Publish the global waypoint path for RViz.

        Input:
            None.

        Output:
            None. Publishes a MarkerArray on /waypoint_markers with the waypoints as a
            line strip in the map frame, closed into a loop when the loop parameter is
            set.
        '''
        xs, ys = self.waypoints[:, 0], self.waypoints[:, 1]
        if self.loop:
            xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
        ma = MarkerArray()
        ma.markers.append(line_marker("map", self.get_clock().now().to_msg(), "waypoints", 0, xs, ys,
                                      ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8), width=0.08))
        self.wp_pub.publish(ma)

    def publish_tree(self, tree, path_map):
        '''
        Publish the RRT tree and the planned path for RViz.

        Input:
            tree: Tree, the tree that was just grown.
            path_map: numpy array of shape (n, 2) with the path in the map frame, or
            None when no path was found.

        Output:
            None. Publishes a MarkerArray on /rrt_markers with the tree edges, and
            either the path line or a delete marker that removes the previous path.
        '''
        stamp = self.get_clock().now().to_msg()
        xs, ys = tree.xy()
        child = np.arange(1, tree.n)
        par = tree.parent[1:tree.n]
        ex = np.column_stack([xs[par], xs[child]]).ravel()
        ey = np.column_stack([ys[par], ys[child]]).ravel()
        ma = MarkerArray()
        ma.markers.append(line_marker(self.oc_frame, stamp, "rrt_tree", 0, ex, ey,
                                      ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6), width=0.02, strip=False))
        if path_map is not None:
            ma.markers.append(line_marker("map", stamp, "rrt_path", 1, path_map[:, 0], path_map[:, 1],
                                          ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0), width=0.07, z=0.1))
        else:
            ma.markers.append(delete_marker("map", "rrt_path", 1))
        self.rrt_pub.publish(ma)

    def clear_rrt_markers(self):
        '''
        Remove the RRT tree and path markers from RViz.

        Input:
            None.

        Output:
            None. Publishes delete markers on /rrt_markers, used when the car switches
            back to pure waypoint following.
        '''
        ma = MarkerArray()
        ma.markers.append(delete_marker(self.oc_frame, "rrt_tree", 0))
        ma.markers.append(delete_marker("map", "rrt_path", 1))
        self.rrt_pub.publish(ma)

    def publish_targets(self, goal_map, target):
        '''
        Publish the current goal and the tracked lookahead point for RViz.

        Input:
            goal_map: (x, y) point, the goal in the map frame.
            target: (x, y) point, the pure pursuit lookahead point in the map frame.

        Output:
            None. Publishes a MarkerArray on /rrt_markers with a green sphere at the
            goal and a yellow sphere at the tracked point.
        '''
        stamp = self.get_clock().now().to_msg()
        ma = MarkerArray()
        ma.markers.append(sphere_marker("map", stamp, "rrt_goal", 2, goal_map[0], goal_map[1],
                                        ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), size=0.3))
        ma.markers.append(sphere_marker("map", stamp, "tracked_point", 3, target[0], target[1],
                                        ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0), size=0.2))
        self.rrt_pub.publish(ma)


def main(args=None):
    '''
    Entry point of the node.

    Input:
        args: list of command line arguments passed to rclpy, or None to use
        sys.argv.

    Output:
        None. Initializes rclpy, spins the RRT node until shutdown, then destroys
        the node and shuts rclpy down.
    '''
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
