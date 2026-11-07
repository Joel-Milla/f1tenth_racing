#!/usr/bin/env python3
'''
Optimized drop-in replacement for rrt_node.py.

Same parameters, topics, messages and decisions as rrt_node.py; only the
implementation is faster:
    - scan geometry (ray cells, trig) is cached and reused while the scan layout
      does not change, rays are reduced to their unique cells once;
    - inflation is a separable dilation on a padded grid instead of fancy indexing;
    - reachability uses scipy.ndimage.label (falls back to the original flood fill);
    - collision checks inside the RRT loop are exact scalar loops with early exit
      that sample the very same points as the numpy versions;
    - choose_parent checks candidates cheapest first and stops at the first clear one;
    - random samples are drawn in one batch per plan;
    - waypoint goal search only transforms the waypoints it actually needs;
    - RViz markers reuse cached Point objects.
'''

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pure_pursuit_smooth import (LASER_TO_REAR_AXLE, SmoothPurePursuit,
                                 CAR_POSE_TOPIC, DEFAULT_WAYPOINTS, SIM_POSE_TOPIC,
                               PurePursuitController, delete_marker, drive_msg,
                               global_to_local, local_to_global, make_marker,
                               pose_from_msg, resample_polyline, sphere_marker)

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

# segments with more samples than this are checked with numpy instead of a Python loop
SCALAR_MAX_SAMPLES = 64
# number of waypoints transformed at a time while searching for the goal
GOAL_CHUNK = 256


def fast_line_marker(frame, stamp, ns, mid, xs, ys, color, width=0.05, z=0.05, strip=True):
    '''
    Same marker as pure_pursuit_node.line_marker, built without per-element numpy scalars.

    Input:
        frame, stamp, ns, mid, color, width, z, strip: as in line_marker.
        xs: array, x coordinates of the points.
        ys: array, y coordinates of the points.

    Output:
        visualization_msgs/Marker, a LINE_STRIP (or LINE_LIST) marker.
    '''
    m = make_marker(frame, stamp, ns, mid, Marker.LINE_STRIP if strip else Marker.LINE_LIST, color, width)
    xs = np.asarray(xs, dtype=float).tolist()
    ys = np.asarray(ys, dtype=float).tolist()
    m.points = [Point(x=x, y=y, z=z) for x, y in zip(xs, ys)]
    return m


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
            (xs, ys) tuple of numpy arrays, each of length n.
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
        cost, children = self.cost, self.children
        while stack:
            k = stack.pop()
            cost[k] += delta
            stack.extend(children[k])


class RRT(Node):
    def __init__(self):
        '''
        Start the RRT ROS node: read parameters, build the occupancy grid and set up I/O.

        Input:
            None (values come from ROS parameters, which fall back to the defaults
            declared in this method).

        Output:
            None. Loads the waypoints, allocates the local occupancy grid and its
            helper masks and caches, resets the planner state, and creates the
            pose/scan subscriptions, the drive and marker publishers and the
            waypoint timer.
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
            "rear_axle_offset": LASER_TO_REAR_AXLE,
            "steer_tau": 0.05,
            "max_steer_rate": 3.0,
            "track_window": 2.0,
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

        if self.use_sim:
            # sim odom pose is already the rear axle (base_link)
            self.rear_axle_offset = 0.0
        self.pp = SmoothPurePursuit(self.wheelbase, self.max_steer, self.rear_axle_offset,
                                    self.steer_tau, self.max_steer_rate, self.track_window)
        self.waypoints = PurePursuitController.load_waypoints(self.waypoints_file)
        self.wp_x = np.ascontiguousarray(self.waypoints[:, 0])
        self.wp_y = np.ascontiguousarray(self.waypoints[:, 1])
        self.get_logger().info(f"loaded {len(self.waypoints)} waypoints from {self.waypoints_file}")

        # oc[ix, iy]: 0 free, 1 occupied (inflated + everything the laser didn't see)
        nx = int(round(self.oc_length / self.oc_resolution))
        ny = int(round(self.oc_width / self.oc_resolution))
        self.nx, self.ny = nx, ny
        self.oc = np.ones((nx, ny), dtype=np.uint8)
        self.oc_origin = (int(round(nx * self.oc_back_frac)), ny // 2)
        self.oc_frame = "ego_racecar/laser"
        self.oc_stamp = None
        self.inflation_offsets = self.build_inflation_offsets()
        self.car_local = (-self.laser_offset_x, 0.0)
        gx, gy = self.grid_to_world(*np.indices((nx, ny)))
        self.near_car = np.hypot(gx - self.car_local[0], gy - self.car_local[1]) < self.inflation_radius
        self.reach = np.zeros((nx, ny), dtype=bool)

        # grids padded by one cell so clipped indices fall on the border:
        # oc_pad border = 1 (blocked), reach_pad border = False (unreachable)
        self.oc_pad = np.ones((nx + 2, ny + 2), dtype=np.uint8)
        self.reach_pad = np.zeros((nx + 2, ny + 2), dtype=bool)
        self.oc_bytes = self.oc.tobytes()
        self.half_res = self.oc_resolution / 2
        self.linspace_cache = {}
        self.build_dilation_plan()
        self.scan_key = None
        self.cell_points = None

        self.have_scan = False
        self.scan_count = 0
        self.scans = deque(maxlen=20)
        self.grid_scan = None
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
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def map_to_laser(self, px, py, pose):
        '''
        Transform points from the global map frame into the laser frame.

        Input:
            px: float or array, map-frame x coordinate(s).
            py: float or array, map-frame y coordinate(s).
            pose: (x, y, yaw) tuple, current car pose in the map frame.

        Output:
            (lx, ly) tuple of floats or arrays, the same points in the laser frame.
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

    def build_dilation_plan(self):
        '''
        Split the inflation disk into one y-span per x offset for a separable dilation.

        Input:
            None (uses self.inflation_offsets).

        Output:
            None. Sets self.dil_pad (padding in cells, the disk radius), self.dil_widths
            (sorted distinct half widths along y) and self.dil_rows (list of
            (dx, ymin, ymax) spans covering the disk exactly).
        '''
        off = self.inflation_offsets
        self.dil_pad = int(np.abs(off).max()) if len(off) else 0
        rows = []
        for dx in np.unique(off[:, 0]) if len(off) else []:
            ys = np.sort(off[off[:, 0] == dx, 1])
            # split into contiguous runs so any disk shape is reproduced exactly
            breaks = np.flatnonzero(np.diff(ys) != 1)
            for run in np.split(ys, breaks + 1):
                rows.append((int(dx), int(run[0]), int(run[-1])))
        self.dil_rows = rows

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
        np.clip(ix, -1, self.nx, out=ix)
        np.clip(iy, -1, self.ny, out=iy)
        blocked = self.oc_pad[ix + 1, iy + 1].astype(bool)
        b = np.flatnonzero(blocked)
        if len(b):
            near_car = np.hypot(xs[b] - self.car_local[0], ys[b] - self.car_local[1]) < self.inflation_radius
            blocked[b[near_car]] = False
        return blocked

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
        rx, ry = resample_polyline(xs, ys, self.half_res)
        return bool(self.points_blocked(rx, ry).any())

    def prepare_scan_geometry(self, n, angle_min, angle_increment):
        '''
        Precompute everything in a scan that only depends on its angular layout.

        Input:
            n: int, number of beams.
            angle_min: float, angle of the first beam.
            angle_increment: float, angle between beams.

        Output:
            None. Caches the beam cos/sin and, for every beam, the distinct grid cells
            its samples visit (ray_cells) with the distance of the first sample in
            each cell (ray_d, inf padded). A cell is carved free by a beam exactly
            when that first distance is below the beam's free range, which is what
            the original per-sample test reduces to.
        '''
        angles = angle_min + np.arange(n) * angle_increment
        self.cos_a, self.sin_a = np.cos(angles), np.sin(angles)

        nx, ny = self.nx, self.ny
        cx, cy = self.oc_origin
        max_d = np.hypot(max(cx, nx - cx), max(cy, ny - cy)) * self.oc_resolution
        d = np.arange(0.0, max_d, self.half_res)
        sx, sy = self.world_to_grid(d[None, :] * self.cos_a[:, None],
                                    d[None, :] * self.sin_a[:, None])
        inb = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
        flat = np.where(inb, sx * ny + sy, -1)
        # samples move monotonically along a beam, so each cell is one contiguous run
        first = np.ones_like(inb)
        first[:, 1:] = flat[:, 1:] != flat[:, :-1]
        keep = first & inb
        rank = np.cumsum(keep, axis=1) - 1
        width = max(int(keep.sum(axis=1).max()), 1)
        rows = np.broadcast_to(np.arange(n)[:, None], keep.shape)[keep]
        self.ray_cells = np.zeros((n, width), dtype=np.intp)
        self.ray_d = np.full((n, width), np.inf)
        self.ray_cells[rows, rank[keep]] = flat[keep]
        self.ray_d[rows, rank[keep]] = np.broadcast_to(d[None, :], keep.shape)[keep]
        self.scan_key = (n, angle_min, angle_increment)

    def build_occupancy_map(self, scan_msg):
        '''
        Rebuild the local occupancy grid from one laser scan.

        Input:
            scan_msg: sensor_msgs/LaserScan, the latest scan.

        Output:
            None. Fills self.oc with 1 (occupied/unknown) everywhere, inflates every
            laser hit by the inflation disk, then carves the rays back to 0 (free) up
            to each measured range, and records the scan frame and stamp. Also
            refreshes the padded/bytes copies of the grid used for collision checks.
        '''
        self.oc_frame = scan_msg.header.frame_id
        self.oc_stamp = scan_msg.header.stamp

        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        key = (len(ranges), scan_msg.angle_min, scan_msg.angle_increment)
        if key != self.scan_key:
            self.prepare_scan_geometry(*key)
        cos_a, sin_a = self.cos_a, self.sin_a

        r_free = np.where(np.isinf(ranges), scan_msg.range_max, ranges)
        valid = np.isfinite(r_free) & (r_free >= scan_msg.range_min)
        r_free = np.where(valid, np.minimum(r_free, scan_msg.range_max), 0.0)

        nx, ny, P = self.nx, self.ny, self.dil_pad

        # inflation: mark hits on a grid padded by the disk radius, then dilate
        hit = valid & (ranges < scan_msg.range_max)
        hx, hy = self.world_to_grid(ranges[hit] * cos_a[hit], ranges[hit] * sin_a[hit])
        hx += P
        hy += P
        ok = (hx >= 0) & (hx < nx + 2 * P) & (hy >= 0) & (hy < ny + 2 * P)
        hits = np.zeros((nx + 2 * P, ny + 2 * P), dtype=bool)
        hits[hx[ok], hy[ok]] = True
        inflated = np.zeros((nx, ny), dtype=bool)
        spans = {}
        for dx, y0, y1 in self.dil_rows:
            band = spans.get((y0, y1))
            if band is None:
                # band[:, j] = any hit at padded column j - y for y in [y0, y1]
                band = np.zeros((nx + 2 * P, ny), dtype=bool)
                for oy in range(y0, y1 + 1):
                    band |= hits[:, P - oy:P - oy + ny]
                spans[(y0, y1)] = band
            inflated |= band[P - dx:P - dx + nx]

        # carving: a beam frees every distinct cell it enters before its range
        free = np.zeros(nx * ny, dtype=bool)
        free[self.ray_cells[self.ray_d < r_free[:, None]]] = True
        free &= ~inflated.ravel()
        np.logical_not(free.reshape(nx, ny), out=self.oc, casting="unsafe")

        self.oc_pad[1:-1, 1:-1] = self.oc
        self.oc_bytes = self.oc.tobytes()

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
        if ndimage is not None:
            labels, _ = ndimage.label(free)
            seeds = np.unique(labels[self.near_car])
            lut = np.zeros(labels.max() + 1, dtype=bool)
            lut[seeds] = True
            lut[0] = False
            return lut[labels]
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
        np.clip(ix, -1, self.nx, out=ix)
        np.clip(iy, -1, self.ny, out=iy)
        return self.reach_pad[ix + 1, iy + 1]

    def scan_callback(self, scan_msg):
        '''
        Handle an incoming laser scan.

        Input:
            scan_msg: sensor_msgs/LaserScan, the new scan.

        Output:
            None. Buffers the scan so the grid can be built from the scan matching
            the pose it is used with, and flags that a scan is available.
        '''
        self.scans.append(scan_msg)
        self.have_scan = True

    def update_grid(self, stamp):
        '''
        Build the occupancy grid from the buffered scan closest in time to a pose.

        The grid lives in the laser frame at scan time while waypoints are moved into
        it with the pose, so both must describe the same instant. The particle filter
        computes a pose from one scan; using a newer scan with that pose rotates and
        shifts the waypoints against the walls, which pushes the goal off the path.

        Input:
            stamp: builtin_interfaces/Time, stamp of the pose message.

        Output:
            None. Rebuilds the occupancy grid and the reachability mask when the chosen
            scan differs from the one already used, and republishes the grid markers
            every oc_viz_every rebuilds.
        '''
        t = stamp.sec + stamp.nanosec * 1e-9
        scan_msg = min(self.scans, key=lambda m: abs(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - t))
        if scan_msg is self.grid_scan:
            return
        self.grid_scan = scan_msg
        self.build_occupancy_map(scan_msg)
        self.reach = self.reachable_cells()
        self.reach_pad[1:-1, 1:-1] = self.reach
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
        self.update_grid(pose_msg.header.stamp)

        wp_start = self.pp.track_index(self.waypoints, pose, self.loop, "waypoints")
        goal = self.select_goal(pose, wp_start)
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
            path, L, speed, loop, start = self.waypoints, self.lookahead, self.speed, self.loop, wp_start
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
            path, L, speed, loop, start = self.rrt_path, self.rrt_lookahead, self.rrt_speed, False, None

        target = self.pp.lookahead_point(path, pose, L, loop, start)
        steer = self.pp.smooth(self.pp.steering_angle(pose, target), now)
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
        self.pp.reset()
        self.drive_pub.publish(drive_msg(self.get_clock().now().to_msg(), 0.0, 0.0))


    def select_goal(self, pose, start=None):
        '''
        Pick the goal point on the waypoint path ahead of the car.

        Input:
            pose: (x, y, yaw) tuple, current car pose in the map frame.
            start: int or None, index of the waypoint closest to the car if already
            known.

        Output:
            None if no usable goal exists, otherwise a tuple
            (goal_local, goal_map, direct_clear) where goal_local is the (x, y) goal
            in the laser frame, goal_map is the (x, y) goal in the map frame and
            direct_clear is a bool saying whether the waypoint path up to the goal is
            free of obstacles. If the waypoint goal is unreachable the nearest
            reachable free cell is used instead and direct_clear is False.
        '''
        wp = self.waypoints
        n = len(wp)
        if start is None:
            start = self.pp.closest_index(wp, pose)
        total = n if self.loop else n - start

        # walk the waypoints ahead in chunks: find the first one in front of the car,
        # then the first one leaving the grid, transforming only what is needed
        k0, end, parts = None, None, []
        for p in range(0, total, GOAL_CHUNK):
            idx = start + np.arange(p, min(p + GOAL_CHUNK, total))
            if self.loop:
                idx %= n
            lx, ly = self.map_to_laser(self.wp_x[idx], self.wp_y[idx], pose)
            lo = 0
            if k0 is None:
                ahead = np.flatnonzero(lx > self.car_local[0])
                if len(ahead) == 0:
                    continue
                k0 = p + int(ahead[0])
                lo = int(ahead[0])
            idx, lx, ly = idx[lo:], lx[lo:], ly[lo:]
            outside = np.flatnonzero(~self.in_grid(lx, ly))
            if len(outside):
                e = int(outside[0])
                parts.append((idx[:e], lx[:e], ly[:e]))
                end = True
                break
            parts.append((idx, lx, ly))
        if k0 is None:
            return None
        idx = np.concatenate([q[0] for q in parts])
        if len(idx) == 0:
            return None
        lx = np.concatenate([q[1] for q in parts])
        ly = np.concatenate([q[2] for q in parts])
        pts = wp[idx]

        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))])
        s += math.hypot(lx[0] - self.car_local[0], ly[0] - self.car_local[1])
        kg = min(int(np.searchsorted(s, self.goal_dist)), len(s) - 1)

        ok = self.points_reachable(lx, ly)
        k = kg
        if not ok[kg]:
            ok_after = np.flatnonzero(ok[kg:])
            # stay on the path: past the blocked part if possible (obstacle on the
            # path), else the farthest reachable waypoint before it (e.g. the path
            # turns behind a wall the laser cannot see through)
            ok_before = np.flatnonzero(ok[:kg] & (s[:kg] >= self.min_goal_dist))
            if len(ok_after):
                k = kg + int(ok_after[0])
            elif len(ok_before):
                k = int(ok_before[-1])
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

    def draw_samples(self, goal, count):
        '''
        Draw all random samples one plan can use in a single batch.

        Input:
            goal: (x, y) tuple, goal point in the laser frame.
            count: int, number of samples.

        Output:
            (xs, ys) tuple of Python float lists. Each sample is the goal with
            probability goal_bias, otherwise a random free cell centre jittered inside
            its cell (same distribution as sample()).
        '''
        fx, fy = self.free_samples
        rng = self.rng
        use_goal = rng.random(count) < self.goal_bias
        i = rng.integers(len(fx), size=count)
        jit = (rng.random((count, 2)) - 0.5) * self.oc_resolution
        xs = fx[i] + jit[:, 0]
        ys = fy[i] + jit[:, 1]
        xs[use_goal] = goal[0]
        ys[use_goal] = goal[1]
        return xs.tolist(), ys.tolist()

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

        max_iter, step = self.max_iter, self.step
        star, r2 = self.use_rrt_star, self.rrt_star_radius ** 2
        goal = (float(goal[0]), float(goal[1]))
        gx, gy = goal
        goal_tol = self.goal_tol
        collides = self.collides
        hypot = math.hypot

        sxs, sys_ = self.draw_samples(goal, max_iter)
        tree = Tree(max_iter + 1, self.car_local)
        tx, ty, tcost = tree.x, tree.y, tree.cost
        goal_parents = []
        stop_at = max_iter
        for it in range(max_iter):
            if it >= stop_at:
                break
            sx, sy = sxs[it], sys_[it]
            n = tree.n
            xs, ys = tx[:n], ty[:n]
            nearest = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))

            x0, y0 = float(tx[nearest]), float(ty[nearest])
            dx, dy = sx - x0, sy - y0
            d = hypot(dx, dy)
            if d <= step:
                new = (sx, sy)
            else:
                new = (x0 + dx * step / d, y0 + dy * step / d)
            if collides(x0, y0, new[0], new[1]):
                continue

            if star:
                near = np.flatnonzero((xs - new[0]) ** 2 + (ys - new[1]) ** 2 <= r2)
                parent, cost = self.choose_parent(tree, near, nearest, new)
                idx = tree.add(new, parent, cost)
                self.rewire(tree, near, idx)
            else:
                idx = tree.add(new, nearest, float(tcost[nearest]) + hypot(new[0] - x0, new[1] - y0))

            if hypot(gx - new[0], gy - new[1]) < goal_tol and not collides(new[0], new[1], gx, gy):
                if not goal_parents:
                    stop_at = it + 1 + (self.rrt_star_extra_iter if star else 0)
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
        Draw one random sample for the tree expansion (plan() uses draw_samples()).

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

    def linspace01(self, n):
        '''
        Cached np.linspace(0, 1, n) as a Python list.

        Input:
            n: int, number of samples.

        Output:
            list of n floats from 0.0 to 1.0.
        '''
        t = self.linspace_cache.get(n)
        if t is None:
            t = self.linspace_cache[n] = np.linspace(0.0, 1.0, n).tolist()
        return t

    def scalar_blocked(self, x, y):
        '''
        points_blocked for a single point, without numpy.

        Input:
            x: float, laser-frame x coordinate.
            y: float, laser-frame y coordinate.

        Output:
            bool, True if the point is occupied or outside the grid and not within the
            inflation radius of the car.
        '''
        cx, cy = self.oc_origin
        ix = math.floor(x / self.oc_resolution + cx)
        iy = math.floor(y / self.oc_resolution + cy)
        if 0 <= ix < self.nx and 0 <= iy < self.ny and not self.oc_bytes[ix * self.ny + iy]:
            return False
        return math.hypot(x - self.car_local[0], y - self.car_local[1]) >= self.inflation_radius

    def collides(self, x0, y0, x1, y1):
        '''
        check_collision on plain floats: same sample points, early exit on the first hit.

        Input:
            x0, y0: floats, start point in the laser frame.
            x1, y1: floats, end point in the laser frame.

        Output:
            bool, True if any point sampled along the segment (at half the grid
            resolution) is blocked.
        '''
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(dist / self.half_res)) + 1)
        div = n - 1
        dx, dy = x1 - x0, y1 - y0
        stx, sty = dx / div, dy / div
        if n > SCALAR_MAX_SAMPLES or (stx == 0.0 and dx != 0.0) or (sty == 0.0 and dy != 0.0):
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            return bool(self.points_blocked(xs, ys).any())

        # same values as np.linspace: start + i * step, last point exactly the end
        res, nx, ny, occ = self.oc_resolution, self.nx, self.ny, self.oc_bytes
        cx, cy = self.oc_origin
        floor = math.floor
        for i in range(n):
            if i == div:
                x, y = x1, y1
            else:
                x, y = i * stx + x0, i * sty + y0
            ix = floor(x / res + cx)
            iy = floor(y / res + cy)
            if 0 <= ix < nx and 0 <= iy < ny and not occ[ix * ny + iy]:
                continue
            if self.scalar_blocked(x, y):
                return True
        return False

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
        return self.collides(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))

    def fan_segment_blocked(self, px, py, x, y, t):
        '''
        One segment of segments_blocked(), checked on plain floats with early exit.

        Input:
            px, py: floats, the common endpoint of the fan.
            x, y: floats, the other endpoint of this segment.
            t: list of floats, the shared interpolation parameters (linspace(0, 1, n)).

        Output:
            bool, True if any of the fan samples on this segment is blocked.
        '''
        res, nx, ny, occ = self.oc_resolution, self.nx, self.ny, self.oc_bytes
        cx, cy = self.oc_origin
        floor = math.floor
        dx, dy = x - px, y - py
        for ti in t:
            qx, qy = px + dx * ti, py + dy * ti
            ix = floor(qx / res + cx)
            iy = floor(qy / res + cy)
            if 0 <= ix < nx and 0 <= iy < ny and not occ[ix * ny + iy]:
                continue
            if self.scalar_blocked(qx, qy):
                return True
        return False

    def fan_samples(self, p, xs, ys):
        '''
        Number of samples segments_blocked() uses for a fan of segments.

        Input:
            p: (x, y) tuple, the common endpoint.
            xs, ys: arrays, the other endpoints.

        Output:
            int, the per-segment sample count (set by the longest segment).
        '''
        return self.fan_count(float(np.max(np.hypot(xs - p[0], ys - p[1]))))

    def fan_count(self, longest):
        '''
        Sample count for a fan whose longest segment is already known.

        Input:
            longest: float, length of the longest segment in the fan.

        Output:
            int, the per-segment sample count.
        '''
        return max(2, int(math.ceil(longest / self.half_res)) + 1)

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
        n = self.fan_samples(p, xs, ys)
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
        pts = path.tolist()
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and self.collides(pts[i][0], pts[i][1], pts[j][0], pts[j][1]):
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
            cost-to-come of the new node. Candidates are tested cheapest first (stable
            order on ties), so the first clear one is the original argmin.
        '''
        best, best_cost = nearest, self.cost(tree, nearest) + self.line_cost(tree.point(nearest), new)
        if len(near) == 0:
            return best, best_cost
        dist = np.hypot(tree.x[near] - new[0], tree.y[near] - new[1])
        costs = tree.cost[near] + dist
        cheaper = np.flatnonzero(costs < best_cost)
        if len(cheaper) == 0:
            return best, best_cost
        cand, cand_cost = near[cheaper], costs[cheaper]
        cx, cy = tree.x[cand], tree.y[cand]
        t = self.linspace01(self.fan_count(float(dist[cheaper].max())))
        cxl, cyl, ccl = cx.tolist(), cy.tolist(), cand_cost.tolist()
        for j in np.argsort(cand_cost, kind="stable").tolist():
            if not self.fan_segment_blocked(new[0], new[1], cxl[j], cyl[j], t):
                return int(cand[j]), ccl[j]
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
        dist = np.hypot(tree.x[near] - p_new[0], tree.y[near] - p_new[1])
        via_new = c_new + dist
        better = via_new < tree.cost[near] - 1e-9
        if not better.any():
            return
        cand, cand_cost = near[better], via_new[better]
        cx, cy = tree.x[cand], tree.y[cand]
        t = self.linspace01(self.fan_count(float(dist[better].max())))
        # all checks first, then reparent in order, exactly like the original
        clear = [j for j, (x, y) in enumerate(zip(cx.tolist(), cy.tolist()))
                 if not self.fan_segment_blocked(p_new[0], p_new[1], x, y, t)]
        cl, ccl = cand.tolist(), cand_cost.tolist()
        for j in clear:
            tree.reparent(cl[j], new_idx, ccl[j])


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
            matching cell (Point objects are cached per cell and reused).
        '''
        if self.cell_points is None:
            cells = np.indices(self.oc.shape).reshape(2, -1)
            xs, ys = self.grid_to_world(cells[0], cells[1])
            self.cell_points = [Point(x=x, y=y, z=0.0) for x, y in zip(xs.tolist(), ys.tolist())]
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
        cp = self.cell_points
        m.points = [cp[i] for i in np.flatnonzero(self.oc.ravel() == value).tolist()]
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
        xs, ys = self.wp_x, self.wp_y
        if self.loop:
            xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
        ma = MarkerArray()
        ma.markers.append(fast_line_marker("map", self.get_clock().now().to_msg(), "waypoints", 0, xs, ys,
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
        ma.markers.append(fast_line_marker(self.oc_frame, stamp, "rrt_tree", 0, ex, ey,
                                           ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.6), width=0.02, strip=False))
        if path_map is not None:
            ma.markers.append(fast_line_marker("map", stamp, "rrt_path", 1, path_map[:, 0], path_map[:, 1],
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
