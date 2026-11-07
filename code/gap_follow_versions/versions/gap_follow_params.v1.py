import rclpy
import numpy as np
import math
import time

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# =============================================================================
#                            TUNABLE PARAMETERS
#           (edit values here; nothing below needs to change)
# =============================================================================

# ---------------- Speed (continuous / dynamic) ----------------
# Commanded speed = min(clearance limit, cornering limit, MAX_SPEED), floored at MIN_SPEED,
# then ramped up no faster than MAX_ACCEL (slowing down is applied immediately).
MAX_SPEED               = 3.5   # [m/s] top speed on long straights (VESC hard limit ~5.0)
MIN_SPEED               = 1.5   # [m/s] never go slower than this while a gap exists
SPEED_NO_GAP            = 0.0   # [m/s] used when no gap is found
BRAKE_DECEL             = 3.0   # [m/s^2] clearance limit: v = sqrt(2 * BRAKE_DECEL * (d_ahead - STOP_MARGIN))
STOP_MARGIN             = 0.4   # [m]  distance kept in reserve in front of the car
LATERAL_ACCEL           = 4.5   # [m/s^2] cornering limit: v = sqrt(LATERAL_ACCEL * turn_radius)
MAX_ACCEL               = 4.0   # [m/s^2] max rate the speed command may increase
SPEED_CONE_ANGLE        = math.radians(10)  # [rad] half-angle of the cone ahead used for d_ahead
SPEED_RANGE_CAP         = 10.0  # [m]  cap on d_ahead (ignores huge/no-return readings)

# ---------------- Steering ----------------
WHEELBASE               = 0.33  # [m]  used to convert steering angle to turn radius
MAX_STEERING            = 0.34  # [rad] clamp on published steering angle
STEERING_GAIN           = 1.0   # [-]  multiplier on target beam angle
STEERING_SMOOTHING      = 0.8   # [-]  0..1, weight of new command (1.0 = no smoothing)

# ---------------- Gap finding ----------------
GAP_THRESHOLD           = 1.0   # [m]  min range for a beam to count as part of a gap
CLIPPING_DISTANCE       = 1.8   # [m]  ranges in the chosen gap are clipped to this value
TARGET_MODE             = 'middle'  # 'middle' -> aim at centre of biggest gap
                                    # 'max'    -> aim at furthest point in biggest gap
GAP_FOV                 = math.radians(110)  # [rad] only beams within +/- this angle can form a gap

# ---------------- Disparity extender ----------------
DISPARITY_THRESHOLD     = 0.3   # [m]  jump between adjacent beams that counts as a disparity
DISPARITY_BUFFER        = 2.5   # [-]  multiplier on car half-width when extending disparities

# ---------------- Safety bubble ----------------
BUBBLE_BUFFER           = 1.1   # [-]  bubble radius = car half-diagonal * this (1.1 = +10%)

# ---------------- Rear-quadrant check ----------------
BACK_QUADRANT_THRESHOLD = 0.4            # [m]   obstacle closer than this behind the car blocks turning toward it
BACK_QUADRANT_ANGLE     = math.pi / 2    # [rad] beams beyond +/- this angle are treated as "behind"

# ---------------- Car dimensions ----------------
CAR_WIDTH               = 0.30  # [m] full width across the tires (0.296 rounded up)
CAR_LENGTH              = 0.57  # [m] bumper to bumper

# ---------------- Topics / QoS ----------------
SCAN_TOPIC              = 'scan'
ODOM_TOPIC              = 'odom'   # real car; use 'ego_racecar/odom' in the simulator
DRIVE_TOPIC             = 'drive'
QOS_DEPTH               = 10

# Visualization topics
PUBLISH_VIZ             = False  # debug scans for rviz; turn off when racing (saves CPU/latency)
VIZ_BUBBLE_TOPIC        = 'scan_bubble'
VIZ_DISPARITY_TOPIC     = 'scan_disparity'
VIZ_BIGGEST_GAP_TOPIC   = 'biggest_gap_'
VIZ_CLIPPED_GAP_TOPIC   = 'clipped_gap_'
VIZ_TARGET_POINT_TOPIC  = 'target_point_'

# ---------------- Debug ----------------
LOG_STEERING            = True  # print speed / steering / timing
LOG_PERIOD              = 0.25  # [s] min time between log lines

# =============================================================================


class GapFollowN(Node):

    def __init__(self):
        super().__init__('gap_follow_node')
        self.subscription_scan = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            QOS_DEPTH)
        self.subscription_scan  # prevent unused variable warning

        self.subscription_odom = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            QOS_DEPTH)
        self.subscription_odom  # prevent unused variable warning

        self.publisher_vel = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, QOS_DEPTH)

        # threshold values (pulled from the parameter block above)
        self.gap_threshold = GAP_THRESHOLD
        self.disparity_threshold = DISPARITY_THRESHOLD
        self.back_quadrant_threshold = BACK_QUADRANT_THRESHOLD
        self.clipping_distance_when_above_threshold = CLIPPING_DISTANCE
        self.disparity_buffer = DISPARITY_BUFFER

        # set dimensions of the car
        self.car_width  = CAR_WIDTH
        self.car_length = CAR_LENGTH
        self.half_width  = self.car_width  / 2.0
        self.half_length = self.car_length / 2.0

        self.speed = 0.0
        self.last_speed_cmd = 0.0
        self.last_steering = 0.0
        self.last_cmd_time = None
        self.last_log_time = 0.0

        # Publishers for visualization purposes
        self.publisher_bubble_ = self.create_publisher(LaserScan, VIZ_BUBBLE_TOPIC, QOS_DEPTH)
        self.scan_disparity_ = self.create_publisher(LaserScan, VIZ_DISPARITY_TOPIC, QOS_DEPTH)
        self.biggest_gap_ = self.create_publisher(LaserScan, VIZ_BIGGEST_GAP_TOPIC, QOS_DEPTH)
        self.clipped_gap_ = self.create_publisher(LaserScan, VIZ_CLIPPED_GAP_TOPIC, QOS_DEPTH)
        self.target_point_ = self.create_publisher(LaserScan, VIZ_TARGET_POINT_TOPIC, QOS_DEPTH)

    def odom_callback(self, msg : Odometry):
        '''
        Updates speed with the current odom readings
        '''
        self.speed = msg.twist.twist.linear.x

    def scan_callback(self, msg : LaserScan):
        '''
        Function to follow the gap given a single scan
        '''
        t_start = time.perf_counter()
        self.latest_scan = msg
        self.ranges = np.array(msg.ranges, dtype=float) # get ranges of type float
        # nan / inf / no-return readings (Hokuyo reports ~65.5 m) -> treat as max range
        self.ranges = np.nan_to_num(self.ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        np.clip(self.ranges, 0.0, msg.range_max, out=self.ranges)
        self.raw_ranges = self.ranges.copy()
        # angles in radians, matched in position with ranges array (built from len() so sizes always agree)
        self.angles = msg.angle_min + msg.angle_increment * np.arange(len(self.ranges))

        self.create_disparity()
        self.draw_safety_bubble()
        biggest_gap = self.find_biggest_gap()

        if TARGET_MODE == 'max':
            middle_indx_point, target_point = self.get_target_point_max(biggest_gap)
        else:
            middle_indx_point, target_point = self.get_target_point_middle(biggest_gap)

        # steering angle
        steering_angle = STEERING_GAIN * self.angles[middle_indx_point]
        steering_angle = self.check_close_3rd_4th_quadrant(steering_angle, target_point)
        steering_angle = float(np.clip(steering_angle, -MAX_STEERING, MAX_STEERING))
        steering_angle = STEERING_SMOOTHING * steering_angle + (1.0 - STEERING_SMOOTHING) * self.last_steering
        self.last_steering = steering_angle

        # speed
        if target_point == 0:
            speed = SPEED_NO_GAP
        else:
            speed = self.get_optimal_linear_speed(steering_angle)
        speed = self.rate_limit_speed(speed)

        if LOG_STEERING:
            now = time.monotonic()
            if now - self.last_log_time > LOG_PERIOD:
                self.last_log_time = now
                self.get_logger().info(
                    f'speed {speed:4.2f} m/s | steer {steering_angle:+.3f} rad | '
                    f'd_ahead {self.d_ahead:4.2f} m | odom {self.speed:4.2f} | '
                    f'calc {(time.perf_counter() - t_start) * 1000:4.1f} ms')
        self.publish_velocity(speed, steering_angle)

    def check_close_3rd_4th_quadrant(self, steering_angle, distance):
        close = self.raw_ranges < self.back_quadrant_threshold
        object_right = bool(np.any(close & (self.angles < -BACK_QUADRANT_ANGLE)))
        object_left  = bool(np.any(close & (self.angles >  BACK_QUADRANT_ANGLE)))

        if steering_angle > 0 and object_left:
            return 0.0

        if steering_angle < 0 and object_right:
            return 0.0

        return steering_angle

    def get_optimal_linear_speed(self, steering_angle):
        '''
        Continuous speed from free distance straight ahead and from how hard we are turning
        '''
        # free distance ahead: closest raw beam inside a narrow cone in front of the car
        cone = np.abs(self.angles) <= SPEED_CONE_ANGLE
        self.d_ahead = float(min(np.min(self.raw_ranges[cone]), SPEED_RANGE_CAP))

        # limit 1: be able to brake within the free distance
        v_clear = math.sqrt(2.0 * BRAKE_DECEL * max(self.d_ahead - STOP_MARGIN, 0.0))

        # limit 2: lateral acceleration in the turn, radius = wheelbase / tan(steer)
        tan_steer = abs(math.tan(steering_angle))
        if tan_steer > 1e-3:
            v_turn = math.sqrt(LATERAL_ACCEL * WHEELBASE / tan_steer)
        else:
            v_turn = MAX_SPEED

        return max(MIN_SPEED, min(MAX_SPEED, v_clear, v_turn))

    def rate_limit_speed(self, speed):
        '''
        Limit how fast the speed command increases; decreases pass through immediately
        '''
        now = time.monotonic()
        if self.last_cmd_time is not None and speed > self.last_speed_cmd:
            dt = min(now - self.last_cmd_time, 0.1)
            speed = min(speed, self.last_speed_cmd + MAX_ACCEL * dt)
        self.last_cmd_time = now
        self.last_speed_cmd = speed
        return speed

    def create_disparity(self):
        disparity_ranges = self.ranges.copy()
        n_beams = len(disparity_ranges)
        differences = self.ranges[:-1] - self.ranges[1:]

        # only visit indices where the jump between neighbours is significant
        for curr_indx in np.nonzero(np.abs(differences) > self.disparity_threshold)[0]:
            curr_range = self.ranges[curr_indx]
            next_range = self.ranges[curr_indx + 1]

            # Means that current beam is greater than next beam -> extend toward lower indices
            if differences[curr_indx] > 0:
                smallest_beam = next_range
                increment = -1
                starting_indx = curr_indx + 1
            else: # current beam is smaller than next beam -> extend toward higher indices
                smallest_beam = curr_range
                increment = 1
                starting_indx = curr_indx

            if smallest_beam == 0:
                smallest_beam = self.latest_scan.range_min # if smallest beam got zero, change to minimum range lidar can detect

            # number of beams needed to cover the buffered half width at that distance
            n = int(math.ceil((self.half_width * self.disparity_buffer) / (smallest_beam * self.latest_scan.angle_increment)))

            if increment > 0:
                lo, hi = starting_indx, min(starting_indx + n, n_beams - 1) + 1
            else:
                lo, hi = max(starting_indx - n, 0), starting_indx + 1
            # if two disparities overlap, keep smallest one
            np.minimum(disparity_ranges[lo:hi], smallest_beam, out=disparity_ranges[lo:hi])

        if PUBLISH_VIZ:
            self.publish_scan(self.scan_disparity_, self.latest_scan, disparity_ranges)
        self.ranges = disparity_ranges

    def draw_safety_bubble(self):
        '''
        Draw a safety bubble around closest point
        '''
        r_rob = math.sqrt(self.half_width**2 + self.half_length**2) # radius of circle around the car, which has length of diagonal using pythagorean theorem
        r_rob *= BUBBLE_BUFFER # inflate as a buffer

        ranges = self.ranges.copy()
        min_indx = int(np.argmin(ranges))
        min_distance = ranges[min_indx]

        # distance of every beam endpoint to the closest point (law of cosines)
        dtheta = (np.arange(len(ranges)) - min_indx) * self.latest_scan.angle_increment
        dist = np.sqrt(np.maximum(ranges**2 + min_distance**2 - 2 * ranges * min_distance * np.cos(dtheta), 0.0))
        outside = dist >= r_rob

        # zero the contiguous run of beams around min_indx that lie inside the bubble
        right = outside[min_indx:]
        hi = min_indx + (int(np.argmax(right)) if right.any() else len(right))
        left = outside[:min_indx + 1][::-1]
        lo = min_indx - (int(np.argmax(left)) if left.any() else min_indx) + 1
        ranges[max(lo, 0):hi] = 0.0

        self.ranges = ranges
        if PUBLISH_VIZ:
            self.publish_scan(self.publisher_bubble_, self.latest_scan, ranges)

    def find_biggest_gap(self):
        '''
        Returns an array of indices of the maximum length sequence found
        '''
        in_gap = (self.ranges >= self.gap_threshold) & (np.abs(self.angles) <= GAP_FOV)

        # find start/end of every run of True values
        padded = np.concatenate(([0], in_gap.astype(np.int8), [0]))
        edges = np.diff(padded)
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0]

        if len(starts) == 0:
            biggest_gap = np.array([], dtype=int)
        else:
            k = int(np.argmax(ends - starts))
            biggest_gap = np.arange(starts[k], ends[k])

        # Debugging, show in rviz the greatest gap
        if PUBLISH_VIZ:
            debugging_ranges = np.zeros_like(self.ranges)
            debugging_ranges[biggest_gap] = self.ranges[biggest_gap]
            self.publish_scan(self.biggest_gap_, self.latest_scan, debugging_ranges)

        return biggest_gap

    def get_target_point_max(self, biggest_gap):
        '''
        Get the target point as the furthest point in the biggest gap
        returns (middle_index_point, target_point)
        '''
        if len(biggest_gap) == 0:
            self.get_logger().info("There is no gap")
            return (int(np.argmin(np.abs(self.angles))), 0) # returns the index of the angle 0

        # Now get the target point, which is the max of the values
        middle_indx_point = int(biggest_gap[np.argmax(self.ranges[biggest_gap])])
        target_point = self.ranges[middle_indx_point]

        if PUBLISH_VIZ:
            local_range = np.zeros_like(self.ranges)
            local_range[middle_indx_point] = target_point
            self.publish_scan(self.target_point_, self.latest_scan, local_range)
        return (middle_indx_point, target_point)

    def get_target_point_middle(self, biggest_gap):
        '''
        First, using the biggest_gap, clip it using a threshold. Then, get the target point (the middle of the sequence)
        returns (middle_index_point, target_point)
        '''
        if len(biggest_gap) == 0:
            self.get_logger().info("There is no gap")
            return (int(np.argmin(np.abs(self.angles))), 0) # returns the index of the angle 0

        # First, clip biggest gap given a threshold value
        local_range = np.zeros_like(self.ranges)
        local_range[biggest_gap] = np.minimum(self.ranges[biggest_gap], self.clipping_distance_when_above_threshold)

        if PUBLISH_VIZ:
            self.publish_scan(self.clipped_gap_, self.latest_scan, local_range) # Publish this for debugging purposes

        # Now get the target point, which is the middle of the values
        middle_indx_point = int(biggest_gap[len(biggest_gap) // 2])
        target_point = local_range[middle_indx_point]

        if PUBLISH_VIZ:
            target_ranges = np.zeros_like(self.ranges)
            target_ranges[middle_indx_point] = target_point
            self.publish_scan(self.target_point_, self.latest_scan, target_ranges)
        return (middle_indx_point, target_point)

    def publish_scan(self, publisher, scan : LaserScan, new_ranges):
        msg = LaserScan()
        msg.header = scan.header
        msg.angle_min = scan.angle_min
        msg.angle_max = scan.angle_max
        msg.angle_increment = scan.angle_increment
        msg.time_increment = scan.time_increment
        msg.scan_time = scan.scan_time
        msg.range_min = scan.range_min
        msg.range_max = scan.range_max
        msg.ranges = new_ranges.tolist()   # must be a list of floats, not ndarray
        publisher.publish(msg)

    def publish_velocity(self, speed, steering_angle):
        '''
        Publishes speed and steering angle
        '''
        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = float(speed)
        ack_msg.drive.steering_angle = float(steering_angle)

        self.publisher_vel.publish(ack_msg)


def main(args=None):
    rclpy.init(args=args)

    node = GapFollowN()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
