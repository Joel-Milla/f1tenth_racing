import rclpy
import numpy as np
import math
import time

from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Joy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# =============================================================================
#                            TUNABLE PARAMETERS
#           (edit values here; nothing below needs to change)
# =============================================================================

# ---------------- Speed (continuous / dynamic) ----------------
# Commanded speed = min(clearance limit, cornering limit, MAX_SPEED), floored at MIN_SPEED,
# then ramped up no faster than MAX_ACCEL (slowing down is applied immediately).
MAX_SPEED               = 4.5   # [m/s] top speed on long straights (VESC hard limit ~5.0 incl. SPEED_COMMAND_GAIN)
MIN_SPEED               = 2.0   # [m/s] never go slower than this while a gap exists
SPEED_NO_GAP            = 0.0   # [m/s] used when no gap is found
BRAKE_DECEL             = 3.1   # [m/s^2] clearance limit: v = sqrt(2 * BRAKE_DECEL * (d_ahead - STOP_MARGIN))
                                #         measured on the car: only ~0.5-1.2 m/s^2 real braking (VESC limits)
STOP_MARGIN             = 0.5   # [m]  distance kept in reserve in front of the car
LATERAL_ACCEL           = 9.0   # [m/s^2] cornering limit: v = sqrt(LATERAL_ACCEL * turn_radius)
MAX_ACCEL               = 5.0   # [m/s^2] max rate the speed command may increase
SPEED_PATH_HALF_WIDTH   = 0.25  # [m]  d_ahead = distance along the arc the car is steering on to the first point
                                #      within this lateral distance of that arc
CORNER_SPEED            = 3.4   # [m/s] speed the car must be down to when it reaches the wall ahead (corner entry):
                                #       v = sqrt(CORNER_SPEED^2 + 2 * BRAKE_DECEL * (d - CORNER_BRAKE_POINT)), d = min(arc, straight)
                                #       (IMU: hard floor grip ~3.3 m/s^2 -> ~1.9 m/s at full lock)
STEER_SPEED_DEADBAND    = math.radians(10)  # [rad] steering below this = gentle/wavy bend: no corner slow-down,
                                #       only the driven arc is checked (0 = off)
CORNER_BRAKE_POINT      = 1.2   # [m]  wall distance where the car must be at CORNER_SPEED (it starts turning ~1.45 m out)
SPEED_RANGE_CAP         = 10.0  # [m]  cap on d_ahead (ignores huge/no-return readings)
SPEED_COMMAND_GAIN      = 1.10  # [-]  published speed = computed speed * this (VESC settles ~10% below command)
BRAKE_BOOST             = 0.8   # [-]  when odom is above the target, command target - BRAKE_BOOST * (odom - target)
                                #      so the VESC brakes hard (never below 0)
ACCEL_BOOST             = 0.5   # [-]  when odom is below the target, command target + ACCEL_BOOST * (target - odom)
VESC_MAX_SPEED          = 5.0   # [m/s] hard cap on the published command (speed_max 23250 erpm / 4614)
LIDAR_X_OFFSET          = 0.27  # [m]  lidar is this far ahead of the rear axle (base_link -> laser tf)

# ---------------- Steering (pure pursuit toward the target point) ----------------
# steer = atan(2 * WHEELBASE * sin(target_angle) / lookahead),
# lookahead = min(target range, closest obstacle in LOOKAHEAD_CONE_ANGLE) clipped to [MIN, MAX]
WHEELBASE               = 0.33  # [m]  used for pure pursuit and turn radius
LOOKAHEAD_CONE_ANGLE    = math.radians(30)  # [rad] obstacles inside +/- this angle shorten the lookahead
LOOKAHEAD_MIN           = 0.5   # [m]  smaller -> sharper reaction to close targets / obstacles
LOOKAHEAD_MAX           = 2.2   # [m]  larger -> calmer on straights (less wobble), but turns in later
LOOKAHEAD_TIME          = 0.5   # [s]  at speed, max lookahead grows to odom speed * this (calmer at high speed)
MAX_STEERING_LEFT       = 0.30  # [rad] servo limit for left turns (vesc.yaml servo_min 0.10)
MAX_STEERING_RIGHT      = 0.31  # [rad] servo limit for right turns (vesc.yaml servo_max 0.85)
STEERING_GAIN           = 1.0   # [-]  multiplier on pure-pursuit steering
STEERING_SMOOTHING      = 0.75  # [-]  0..1, weight of new command (1.0 = no smoothing)
MAX_STEERING_RATE       = 2.5   # [rad/s] max change of steering command (0 -> full lock in ~0.08 s)

# ---------------- Corner anticipation (turn in sooner) ----------------
# When a wall is ahead (straight-ahead corridor) closer than TURN_IN_DISTANCE, add a steering bias toward
# the more open side, growing from 0 at TURN_IN_DISTANCE to TURN_IN_GAIN at STOP_MARGIN.
TURN_IN_DISTANCE        = 2.4   # [m]  start leaning into the corner when the wall ahead is this close
TURN_IN_GAIN            = 0.0   # [rad] max extra steering toward the open side (0 = off)
TURN_IN_MIN_DIFFERENCE  = 0.4   # [m]  sides must differ by this much (median range 20-90 deg) to count as a corner

# ---------------- Gap finding ----------------
GAP_THRESHOLD           = 1.0   # [m]  min range for a beam to count as part of a gap
CLIPPING_DISTANCE       = 1.8   # [m]  ranges in the chosen gap are clipped to this value
TARGET_MODE             = 'max'     # 'middle' -> aim at centre of biggest gap (turns in late)
                                    # 'max'    -> aim at furthest region in biggest gap
TARGET_DEEP_FRACTION    = 0.98  # [-]  'max' mode: aim at middle of beams >= this * furthest range (stabilises target)
GAP_FOV                 = math.radians(110)  # [rad] only beams within +/- this angle can form a gap
GAP_SELECTION           = 'deepest' # 'deepest' -> gap containing the furthest beam (stable, follows the track)
                                    # 'widest'  -> gap with the most beams (can pick short side openings)

# ---------------- Disparity extender ----------------
DISPARITY_THRESHOLD     = 0.3   # [m]  jump between adjacent beams that counts as a disparity
DISPARITY_BUFFER        = 1.6   # [-]  multiplier on car half-width when extending disparities

# ---------------- Safety bubble ----------------
BUBBLE_BUFFER           = 1.0   # [-]  bubble radius = car half-diagonal * this (1.1 = +10%)

# ---------------- Rear-quadrant check ----------------
BACK_QUADRANT_THRESHOLD = 0.0            # [m]   obstacle closer than this behind the car blocks turning toward it
                                         #       (0 = off; it snapped steering to 0 mid-corner, side clearance check replaces it)
BACK_QUADRANT_ANGLE     = math.pi / 2    # [rad] beams beyond +/- this angle are treated as "behind"

# ---------------- Side clearance check ----------------
# Anything inside a box beside the car (lidar frame: x forward, y left) blocks steering toward that side
# until it has cleared, so the car does not cut into an obstacle it is still passing.
SIDE_CHECK              = False   # off for the narrow race track (blocked turns into the inner wall); True for single obstacles
SIDE_CLEARANCE          = 0.25  # [m]  lateral distance from car centreline (half width 0.15 + margin)
SIDE_CHECK_FRONT        = 0.20  # [m]  box starts this far ahead of the lidar
SIDE_CHECK_BACK         = 0.45  # [m]  box ends this far behind the lidar (rear of car)
# the lidar is blind past +/-135 deg, so beside the rear of the car is not visible: once an obstacle
# was seen beside the car, keep blocking until the car has driven SIDE_HOLD_DISTANCE further (odom)
SIDE_HOLD_DISTANCE      = 0.45  # [m]
SIDE_HOLD_MAX_TIME      = 0.6   # [s]  safety cap on the hold (e.g. if odom is not updating)

# ---------------- Car dimensions ----------------
CAR_WIDTH               = 0.30  # [m] full width across the tires (0.296 rounded up)
CAR_LENGTH              = 0.57  # [m] bumper to bumper

# ---------------- Topics / QoS ----------------
SCAN_TOPIC              = 'scan'
ODOM_TOPIC              = 'odom'   # real car; use 'ego_racecar/odom' in the simulator
VESC_STATE_TOPIC        = 'sensors/core'  # backup speed source (erpm) if odom stops publishing
SPEED_TO_ERPM_GAIN      = 4614.0   # vesc.yaml speed_to_erpm_gain
ODOM_TIMEOUT            = 0.2      # [s] odom older than this -> use the VESC speed instead

# ---------------- Launch boost (race start) ----------------
LAUNCH_ENABLED          = True     # first LAUNCH_TIME after R1 is first pressed: full speed, steering from gap follow
LAUNCH_TIME             = 1.5      # [s]
LAUNCH_SPEED            = 5.0      # [m/s] command during launch (VESC max ~5.0; the car reaches ~2.5-3.5 in 1 s)
LAUNCH_ABORT_DIST       = 1.0      # [m] skip the launch if anything is this close straight ahead (0 = never skip)
LAUNCH_MOVING_SPEED     = 0.2      # [m/s] LAUNCH_TIME is counted from when the car really moves (VESC wakes up ~3 s late)
LAUNCH_MAX_WAIT         = 4.0      # [s] give up the launch if the car has not moved by then

# ---------------- Deadman switch (hold R1 to drive) ----------------
DEADMAN_ENABLED         = True     # False = drive as soon as the node starts (old behaviour)
DEADMAN_BUTTON          = 5        # joy button index: 5 = R1 / RB (same as joy_teleop autonomous_control)
JOY_TOPIC               = 'joy'
DEADMAN_TIMEOUT         = 0.3      # [s] no /joy message for this long (controller lost) -> stop
DEADMAN_STOP_TIME       = 0.5      # [s] after release, keep sending stop commands this long, then go quiet
                                   #     so manual driving (L1) through the mux works again
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

        # backup speed straight from the VESC (only on the real car, where vesc_msgs exists)
        self.last_odom_time = 0.0
        try:
            from vesc_msgs.msg import VescStateStamped
            self.subscription_vesc = self.create_subscription(
                VescStateStamped,
                VESC_STATE_TOPIC,
                self.vesc_state_callback,
                QOS_DEPTH)
        except ImportError:
            self.get_logger().warn('vesc_msgs not found, speed only from odom')

        self.publisher_vel = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, QOS_DEPTH)

        # deadman switch: only drive while R1 is held
        self.deadman_pressed = False
        self.last_joy_time = 0.0
        self.deadman_release_time = 0.0
        self.subscription_joy = self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, QOS_DEPTH)

        # launch boost: runs once, starting the first time the car is allowed to drive
        self.launch_start = None
        self.launch_moving_start = None
        self.launch_done = not LAUNCH_ENABLED

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
        self.side_hold = {'L': [0.0, 0.0], 'R': [0.0, 0.0]} # [remaining distance, remaining time] per side
        self.side_last_time = time.monotonic()
        self.side_block = ''
        self.d_ahead = 0.0
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
        self.last_odom_time = time.monotonic()

    def joy_callback(self, msg : Joy):
        '''
        Tracks the deadman button (R1)
        '''
        pressed = len(msg.buttons) > DEADMAN_BUTTON and msg.buttons[DEADMAN_BUTTON] == 1
        if self.deadman_pressed and not pressed:
            self.deadman_release_time = time.monotonic()
        self.deadman_pressed = pressed
        self.last_joy_time = time.monotonic()

    def launch_active(self):
        '''
        True during the first LAUNCH_TIME seconds of driving (once per node run)
        '''
        if self.launch_done:
            return False
        now = time.monotonic()
        if self.launch_start is None:
            x = self.raw_ranges * np.cos(self.angles)
            y = self.raw_ranges * np.sin(self.angles)
            ahead = (x > 0) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
            d_ahead = float(np.min(x[ahead])) if np.any(ahead) else SPEED_RANGE_CAP
            if d_ahead < LAUNCH_ABORT_DIST:
                self.get_logger().info(f'launch skipped: obstacle {d_ahead:.2f} m ahead')
                self.launch_done = True
                return False
            self.launch_start = now
            self.get_logger().info('launch!')
        if self.launch_moving_start is None and self.speed > LAUNCH_MOVING_SPEED:
            self.launch_moving_start = now
        if self.launch_moving_start is None:
            if now - self.launch_start < LAUNCH_MAX_WAIT:
                return True  # still waiting for the wheels to turn
        elif now - self.launch_moving_start < LAUNCH_TIME:
            return True
        # hand over to normal control from the current speed so it does not drop suddenly
        self.launch_done = True
        self.last_speed_cmd = max(self.speed, 0.0)
        self.last_cmd_time = now
        return False

    def deadman_active(self):
        '''
        True while R1 is held and the controller is still sending messages
        '''
        if not DEADMAN_ENABLED:
            return True
        now = time.monotonic()
        if self.deadman_pressed and now - self.last_joy_time > DEADMAN_TIMEOUT:
            # controller went quiet while held -> treat as released
            self.deadman_pressed = False
            self.deadman_release_time = now
        return self.deadman_pressed

    def vesc_state_callback(self, msg):
        '''
        Uses the VESC motor speed when odom has stopped arriving
        '''
        if time.monotonic() - self.last_odom_time > ODOM_TIMEOUT:
            self.speed = msg.state.speed / SPEED_TO_ERPM_GAIN

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

        # steering angle: pure pursuit toward the target point
        target_angle = self.angles[middle_indx_point]
        # an obstacle close in front shortens the lookahead -> steer harder around it
        closest_front = float(np.min(self.raw_ranges[np.abs(self.angles) <= LOOKAHEAD_CONE_ANGLE]))
        lookahead_max = max(LOOKAHEAD_MAX, abs(self.speed) * LOOKAHEAD_TIME)
        lookahead = min(max(min(target_point, closest_front), LOOKAHEAD_MIN), lookahead_max)
        steering_angle = STEERING_GAIN * math.atan2(2.0 * WHEELBASE * math.sin(target_angle), lookahead)
        steering_angle += self.get_turn_in_bias()
        steering_angle = self.check_close_3rd_4th_quadrant(steering_angle, target_point)
        steering_angle = float(np.clip(steering_angle, -MAX_STEERING_RIGHT, MAX_STEERING_LEFT))
        required_steering = steering_angle # before smoothing, used for the corner speed limit
        steering_angle = STEERING_SMOOTHING * steering_angle + (1.0 - STEERING_SMOOTHING) * self.last_steering
        steering_angle = self.check_side_clearance(steering_angle)
        # rate limit -> one smooth steering motion instead of jumps
        max_step = MAX_STEERING_RATE * (msg.scan_time if msg.scan_time > 0 else 0.025)
        steering_angle = float(np.clip(steering_angle, self.last_steering - max_step, self.last_steering + max_step))
        self.last_steering = steering_angle

        # speed
        if target_point == 0:
            speed = SPEED_NO_GAP
        else:
            speed = self.get_optimal_linear_speed(required_steering)
        speed = self.rate_limit_speed(speed)

        if LOG_STEERING:
            now = time.monotonic()
            if now - self.last_log_time > LOG_PERIOD:
                self.last_log_time = now
                self.get_logger().info(
                    f'speed {speed:4.2f} m/s | steer {steering_angle:+.3f} rad | '
                    f'd_ahead {self.d_ahead:4.2f} m | odom {self.speed:4.2f} | '
                    f'side_block {self.side_block or "-"} | '
                    f'calc {(time.perf_counter() - t_start) * 1000:4.1f} ms')
        if self.deadman_active():
            if self.launch_active():
                self.publish_velocity(LAUNCH_SPEED, steering_angle)
            else:
                self.publish_velocity(self.boost_speed_command(speed), steering_angle)
        else:
            # not held: reset the speed ramp so it starts gently next time, send stop briefly, then go quiet
            self.last_speed_cmd = 0.0
            if time.monotonic() - self.deadman_release_time < DEADMAN_STOP_TIME:
                self.publish_velocity(0.0, 0.0)

    def check_close_3rd_4th_quadrant(self, steering_angle, distance):
        close = self.raw_ranges < self.back_quadrant_threshold
        object_right = bool(np.any(close & (self.angles < -BACK_QUADRANT_ANGLE)))
        object_left  = bool(np.any(close & (self.angles >  BACK_QUADRANT_ANGLE)))

        if steering_angle > 0 and object_left:
            return 0.0

        if steering_angle < 0 and object_right:
            return 0.0

        return steering_angle

    def get_turn_in_bias(self):
        '''
        Lean toward the more open side when a wall is coming up ahead, before the corner opening is visible
        '''
        if TURN_IN_GAIN <= 0:
            return 0.0
        x = self.raw_ranges * np.cos(self.angles)
        y = self.raw_ranges * np.sin(self.angles)
        ahead = (x > 0) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
        d_wall = float(np.min(x[ahead])) if np.any(ahead) else SPEED_RANGE_CAP
        if d_wall >= TURN_IN_DISTANCE:
            return 0.0
        left = float(np.median(self.raw_ranges[(self.angles > math.radians(20)) & (self.angles < math.radians(90))]))
        right = float(np.median(self.raw_ranges[(self.angles < -math.radians(20)) & (self.angles > -math.radians(90))]))
        if abs(left - right) < TURN_IN_MIN_DIFFERENCE:
            return 0.0
        weight = min((TURN_IN_DISTANCE - d_wall) / max(TURN_IN_DISTANCE - STOP_MARGIN, 1e-3), 1.0)
        return TURN_IN_GAIN * weight * (1.0 if left > right else -1.0)

    def check_side_clearance(self, steering_angle):
        '''
        Hold steering straight while an obstacle beside the car is on the side we want to turn to
        '''
        self.side_block = ''
        if not SIDE_CHECK:
            return steering_angle
        now = time.monotonic()
        dt = min(now - self.side_last_time, 0.1)
        self.side_last_time = now
        travelled = abs(self.speed) * dt
        for side in self.side_hold:
            self.side_hold[side][0] -= travelled  # remaining distance
            self.side_hold[side][1] -= dt         # remaining time

        x = self.raw_ranges * np.cos(self.angles)
        y = self.raw_ranges * np.sin(self.angles)
        beside = (x <= SIDE_CHECK_FRONT) & (x >= -SIDE_CHECK_BACK) & (self.raw_ranges > self.latest_scan.range_min)
        if np.any(beside & (y > 0) & (y < SIDE_CLEARANCE)):
            self.side_hold['L'] = [SIDE_HOLD_DISTANCE, SIDE_HOLD_MAX_TIME]
        if np.any(beside & (y < 0) & (y > -SIDE_CLEARANCE)):
            self.side_hold['R'] = [SIDE_HOLD_DISTANCE, SIDE_HOLD_MAX_TIME]

        def held(side):
            return self.side_hold[side][0] > 0 and self.side_hold[side][1] > 0

        # walls close on both sides (narrow section): holding straight never helps, so do not block
        if held('L') and held('R'):
            return steering_angle

        if steering_angle > 0 and held('L'):
            self.side_block = 'L'
            return 0.0
        if steering_angle < 0 and held('R'):
            self.side_block = 'R'
            return 0.0
        return steering_angle

    def get_optimal_linear_speed(self, steering_angle):
        '''
        Continuous speed from free distance along the path we are steering on and from how hard we are turning
        '''
        # free distance along the arc driven with this steering angle (rear axle frame), inside a
        # corridor as wide as the car + margin
        x = self.raw_ranges * np.cos(self.angles) + LIDAR_X_OFFSET
        y = self.raw_ranges * np.sin(self.angles)
        curvature = math.tan(steering_angle) / WHEELBASE
        if abs(curvature) < 1e-3:
            lateral_error = np.abs(y)
            arc_length = x
        else:
            radius = 1.0 / curvature # signed, centre of the turn at (0, radius)
            lateral_error = np.abs(np.hypot(x, y - radius) - abs(radius))
            arc_angle = np.arctan2(x, np.sign(radius) * (radius - y))
            arc_length = np.where(arc_angle > 0, abs(radius) * arc_angle, -1.0)
        in_path = (arc_length > LIDAR_X_OFFSET) & (lateral_error <= SPEED_PATH_HALF_WIDTH)
        if np.any(in_path):
            self.d_ahead = float(min(np.min(arc_length[in_path]) - LIDAR_X_OFFSET, SPEED_RANGE_CAP))
        else:
            self.d_ahead = SPEED_RANGE_CAP

        # limit 1: be down to CORNER_SPEED when reaching the wall ahead (along the arc or straight ahead,
        # whichever is closer) -> full speed on straights, hard braking right before the corner
        straight = (x > LIDAR_X_OFFSET) & (np.abs(y) <= SPEED_PATH_HALF_WIDTH)
        d_straight = float(np.min(x[straight]) - LIDAR_X_OFFSET) if np.any(straight) else SPEED_RANGE_CAP
        gentle = abs(steering_angle) < STEER_SPEED_DEADBAND
        # gentle (wavy) bend: keep speed if either the driven arc or straight ahead is clear,
        # since the car will keep adjusting its steering through the waves
        # committed turn: the wall straight ahead is the one the car is turning away from, so only the
        # driven arc counts (it was cutting speed mid-corner at full lock)
        d_brake = max(self.d_ahead, d_straight) if gentle else self.d_ahead
        v_clear = math.sqrt(CORNER_SPEED**2 + 2.0 * BRAKE_DECEL * max(d_brake - CORNER_BRAKE_POINT, 0.0))
        # inside the brake point, something close ahead must still slow the car toward a stop
        v_clear = min(v_clear, math.sqrt(2.0 * BRAKE_DECEL * max(d_brake - STOP_MARGIN, 0.0)) + CORNER_SPEED * 0.5)

        # limit 2: lateral acceleration in the turn, radius = wheelbase / tan(steer)
        tan_steer = abs(math.tan(steering_angle))
        if tan_steer > 1e-3 and not gentle:
            v_turn = math.sqrt(LATERAL_ACCEL * WHEELBASE / tan_steer)
        else:
            v_turn = MAX_SPEED

        return max(MIN_SPEED, min(MAX_SPEED, v_clear, v_turn))

    def boost_speed_command(self, speed):
        '''
        Over-command the VESC speed loop so it brakes / accelerates harder toward the target speed
        '''
        if speed <= 0.0:
            return 0.0
        error = speed - self.speed  # target - odom
        if error < 0.0:
            command = speed + BRAKE_BOOST * error
        else:
            command = speed + ACCEL_BOOST * error
        return float(np.clip(command * SPEED_COMMAND_GAIN, 0.0, VESC_MAX_SPEED))

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
            if GAP_SELECTION == 'widest':
                k = int(np.argmax(ends - starts))
            else:
                k = int(np.argmax([self.ranges[s:e].max() for s, e in zip(starts, ends)]))
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

        # Now get the target point: middle of the deep region (beams close to the max) around the furthest beam,
        # which is much steadier than the single furthest beam
        gap_ranges = self.ranges[biggest_gap]
        deepest = int(np.argmax(gap_ranges))
        deep = gap_ranges >= TARGET_DEEP_FRACTION * gap_ranges[deepest]
        lo = deepest
        while lo > 0 and deep[lo - 1]:
            lo -= 1
        hi = deepest
        while hi < len(deep) - 1 and deep[hi + 1]:
            hi += 1
        middle_indx_point = int(biggest_gap[(lo + hi) // 2])
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
