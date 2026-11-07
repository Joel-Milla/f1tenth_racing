import rclpy
import numpy as np
import math

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


# =============================================================================
#                            TUNABLE PARAMETERS
#           (edit values here; nothing below needs to change)
# =============================================================================

# ---------------- Speed ----------------
SPEED_FAST              = 3.5   # [m/s] used when target point is far (> SPEED_DISTANCE_THRESHOLD)
SPEED_SLOW              = 2.0   # [m/s] used when target point is close
SPEED_NO_GAP            = 0.0   # [m/s] used when no gap is found (target distance == 0)
SPEED_DISTANCE_THRESHOLD = 1.1  # [m]  target distance above which SPEED_FAST is used

# ---------------- Gap finding ----------------
GAP_THRESHOLD           = 1.0   # [m]  min range for a beam to count as part of a gap
CLIPPING_DISTANCE       = 1.8   # [m]  ranges in the chosen gap are clipped to this value
TARGET_MODE             = 'middle'  # 'middle' -> aim at centre of biggest gap
                                    # 'max'    -> aim at furthest point in biggest gap

# ---------------- Disparity extender ----------------
DISPARITY_THRESHOLD     = 0.3   # [m]  jump between adjacent beams that counts as a disparity
DISPARITY_BUFFER        = 3     # [-]  multiplier on car half-width when extending disparities

# ---------------- Safety bubble ----------------
BUBBLE_BUFFER           = 1.1   # [-]  bubble radius = car half-diagonal * this (1.1 = +10%)

# ---------------- Rear-quadrant check ----------------
BACK_QUADRANT_THRESHOLD = 0.5            # [m]   obstacle closer than this behind the car blocks turning toward it
BACK_QUADRANT_ANGLE     = math.pi / 2    # [rad] beams beyond +/- this angle are treated as "behind"

# ---------------- Car dimensions ----------------
CAR_WIDTH               = 0.30  # [m] full width across the tires (0.296 rounded up)
CAR_LENGTH              = 0.57  # [m] bumper to bumper

# ---------------- Topics / QoS ----------------
SCAN_TOPIC              = 'scan'
ODOM_TOPIC              = 'ego_racecar/odom'
DRIVE_TOPIC             = 'drive'
QOS_DEPTH               = 10

# Visualization topics
VIZ_BUBBLE_TOPIC        = 'scan_bubble'
VIZ_DISPARITY_TOPIC     = 'scan_disparity'
VIZ_BIGGEST_GAP_TOPIC   = 'biggest_gap_'
VIZ_CLIPPED_GAP_TOPIC   = 'clipped_gap_'
VIZ_TARGET_POINT_TOPIC  = 'target_point_'

# ---------------- Debug ----------------
LOG_STEERING            = True  # print steering angle every scan

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
        self.distance_threshold = SPEED_DISTANCE_THRESHOLD
        self.clipping_distance_when_above_threshold = CLIPPING_DISTANCE
        self.disparity_buffer = DISPARITY_BUFFER

        # set dimensions of the car
        self.car_width  = CAR_WIDTH
        self.car_length = CAR_LENGTH
        self.half_width  = self.car_width  / 2.0
        self.half_length = self.car_length / 2.0

        self.speed = 0.0

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
        self.latest_scan = msg
        self.ranges = np.array(msg.ranges, dtype=float) # get ranges of type float
        np.nan_to_num(self.ranges, False, np.inf, np.inf, np.inf) # in place, change nan/posinf/neginf to inf values
        self.angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment, float) # get angles in radians, which their postiion are matched in ranges array

        self.create_disparity()
        self.draw_safety_bubble()
        biggest_gap = self.find_biggest_gap()

        if TARGET_MODE == 'max':
            middle_indx_point, target_point = self.get_target_point_max(biggest_gap)
        else:
            middle_indx_point, target_point = self.get_target_point_middle(biggest_gap)

        # publish speed and steering angle
        speed = self.get_optimal_linear_speed(distance=target_point)
        steering_angle  = self.angles[middle_indx_point]
        steering_angle = self.check_close_3rd_4th_quadrant(steering_angle, target_point)
        if LOG_STEERING:
            self.get_logger().info(f'Steering angle: {steering_angle}')
        self.publish_velocity(speed, steering_angle)

    def check_close_3rd_4th_quadrant(self, steering_angle, distance):
        object_right = False
        object_left = False
        for angle, curr_range in zip(self.angles, self.ranges):
            if angle < -BACK_QUADRANT_ANGLE:
                if curr_range < self.back_quadrant_threshold:
                    object_right = True

            if angle > BACK_QUADRANT_ANGLE:
                if curr_range < self.back_quadrant_threshold:
                    object_left = True

        if steering_angle > 0 and object_left:
            return 0.0

        if steering_angle < 0 and object_right:
            return 0.0

        return steering_angle

    def get_optimal_linear_speed(self, distance):
        if distance > self.distance_threshold:
            return SPEED_FAST
        elif distance == 0:
            return SPEED_NO_GAP
        else:
            return SPEED_SLOW

    def create_disparity(self):
        disparity_ranges = self.ranges.copy()
        curr_indx = 0
        while (curr_indx < (len(disparity_ranges) - 1)): # traverse to the second to last value
            curr_range = self.ranges[curr_indx]
            next_range = self.ranges[curr_indx + 1]

            difference = curr_range - next_range

            # if difference is significant (i.e. greater than threshold)
            if abs(difference) > self.disparity_threshold:
                # Means that current beam is greater than next beam
                increment = 0
                starting_indx = 0
                if difference > 0:
                    smallest_beam = next_range

                    # set increment
                    increment = -1
                    starting_indx = curr_indx + 1 #start on next index, and go to right of car (or left of array)
                else: #means that current beam is smaller than next beam
                    smallest_beam = curr_range

                    increment = 1 #set increment
                    starting_indx = curr_indx

                if smallest_beam == 0:
                    smallest_beam = self.latest_scan.range_min # if smallest beam got zero, change to minimum range lidar can detect

                # calculate n, which is the number of indices that need to travel and set to specific width
                n = (self.half_width * self.disparity_buffer) / (smallest_beam * self.latest_scan.angle_increment)
                n = int(math.ceil(n)) # round it up, and change type to int

                # starting on next value, cover its right side (-1 when going to right side because decreasing index as
                # indices closer to zero means going to right of beam and +1 are incrementing index postiion and going to left of car),
                # with n steps, and substitute with smallest_beam value.
                self.set_values_in_range(disparity_ranges, starting_indx, n, increment, smallest_beam)
            curr_indx += 1

        self.publish_scan(self.scan_disparity_, self.latest_scan, disparity_ranges)
        self.ranges = disparity_ranges

    def draw_safety_bubble(self):
        '''
        Draw a safety bubble around closest point
        '''
        r_rob = math.sqrt(self.half_width**2 + self.half_length**2) # radius of circle around the car, which has length of diagonal using pythagorean theorem
        r_rob *= BUBBLE_BUFFER # inflate as a buffer

        ranges = self.ranges.copy()
        min_indx = np.argmin(ranges)
        min_distance = ranges[min_indx]

        angle_increment = self.latest_scan.angle_increment

        # Distance to a specific center
        def dist_to_center(indx):
            dtheta = (indx - min_indx) * angle_increment
            # using law of cosines, get distance between two points
            return math.sqrt(ranges[indx]**2 + min_distance**2
                     - 2*ranges[indx]*min_distance*math.cos(dtheta))

        indx = min_indx
        while indx < len(ranges) and dist_to_center(indx) < r_rob: # check if index within circle of r_rob
            ranges[indx] = 0.0
            indx += 1

        indx = min_indx
        while indx > 0 and dist_to_center(indx) < r_rob: # check if index within circle of r_rob
            ranges[indx] = 0.0
            indx -= 1

        self.ranges = ranges
        self.publish_scan(self.publisher_bubble_, self.latest_scan, ranges)

    def set_values_in_range(self, array, starting_indx, iterations, increment, new_value):
        '''
        Given an array, an starting index, number of interations, and increment to be made, traverse the array by updating ...
        with value given
        @increment -> +1 or -1
        '''
        end_indx = starting_indx + (iterations * increment)
        if end_indx >= len(array):
            end_indx = len(array) - 1 # last index of array

        if end_indx < 0:
            end_indx = 0

        for indx in range(starting_indx, end_indx + increment, increment):
            array[indx] = (min(array[indx], new_value)) # if two disparities overlap, select smallest one

    def find_biggest_gap(self):
        '''
        Returns an array of indices of the maximum length sequence found
        '''

        biggest_gap = []
        local_sequence = []

        for indx, curr_range in enumerate(self.ranges):
            if curr_range >= self.gap_threshold:
                local_sequence.append(indx)
            else:
                if len(local_sequence) > len(biggest_gap):
                    biggest_gap = local_sequence
                local_sequence = []

        if len(local_sequence) > len(biggest_gap):
            biggest_gap = local_sequence

        # Debugging, show in rviz the greatest gap
        debugging_ranges = []
        for indx, curr_range in enumerate(self.ranges):
            if indx not in biggest_gap:
                debugging_ranges.append(0.0)
            else:
                debugging_ranges.append(curr_range)
        self.publish_scan(self.biggest_gap_, self.latest_scan, np.array(debugging_ranges))

        return biggest_gap

    def get_target_point_max(self, biggest_gap):
        '''
        Get the target point as the furthest point in the biggest gap
        returns (middle_index_point, target_point)
        '''
        if len(biggest_gap) == 0:
            self.get_logger().info("There is no gap")
            return (len(self.ranges) // 2, 0) # returns the index of the angle 0

        local_range = self.ranges.copy()
        for indx in range(len(local_range)):
            if indx not in biggest_gap: # if not in biggest gap, make range 0.0
                local_range[indx] = 0.0

        # Now get the target point, which is the max of the values
        middle_indx_point = np.argmax(local_range)
        for indx in range(len(local_range)): # If not target point, close that to zero
            if indx != middle_indx_point:
                local_range[indx] = 0.0

        self.publish_scan(self.target_point_, self.latest_scan, np.array(local_range))
        target_point = local_range[middle_indx_point]
        return (middle_indx_point, target_point)

    def get_target_point_middle(self, biggest_gap):
        '''
        First, using the biggest_gap, clip it using a threshold. Then, get the target point (the middle of the sequence)
        returns (middle_index_point, target_point)
        '''
        if len(biggest_gap) == 0:
            self.get_logger().info("There is no gap")
            return (len(self.ranges) // 2, 0) # returns the index of the angle 0
        local_range = self.ranges.copy()

        # First, clip biggest gap given a threshold value
        for indx, curr_range in enumerate(local_range):
            if indx not in biggest_gap:
                local_range[indx] = 0.0
            else: # check if bigger than threshold, to reset value
                if curr_range > self.clipping_distance_when_above_threshold:
                    local_range[indx] = self.clipping_distance_when_above_threshold

        self.publish_scan(self.clipped_gap_, self.latest_scan, np.array(local_range)) # Publish this for debugging purposes

        # Now get the target point, which is the middle of the values
        middle_indx_point = biggest_gap[len(biggest_gap) // 2]
        for indx in range(len(local_range)): # If not target point, close that to zero
            if indx != middle_indx_point:
                local_range[indx] = 0.0

        self.publish_scan(self.target_point_, self.latest_scan, np.array(local_range))
        target_point = local_range[middle_indx_point]
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
        ack_msg.drive.speed = speed
        ack_msg.drive.steering_angle = steering_angle

        self.publisher_vel.publish(ack_msg)


def main(args=None):
    rclpy.init(args=args)

    node = GapFollowN()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()