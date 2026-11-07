# Circle grip test: full left lock, stepped speeds, records odom + IMU yaw rate.
# Safety: stops if anything is closer than STOP_DIST in front (+-60 deg) or on Ctrl+C.
import rclpy, numpy as np, time, sys, pickle, math
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescStateStamped

SPEEDS = [float(s) for s in sys.argv[1:]] or [1.5, 2.0, 2.5, 3.0]
HOLD = 3.0          # [s] per speed step
STEER = 0.30        # [rad] full left lock
STOP_DIST = 0.45    # [m] obstacle in front -> stop
rclpy.init(); n = rclpy.create_node('grip_test')
pub = n.create_publisher(AckermannDriveStamped, 'drive', 10)
st = {'v': 0.0, 'wz': 0.0, 'front': 10.0}; rows = []
def scan_cb(m):
    r = np.nan_to_num(np.array(m.ranges), nan=30, posinf=30); a = m.angle_min + m.angle_increment * np.arange(len(r))
    st['front'] = float(np.min(np.where((np.abs(a) < math.radians(60)) & (r > m.range_min), r, 30)))
n.create_subscription(LaserScan, 'scan', scan_cb, 10)
# speed straight from the VESC (erpm / speed_to_erpm_gain), works even if /odom is down
n.create_subscription(VescStateStamped, 'sensors/core', lambda m: st.update(v=m.state.speed / 4614.0), 10)
n.create_subscription(Imu, 'sensors/imu/raw', lambda m: st.update(wz=m.angular_velocity.z), 50)
def send(v, s):
    msg = AckermannDriveStamped(); msg.drive.speed = float(v); msg.drive.steering_angle = float(s); pub.publish(msg)
stopped = ''
try:
    for v_cmd in SPEEDS:
        end = time.time() + HOLD
        while time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.0)
            if st['front'] < STOP_DIST:
                stopped = f'obstacle {st["front"]:.2f} m in front at {v_cmd} m/s'; raise KeyboardInterrupt
            send(v_cmd, STEER); rows.append((time.time(), v_cmd, st['v'], st['wz']))
            time.sleep(0.02)
except KeyboardInterrupt:
    pass
for _ in range(25):
    send(0.0, 0.0); time.sleep(0.02)
pickle.dump(rows, open('grip_test.pkl', 'wb'))
print('STOPPED EARLY: ' + stopped if stopped else 'completed all steps')
# analysis: IMU wz is deg/s, ~150 ms lag; expected yaw rate = v * tan(steer) / wheelbase
rows = np.array(rows); t, vc, v, wz = rows.T; wz = np.radians(np.roll(wz, -8))
for s in SPEEDS:
    m = (vc == s) & (t > t[vc == s][0] + 1.5) if np.any(vc == s) else None   # last 1.5 s of each step (steady state)
    if m is None or m.sum() < 10: continue
    vv = np.median(v[m]); w = np.median(np.abs(wz[m])); exp = vv * math.tan(STEER) / 0.33
    print(f'cmd {s:.1f} m/s: real {vv:.2f} m/s  turn achieved {w/exp:.2f} of steering  lateral accel {vv*w:.2f} m/s^2')
