# Records scan + drive + speed + IMU + joystick R1 until killed; saves every 10 s to rec4.pkl
import rclpy, numpy as np, time, sys, pickle, signal
from sensor_msgs.msg import LaserScan, Imu, Joy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
rclpy.init(); n=rclpy.create_node('recorder4'); st={'v':0.,'cv':0.,'cs':0.,'wz':0.,'ay':0.,'r1':0}; rows=[]
n.create_subscription(AckermannDriveStamped,'drive',lambda m: st.update(cv=m.drive.speed,cs=m.drive.steering_angle),10)
n.create_subscription(Odometry,'odom',lambda m: st.update(v=m.twist.twist.linear.x),10)
n.create_subscription(Imu,'sensors/imu/raw',lambda m: st.update(wz=m.angular_velocity.z,ay=m.linear_acceleration.y),50)
n.create_subscription(Joy,'joy',lambda m: st.update(r1=int(len(m.buttons)>5 and m.buttons[5]==1)),10)
try:
    from vesc_msgs.msg import VescStateStamped
    n.create_subscription(VescStateStamped,'sensors/core',lambda m: st.update(v=m.state.speed/4614.0) if st.get('odom_stale',True) else None,10)
except ImportError: pass
n.create_subscription(LaserScan,'scan',lambda m: rows.append((time.time(),st['cv'],st['cs'],st['v'],np.array(m.ranges,dtype=np.float32),st['wz'],st['ay'],st['r1'])),10)
stop=[False]; signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0,True)); signal.signal(signal.SIGINT, lambda *a: stop.__setitem__(0,True))
last=time.time()
while not stop[0]:
    rclpy.spin_once(n,timeout_sec=0.1)
    if time.time()-last>10: pickle.dump(rows,open('rec4.pkl','wb')); last=time.time()
pickle.dump(rows,open('rec4.pkl','wb')); print('saved',len(rows))
