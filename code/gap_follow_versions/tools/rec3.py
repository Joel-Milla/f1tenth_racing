import rclpy, numpy as np, time, sys, pickle
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
rclpy.init(); n=rclpy.create_node('recorder3'); st={'v':0.,'cv':0.,'cs':0.,'wz':0.,'ay':0.}; rows=[]
n.create_subscription(AckermannDriveStamped,'drive',lambda m: st.update(cv=m.drive.speed,cs=m.drive.steering_angle),10)
n.create_subscription(Odometry,'odom',lambda m: st.update(v=m.twist.twist.linear.x),10)
n.create_subscription(Imu,'sensors/imu/raw',lambda m: st.update(wz=m.angular_velocity.z,ay=m.linear_acceleration.y),50)
n.create_subscription(LaserScan,'scan',lambda m: rows.append((time.time(),st['cv'],st['cs'],st['v'],np.array(m.ranges,dtype=np.float32),st['wz'],st['ay'])),10)
end=time.time()+float(sys.argv[1])
while time.time()<end: rclpy.spin_once(n,timeout_sec=0.1)
pickle.dump(rows,open('rec3.pkl','wb')); print(len(rows))
