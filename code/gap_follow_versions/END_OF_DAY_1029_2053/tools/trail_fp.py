import pickle, importlib.util, rclpy, numpy as np, types, sys
from sensor_msgs.msg import LaserScan
rclpy.init()
s=importlib.util.spec_from_file_location('g',sys.argv[1]); g=importlib.util.module_from_spec(s); s.loader.exec_module(g)
for rec in sys.argv[2:]:
    rows=[x[:5] for x in pickle.load(open(rec,'rb'))]
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False
    n=g.GapFollowN(); n.launch_done=True; n.publish_velocity=lambda v,a: None
    det=trail=mv=0; dists=[]
    for t,cv,cs,v,r in rows:
        clock[0]=t; m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=r.tolist()
        n.speed=v; n.scan_callback(m)
        if v>0.8:
            mv+=1; det+= n.opp_dist is not None; trail+=n.trailing or bool(getattr(n,"pass_side",""))
            if n.opp_dist is not None: dists.append(n.opp_dist)
    print(f"{rec}: moving scans {mv}  'car' detected {det} ({det/max(mv,1)*100:.1f}%)  TRAILING {trail} ({trail/max(mv,1)*100:.1f}%)" + (f"  det dist median {np.median(dists):.2f}" if dists else ''))
    n.destroy_node()
