import pickle, importlib.util, rclpy, numpy as np, types, sys
from sensor_msgs.msg import LaserScan
rows=pickle.load(open('rate4.pkl','rb')); res={}
rclpy.init()
for name,path in [('before','gap_follow_params.L1_rate4.py'),('sooner','/home/f1tenth/f1tenth_ws/src/gap_follow/gap_follow/gap_follow_params.py')]:
    s=importlib.util.spec_from_file_location(name,path); g=importlib.util.module_from_spec(s); s.loader.exec_module(g)
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False
    n=g.GapFollowN(); out={}; n.publish_velocity=lambda v,a: out.update(v=v,a=a); st=[]
    for t,cv,cs,v,r in rows:
        clock[0]=t; m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=r.tolist()
        n.speed=v; n.scan_callback(m); st.append(out['a'])
    res[name]=np.array(st); n.destroy_node()
b,s_=res['before'],res['sooner']; t=np.array([r[0] for r in rows])-rows[0][0]
# for each turn-in (before crosses 0.15), how many ms earlier does 'sooner' cross 0.15
on=np.nonzero((np.abs(b[1:])>0.15)&(np.abs(b[:-1])<=0.15))[0]+1; leads=[]
for i in on:
    j=i
    while j>0 and j>i-40 and np.abs(s_[j-1])>0.15: j-=1
    leads.append((i-j)*25)
print('turn-ins:',len(on),' sooner turns in earlier by (ms):',leads)
d=np.diff(s_); rev=((d[1:]*d[:-1])<0)&(np.abs(d[1:])>0.02); db=np.diff(b); revb=((db[1:]*db[:-1])<0)&(np.abs(db[1:])>0.02)
print(f'direction reversals/s before {revb.sum()/(len(b)/40):.2f}  sooner {rev.sum()/(len(b)/40):.2f}')
lags=range(-20,21); c=[np.corrcoef(b[20:-20], s_[20+L:len(s_)-20+L])[0,1] for L in lags]
L=list(lags)[int(np.argmax(c))]; print(f'best alignment: sooner is {-L*25} ms earlier than before (corr {max(c):.3f})')
# also: where does the target point when the car first sees a corner? compare mean |steer| when d_ahead-ish front distance 1.5-2.5
big=np.nonzero((np.abs(b[1:])>0.25)&(np.abs(b[:-1])<=0.25))[0]+1
for k in [16,12,8,4]:
    sb=np.mean([np.sign(b[i])*b[i-k] for i in big if i>=k]); ss=np.mean([np.sign(b[i])*s_[i-k] for i in big if i>=k])
    print(f'{k*25:4d} ms before a hard turn: steering toward the corner  before {sb:+.3f}  new {ss:+.3f}')
