import pickle, importlib.util, rclpy, numpy as np, types, sys
from sensor_msgs.msg import LaserScan
rclpy.init()
recs=['race1.pkl','fast3b.pkl']
variants=[('current (RACE_SAFE_pass)','/home/f1tenth/gap_follow_backups/RACE_SAFE_pass.py',{}),
          ('A+B (0.2 s)','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{}),
          ('A+B, filter 0.4 s','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{'TARGET_FILTER_TAU':0.4}),
          ('A+B + steer rate 1.8','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{'MAX_STEERING_RATE':1.8}),
          ('A+B + lookahead max 3.0','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{'LOOKAHEAD_MAX':3.0}),
          ('A+B + deep frac 0.90','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{'TARGET_DEEP_FRACTION':0.90}),
          ('A+B 0.4s + LA 3.0','/home/f1tenth/gap_follow_backups/RACE_SMOOTH.py',{'TARGET_FILTER_TAU':0.4,'LOOKAHEAD_MAX':3.0})]
base=None
for label,path,over in variants:
    allst=[]; allv=[]
    for rec in recs:
        rows=[x[:5] for x in pickle.load(open(rec,'rb'))]
        s=importlib.util.spec_from_file_location('m'+str(abs(hash(label))),path); g=importlib.util.module_from_spec(s); s.loader.exec_module(g)
        for k,v in over.items(): setattr(g,k,v)
        clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False; g.LAUNCH_ENABLED=False
        n=g.GapFollowN(); n.launch_done=True; out={'a':0.0}; n.publish_velocity=lambda v,a: out.update(a=a)
        for t,cv,cs,v,r in rows:
            clock[0]=t; m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=r.tolist()
            n.speed=v; n.scan_callback(m); allst.append(out['a']); allv.append(v)
        n.destroy_node()
    st=np.array(allst); v=np.array(allv); mv=v>0.8
    lp=np.convolve(st,np.ones(60)/60,'same'); hf=st-lp                       # >2 Hz part = weave
    f=np.fft.rfft(st[mv]-st[mv].mean()); fr=np.fft.rfftfreq(mv.sum(),1/40); band=(fr>0.8)&(fr<2.5)
    weave=np.sqrt((np.abs(f[band])**2).sum()/ (np.abs(f)**2).sum())
    d=np.diff(st); rev=((d[1:]*d[:-1])<0)&(np.abs(d[1:])>0.02)
    hard=np.abs(lp)>0.2
    if base is None: base=st
    lags=range(-12,13); c=[np.corrcoef(base[20:-20], st[20+L:len(st)-20+L])[0,1] for L in lags]; lag=list(lags)[int(np.argmax(c))]*25
    print(f"{label:26s} weave(0.8-2.5Hz share) {weave:.2f}  wiggle std {np.std(hf[mv]):.3f} rad  reversals/s {rev.sum()/(len(st)/40):.2f}  |steer| in corners {np.abs(st[hard]).mean():.3f}  turn-in vs current {-lag:+d} ms")
