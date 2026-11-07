import importlib.util, rclpy, numpy as np, types, math, sys
from sensor_msgs.msg import LaserScan
rclpy.init(); A=-2.356194+0.0043633*np.arange(1081)
def raycast(px,py,yaw,W,ob):
    ang=yaw+A; c,s=np.cos(ang),np.sin(ang); t=np.full(1081,10.0)
    with np.errstate(divide='ignore',invalid='ignore'):
        for wy in (W/2,-W/2):
            tt=(wy-py)/s; t=np.where((tt>0)&(tt<t),tt,t)
        ox,oy,L,Wd=ob  # box centre, length, width
        for (x0,y0,x1,y1) in [(ox-L/2,oy-Wd/2,ox+L/2,oy-Wd/2),(ox-L/2,oy+Wd/2,ox+L/2,oy+Wd/2),(ox-L/2,oy-Wd/2,ox-L/2,oy+Wd/2),(ox+L/2,oy-Wd/2,ox+L/2,oy+Wd/2)]:
            dx,dy=x1-x0,y1-y0; den=c*dy-s*dx
            tt=((x0-px)*dy-(y0-py)*dx)/den; uu=((x0-px)*s-(y0-py)*c)/den
            ok=(tt>0)&(uu>=0)&(uu<=1)&(tt<t); t=np.where(ok,tt,t)
    m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=t.tolist(); return m
def run(path,W,oy,vo,label,T=8.0):
    s=importlib.util.spec_from_file_location('g'+label,path); g=importlib.util.module_from_spec(s); s.loader.exec_module(g)
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False
    n=g.GapFollowN(); n.launch_done=True; out={'a':0.0}; n.publish_velocity=lambda v,st: out.update(v=v,a=st)
    x,y,yaw,v=0.0,0.0,0.0,2.5; ox=1.8; dt=0.025; t=0; coll=False; wall=False; passed=None; trail_t=pass_t=0; mind=9
    n.last_speed_cmd=v
    while t<T:
        clock[0]=t; lx,ly=x+0.27*math.cos(yaw),y+0.27*math.sin(yaw); n.speed=v
        n.scan_callback(raycast(lx,ly,yaw,W,(ox,oy,0.5,0.3)))
        target=n.last_speed_cmd; st=out['a']
        v+=(target-v)*dt/0.3; x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt; yaw+=v*math.tan(st)/0.33*dt; ox+=vo*dt
        cx,cy=x+0.14*math.cos(yaw),y+0.14*math.sin(yaw)   # car body centre
        if abs(cx-ox)<0.53 and abs(cy-oy)<0.30: coll=True
        if abs(cy)>W/2-0.15: wall=True
        mind=min(mind,math.hypot(cx-ox,cy-oy)); trail_t+=n.trailing*dt; pass_t+=bool(getattr(n,'pass_side',''))*dt
        if passed is None and cx-0.29>ox+0.25: passed=t
        t+=dt
    print(f"  {label:10s} W={W:.1f} opp y={oy:+.2f}: {'PASSED at %.1fs'%passed if passed else 'not passed'}  collision={coll} wall={wall}  trail {trail_t:.1f}s pass-mode {pass_t:.1f}s  min centre dist {mind:.2f}")
    n.destroy_node()
import sys as _s
files=[a.split('=') for a in _s.argv[1:]] or [('trail-only','/home/f1tenth/gap_follow_backups/FAST6_trail.py'),('PASS','/home/f1tenth/gap_follow_backups/FAST6_pass.py')]
for W in [1.1,1.4]:
    for oy in [0.0,0.25,-0.25]:
        for label,path in files:
            run(path,W,oy,1.5,label)
