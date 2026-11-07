# Closed-loop sim: car drives through a wavy 1.1 m corridor (bends + small wall bumps). Ray-cast lidar on polyline walls.
import importlib.util, rclpy, numpy as np, types, math, sys
from sensor_msgs.msg import LaserScan
rclpy.init(); A=-2.356194+0.0043633*np.arange(1081)
X=np.linspace(-2,40,1400)
def walls(amp=0.30, lam=3.2, bump=0.06, blam=0.7, half=0.55):
    yc=amp*np.sin(2*np.pi*X/lam)
    left=yc+half+bump*np.sin(2*np.pi*X/blam); right=yc-half+bump*np.sin(2*np.pi*X/blam+1.3)
    segs=[]
    for wy in (left,right): segs.append(np.stack([X[:-1],wy[:-1],X[1:],wy[1:]],1))
    return np.vstack(segs), yc
def raycast(px,py,yaw,segs):
    near=segs[(np.abs(segs[:,0]-px)<6)]
    ang=yaw+A; c,s=np.cos(ang)[:,None],np.sin(ang)[:,None]
    x0,y0,x1,y1=[near[:,i][None,:] for i in range(4)]; dx,dy=x1-x0,y1-y0
    with np.errstate(divide='ignore',invalid='ignore'):
        den=c*dy-s*dx; t=((x0-px)*dy-(y0-py)*dx)/den; u=((x0-px)*s-(y0-py)*c)/den
        t=np.where((t>0)&(u>=0)&(u<=1),t,np.inf)
    r=np.minimum(t.min(axis=1),10.0)
    m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=r.tolist(); return m
GEOM=dict(amp=0.30, lam=3.2, bump=0.06, blam=0.7)
def run(label,path,over={},T=10.0):
    segs,yc=walls(**GEOM)
    s=importlib.util.spec_from_file_location('w'+str(abs(hash(label))),path); g=importlib.util.module_from_spec(s); s.loader.exec_module(g)
    for k,v in over.items(): setattr(g,k,v)
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False; g.LAUNCH_ENABLED=False
    n=g.GapFollowN(); n.launch_done=True; out={'a':0.0}; n.publish_velocity=lambda v,a: out.update(a=a)
    st_act=0.0
    x,y,yaw,v=0.0,0.0,0.0,2.0; dt=0.025; t=0; sts=[]; contact=0; vs=[]; n.last_speed_cmd=v; ys=[]
    while t<T and x<36:
        clock[0]=t; lx,ly=x+0.27*math.cos(yaw),y+0.27*math.sin(yaw); n.speed=v
        n.scan_callback(raycast(lx,ly,yaw,segs)); target=n.last_speed_cmd
        st_act=st_act+(out['a']-st_act)*dt/0.10   # servo lag 0.1 s
        st=st_act
        v+=(target-v)*dt/0.3
        # steering servo lag ~0.08 s
        x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt; yaw+=v*math.tan(st)/0.33*dt; t+=dt
        cyc=np.interp(x+0.14,X,yc); off=y-cyc
        if abs(off)>0.55-0.17: contact+=1
        sts.append(st); vs.append(v); ys.append(off)
    st=np.array(sts); d=np.diff(st); rev=((d[1:]*d[:-1])<0)&(np.abs(d[1:])>0.02)
    path_len=np.sum(np.abs(np.diff(st)))
    print(f"  {label:28s} dist {x:5.1f} m in {t:4.1f} s (avg {x/t:4.2f} m/s)  steer total travel {path_len:5.1f} rad  reversals/s {rev.sum()/t:4.2f}  steer std {st.std():.3f}  wall-contact scans {contact}")
    n.destroy_node()
if __name__=='__main__':
    import os
    if os.environ.get('GEOM'): GEOM.update(eval(os.environ['GEOM']))
    print('geometry', GEOM)
    for arg in sys.argv[1:]:
        label,path,*ov=arg.split('|'); over={}
        for o in ov:
            k,vv=o.split('='); over[k]=eval(vv)
        run(label,path,over)
