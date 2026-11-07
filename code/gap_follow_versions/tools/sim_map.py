# Closed-loop sim on the real track map (track2_clean.pgm). Sphere-traced lidar on the distance transform.
import importlib.util, rclpy, numpy as np, types, math, sys, cv2
from sensor_msgs.msg import LaserScan
rclpy.init(); A=-2.356194+0.0043633*np.arange(1081)
import os
M=cv2.imread(os.environ.get('MAP','/home/f1tenth/f1tenth_ws/track2_clean.pgm'),-1); RES=0.05; OX,OY=-4.99,-2.04; H,W=M.shape
free=(M==254).astype(np.uint8)
DT=cv2.distanceTransform(free,cv2.DIST_L2,5)*RES          # metres to nearest non-free cell
def to_px(x,y): return ((x-OX)/RES), (H-1-(y-OY)/RES)
def dist_at(x,y):
    c,r=to_px(x,y); c=np.clip(c.astype(int) if isinstance(c,np.ndarray) else int(c),0,W-1); r=np.clip(r.astype(int) if isinstance(r,np.ndarray) else int(r),0,H-1); return DT[r,c]
def scan(px,py,yaw):
    ang=yaw+A; c,s=np.cos(ang),np.sin(ang); t=np.zeros(1081); alive=np.ones(1081,bool)
    for _ in range(60):
        d=dist_at(px+t*c,py+t*s); step=np.maximum(d,0.01)
        t=np.where(alive,t+step,t); alive&=(d>0.012)&(t<10)
        if not alive.any(): break
    t=np.minimum(t,10.0)
    m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=t.tolist(); return m
fr,fc=np.nonzero(free); CX=OX+fc.mean()*RES; CY=OY+(H-1-fr.mean())*RES      # loop centre for lap counting
def start_pose():
    if os.environ.get('START'):   # "row,col,heading_deg"
        r,c,h=[float(v) for v in os.environ['START'].split(',')]
        return OX+c*RES, OY+(H-1-r)*RES, math.radians(h)
    if 'track2' in os.environ.get('MAP','track2'):
        r=100; row=free[r]; cols=np.nonzero(row)[0]; right=cols[cols>W*0.6]; c=(right.min()+right.max())/2
        x=OX+c*RES; y=OY+(H-1-r)*RES; return x,y,math.radians(68)
    # generic: widest point of the track, heading along the band (anticlockwise around the loop centre)
    rr,cc=np.unravel_index(np.argmax(DT),DT.shape); x=OX+cc*RES; y=OY+(H-1-rr)*RES
    ys,xs=np.nonzero(free); px=OX+xs*RES; py=OY+(H-1-ys)*RES; near=np.hypot(px-x,py-y)<0.8
    cov=np.cov(np.stack([px[near]-x,py[near]-y])); w,v=np.linalg.eigh(cov); d=v[:,np.argmax(w)]
    radial=np.array([x-CX,y-CY]); tang=np.array([-radial[1],radial[0]])          # anticlockwise tangent
    if np.dot(d,tang)<0: d=-d
    return x,y,math.atan2(d[1],d[0])
def run(label,path,over={},laps=3,T=40.0,quiet=False):
    s_=importlib.util.spec_from_file_location('mp'+str(abs(hash(label))),path); g=importlib.util.module_from_spec(s_); s_.loader.exec_module(g)
    for k,v in over.items(): setattr(g,k,v)
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False; g.LAUNCH_ENABLED=False
    n=g.GapFollowN(); n.launch_done=True; out={'a':0.0}; n.publish_velocity=lambda v,a: out.update(a=a)
    x,y,yaw=start_pose(); v=1.0; dt=0.025; t=0; st_act=0.0; n.last_speed_cmd=v
    ang0=math.atan2(y-CY,x-CX); prog=0.0; last=ang0; lap_t=[]; lap_start=0.0
    clear=[]; yawabs=0.0; cmds=[]; vs=[]; crash=False
    while t<T and len(lap_t)<laps:
        clock[0]=t; lx,ly=x+0.27*math.cos(yaw),y+0.27*math.sin(yaw); n.speed=v
        n.scan_callback(scan(lx,ly,yaw)); target=n.last_speed_cmd; cmds.append(out['a'])
        st_act+=(out['a']-st_act)*dt/0.10
        v+=(target-v)*dt/0.3; x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt; dyaw=v*math.tan(st_act)/0.33*dt; yaw+=dyaw; t+=dt
        cx,cy=x+0.14*math.cos(yaw),y+0.14*math.sin(yaw); cl=float(dist_at(cx,cy)); clear.append(cl); vs.append(v)
        if lap_t: yawabs+=abs(dyaw)
        if cl<0.05: crash=True; break
        a=math.atan2(cy-CY,cx-CX); da=(a-last+math.pi)%(2*math.pi)-math.pi; prog+=da; last=a
        if abs(prog)>=2*math.pi*(len(lap_t)+1): lap_t.append(t-lap_start); lap_start=t
    cm=np.array(cmds); dd=np.diff(cm); rev=((dd[1:]*dd[:-1])<0)&(np.abs(dd[1:])>0.02); clear=np.array(clear)
    laps_done=lap_t[1:] if len(lap_t)>1 else lap_t     # first lap includes the standing start
    res=dict(label=label,laps=[round(l,2) for l in lap_t],crash=crash,min_clear=clear.min(),p5=np.percentile(clear,5),
             heading=yawabs/max(len(laps_done),1),rev=rev.sum()/t,vavg=np.mean(vs))
    if not quiet:
        print(f"  {label:30s} laps {res['laps']}  {'CRASH' if crash else ''} avg v {res['vavg']:.2f}  heading change/lap {math.degrees(res['heading']):5.0f} deg  steer reversals/s {res['rev']:.2f}  clearance min {res['min_clear']:.2f} p5 {res['p5']:.2f} m")
    n.destroy_node(); return res
if __name__=='__main__':
    for arg in sys.argv[1:]:
        label,path,*ov=arg.split('|'); over={}
        for o in ov:
            k,vv=o.split('='); over[k]=eval(vv)
        run(label,path,over)
