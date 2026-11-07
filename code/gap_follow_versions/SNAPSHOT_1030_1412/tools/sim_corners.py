# Closed-loop sim on a corridor with real corners: straight, 90 deg left, straight, 90 deg right, S-bend ... 1.1 m wide
import numpy as np, math, sys, os
sys.argv_saved=sys.argv; sys.argv=['x']
exec(open('/home/f1tenth/gap_follow_backups/tools/sim_wavy.py').read().split("if __name__=='__main__':")[0])
def centerline():
    pts=[(0,0)]; h=0.0; x=y=0.0
    def straight(L):
        nonlocal x,y
        for _ in range(int(L/0.05)): x+=0.05*math.cos(h); y+=0.05*math.sin(h); pts.append((x,y))
    def turn(R,ang):
        nonlocal x,y,h
        n=int(abs(ang)*R/0.05)
        for _ in range(n): h+=math.copysign(0.05/R,ang); x+=0.05*math.cos(h); y+=0.05*math.sin(h); pts.append((x,y))
    straight(4); turn(1.3,math.pi/2); straight(2.5); turn(1.3,-math.pi/2); straight(1.5); turn(1.6,-math.pi/3); turn(1.6,math.pi/3); straight(3); turn(1.3,math.pi/2); straight(4)
    return np.array(pts)
C=centerline(); d=np.diff(C,axis=0); nrm=np.stack([-d[:,1],d[:,0]],1); nrm/=np.linalg.norm(nrm,axis=1)[:,None]; nrm=np.vstack([nrm,nrm[-1:]])
HALF=0.55
L=C+HALF*nrm; R=C-HALF*nrm
SEGS=np.vstack([np.hstack([L[:-1],L[1:]]),np.hstack([R[:-1],R[1:]])])
def raycast2(px,py,yaw):
    near=SEGS[(np.hypot(SEGS[:,0]-px,SEGS[:,1]-py)<6)]
    ang=yaw+A; c,s=np.cos(ang)[:,None],np.sin(ang)[:,None]
    x0,y0,x1,y1=[near[:,i][None,:] for i in range(4)]; dx,dy=x1-x0,y1-y0
    with np.errstate(divide='ignore',invalid='ignore'):
        den=c*dy-s*dx; t=((x0-px)*dy-(y0-py)*dx)/den; u=((x0-px)*s-(y0-py)*c)/den
        t=np.where((t>0)&(u>=0)&(u<=1),t,np.inf)
    r=np.minimum(t.min(axis=1),10.0)
    m=LaserScan(); m.angle_min=-2.356194; m.angle_max=2.356194; m.angle_increment=0.0043633; m.range_min=0.02; m.range_max=30.0; m.ranges=r.tolist(); return m
def run2(label,path,over={}):
    s_=importlib.util.spec_from_file_location('c'+str(abs(hash(label))),path); g=importlib.util.module_from_spec(s_); s_.loader.exec_module(g)
    for k,v in over.items(): setattr(g,k,v)
    clock=[0.0]; g.time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]); g.LOG_STEERING=False; g.DEADMAN_ENABLED=False; g.LAUNCH_ENABLED=False
    n=g.GapFollowN(); n.launch_done=True; out={'a':0.0}; n.publish_velocity=lambda v,a: out.update(a=a)
    x,y,yaw,v=0.0,0.0,0.0,2.0; dt=0.025; t=0; st_act=0.0; cmds=[]; offs=[]; contact=0; n.last_speed_cmd=v; done=False
    while t<20:
        clock[0]=t; lx,ly=x+0.27*math.cos(yaw),y+0.27*math.sin(yaw); n.speed=v
        n.scan_callback(raycast2(lx,ly,yaw)); target=n.last_speed_cmd; cmds.append(out['a'])
        st_act+=(out['a']-st_act)*dt/0.10
        v+=(target-v)*dt/0.3; x+=v*math.cos(yaw)*dt; y+=v*math.sin(yaw)*dt; yaw+=v*math.tan(st_act)/0.33*dt; t+=dt
        cx,cy=x+0.14*math.cos(yaw),y+0.14*math.sin(yaw); i=np.argmin(np.hypot(C[:,0]-cx,C[:,1]-cy)); off=np.dot([cx-C[i,0],cy-C[i,1]],nrm[i]); offs.append(off)
        if abs(off)>HALF-0.17: contact+=1
        if i>=len(C)-3: done=True; break
    cm=np.array(cmds); dd=np.diff(cm); rev=((dd[1:]*dd[:-1])<0)&(np.abs(dd[1:])>0.02); offs=np.array(offs)
    print(f"  {label:30s} {'FINISHED' if done else 'did not finish'} in {t:5.2f} s  lateral swing std {offs.std():.3f} m  max off-centre {np.abs(offs).max():.2f} m  wall-contact scans {contact:3d}  steer travel {np.sum(np.abs(dd)):5.1f}  reversals/s {rev.sum()/t:.2f}")
    n.destroy_node()
for arg in sys.argv_saved[1:]:
    label,path,*ov=arg.split('|'); over={}
    for o in ov:
        k,vv=o.split('='); over[k]=eval(vv)
    run2(label,path,over)
