import pickle, numpy as np, sys
rows=pickle.load(open(sys.argv[1],'rb'))
a=-2.356194+0.0043633*np.arange(1081)
t=np.array([r[0] for r in rows])-rows[0][0]; cv=np.array([r[1] for r in rows]); cs=np.array([r[2] for r in rows]); v=np.array([r[3] for r in rows])
mins=[]
for r in rows:
    rr=np.clip(np.nan_to_num(r[4],nan=30,posinf=30),0,30); x=rr*np.cos(a)+0.27; y=rr*np.sin(a)
    # distance from the car body (rectangle x -0.15..0.42 rear-axle frame, |y|<0.15)
    dx=np.maximum(np.maximum(-0.15-x,x-0.42),0); dy=np.maximum(np.abs(y)-0.15,0); mins.append(np.hypot(dx,dy).min())
mins=np.array(mins); mv=v>0.3
dah=None
x=v-v.mean()
print(f"moving {mv.mean():.2f}  odom mean {v[mv].mean():.2f} p90 {np.percentile(v[mv],90):.2f} max {v.max():.2f} | cmd mean {cv[mv].mean():.2f}")
print(f"body clearance: min {mins[mv].min():.3f} m  p1 {np.percentile(mins[mv],1):.3f}  p5 {np.percentile(mins[mv],5):.3f}   scans <5cm: {(mins[mv]<0.05).sum()}  stops(cmd 0 while moving): {((cv==0)&mv).sum()}")
vv=np.convolve(v,np.ones(8)/8,'same'); acc=np.gradient(vv,t); print(f"max real decel {-np.percentile(acc[mv],1):.2f} m/s2")
