import pickle, numpy as np, re, sys
log=sys.argv[1]; t_start=float(re.search(r'\[(\d+\.\d+)\]', open(log).readline()).group(1))
rows=[r for r in pickle.load(open('rec4_snap.pkl','rb')) if r[0]>=t_start]
a=-2.356194+0.0043633*np.arange(1081)
t=np.array([r[0] for r in rows])-t_start; v=np.array([r[3] for r in rows]); cs=np.array([r[2] for r in rows]); r1=np.array([r[7] for r in rows])
mv=v>0.8
print(f"since node start: {t[-1]:.0f} s recorded | R1 held {r1.mean()*100:.0f}% | moving {mv.mean()*100:.0f}%")
if mv.sum()<80: print('not enough driving yet'); sys.exit()
F=[];clr=[]
for r in rows:
    rr=np.clip(np.nan_to_num(r[4],nan=30,posinf=30),0,30); x=rr*np.cos(a); y=rr*np.sin(a); s=(x>0)&(np.abs(y)<0.25); F.append(x[s].min() if s.any() else 10)
    xx=x+0.27; dx=np.maximum(np.maximum(-0.15-xx,xx-0.42),0); dy=np.maximum(np.abs(y)-0.15,0); clr.append(np.hypot(dx,dy).min())
F=np.array(F); clr=np.array(clr)
Fm=F[mv]-F[mv].mean(); ac=np.correlate(Fm,Fm,'full')[len(Fm)-1:]; lag=np.argmax(ac[160:min(600,len(ac))])+160
d=np.diff(cs); rev=((d[1:]*d[:-1])<0)&(np.abs(d[1:])>0.02); m2=mv[2:]
vv=np.convolve(v,np.ones(6)/6,'same'); acc=np.gradient(vv,t)
print(f"lap period ~ {lag/40:.2f} s")
print(f"real speed while moving: avg {v[mv].mean():.2f}  p90 {np.percentile(v[mv],90):.2f}  max {v.max():.2f} m/s")
print(f"wall clearance while moving: p5 {np.percentile(clr[mv],5)*100:.1f} cm  scans <5 cm: {(clr[mv]<0.05).sum()}  touching (<1 cm): {(clr[mv]<0.01).sum()}")
print(f"steering reversals/s {rev[m2].sum()/(m2.sum()/40):.2f}   hardest decel {-np.percentile(acc[mv],0.5):.1f} m/s2 (>3.5 = likely impact)")
i=np.argmax(mv); j=np.searchsorted(t,t[i]+1.5); print(f"launch: speed 1.5 s after first motion {v[min(j,len(v)-1)]:.2f} m/s")
