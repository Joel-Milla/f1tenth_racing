import pickle, numpy as np, sys
rows=pickle.load(open(sys.argv[1],'rb')); t=np.array([r[0] for r in rows])-rows[0][0]
cs=np.array([r[2] for r in rows]); v=np.array([r[3] for r in rows]); wz=np.array([r[5] for r in rows]); ay=np.array([r[6] for r in rows])
exp=v*np.tan(cs)/0.33                     # yaw rate the steering should give with no sliding
turning=(np.abs(cs)>0.15)&(v>1.0)
ratio=np.abs(wz[turning])/np.maximum(np.abs(exp[turning]),1e-3)
print(f"IMU yaw rate vs expected in turns: median {np.median(ratio):.2f} (1.0 = full grip, <1 = sliding wide / understeer, >1 = rear stepping out / drift)")
print(f"  understeer (<0.7): {(ratio<0.7).mean()*100:.0f}% of turning time   oversteer (>1.3): {(ratio>1.3).mean()*100:.0f}%")
print(f"  peak lateral accel {np.percentile(np.abs(ay[v>0.5]),99):.1f} m/s2   sign check corr(wz,exp) {np.corrcoef(wz[turning],exp[turning])[0,1]:.2f}")
