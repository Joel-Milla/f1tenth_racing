import pickle, numpy as np, sys
rows=pickle.load(open(sys.argv[1],'rb')); a=-2.356194+0.0043633*np.arange(1081)
F=[]
for r in rows:
    rr=np.clip(np.nan_to_num(r[4],nan=30,posinf=30),0,30); x=rr*np.cos(a); y=rr*np.sin(a); s=(x>0)&(np.abs(y)<0.25); F.append(x[s].min() if s.any() else 10)
F=np.array(F); x=F-F.mean(); ac=np.correlate(x,x,'full')[len(x)-1:]; lag=np.argmax(ac[160:600])+160; print('lap period ~',round(lag/40,2),'s')
