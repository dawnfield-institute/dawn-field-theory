"""Does the RETURN-CHANNEL RATE set the domain scale? If the sink is the boundary with the
parent node, its rate is physical and the structure scale should track it.
Plain decimals only in the rules."""
import numpy as np
exec(open("/private/tmp/claude-501/-Users-petergroom-repos-core-workspace/4df44f1a-10d2-4b18-8a87-d921a99b5dd3/scratchpad/postsym_redistribute.py").read().split('def spatial_corr')[0])

def corr_len(e):
    e=e-e.mean()
    if e.std()==0: return float('nan')
    c=[float(np.mean(e*np.roll(e,k,0))/e.var()) for k in range(1,40)]
    for k,v in enumerate(c,1):
        if v < np.exp(-1):
            prev=c[k-2] if k>=2 else 1.0
            return k-1+(prev-np.exp(-1))/max(prev-v,1e-12)
    return float('nan')

print("sink rate vs domain scale   (diff=0.20 fixed; non-closure = 1 - mem_decay)")
print(f"{'mem_decay':>10}{'non-closure':>13}{'firing(late)':>13}{'CoV ev':>9}{'corr length':>13}")
rows=[]
for md in (0.999,0.995,0.99,0.98,0.95,0.90,0.80):
    r=run("redistribute_sink", mem_decay=md, seed=11)
    e=r['events']; late=r['alive'][-200:].mean()
    L=corr_len(e)
    rows.append((1-md,L,late))
    print(f"{md:10.3f}{1-md:13.4f}{late:13.4f}{e.std()/max(e.mean(),1e-9):9.4f}{L:13.3f}")
x=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows])
ok=np.isfinite(y)&(y>0)
if ok.sum()>=3:
    s=np.polyfit(np.log(x[ok]),np.log(y[ok]),1)
    print(f"\n  corr_length ~ (non-closure)^{s[0]:+.3f}   over {ok.sum()} points")
    print(f"  correlation of logs = {np.corrcoef(np.log(x[ok]),np.log(y[ok]))[0,1]:+.4f}")
