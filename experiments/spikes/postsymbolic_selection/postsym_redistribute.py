"""Reinforcement = selection pressure (local). Release = REDISTRIBUTION (diffusive, conserved).
Contrast against the version where release merely deletes the potential.
Rules: plain decimals only -- no phi, Xi, Fibonacci, pi in the dynamics."""
import numpy as np

def lap(m):
    return (np.roll(m,1,0)+np.roll(m,-1,0)+np.roll(m,1,1)+np.roll(m,-1,1)-4*m)

def run(mode, n=128, steps=800, growth=0.05, couple=0.05, edecay=0.90, damping=0.10,
        th_info=0.40, th_energy=0.05, gain=0.05, mem_decay=0.98, diff=0.20, seed=0):
    rng=np.random.default_rng(seed)
    info=rng.random((n,n)); energy=rng.random((n,n))
    exc=np.zeros((n,n))                      # memory EXCESS above baseline
    events=np.zeros((n,n)); alive=[]; total=[]
    for t in range(steps):
        info = info + growth*(rng.random((n,n))-0.5) + couple*np.roll(info,1,0)
        info -= exc*damping
        np.clip(info,0.0,1.0,out=info)
        energy = energy + couple*np.roll(energy,1,1); np.clip(energy,0.0,1.0,out=energy)
        if mode=="delete":                   # local decay: potential is DESTROYED
            exc *= mem_decay
        elif mode=="redistribute":           # Laplacian diffusion: potential MOVES, conserved
            exc = exc + diff*lap(exc)
        elif mode=="redistribute_sink":      # diffusion + small uniform return to potential
            exc = exc + diff*lap(exc); exc *= mem_decay
        fire=(info>th_info)&(energy>th_energy)
        if fire.any():
            energy[fire]*=edecay
            exc[fire]+=gain                  # REINFORCEMENT = the selection pressure
            events[fire]+=1
        alive.append(fire.mean()); total.append(exc.sum())
    return dict(events=events,exc=exc,alive=np.array(alive),total=np.array(total))

def spatial_corr(m):
    m=m-m.mean()
    if m.std()==0: return float('nan')
    return float(np.mean(m*np.roll(m,1,0))/m.var())

print(f"{'mode':<20}{'firing(late)':>13}{'CoV events':>12}{'spatial corr':>14}"
      f"{'total exc drift':>17}{'conserved?':>12}")
for mode in ("delete","redistribute","redistribute_sink"):
    r=run(mode,seed=5)
    e=r['events']; a=r['alive']; tot=r['total']
    late=a[-200:].mean()
    # conservation check: between firings, does total excess hold?
    drift = (tot[-1]-tot[0])
    print(f"{mode:<20}{late:13.4f}{e.std()/max(e.mean(),1e-9):12.4f}{spatial_corr(e):14.4f}"
          f"{drift:17.1f}{'exact' if mode=='redistribute' else 'no':>12}")

print("\nDoes redistribution create DOMAINS? (spatial autocorrelation of the event map)")
print(f"{'mode':<20}{'lag1':>9}{'lag2':>9}{'lag4':>9}{'lag8':>9}")
for mode in ("delete","redistribute","redistribute_sink"):
    r=run(mode,seed=6); e=r['events']-r['events'].mean()
    if e.std()==0: continue
    row=[float(np.mean(e*np.roll(e,k,0))/e.var()) for k in (1,2,4,8)]
    print(f"{mode:<20}" + "".join(f"{v:9.4f}" for v in row))

print("\nIs the diffusion CONSERVING? (total excess vs cumulative reinforcement)")
r=run("redistribute",seed=7)
expected=r['events'].sum()*0.05
print(f"   total excess at end : {r['exc'].sum():.3f}")
print(f"   events * gain       : {expected:.3f}")
print(f"   relative difference : {abs(r['exc'].sum()-expected)/max(expected,1e-9):.2e}")
