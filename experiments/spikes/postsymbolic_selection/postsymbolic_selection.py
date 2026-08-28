"""EXPLORATORY -- post-symbolic selection pressure. Signature hunt, no scoring.

Spirit (NOT a port) of era1 brain.py: fields with history, no symbols, no labels. Collapse
reinforces a local memory; that memory suppresses future collapse at the same site; so
activity must move and regions differentiate. That is selection pressure implemented
dynamically rather than bookkept.

*** WHAT THE RULES CONTAIN (stated up front, per STANDARDS 2.8) ***
Plain decimals ONLY: growth 0.05, couple 0.05, decay 0.90, damping 0.02, thresholds
0.40/0.05, reinforcement 1.05, cap 2.0. NO phi, NO Xi, NO Fibonacci, NO pi anywhere in the
dynamics. Anything resembling a framework constant in the OUTPUT was not put in.

*** THE CONTROL ***
Sweep reinforcement r and cap c. A statistic that MOVES with the knobs is a knob readout.
A statistic that stays put while r and c vary is a candidate invariant. Only the second is
interesting, and the sweep is what tells them apart.
"""
import numpy as np

def run(n=128, steps=600, growth=0.05, couple=0.05, decay=0.90, damping=0.02,
        th_info=0.40, th_energy=0.05, reinforce=1.05, cap=2.0, seed=0):
    rng = np.random.default_rng(seed)
    info = rng.random((n, n)); energy = rng.random((n, n))
    mem = np.ones((n, n)); events = np.zeros((n, n))
    last = np.full((n, n), -1.0); gaps = []
    for t in range(steps):
        info = info + growth*(rng.random((n,n))-0.5) + couple*np.roll(info,1,0)
        info -= mem*damping
        np.clip(info, 0.0, 1.0, out=info)
        energy = energy + couple*np.roll(energy,1,1)
        np.clip(energy, 0.0, 1.0, out=energy)
        fire = (info > th_info) & (energy > th_energy)
        if fire.any():
            energy[fire] *= decay
            mem[fire] = np.minimum(mem[fire]*reinforce, cap)
            prev = last[fire]
            seen = prev >= 0
            if seen.any(): gaps.extend((t - prev[seen]).tolist())
            last[fire] = t
            events[fire] += 1
    return dict(events=events, mem=mem, gaps=np.array(gaps), info=info)

print("A. does the field DIFFERENTIATE, or fire uniformly?  (baseline knobs)")
r = run()
ev = r['events']
print(f"   events/cell: mean {ev.mean():.1f}  sd {ev.std():.1f}  CoV {ev.std()/ev.mean():.4f}")
print(f"   memory at end: min {r['mem'].min():.3f}  max {r['mem'].max():.3f}  "
      f"frac at cap {np.mean(r['mem']>=1.999):.3f}")
print(f"   inter-collapse gaps: n={len(r['gaps'])}  mean {r['gaps'].mean():.3f}  "
      f"median {np.median(r['gaps']):.1f}")

print("\nB. KNOB SWEEP -- which statistics move with (reinforce, cap), which stay put?")
print(f"{'reinforce':>10}{'cap':>6}{'steps_to_cap':>14}{'mean gap':>10}{'CoV events':>12}"
      f"{'gap/steps_to_cap':>18}")
inv = []
for reinf in (1.02, 1.05, 1.10, 1.20):
    for cap in (1.5, 2.0, 3.0):
        rr = run(reinforce=reinf, cap=cap, steps=600)
        stc = np.log(cap)/np.log(reinf)          # arithmetic: firings needed to saturate
        g = rr['gaps'].mean() if len(rr['gaps']) else np.nan
        e = rr['events']
        ratio = g/stc
        inv.append(ratio)
        print(f"{reinf:10.2f}{cap:6.1f}{stc:14.2f}{g:10.3f}{e.std()/e.mean():12.4f}{ratio:18.4f}")
inv = np.array(inv)
print(f"\n   gap/steps_to_cap across all 12 settings: mean {inv.mean():.4f}  sd {inv.sd() if hasattr(inv,'sd') else inv.std():.4f}"
      f"  CoV {inv.std()/inv.mean():.4f}")
print("   (small CoV => the ratio is knob-independent; large => it is a knob readout)")
