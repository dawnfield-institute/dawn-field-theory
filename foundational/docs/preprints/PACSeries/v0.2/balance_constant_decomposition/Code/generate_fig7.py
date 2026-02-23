"""Generate fig7: Möbius field dynamics Ξ_L2 convergence for Paper 2."""
import json, os, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12,
    'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)
GAMMA = 0.5772156649015329
XI_ANALYTIC = GAMMA + LN_PHI
XI_FIB = 1 + np.pi / 55

# Load data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'results')
data = None
for f in sorted(os.listdir(data_dir)):
    if f.startswith('exp_15') and f.endswith('.json'):
        with open(os.path.join(data_dir, f)) as fh:
            data = json.load(fh)
        break

if data is None:
    print("ERROR: No exp_15 data found. Run exp_15_mobius_field_dynamics.py first.")
    sys.exit(1)

trace = data['xi_trace']
ts = [r['t'] for r in trace]
xis = [r['xi_L2'] for r in trace]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})

# Left: Full convergence trajectory
ax1.plot(ts, xis, color='#2196F3', linewidth=1.0, alpha=0.8,
         label=r'$\Xi_{L2}(t)$')
ax1.axhline(y=XI_ANALYTIC, color='red', linestyle='--', linewidth=1.5,
            label=f'$\\gamma + \\ln\\varphi = {XI_ANALYTIC:.5f}$')
ax1.axhline(y=XI_FIB, color='orange', linestyle=':', linewidth=1.5,
            label=f'$1 + \\pi/55 = {XI_FIB:.5f}$')
ax1.set_xlabel('Time step')
ax1.set_ylabel(r'$\Xi_{L2}$ (anti/sym energy ratio)')
ax1.set_title(r'(a) Convergence of $\Xi_{L2}$ from M\"obius Field Dynamics')
ax1.set_ylim(0, max(xis) * 1.1)
ax1.legend(loc='lower right', framealpha=0.9)

# Right: Zoomed steady state (last 5000 steps)
late_trace = [(t, x) for t, x in zip(ts, xis) if t >= 5000]
ts_late = [r[0] for r in late_trace]
xis_late = [r[1] for r in late_trace]
ax2.plot(ts_late, xis_late, color='#2196F3', linewidth=1.0, alpha=0.8)
ax2.axhline(y=XI_ANALYTIC, color='red', linestyle='--', linewidth=1.5)
ax2.axhline(y=XI_FIB, color='orange', linestyle=':', linewidth=1.5)
ax2.set_xlabel('Time step')
ax2.set_title('(b) Steady State (t > 5000)')
xi_mean_late = np.mean(xis_late)
xi_std_late = np.std(xis_late)
ax2.set_ylim(xi_mean_late - 10*xi_std_late, xi_mean_late + 10*xi_std_late)

# Annotation
textstr = (f'Final: {xis[-1]:.6f}\n'
           f'$\\gamma+\\ln\\varphi$: {XI_ANALYTIC:.6f}\n'
           f'Error: {abs(xis[-1]-XI_ANALYTIC)/XI_ANALYTIC*100:.4f}%')
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'Figures')
os.makedirs(fig_dir, exist_ok=True)
path = os.path.join(fig_dir, 'fig7_mobius_convergence.png')
plt.savefig(path, dpi=300)
print(f'Saved: {path}')
plt.close()
