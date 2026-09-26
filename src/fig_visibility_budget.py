"""Figure: temporal overlap between reconstructed delivery and modeled water.

(a) The association between fast transition-zone structure and reconstructed
subduction decreases with delivery age. The shaded band is the nominal fitted
association scale, not a limit on slab age or seismic visibility.

(b) The water inventory predicted to be in the transition zone today,
cumulated by delivery age. The two panels share an age axis to show their
temporal overlap without equating delivery age with detectability.
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:
    OUT, FIG = 'out', 'figures'
import figstyle as F

ap = argparse.ArgumentParser()
ap.add_argument('--tau-min', type=float, default=16.0,
                help='lower bound of the converged association-scale fits, Myr')
ap.add_argument('--tau-max', type=float, default=50.0,
                help='upper bound of the converged association-scale fits, Myr')
a = ap.parse_args()
F.apply(9.5)

p = pd.read_csv(os.path.join(OUT, 'persistence_decay.csv'))
z = np.load(os.path.join(OUT, 'water_forward.npz'))
by, ab = z['by_age'], z['abins']
tot = np.array([b.sum() for b in by]); share = tot / tot.sum()
edges = np.concatenate([[ab[0][0]], ab[:, 1]])
cum = np.concatenate([[0.0], np.cumsum(share)])

fig, ax = plt.subplots(1, 2, figsize=(17.5 * F.CM, 7.4 * F.CM), constrained_layout=True,
                       sharex=True)
for x in ax:
    x.axvspan(a.tau_min, a.tau_max, color='#eef2ee', zorder=0)
    x.set_xlim(0, 400)
    x.set_xlabel('Age of reconstructed delivery (Ma)')

# the exponential-plus-floor fit the text quotes, read from the file
# persistence_decay.py wrote rather than refitted here, so that the curve on
# the figure is the fit in the text (as fig_three_clocks.py does)
q = pd.read_csv(os.path.join(OUT, 'persistence_decay_fit.csv')).iloc[0]
tt = np.linspace(float(p.t_mid.min()), 400, 401)   # drawn over the fitted points, not extrapolated to 0 Ma
ax[0].plot(tt, q.Einf + (q.E0 - q.Einf) * np.exp(-tt / q.tau), '-', color=F.BLU, lw=1.6, zorder=2)
ax[0].plot(p.t_mid, p.enrichment, 'o-', color=F.ACC, ms=4.5, lw=1.5, zorder=3)
ax[0].axhline(1.0, color=F.GRY, lw=0.9, ls=(0, (4, 3)))
ax[0].set_ylabel('Enrichment of fast\ntransition-zone structure')
ax[0].text(a.tau_max + 12, ax[0].get_ylim()[1] * 0.92,
           f'Sensitivity range\n{a.tau_min:.0f}\u2013{a.tau_max:.0f} Myr',
           color=F.GRY, va='top')
ax[0].text(395, 1.12, 'No enrichment', ha='right', va='bottom', color=F.GRY)

ax[1].step(edges, 100 * cum, where='post', color=F.BLU, lw=1.8)
ax[1].fill_between(edges, 0, 100 * cum, step='post', color=F.BLU, alpha=0.12)
youngest = 100 * share[0]
young_mid = float(np.mean(ab[0]))
ax[1].plot([young_mid], [youngest], 'o', color=F.ACC, ms=7, mec='white', mew=0.7, zorder=5)
# the callout goes in the empty upper left rather than beside the curve, where
# it crowded the steps; the leader bows right to clear the shaded window band
ax[1].annotate(f'{youngest:.1f}% in youngest resolved bin\n(8\u201325 Ma)',
               xy=(young_mid, youngest), xytext=(58, 92), ha='left', va='top',
               color=F.INK,
               arrowprops=dict(arrowstyle='->', color=F.INK, lw=0.9,
                               shrinkB=6, connectionstyle='arc3,rad=0.28'))
ax[1].set_ylim(0, 104); ax[1].set_ylabel('Modelled transition-zone water,\n'
                                         'cumulative share (%)')
for i, x in enumerate(ax):
    x.text(0.0, 1.02, 'ab'[i], transform=x.transAxes, ha='left', va='bottom',
           fontweight='bold', fontsize=matplotlib.rcParams['font.size'] + 1.0)
F.check(fig)
os.makedirs(FIG, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_visibility_budget.{ext}'), dpi=400,
                bbox_inches='tight', facecolor='white')
print(f'wrote {FIG}/fig_visibility_budget   {youngest:.1f}% of modelled water '
      'in the youngest resolved bin')
