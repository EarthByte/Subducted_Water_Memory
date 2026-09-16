"""Figure: province-scale delivery timing and the recent-delivery proxy.

(a) Provinces classified fast are those whose reconstructed delivery is timed to
put the parcel at transition-zone depth when they erupted. Each province is one
point, because a province is one observation; the fields inside it are not
independent.

(b) Recent modelled water delivery against the seismic anomaly. The positive
relation reveals covariance with the thermal slab carrier; it is not interpreted
as a direct test of water content.
"""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:
    OUT, FIG = 'out', 'figures'
import figstyle as F
from province_pivot import provinces, CAT, RADIUS_DEG
from age_limit_scan import footprint

ap = argparse.ArgumentParser()
ap.add_argument('--band', default='410-520')
# 66 Ma: the 20 Cenozoic provinces are the paper's primary comparison (Section
# 3.4); the figure and the text quote the same sample.
ap.add_argument('--max-age', type=float, default=66.0)
a = ap.parse_args()
F.apply(9.5)

d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
pn = provinces(d)
e = pd.read_csv(os.path.join(OUT, 'province_eruption_context.csv')); e['province'] = pn
c = pd.read_csv(os.path.join(OUT, 'province_classes_continental.csv'))
t = pd.read_csv(os.path.join(OUT, 'province_scores_continental.csv'))
z = np.load(os.path.join(OUT, 'water_forward.npz'))
S = t[t.band == a.band].groupby('province').S.mean()
g = e.groupby('province').agg(n=('age_Ma', 'size'), age=('age_Ma', 'median'),
                              tz=('tz_timed', 'mean')).reset_index()
g['water'] = pd.Series(footprint(z['by_age'][0], z['lon'], z['lat'], d)
                       ).groupby(pd.Series(pn)).mean().reindex(g.province).values
s = g.merge(c[['province', 'cls']], on='province')
s['S'] = s.province.map(S)
s = s[s.age <= a.max_age]
fast = s.cls == 'fast'

fig, ax = plt.subplots(1, 2, figsize=(18.4 * F.CM, 7.6 * F.CM), constrained_layout=True)

# (a) timed delivery, fast against the rest
rng = np.random.default_rng(0)
for k, (m, lab, col) in enumerate(((fast, 'fast', F.ACC), (~fast, 'all others', F.BLU))):
    y = 100 * s.tz[m].values
    x = k + rng.uniform(-0.13, 0.13, len(y))
    ax[0].plot(x, y, 'o', color=col, ms=5.5, mec='white', mew=0.5, alpha=0.9)
    ax[0].plot([k - 0.28, k + 0.28], [np.median(y)] * 2, '-', color=col, lw=2.2)
    ax[0].text(k, 104, f'{lab}\nn = {int(m.sum())}', ha='center', va='bottom',
               color=col, fontweight='bold')
ax[0].set_xlim(-0.55, 1.55); ax[0].set_ylim(-6, 118)
ax[0].set_xticks([]); ax[0].set_ylabel('Fields with delivery timed to the\n'
                                       'transition zone at eruption (%)')
ax[0].set_yticks([0, 25, 50, 75, 100])
ax[0].spines['bottom'].set_visible(False)

# (b) predicted water against the anomaly
j = s[['water', 'S', 'cls']].dropna(); j = j[j.water > 0]
r, p = spearmanr(j.water, j.S)
for m, col in ((j.cls == 'fast', F.ACC), (j.cls != 'fast', F.BLU)):
    ax[1].plot(np.log10(j.water[m]), j.S[m], 'o', color=col, ms=5.5, mec='white', mew=0.5)

ax[1].axhline(0, color=F.GRY, lw=0.8, ls=(0, (4, 3)))
ax[1].set_xlabel('Modelled water delivery, 8–25 Ma\n(log10 of % of maximum)')
ax[1].set_ylabel(f'Standardised anomaly, {a.band} km')
ax[1].text(0.03, 0.94, f'Spearman ρ = {r:+.2f}\np = {p:.3f}   n = {len(j)}',
           transform=ax[1].transAxes, va='top', color=F.INK)
ax[1].text(0.97, 0.06, 'recent delivery covaries\nwith fast structure',
           transform=ax[1].transAxes, ha='right', va='bottom', color=F.GRY)
for i, x in enumerate(ax):
    x.text(0.0, 1.10 if i == 0 else 1.02, 'ab'[i], transform=x.transAxes,
           ha='left', va='bottom', fontweight='bold',
           fontsize=matplotlib.rcParams['font.size'] + 1.0)
F.check(fig)
os.makedirs(FIG, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_geological_test.{ext}'), dpi=400,
                bbox_inches='tight', facecolor='white')
print(f'wrote {FIG}/fig_geological_test   rho {r:+.3f} p {p:.4f} n {len(j)}; '
      f'fast median {100 * s.tz[fast].median():.0f}%, others {100 * s.tz[~fast].median():.0f}%')
