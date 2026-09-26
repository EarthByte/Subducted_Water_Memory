"""Figure: the depth localisation against the one free parameter in the attribution.

The down-dip displacement decides how far from its trench subducted material is
attributed, and the value used throughout was chosen because the observed fast
transition zone preferred it. If that choice were carrying the result, the
significance would peak there. It does not: every volume improves monotonically
with the displacement, and the adopted value is neither the shortest geometry
allows nor the one that maximises significance.
"""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:
    OUT, FIG = 'out', 'figures'
import figstyle as F

ap = argparse.ArgumentParser()
ap.add_argument('--adopted', type=float, default=300.0)
a = ap.parse_args()
F.apply(9.5)

d = pd.read_csv(os.path.join(OUT, 'depth_offset_profile.csv'))
p = d.pivot_table(index='model', columns='offset', values='p_adj')
STYLE = [('REVEAL', F.ACC, 'o-'), ('RevealLO', F.BLU, 's-'),
         ('GLAD-M35', '#3F7A4B', '^-'), ('SPiRaL', '#7B5EA7', 'D-'),
         ('SEMUCB-WM1', '#C98A20', 'v-')]
offs = np.array(sorted(p.columns), float)

fig, ax = plt.subplots(1, 2, figsize=(17.6 * F.CM, 7.4 * F.CM),
                       constrained_layout=True)

# (a) the adjusted probability against the displacement
ax[0].axvspan(a.adopted - 6, a.adopted + 6, color='#eef2ee', zorder=0)
for lab, col, st in STYLE:
    if lab not in p.index:
        continue
    y = p.loc[lab, offs].values.astype(float)
    ax[0].plot(offs, y, st, color=col, ms=4.5, lw=1.5, label=lab, mec='white',
               mew=0.4)
ax[0].axhline(0.05, color=F.GRY, lw=0.9, ls=(0, (4, 3)))
ax[0].text(offs[-1], 0.054, '0.05', ha='right', va='bottom', color=F.GRY)
ax[0].annotate('Adopted', xy=(a.adopted, 0.985), xytext=(a.adopted, 0.985),
               xycoords=('data', 'axes fraction'), ha='center', va='top',
               color=F.GRY)
ax[0].set_yscale('log'); F.decimal_ticks(ax[0], 'y')
ax[0].set_xlabel('Down-dip displacement (km)')
ax[0].set_ylabel('Adjusted probability at 410–520 km')
# the same translucent key as Figure S3: the REVEAL curve runs under it
lg = ax[0].legend(frameon=True, loc='lower left', ncol=1, borderpad=0.3,
                  framealpha=0.8, facecolor='white', edgecolor='#bdbdbd', fancybox=False)
lg.get_frame().set_linewidth(0.6)

# (b) the dip each displacement implies, and how many volumes survive
n = [(p[o] <= 0.05).sum() for o in offs]
ax[1].plot(offs, n, 'o-', color=F.INK, ms=5.5, lw=1.6, mec='white', mew=0.5)
ax[1].set_ylim(-0.4, 5.4); ax[1].set_yticks(range(6))
ax[1].set_xlabel('Down-dip displacement (km)')
ax[1].set_ylabel('Volumes surviving the depth correction')
sec = ax[1].secondary_xaxis('top', functions=(
    lambda x: np.degrees(np.arctan2(410.0, np.maximum(x, 1e-6))),
    lambda t: 410.0 / np.tan(np.radians(np.clip(t, 1e-6, 89.9)))))
sec.set_xlabel('Implied slab dip (degrees)')

for i, x in enumerate(ax):
    x.text(0.0, 1.02 if i == 0 else 1.16, 'ab'[i], transform=x.transAxes,
           ha='left', va='bottom', fontweight='bold',
           fontsize=matplotlib.rcParams['font.size'] + 1.0)
F.check(fig)
os.makedirs(FIG, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_depth_offset.{ext}'), dpi=400,
                bbox_inches='tight', facecolor='white')
print(f'wrote {FIG}/fig_depth_offset   survivors ' +
      ', '.join(f'{int(o)}km:{int(k)}' for o, k in zip(offs, n)))
