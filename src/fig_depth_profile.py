"""Figure: anomaly beneath continental intraplate volcanism against depth.

The job of the data is magnitude against a continuous ordinal axis, with an
uncertainty band, so the form is a profile: depth on the vertical axis increasing
downward as geological convention requires, anomaly on the horizontal, and the two
null envelopes as bands behind the observation.

Two panels because the structure is all in the top 800 km while the claim that the
fast anomaly does not continue needs the whole column to be visible.

The measurement lives in depth_profile.py; this script only draws what that wrote,
so the figure and the reported probabilities come from one set of rotations.

House style from the project's other figure scripts. The house blue #2E5E8E fails
the chroma floor - it reads gray - so it is snapped to #1D6BAA, the nearest step
that passes; the pair then clears CVD separation at dE 18.8 protan against a floor
of 8.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import paths as P
DATA = P.TOMO
CAT = os.environ.get('CATALOGUE', P.IPV_V3)
OUT = P.FIG
os.makedirs(OUT, exist_ok=True)

import figstyle as F
F.apply(11.5)
CM, INK, BLU, ACC, GRY = F.CM, F.INK, F.BLU, F.ACC, F.GRY

EDGES = [0, 100, 200, 300, 410, 520, 660, 800, 1000, 1300, 1600, 2000, 2500]
BANDS = list(zip(EDGES[:-1], EDGES[1:]))

# Read the measured profile and its nulls rather than drawing fresh rotations, so
# that the figure and the reported probabilities cannot drift apart.
NPZ = os.path.join(P.OUT, 'depth_profile_nulls.npz')
z = np.load(NPZ)
if [tuple(b) for b in z['bands'].astype(int)] != [tuple(b) for b in BANDS]:
    raise SystemExit(f'{NPZ} was written for different depth bands; rerun '
                     'depth_profile.py')
obs, free, cont = z['obs'], z['free'], z['cont']
N_SPIN = int(z['n_spin'])
d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180'])
print(f'{len(d)} fields, {N_SPIN} rotations each null, from {NPZ}')

mid = np.array([0.5 * (a + b) for a, b in BANDS])
def step(y, x):                      # band values as a staircase against depth
    yy = np.repeat(EDGES, 2)[1:-1]
    return np.repeat(x, 2), yy

fig, ax = plt.subplots(1, 2, figsize=(17.5 * CM, 10.4 * CM), constrained_layout=True)
# Two panels because one linear scale cannot show both: the lithospheric anomaly is
# five times anything below it, so a shared range compresses the transition-zone
# structure - the actual result - into invisibility.
# panel letters rather than titles: the description of each panel is in the caption
PANELS = [(0, 800, -4.6, 2.2, 'a'),
          (300, 2000, -0.8, 1.25, 'b')]
for k, (zmin, zmax, x0, x1, label) in enumerate(PANELS):
    a = ax[k]
    for band, col, name in ((free, BLU, 'Free rotation'),
                            (cont, ACC, 'Continental rotation')):
        lo_, hi_ = np.percentile(band, [2.5, 97.5], axis=0)
        xl, yy = step(None, lo_); xh, _ = step(None, hi_)
        a.fill_betweenx(yy, xl, xh, color=col, alpha=0.18, lw=0,
                        label=name if k == 1 else None)
        xm, _ = step(None, np.median(band, axis=0))
        a.plot(xm, yy, color=col, lw=0.9, alpha=0.9)
    xo, yy = step(None, obs)
    a.plot(xo, yy, color=INK, lw=2.0, solid_joinstyle='miter',
           label='Observed' if k == 1 else None, zorder=5)
    a.axvline(0, color=GRY, lw=0.8, zorder=0)
    for z in (410, 660):
        if zmin <= z <= zmax:
            a.axhline(z, color=GRY, lw=0.7, ls=(0, (4, 3)), zorder=0)
            a.text(x1, z - 12, f'{z}', fontsize=11, color=GRY, ha='right',
                   va='bottom')
    a.set_ylim(zmax, zmin)
    a.set_xlim(x0, x1)
    a.set_xlabel('Shear-velocity anomaly (per cent)')
    a.set_ylabel('Depth (km)' if k == 0 else None)
    a.text(0.0, 1.02, label, transform=a.transAxes, ha='left', va='bottom',
           fontweight='bold', fontsize=13)
    a.grid(axis='x', color='#ebebeb', lw=0.6, zorder=0)
    a.set_axisbelow(True)

# The three explanatory notes that used to sit on the panels at under 7 point are
# in the caption instead. A reader looks there for them anyway, and nothing on a
# figure should need magnifying.
h, l = ax[1].get_legend_handles_labels()
# 'outside lower center' rather than an anchored 'lower center': a figure legend
# placed by anchor is not part of the constrained layout, so it lands wherever
# the fraction puts it, which here was directly on top of both x-axis labels.
fig.legend(h, l, loc='outside lower center', ncol=3, frameon=False,
           handlelength=1.9, columnspacing=2.4)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_depth_profile.{ext}', bbox_inches='tight',
                facecolor='white')
print(f'wrote {OUT}/fig_depth_profile.pdf and .png')
