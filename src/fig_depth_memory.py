"""Figure 4: how deep the association with subduction can be followed.

Everything is read from out/depth_fwer.csv, which depth_fwer.py writes, so the
figure and the probabilities quoted beside it cannot disagree. The earlier
version read three separate depth_decay files and had to be edited whenever a
volume was added; this one draws whatever the scan produced.

Two panels, because the claim needs both. (a) is the measurement: E*, the
enrichment at the best-explaining age, against the 95th percentile of the
rotated null. E* is the largest of ten ratios and so exceeds one with no signal
at all, which is why the null and not unity is the reference. (b) is the result:
the probability after correcting for the ten depths searched in each volume,
which is what decides whether a band means anything.
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paths as P
import figstyle as F

F.apply(10.0)
CM, INK, GRY = F.CM, F.INK, F.GRY
# One colour and one marker per volume; the two REVEAL-family models share a
# family but not a line, because the paper's argument turns on telling them apart.
STYLE = [('REVEAL',    '#B4442E', 'o'),
         ('RevealLO',  '#1D6BAA', 's'),
         ('GLADM35',   '#3F7A4B', '^'),
         ('SPiRaL',    '#7B5AA6', 'D'),
         ('SEMUCBWM1', '#C77B2B', 'v')]
NICE = {'GLADM35': 'GLAD-M35', 'SEMUCBWM1': 'SEMUCB-WM1'}
ALPHA = 0.05

d = pd.read_csv(os.path.join(P.OUT, 'depth_fwer.csv'))
have = [(t, c, m) for t, c, m in STYLE if (d.model == t).any()]
print(f'{len(have)} volumes: ' + ', '.join(NICE.get(t, t) for t, _, _ in have))

fig, ax = plt.subplots(1, 2, figsize=(19.0 * CM, 10.6 * CM), constrained_layout=True)
for tag, col, mk in have:
    g = d[d.model == tag].sort_values('z_mid' if 'z_mid' in d else 'z0')
    z = 0.5 * (g.z0 + g.z1)
    ax[0].plot(g.e_best, z, mk + '-', color=col, ms=4.2, lw=1.4,
               label=NICE.get(tag, tag))
    ax[0].plot(g.null_p95, z, ':', color=col, lw=1.1, alpha=0.85)
    ax[1].plot(g.p_adj, z, mk + '-', color=col, ms=4.2, lw=1.4)

for a in ax:
    a.axhspan(410, 660, color='#eef2ee', zorder=0)
    for z_ in (410, 660):
        a.axhline(z_, color=GRY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    a.set_ylim(2500, 350)
    a.set_ylabel('Depth (km)')
ax[0].set_xlabel('Enrichment at the best-explaining age, E*')
ax[0].set_xlim(1.4, 4.4)
ax[0].text(1.5, 535, 'Transition\nzone', color=GRY, ha='left', va='center')
# A white box at 80 per cent opacity behind the key: the curves still show
# faintly through it, and the labels stay readable where they cross.
lg = ax[0].legend(frameon=True, loc='lower right', framealpha=0.8, facecolor='white',
                  edgecolor='#bdbdbd', fancybox=False,
                  title='Solid: E*\nDotted: chance level',
                  title_fontsize=matplotlib.rcParams['font.size'])
lg.get_frame().set_linewidth(0.6)
ax[1].set_xscale('log')
ax[1].set_xlim(2e-4, 1.4)
# Plain decimals rather than powers of ten: matplotlib sets the exponent as a
# superscript about three fifths the size of the mantissa, which prints below
# the legibility floor even when the label as a whole passes.
# 0.05 is drawn as a line rather than a tick: as a tick it sits close enough to
# 0.1 that the two labels touch.
ax[1].set_xticks([0.001, 0.01, 0.1, 1.0])
ax[1].set_xticklabels(['0.001', '0.01', '0.1', '1'])
ax[1].set_xticks([], minor=True)
ax[1].axvline(ALPHA, color=INK, lw=1.0, ls=(0, (5, 3)))
ax[1].set_xlabel('Adjusted probability')
ax[1].text(ALPHA * 0.85, 1950, f'{ALPHA:g}', color=INK, ha='right', va='center')
# the arrow has to point into the region that survives, which is to the LEFT of
# the threshold; pointing at the line reads as the opposite
ax[1].annotate('Survives', xy=(0.0035, 2180), xytext=(ALPHA * 0.85, 2180),
               color=INK, ha='right', va='center',
               arrowprops=dict(arrowstyle='->', color=INK, lw=0.9))
for i, a in enumerate(ax):
    a.text(0.0, 1.015, 'ab'[i], transform=a.transAxes, ha='left', va='bottom',
           fontweight='bold', fontsize=matplotlib.rcParams['font.size'] + 1.0)
F.check(fig)
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_depth_memory.{e}'), bbox_inches='tight',
                facecolor='white')
print(f'wrote {P.FIG}/fig_depth_memory.pdf and .png')
