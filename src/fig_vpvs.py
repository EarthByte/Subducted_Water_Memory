"""Figure: the P to S ratio in the transition zone, against a thermal reference.

Wang and Wang (2022) give about 4 per cent in Vp against 8 per cent in Vs over the
temperature range, so a thermal anomaly sits near an angle of 63.4 degrees and a
hydrous one at or below 45. The left panel is the discriminant against anomaly
amplitude, which is the test that matters: a real hydrous population strengthens
with amplitude, and this one vanishes with it. The right panel is the same
statistic shell by shell, and nothing happens at either discontinuity.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as F

F.apply(9.0)
CM, INK, ACC, BLU, GRY = F.CM, F.INK, F.ACC, F.BLU, F.GRY
THERMAL, HYDROUS = 63.4, 45.0

import os
import paths as P
z = np.load(os.path.join(P.OUT, 'vpvs_angle_field.npz'))
th, dvp, dvs, good = z['theta'], z['dvp'], z['dvs'], z['good']
# the file holds fractions; the paper quotes per cent, and 3.939 per cent is
# the largest anomaly, which is how this scaling was checked
amp = 100.0 * np.hypot(dvp, dvs)
k = good & np.isfinite(th) & np.isfinite(amp)
a, t = amp[k], th[k]

fig, ax = plt.subplots(1, 2, figsize=(15.8 * CM, 7.0 * CM), constrained_layout=True)

# left: the discriminant against amplitude, in equal-count bins
q = np.quantile(a, np.linspace(0, 1, 9))
mid, med, lo, hi, frac = [], [], [], [], []
for i in range(len(q) - 1):
    m = (a >= q[i]) & (a < q[i + 1])
    if m.sum() < 200:
        continue
    mid.append(np.median(a[m])); med.append(np.median(t[m]))
    lo.append(np.percentile(t[m], 25)); hi.append(np.percentile(t[m], 75))
    frac.append(100 * (t[m] < HYDROUS).mean())
b = ax[0]
b.axhspan(HYDROUS - 45, HYDROUS, color='#dce6f1', zorder=0)
b.axhline(THERMAL, color=GRY, lw=1.0, ls='--', zorder=1)
b.axhline(HYDROUS, color=BLU, lw=1.0, ls=':', zorder=1)
b.fill_between(mid, lo, hi, color=ACC, alpha=0.18, lw=0)
b.plot(mid, med, 'o-', color=ACC, ms=5, lw=1.6)
b.text(mid[-1], THERMAL + 1.2, 'Thermal, 63.4°', color=GRY, ha='right', va='bottom')
b.text(mid[-1], HYDROUS - 2.0, 'Hydrous, 45° and below', color=BLU, ha='right', va='top')
b.set_xscale('log')
# the log minor tick labels collide with each other at this width
b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f'{v:g}'))
b.set_xlabel('Anomaly amplitude (per cent)')
b.set_ylabel('Angle (degrees)')   # defined in the caption
b.set_ylim(20, 80)
# No key and no annotation on the panel. The only free corner is inside the
# shaded hydrous band, where a label reads as data, and the caption is where a
# reader looks for what the band and the dashed line are.
b2 = b.twinx()
b2.plot(mid, frac, 's--', color=INK, ms=4, lw=1.0, alpha=0.75)
b2.set_ylabel('Per cent below 45 degrees', color=INK)
b2.set_ylim(0, 45)
b.text(-0.14, 1.03, 'a', transform=b.transAxes, fontsize=13, fontweight='bold',
       va='bottom', ha='right')

# right: shell by shell
p = pd.read_csv('out/vpvs_angle_profile.csv')
c = ax[1]
c.axvspan(0, HYDROUS, color='#dce6f1', zorder=0)
c.axvline(THERMAL, color=GRY, lw=1.0, ls='--', zorder=1)
c.plot(p.median_angle, p.depth, '-', color=ACC, lw=2.0)
for d in (410, 660):
    c.axhline(d, color=GRY, lw=0.8, ls=(0, (4, 3)), zorder=0)
    c.text(30, d - 8, f'{d}', color=GRY, ha='left', va='bottom')
c.set_ylim(p.depth.max(), p.depth.min())
c.set_xlim(25, 75)
c.set_xlabel('Median angle (degrees)')
c.set_ylabel('Depth (km)')
c.text(-0.16, 1.03, 'b', transform=c.transAxes, fontsize=13, fontweight='bold',
       va='bottom', ha='right')

F.check(fig)
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_vpvs.{e}'), bbox_inches='tight', facecolor='white')
print('wrote figures/fig_vpvs')
