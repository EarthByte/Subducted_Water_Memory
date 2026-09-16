"""Three clocks on the same subducted water.

  (a) Seismic visibility. Excess occurrence of fast 410-660 km structure at
      reconstructed delivery locations against delivery age (Table S4), with the
      exponential-plus-floor fit: the delivering slab stops being identifiable
      after a few tens of Myr.
  (b) Chemical survival. MORB H2O/Ce, one median per 300 km cluster, against
      the hydration age of the mantle beneath it; clusters that a craton later
      passed over are filled. The Pacific median, where no water arrived in
      400 Myr, is the reference line. The signature does not decline with age
      and is not removed by keel passage.
  (c) Chemical removal. The same cluster medians in the Atlantic-Arctic
      corridor against the time a spreading ridge has spent within 200 km of the
      site, with the exponential fit of the excess over the Pacific median.

Reads out/persistence_decay.csv (Table S4), out/morb_kinematics.csv and the Dixon table;
writes figures/fig_three_clocks. Every number is one an earlier step wrote.
"""
import os, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import paths as P
import figstyle as F
import morb

F.apply(11.0)


def table_s4():
    """The enrichment curve, as persistence_decay.py wrote it (Table S4)."""
    p = pd.read_csv(os.path.join(P.OUT, 'persistence_decay.csv'))
    return np.column_stack([p.t_mid.values, p.enrichment.values, p.sigma.values])


k = pd.read_csv(os.path.join(P.OUT, 'morb_kinematics.csv'))
d = morb.load()
assert (k.Sample.values == d.Sample.values).all()
plume = morb.plume_mask(d)

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16.0 * F.CM, 7.4 * F.CM),
                                       gridspec_kw=dict(wspace=0.42))

# (a) seismic visibility
t, e, s = table_s4().T
ax_a.errorbar(t, e, yerr=s, fmt='o', color=F.INK, ms=4, lw=1, capsize=2, zorder=3)
f = lambda t, e0, ei, tau: ei + (e0 - ei) * np.exp(-t / tau)
pp, _ = curve_fit(f, t, e, p0=[4, 0.3, 37], sigma=s, bounds=([0, -1, 3], [60, 5, 600]))
tt = np.linspace(0, 400, 400)
ax_a.plot(tt, f(tt, *pp), color=F.BLU, lw=1.6, zorder=2)
ax_a.axhline(1, color=F.GRY, lw=0.8, ls='--', zorder=1)
ax_a.set_xlabel('delivery age (Ma)')
ax_a.set_ylabel('excess fast 410–660 km')
ax_a.set_xlim(0, 400); ax_a.set_ylim(0, 4.6)
ax_a.text(0.97, 0.90, f'{pp[2]:.0f} Myr', transform=ax_a.transAxes, ha='right', va='top',
          color=F.BLU)
print(f'(a) e-folding {pp[2]:.1f} Myr from Table S4')

# (b) chemical survival
pac = morb.pacific(k)
clp = morb.clusters(k.Longitude.values[pac], k.Latitude.values[pac])
base = float(np.median(morb.cluster_medians(k.H2O_Ce.values[pac], clp)['median']))
hyd = k.hydrated.values & ~plume
cl = morb.clusters(k.Longitude.values[hyd], k.Latitude.values[hyd])
cm = morb.cluster_medians(k.H2O_Ce.values[hyd], cl,
                          extra=dict(age=k.age.values[hyd], craton=k.t_craton.values[hyd],
                                     dwell=k.t_ridge_200.values[hyd]))
over = cm['craton'].values > 0
ax_b.axhline(base, color=F.GRY, lw=0.8, ls='--', zorder=1)
ax_b.scatter(cm['age'][~over], cm['median'][~over], s=26, facecolor='white',
             edgecolor=F.INK, lw=1.0, zorder=3)
ax_b.scatter(cm['age'][over], cm['median'][over], s=26, color=F.ACC, edgecolor='none',
             zorder=4, label='craton passed over')
ax_b.set_xlabel('hydration age (Ma)')
ax_b.set_ylabel('H$_2$O/Ce, cluster median')
ax_b.set_xlim(100, 400); ax_b.set_ylim(120, 460)
print(f'(b) {len(cm)} hydrated clusters without plume segments, {int(over.sum())} overrun; '
      f'Pacific median {base:.0f}')

# (c) chemical removal, corridor, hydrated or not
cor = morb.corridor(k) & ~plume
clc = morb.clusters(k.Longitude.values[cor], k.Latitude.values[cor])
cc = morb.cluster_medians(k.H2O_Ce.values[cor], clc,
                          extra=dict(dwell=k.t_ridge_200.values[cor],
                                     hyd=k.hydrated.values[cor].astype(float)))
dw, ex = cc['dwell'].values, cc['median'].values - base
g = lambda t, e0, tau: e0 * np.exp(-t / tau)
qq, _ = curve_fit(g, dw, ex, p0=[300, 30], bounds=([0, 5], [800, 1000]))
isH = cc['hyd'].values >= 0.5
ax_c.axhline(base, color=F.GRY, lw=0.8, ls='--', zorder=1)
ax_c.scatter(dw[~isH], cc['median'].values[~isH], s=26, facecolor='white', edgecolor=F.INK,
             lw=1.0, zorder=3)
ax_c.scatter(dw[isH], cc['median'].values[isH], s=26, color=F.BLU, edgecolor='none',
             zorder=4)
td = np.linspace(0, 160, 200)
ax_c.plot(td, base + g(td, *qq), color=F.BLU, lw=1.6, zorder=2)
ax_c.set_xlabel('ridge residence (Myr)')
ax_c.set_ylabel('H$_2$O/Ce, cluster median')
ax_c.set_xlim(0, 160); ax_c.set_ylim(120, 460)
ax_c.text(0.97, 0.90, f'{qq[1]:.0f} Myr', transform=ax_c.transAxes, ha='right', va='top',
          color=F.BLU)
print(f'(c) {len(cc)} corridor clusters, e-folding {qq[1]:.1f} Myr, excess at zero {qq[0]:.0f}')

for ax, letter in zip((ax_a, ax_b, ax_c), 'abc'):
    ax.text(-0.22, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')
fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.2)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_three_clocks.{ext}'), bbox_inches='tight')
print('wrote figures/fig_three_clocks.pdf and .png')
