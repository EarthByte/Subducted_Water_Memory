"""The Atlantic-Arctic corridor on one latitude axis: chemistry, delivery, what
passed over the mantle since, and what the transition zone shows today.

Four panels, 50W to 5E plus the Arctic ridges, bottom to top:

  (a) MORB H2O/Ce, every sample, hydrated sites coloured by the time a
      spreading ridge has lain within 200 km of the site since its hydration
      (the variable this paper reports; the hydration age itself is Dixon et
      al.'s Fig. 4 and is not re-plotted), with the median of each 5 degree bin;
  (b) for hydrated sites, Myr since hydration that a spreading ridge has sat
      within 200 km of the site, and Myr under a cratonic keel (>= 200 km
      lithosphere), bin medians; bins with no hydrated site are empty;
  (c) shear-velocity anomaly at 410-520 km beneath the sites in five models,
      bin medians, which is where recent slab delivery would show;
  (d) the same at 100-200 km, the depth range decompression melting samples.

Reads out/morb_kinematics.csv and the band files; writes figures/fig_morb_corridor.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
import paths as P
import figstyle as F
import morb
import morb_tomography as MT

F.apply(11.0)
from cmcrameri import cm as _cm
BOUNDS = [0, 20, 40, 60, 80, 100, 160]      # ridge residence since hydration, Myr
cmap = _cm.nuuk.resampled(len(BOUNDS) - 1)
norm = BoundaryNorm(BOUNDS, cmap.N)
MODEL_COLOURS = {'REVEAL': F.BLU, 'RevealLO': '#5aa0d8', 'GLADM35': F.ACC,
                 'SPiRaL': '#5c8a3a', 'SEMUCBWM1': '#8a5ca8'}

k = pd.read_csv(os.path.join(P.OUT, 'morb_kinematics.csv'))
d = morb.load()
assert (k.Sample.values == d.Sample.values).all()
c = morb.corridor(k)
k = k[c].reset_index(drop=True)
lat = k.Latitude.values
edges = np.arange(-60, 90.1, 5.0)
mid = 0.5 * (edges[:-1] + edges[1:])


def binned(v, fn=np.nanmedian, min_n=3):
    out = np.full(len(mid), np.nan)
    for i in range(len(mid)):
        m = (lat >= edges[i]) & (lat < edges[i + 1]) & np.isfinite(v)
        if m.sum() >= min_n:
            out[i] = fn(v[m])
    return out


models = MT.load_models()
fig, axes = plt.subplots(4, 1, figsize=(16.0 * F.CM, 22.0 * F.CM), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.3, 1.0, 1.0, 1.0], hspace=0.18))
ax_a, ax_b, ax_c, ax_d = axes

# (a) chemistry
hyd = k.hydrated.values
ax_a.scatter(lat[~hyd], k.H2O_Ce.values[~hyd], s=9, c='#c9c9c9', edgecolor='none',
             zorder=2, rasterized=True)
ax_a.scatter(lat[hyd], k.H2O_Ce.values[hyd], s=11, c=k.t_ridge_200.values[hyd], cmap=cmap,
             norm=norm, edgecolor='none', zorder=3, rasterized=True)
ax_a.plot(mid, binned(k.H2O_Ce.values), color=F.INK, lw=1.8, zorder=4)
ax_a.axhline(250, color=F.GRY, lw=0.8, ls='--', zorder=1)
ax_a.set_ylim(80, 520)
ax_a.set_ylabel('H$_2$O/Ce')
cax = ax_a.inset_axes([0.14, 0.80, 0.30, 0.06])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation='horizontal')
cb.set_label('ridge residence (Myr)', labelpad=2)
cb.set_ticks([0, 40, 80, 160])
cax.xaxis.set_ticks_position('bottom')

# (b) what passed over
# hydrated sites only, so that both bars count from the same clock
tr = np.where(hyd, k.t_ridge_200.values, np.nan)
tk = np.where(hyd, k.t_keel.values, np.nan)
ax_b.bar(mid - 1.0, binned(tr), width=2.0, color=F.BLU, label='ridge within 200 km')
ax_b.bar(mid + 1.0, binned(tk), width=2.0, color=F.ACC, label='under a keel')
ax_b.set_ylabel('Myr since hydration')
ax_b.legend(frameon=False, loc='upper left', ncol=2, handlelength=1.0)
ax_b.set_ylim(0, 170)

# (c), (d) tomography beneath the sites
for ax, key, lab in ((ax_c, '410_520', '410–520 km'), (ax_d, '100_200', '100–200 km')):
    for tag, mdl in models.items():
        v = MT.sample(mdl, key, k.Longitude.values, lat)
        ax.plot(mid, binned(v), color=MODEL_COLOURS[tag], lw=1.5, label=tag)
    ax.axhline(0, color=F.GRY, lw=0.8, zorder=1)
    ax.set_ylabel(f'dVs/Vs (%)\n{lab}')
ax_c.set_ylim(-1.6, 1.6)
ax_d.legend(frameon=False, loc='lower right', ncol=3, handlelength=1.2)
ax_d.set_xlabel('latitude (°N)')
ax_d.set_xlim(-60, 88)
for ax, letter in zip(axes, 'abcd'):
    ax.text(0.01, 0.92, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.07)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_morb_corridor.{ext}'), bbox_inches='tight')
print('wrote figures/fig_morb_corridor.pdf and .png')
