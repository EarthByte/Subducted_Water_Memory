"""The kinematic history of the hydrated ridge sites, on the map.

  (a) The time each hydrated site later spent beneath a craton outline of
      Shirmard et al. (2025), the outlines drawn; sites never overlain are
      open. This is the variable the keel test uses.
  (b) The time a spreading ridge has lain within 200 km of each hydrated site
      since its hydration: the ridge residence the chemistry records.
  (c) The time at which a ridge first arrived within 200 km of the site.

Unhydrated sites (no subducted water in 410 Myr) are open in every panel. The
compilation itself and the hydration age of each site are Figures 1 and 4 of
Dixon et al. (in review) and are not re-plotted here. Reads
out/morb_kinematics.csv and the craton shapefile; writes figures/fig_morb_map.
"""
import os, numpy as np, pandas as pd, geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from cmcrameri import cm as ccm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import paths as P
import figstyle as F
import plate_boundaries as PB

F.apply(11.0)
# the present-day plate boundaries of the same model the histories were
# computed with, so the ridges on the map are the ridges of the residence test
from gplately import PlateReconstruction
from plate_model_manager import PlateModelManager
pmm = PlateModelManager().get_model('zahirovic2022', data_dir=P.MODELS)
recon = PlateReconstruction(pmm.get_rotation_model(), topology_features=pmm.get_topologies(),
                            static_polygons=pmm.get_static_polygons())

k = pd.read_csv(os.path.join(P.OUT, 'morb_kinematics.csv'))
cr = gpd.read_file(P.CRATONS)
hyd = k.hydrated.values.astype(bool)
print(f'{len(k)} samples, {int(hyd.sum())} hydrated, '
      f'{int((k.t_craton[hyd] > 0).sum())} of those later beneath a craton outline')

proj = ccrs.Robinson(central_longitude=-30)
geo = ccrs.PlateCarree()
fig, axes = plt.subplots(3, 1, figsize=(16.0 * F.CM, 24.5 * F.CM),
                         subplot_kw=dict(projection=proj))
fig.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.07, hspace=0.36)


def base(ax, cratons=False):
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=F.LAND, edgecolor='none', zorder=1)
    ax.coastlines('110m', lw=0.4, color='#b0aba4', zorder=2)
    ax.spines['geo'].set_linewidth(0.8)
    F.map_grid(ax, left=True, bottom=True)
    # ridges run under the sample symbols, trenches carry teeth towards the
    # overriding plate
    PB.ridges(ax, recon, 0.0, geo, zorder=3)
    PB.trenches(ax, recon, 0.0, geo, zorder=3, tooth_deg=1.3, spacing_deg=9.0)
    if cratons:
        ax.add_geometries(cr.geometry, crs=geo, facecolor='none', edgecolor=F.ACC,
                          lw=0.9, zorder=3)


def draw(ax, mask, vals, cmap, norm, label, ticks):
    ax.scatter(k.Longitude[~mask], k.Latitude[~mask], s=9, facecolor='white',
               edgecolor=F.INK, lw=0.35, transform=geo, zorder=4)
    sc = ax.scatter(k.Longitude[mask], k.Latitude[mask], c=vals[mask], s=10, cmap=cmap,
                    norm=norm, edgecolor=F.INK, lw=0.25, transform=geo, zorder=5)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.045, pad=0.09,
                      aspect=32, ticks=ticks, extend='max' if norm.clip is False else 'neither')
    cb.set_label(label)
    cb.ax.tick_params(length=2.5)
    return sc


# (a) craton residence since hydration
base(axes[0], cratons=True)
over = hyd & (k.t_craton.values > 0)
cb_c = [0, 25, 50, 100, 150, 250]
norm_c = BoundaryNorm(cb_c, 256, clip=False)
cmap_c = ListedColormap(ccm.oslo_r(np.linspace(0.25, 0.95, 256)))
draw(axes[0], over, k.t_craton.values, cmap_c, norm_c,
     'Time beneath a craton outline since hydration (Myr)', cb_c)

# (b) ridge residence since hydration
base(axes[1])
cb_b = [0, 20, 40, 60, 80, 100, 160]
norm_b = BoundaryNorm(cb_b, 256, clip=False)
cmap_b = ListedColormap(ccm.lajolla_r(np.linspace(0.1, 0.9, 256)))
draw(axes[1], hyd, k.t_ridge_200.values, cmap_b, norm_b,
     'Ridge residence since hydration (Myr)', cb_b)

# (c) time of ridge arrival
base(axes[2])
cb_o = [0, 25, 50, 75, 100, 150]
norm_o = BoundaryNorm(cb_o, 256, clip=False)
cmap_o = ListedColormap(ccm.batlowK_r(np.linspace(0.1, 0.9, 256)))
arrived = hyd & np.isfinite(k.ridge_onset_200.values)
draw(axes[2], arrived, k.ridge_onset_200.values, cmap_o, norm_o,
     'Arrival of a ridge within 200 km (Ma)', cb_o)

for ax, letter in zip(axes, 'abc'):
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='bottom', ha='left')
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_morb_map.{ext}'), bbox_inches='tight')
print('wrote figures/fig_morb_map.pdf and .png')
