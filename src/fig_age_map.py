"""Figure 2: when subduction last delivered material to the transition zone.

For every point on the globe it shows the age of the youngest subduction that
could have put a slab beneath it, given the descent geometry and sinking rate
used throughout, with the observed fast transition zone drawn over the top so
the correspondence can be judged rather than asserted.

Two encodings, kept distinct. Age is a magnitude, so it takes one perceptually
uniform sequential ramp. The observed fast transition zone is an identity, so it
takes an outline in a hue the ramp does not contain. nuuk runs from deep blue
through olive to pale yellow and holds no red at any step, which is what makes a
red outline readable everywhere on it; on a ramp with red in it, as here
previously, the outline disappeared over half the map.

Grey is where no subduction in 400 Myr can account for anything, which is 42 per
cent of the surface and is the reason the attribution is not vacuous. It has to
sit outside the ramp too, so it is a neutral grey against a ramp whose light end
is a saturated cream.
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import paths as P
import figstyle as F

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit('\nthis figure needs cartopy\n')

F.apply(12.0)
z = np.load(os.path.join(P.OUT, 'hydration_age_map.npz'))
lon, lat, t_min, obs = z['lon'], z['lat'], z['t_min'], z['observed']
w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))
cov = np.isfinite(t_min)
print(f'{100 * (w[cov].sum() / w.sum()):.1f} per cent of the surface has an '
      f'explanation within 400 Myr')

ORDER = os.environ.get('AGE_CMAP', 'nuuk')
from cmcrameri import cm as _cm
BOUNDS = [10, 25, 50, 75, 100, 150, 200, 300, 400]
cmap = getattr(_cm, ORDER).resampled(len(BOUNDS) - 1)
norm = BoundaryNorm(BOUNDS, cmap.N)
NODATA, LANDC, COAST = '#e4e7ea', '#d2d6d9', '#83878b'
OUTLINE = F.ACC          # red: nuuk contains no red at any step

pc = ccrs.PlateCarree()
fig = plt.figure(figsize=(19.0 * F.CM, 12.2 * F.CM))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
ax.set_global()
ax.set_facecolor(NODATA)
ax.add_feature(cfeature.LAND, facecolor=LANDC, edgecolor='none', zorder=1)
m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(t_min), transform=pc,
                  cmap=cmap, norm=norm, shading='nearest', zorder=2,
                  rasterized=True)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=COAST, zorder=3)
ax.contour(lon, lat, obs.astype(float), levels=[0.5], transform=pc,
           colors=[OUTLINE], linewidths=1.5, zorder=5)

cb = fig.colorbar(m, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06,
                  aspect=42, ticks=BOUNDS, spacing='uniform')
cb.set_label('Age of the youngest subduction that can account for a slab (Ma)')
cb.ax.tick_params(length=4)
handles = [Line2D([], [], color=OUTLINE, lw=1.8,
                  label='observed fast transition zone'),
           Patch(facecolor=NODATA, edgecolor='#8f9297', linewidth=0.5,
                 label='no subduction in 400 Myr')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, -0.075),
          ncol=2, frameon=False, handlelength=1.9, columnspacing=2.2,
          borderpad=0.0)
ax.spines['geo'].set_edgecolor('#cfcac3')
ax.spines['geo'].set_linewidth(0.7)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20)
F.check(fig)
tag = '' if ORDER == 'nuuk' else '_' + ORDER
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_age_map{tag}.{ext}'))
print(f'wrote {P.FIG}/fig_age_map{tag}.pdf and .png  [{ORDER}]')
