"""Figure 1: the fast transition-zone bodies, their attribution, and the volcanism.

The figure has to carry three things at once, because the paper's argument needs
all three in the same frame: where the fast transition zone is, which parts of it
a reconstruction can tie to a trench, and where the volcanism sits relative to
both. The continental mask is drawn rather than a coastline, because the mask is
what the restricted null uses and the reader should see the domain that null
occupies.

Attributed and unattributed bodies are the same colour, because they are the same
kind of object: a fast anomaly. What separates them is provenance, so the
unattributed ones are hatched rather than recoloured. The hatch also survives
greyscale printing and colour-vision deficiency, where a second hue would not.
"""
import os, numpy as np, pandas as pd, dataclasses
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import paths as P
import plume_classifier as pc
import contmask

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit('\nthis figure needs cartopy\n  conda install -c conda-forge cartopy\n')

import figstyle as F
F.apply(12.0)
CM, INK, BLU, ACC, GRY, LAND = F.CM, F.INK, F.BLU, F.ACC, F.GRY, F.LAND
matplotlib.rcParams.update({'hatch.color': BLU, 'hatch.linewidth': 0.9})

P.need(P.REVEAL, 'TOMO_DIR', 'REVEAL_vs_full.nc')
BODIES = os.path.join(P.OUT, 's19', 'slab_bodies.csv')
P.need(BODIES, 'PAPER_OUT', 'slab_bodies.csv; run s19_slab_provenance.py first')

SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
depth, lat, lon, arr = pc.load_anomaly(SPEC, P.REVEAL)
band = np.nanmean(arr[(depth >= 410) & (depth < 660)], axis=0)
f = np.isfinite(band)
mask = f & (band >= np.percentile(band[f], 90.0))

# label with the longitude seam stitched, exactly as the attribution stage does,
# so the body identifiers match the ones it wrote
lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
left, right = lab[:, 0], lab[:, -1]
both = (left > 0) & (right > 0)
pairs = {(min(a, b), max(a, b)) for a, b in zip(left[both], right[both]) if a != b}
if pairs:
    parent = np.arange(n + 1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    lab = np.array([find(x) if x > 0 else 0 for x in range(n + 1)])[lab]

bodies = pd.read_csv(BODIES)
att = set(bodies.loc[bodies.attributed, 'body'].astype(int))
una = set(bodies.loc[~bodies.attributed, 'body'].astype(int))
A = np.isin(lab, list(att))
U = np.isin(lab, list(una))
print(f'{len(att)} attributed and {len(una)} unattributed bodies, '
      f'{A.sum()} and {U.sum()} cells')

clon, clat, cmask = contmask.load()
ipv = pd.read_csv(P.IPV_V3).dropna(subset=['lat', 'lon_180'])
print(f'{len(ipv)} volcanic fields')

pcarree = ccrs.PlateCarree()
fig = plt.figure(figsize=(19.0 * CM, 11.6 * CM))
ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
ax.set_global()

ax.pcolormesh(clon, clat, np.where(cmask, 1.0, np.nan), transform=pcarree,
              cmap=matplotlib.colors.ListedColormap([LAND]), shading='nearest',
              zorder=1, rasterized=True)
ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor='#b9b4ad', zorder=2)

ax.pcolormesh(lon, lat, np.where(A, 1.0, np.nan), transform=pcarree,
              cmap=matplotlib.colors.ListedColormap([BLU]), shading='nearest',
              alpha=0.85, zorder=3, rasterized=True)
ax.contourf(lon, lat, U.astype(float), levels=[0.5, 1.5], transform=pcarree,
            colors='none', hatches=['////'], zorder=3)
ax.contour(lon, lat, U.astype(float), levels=[0.5], transform=pcarree,
           colors=[BLU], linewidths=0.7, zorder=4)
ax.plot(ipv.lon_180.values, ipv.lat.values, transform=pcarree, linestyle='none',
        marker='o', markersize=2.6, markerfacecolor=ACC, markeredgecolor='white',
        markeredgewidth=0.35, zorder=6)

# Four labels, not five, and at reading size. The rest of the geography is named
# in the caption, which is where it belongs.
LABELS = [('northeast Asia', 134, 54, 'left'),
          ('southeast Asia', 112, -4, 'center'),
          ('Mediterranean', 20, 28, 'right'),
          ('South America', -72, -40, 'right')]
for text, x, y, ha in LABELS:
    ax.text(x, y, text, transform=pcarree, fontsize=12, color=INK, ha=ha,
            va='center', zorder=7, linespacing=1.15,
            path_effects=[pe.withStroke(linewidth=3.4, foreground='white')])

handles = [Patch(facecolor=BLU, alpha=0.85, edgecolor='none',
                 label='fast transition zone, attributed to a trench'),
           Patch(facecolor='white', edgecolor=BLU, hatch='////', linewidth=0.7,
                 label='fast transition zone, unattributed'),
           Line2D([], [], linestyle='none', marker='o', markersize=3.4,
                  markerfacecolor=ACC, markeredgecolor='white',
                  markeredgewidth=0.4, label='continental intraplate volcanic field'),
           Patch(facecolor=LAND, edgecolor='#b9b4ad', linewidth=0.4,
                 label='continental lithosphere')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.005, -0.13),
          ncol=2, frameon=False, handlelength=1.8, handleheight=1.2,
          columnspacing=1.8, labelspacing=0.7, borderpad=0.0)
ax.spines['geo'].set_edgecolor('#cfcac3')
ax.spines['geo'].set_linewidth(0.6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.17)
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_map.{ext}'), dpi=400)
print(f'wrote {P.FIG}/fig_map.pdf and .png')
