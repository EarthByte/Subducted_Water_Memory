"""Figure 1: the field every other figure in this paper is computed from.

The analysis reduces three tomographic models to a single derived object, the
top decile of the transition-zone anomaly, and then never shows the reader the
models themselves. This does. Each panel is the shear-velocity anomaly averaged
over 410-660 km, on one shared scale, with the top-decile contour drawn on it,
so three things the text asserts as numbers can be looked at instead:

  - the fast transition zone is where subduction has been, in every model;
  - the models agree on the large features and disagree on the small ones,
    which is why the persistence constant is quoted across all three and why no
    claim is made about ages beyond 150 Ma;
  - the top decile is not a set of discrete bodies. Its largest connected
    component holds 21 per cent of the fast set in REVEAL and 73 per cent in
    GLAD-M35: the set percolates as the model resolves better, which is what
    rules out any statistic computed per body.

The fourth panel is GLAD-M35's P anomaly on the same colour scale as the three
S panels, not on its own. It looks weaker because it is weaker, by about the
factor of two that a thermal anomaly gives, and that is the discriminant of
section 4.3 shown as a picture rather than as an angle.

    python3 src/fig_tomography.py

REVEAL is read from the model file and cached to out/tz_REVEAL.npz, because
reading 400 MB to draw one map is not worth repeating; pass --refresh to redo
it. The other two models are read from the band files extract_bands.py writes.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import paths as P
import figstyle as F

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit('\nthis figure needs cartopy\n'
                     '  conda install -c conda-forge cartopy\n')

Z0, Z1 = 410.0, 660.0
LIM = 2.5          # per cent, symmetric; set from the models, see _limits below
PCTL = 90.0        # the top decile, as everywhere else in the paper


def reveal_band(cache, refresh=False):
    """REVEAL's transition-zone shear anomaly, cached.

    Voigt average and lateral-mean removal come from plume_classifier, so this
    panel is the field the statistics are computed on rather than a second
    definition that happens to look similar. P is not read here: the model file
    the paper uses carries vsv and vsh only, and the P panel is GLAD-M35's.
    """
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        return z['lat'], z['lon'], z['dvs']

    import dataclasses
    import plume_classifier as pc
    P.need(P.REVEAL, 'TOMO_DIR', 'the REVEAL model file')
    spec = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
    depth, lat, lon, arr = pc.load_anomaly(spec, P.REVEAL)
    dvs = np.nanmean(arr[(depth >= Z0) & (depth < Z1)], axis=0)
    np.savez_compressed(cache, lat=lat, lon=lon, dvs=dvs.astype(np.float32))
    print(f'cached {cache}')
    return lat, lon, dvs


def band_file(path):
    """410-660 km from the two bands extract_bands.py writes, thickness-weighted.

    The bands are 410-520 and 520-660, so a plain mean of the two would weight
    110 km of mantle as heavily as 140 km and tilt the average towards the
    upper transition zone, which is the half carrying the signal. That would
    flatter the figure.
    """
    z = np.load(path, allow_pickle=True)
    a, b = z['410_520'].astype(float), z['520_660'].astype(float)
    w0, w1 = 520.0 - 410.0, 660.0 - 520.0
    return z['lat'], z['lon'], (w0 * a + w1 * b) / (w0 + w1)


def draw(ax, lon, lat, field, label, contour=True):
    pc_ = ccrs.PlateCarree()
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=F.LAND, zorder=1)
    m = ax.pcolormesh(lon, lat, field, transform=pc_, cmap='RdBu',
                      vmin=-LIM, vmax=LIM, shading='nearest', zorder=2,
                      rasterized=True)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#8f8a84',
                   zorder=3)
    if contour:
        f = np.isfinite(field)
        thr = np.percentile(field[f], PCTL)
        ax.contour(lon, lat, np.where(f, field, np.nan), levels=[thr],
                   transform=pc_, colors=['#1a1a1a'], linewidths=0.55, zorder=4)
    ax.spines['geo'].set_edgecolor('#cfcac3')
    ax.spines['geo'].set_linewidth(0.6)
    # The panel letter and the model name are one label, set outside the map so
    # nothing is drawn over the data.
    ax.text(0.0, 1.035, label, transform=ax.transAxes, ha='left',
            va='bottom', color=F.INK, fontweight='bold',
            fontsize=matplotlib.rcParams['font.size'] + 1.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true',
                    help='re-read REVEAL from the model file')
    ap.add_argument('--bands-dir', default=os.path.join(
                        getattr(P, 'TOMO', '..'), 'REVEAL_mantle_tomography'),
                    help='where extract_bands.py wrote bands_<model>.npz')
    a = ap.parse_args()

    F.apply(10.0)
    cache = os.path.join(P.OUT, 'tz_REVEAL.npz')
    rlat, rlon, rvs = reveal_band(cache, a.refresh)

    lolat, lolon, lovs = band_file(os.path.join(a.bands_dir, 'bands_RevealLO.npz'))
    glat, glon, gvs = band_file(os.path.join(a.bands_dir, 'bands_GLADM35.npz'))

    # GLAD-M35's P anomaly comes from the field the Vp/Vs work already wrote,
    # so the panel and section 4.3 cannot drift apart.
    vp = np.load(os.path.join(P.OUT, 'vpvs_angle_field.npz'))
    vlat, vlon = vp['lat'], vp['lon']
    gvp = 100.0 * vp['dvp']          # that file holds fractions, not per cent

    panels = [(rlon, rlat, rvs, 'a   REVEAL, S-wave', 'S'),
              (lolon, lolat, lovs, 'b   RevealLO, S-wave', 'S'),
              (glon, glat, gvs, 'c   GLAD-M35, S-wave', 'S'),
              (vlon, vlat, gvp, 'd   GLAD-M35, P-wave', 'P')]
    for _, _, fld, nm, w in panels:
        f = np.isfinite(fld)
        print(f'  {nm:24}: {fld[f].min():+.2f} to {fld[f].max():+.2f} %, '
              f'top decile above {np.percentile(fld[f], PCTL):+.2f} %')

    fig = plt.figure(figsize=(19.0 * F.CM, 14.0 * F.CM))
    proj = ccrs.Robinson(central_longitude=0)
    m = None
    for i, (lon, lat, fld, label, which) in enumerate(panels, 1):
        ax = fig.add_subplot(2, 2, i, projection=proj)
        m = draw(ax, lon, lat, fld, label, contour=(which == 'S'))
        # latitude on the left column, longitude on the bottom row
        F.map_grid(ax, left=i in (1, 3), bottom=i in (3, 4))

    cax = fig.add_axes([0.20, 0.10, 0.33, 0.026])
    cb = fig.colorbar(m, cax=cax, orientation='horizontal',
                      extend='both', ticks=np.arange(-2, 2.1, 1))
    cb.set_label('Velocity anomaly, 410–660 km (per cent)')
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor('#cfcac3')
    # beside the bar, not under it: under it the key sits on the tick labels
    fig.legend(handles=[Line2D([], [], color='#1a1a1a', lw=0.8,
                               label='Top decile, the fast transition zone')],
               loc='center left', bbox_to_anchor=(0.60, 0.113), frameon=False,
               handlelength=1.8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.24,
                        wspace=0.03, hspace=0.20)
    F.check(fig)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(P.FIG, f'fig_tomography.{ext}'), dpi=400)
    print(f'wrote {P.FIG}/fig_tomography.pdf and .png')


if __name__ == '__main__':
    main()
