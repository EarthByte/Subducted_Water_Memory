# -*- coding: utf-8 -*-
"""The depth of the 660 over reconstructed water delivery, with the inverted water beside it.

Three panels. (a) the depth anomaly of the 660 km discontinuity, with the outline of recent
reconstructed delivery over it, which is the comparison the paper makes. (b) the enrichment of a
deep discontinuity above delivery against the age of that delivery, for the three published maps,
with the fitted decline on the one that carries it. (c) the water content inferred from the joint
inversion with shear velocity, as the cross-check: its global mean against the published value and
its own enrichment curve, drawn for the five velocity models so that the spread is the statement.

No titles on the panels; the descriptive text belongs in the caption. Lettering follows the EPSL
guidance through `figstyle`, and `figstyle.check` refuses text below the readable floor or labels
that collide, so the figure is not checked by eye.

    python3 src/fig_d660_delivery.py
    python3 src/fig_d660_delivery.py --model REVEAL --d660 waszek
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:                                            # pragma: no cover
    OUT, FIG = 'out', 'figures'
import figstyle as F
from water_decay import DECILE, enrichment, model as decay_model, top_decile
from water_inversion import (MODELS, area_weights, invert, read_d660, read_vs, smooth)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:                                          # pragma: no cover
    raise SystemExit('\nthis figure needs cartopy\n')

PRODUCTS = ('waszek', 'dai', 'huang')
LABELS = {'waszek': 'Waszek et al. (2021)', 'dai': 'Dai et al. (2026)',
          'huang': 'Huang et al. (2019)'}
PUBLISHED_MEAN = 0.14            # wt%, the global mean of Liu et al. (2026)


def curves(model_name, product, delivery_by_age, abins, glat, glon, w, which):
    """Enrichment against delivery age for one field, and the fitted decline."""
    glat_, glon_, dd, derr, _ = read_d660(product)
    _, _, dvs, band = read_vs(model_name)
    dT, W = invert(dvs, dd)
    solvable = np.isfinite(W) & (dT > -600) & (dT < 400)
    field = {'d660': smooth(dd, glat_),
             'water': smooth(np.where(solvable, np.clip(W, 0.0, None), np.nan), glat_)}[which]
    mask = top_decile(field, w, DECILE)
    t, e, s = [], [], []
    for k, (t0, t1) in enumerate(abins):
        val, _, sig = enrichment(mask, np.nan_to_num(smooth(delivery_by_age[k], glat_)), w)
        t.append(0.5 * (t0 + t1))
        e.append(val)
        s.append(max(sig, 1e-3))
    return (np.asarray(t, float), np.asarray(e, float), band, field, mask,
            np.asarray(s, float))


def fit(t, e, sigma):
    """The same weighted, bounded fit that `water_decay.py` reports, so the two agree."""
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(decay_model, t, e, p0=[max(e[0], 0.1), max(e[-1], 0.1), 60.0],
                            sigma=sigma, absolute_sigma=True,
                            bounds=([0, 0, 5], [10, 3, 500]), maxfev=40000)
        return popt
    except Exception:
        return None


def main():
    """Draw the three panels of the discontinuity result and check them before writing.

    Panel a is the depth anomaly with recent delivery outlined, b the enrichment curve for
    the three published maps, c the same for the water inferred with each velocity model.
    The fitted time constant is printed rather than written on the artwork, where it would
    sit on another model's curve.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='REVEAL', help='the model drawn in panel a and c')
    ap.add_argument('--d660', default='waszek', help='the map drawn in panel a')
    a = ap.parse_args()
    F.apply(9.5)

    z = np.load(os.path.join(OUT, 'water_forward.npz'), allow_pickle=True)
    glat, glon = z['lat'], z['lon']
    by_age, abins = np.asarray(z['by_age']), np.asarray(z['abins'], float)
    recent = np.asarray(by_age)[0] + np.asarray(by_age)[1]
    w = area_weights(glat, glon)

    t_d, e_d, band, dd_field, dd_mask, s_d = curves(a.model, a.d660, by_age, abins,
                                                   glat, glon, w, 'd660')

    # Each line panel carries its key on one row in the space above it, rather than inside
    # the axes where it sat on the curves it names. That space is what the taller figure and
    # the wider gap between panels are for.
    fig = plt.figure(figsize=(16.0 * F.CM, 19.4 * F.CM))
    gs = fig.add_gridspec(3, 1, height_ratios=(1.8, 1.0, 1.0), hspace=0.58)

    # (a) the discontinuity, with recent delivery outlined on it
    ax = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    lim = float(np.nanpercentile(np.abs(dd_field), 98))
    m = ax.pcolormesh(glon, glat, dd_field, transform=ccrs.PlateCarree(), cmap='RdBu',
                      vmin=-lim, vmax=lim, shading='auto', rasterized=True)
    ax.contour(glon, glat, (recent > 0).astype(float), levels=[0.5], colors=[F.ACC],
               linewidths=0.8, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#555555')
    ax.set_global()
    cb = fig.colorbar(m, ax=ax, orientation='horizontal', pad=0.02, fraction=0.048,
                      aspect=46, shrink=0.86)
    cb.set_label('Depth anomaly of the 660 km discontinuity (km)')
    cb.outline.set_linewidth(0.6)
    ax.text(0.012, 0.97, '(a)', transform=ax.transAxes, va='top', ha='left',
            fontweight='bold')

    # (b) the discontinuity enrichment, for the three maps
    ax = fig.add_subplot(gs[1])
    for product, colour in zip(PRODUCTS, (F.ACC, '#4b6584', '#b07b2b')):
        t, e, _, _, _, s = curves(a.model, product, by_age, abins, glat, glon, w, 'd660')
        ax.plot(t, e, 'o-', ms=3.2, lw=1.0, color=colour, label=LABELS[product])
        p = fit(t, e, s)
        if p is not None and product == a.d660:
            # the fitted decline is drawn, and its time constant quoted in the caption rather
            # than written on the artwork, where it would sit on another model's curve
            tt = np.linspace(t.min(), t.max(), 200)
            ax.plot(tt, decay_model(tt, *p), lw=0.9, color=colour, alpha=0.45)
            print('   fitted decline on {}: {:.0f} Myr to a floor of {:.2f}'.format(
                LABELS[product], p[2], p[1]))
    ax.axhline(1.0, lw=0.7, color='#888888')
    ax.set_xlabel('Age of reconstructed delivery (Ma)')
    ax.set_ylabel('Enrichment')
    F.legend_clear(ax, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.015),
                   ncol=3, columnspacing=1.4, handletextpad=0.5)
    ax.text(0.012, 0.97, '(b)', transform=ax.transAxes, va='top', ha='left',
            fontweight='bold')

    # (c) the inverted water as the cross-check, five models
    ax = fig.add_subplot(gs[2])
    means = []
    for model_name, colour in zip(MODELS, (F.ACC, '#4b6584', '#b07b2b', '#6b7f3a', '#7a4b6b')):
        t, e, _, field, _, _ = curves(model_name, a.d660, by_age, abins, glat, glon,
                                      w, 'water')
        ax.plot(t, e, 'o-', ms=3.0, lw=0.9, color=colour, label=model_name)
        means.append(float(np.nansum(field * w) / np.nansum(w * np.isfinite(field))))
    ax.axhline(1.0, lw=0.7, color='#888888')
    ax.set_xlabel('Age of reconstructed delivery (Ma)')
    ax.set_ylabel('Enrichment')
    # five models, so the row is tighter than the one above it
    F.legend_clear(ax, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.015),
                   ncol=5, columnspacing=1.0, handletextpad=0.4)
    ax.text(0.012, 0.97, '(c)', transform=ax.transAxes, va='top', ha='left',
            fontweight='bold')

    fig.subplots_adjust(left=0.115, right=0.98, top=0.965, bottom=0.10)
    F.check(fig)
    os.makedirs(FIG, exist_ok=True)
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, 'fig_d660_delivery.' + e), dpi=400,
                    bbox_inches='tight', facecolor='white')
    print('wrote {}/fig_d660_delivery, band {}, inferred water {:.2f} to {:.2f} wt% across the '
          'five models against {:.2f} published'.format(FIG, band, min(means), max(means),
                                                        PUBLISHED_MEAN))


if __name__ == '__main__':
    main()
