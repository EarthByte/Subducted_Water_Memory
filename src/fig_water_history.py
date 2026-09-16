"""The subducted-water history: where water entered the transition zone, and when.

This is the initial-condition half of the argument. A plate reconstruction says
how much water went down at each trench and in which epoch; `water_forward.py`
displaces each parcel down dip and accumulates it by delivery age. What the
reconstruction cannot say is what happened next, and the figure makes the gap
visible by drawing the observed fast transition zone over each panel: young water
coincides with it, old water does not, and most of the water is old.

Four panels rather than the eight age bands the file carries, because eight
world maps at column width are unreadable and the grouping is what the argument
turns on: inside the memory window, just outside it, and long past it.

    python3 src/fig_water_history.py
    python3 src/fig_water_history.py --model REVEAL
"""
import argparse, os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:
    OUT, FIG = 'out', 'figures'
import figstyle as F
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    raise SystemExit('\nthis figure needs cartopy\n')

GROUPS = [((0, 1), '8–50 Ma', 'inside the memory window'),
          ((2, 3), '50–100 Ma', 'just outside it'),
          ((4, 5), '100–200 Ma', 'long past it'),
          ((6, 7), '200–400 Ma', 'long past it')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=os.path.join(OUT, 'water_forward.npz'))
    a = ap.parse_args()
    F.apply(9.5)
    z = np.load(a.npz)
    lon, lat, by, fast = z['lon'], z['lat'], z['by_age'], z['fast']
    tot = np.array([b.sum() for b in by])

    fields, labels = [], []
    for (i0, i1), lab, note in GROUPS:
        fields.append(by[i0] + by[i1])
        labels.append((lab, note, (tot[i0] + tot[i1]) / tot.sum()))
    # rescaled so the colour bar carries readable decade labels: the raw model
    # units run to about 0.05 and their decades print as superscripts too small
    # to read at column width
    scale = 100.0 / max(f.max() for f in fields)
    fields = [f * scale for f in fields]
    hi = 100.0
    lo = hi / 3e3

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(19.0 * F.CM, 12.6 * F.CM))
    for k, (fld, (lab, note, share)) in enumerate(zip(fields, labels), 1):
        ax = fig.add_subplot(2, 2, k, projection=ccrs.Robinson(central_longitude=160))
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='#eeebe7', zorder=0)
        m = ax.pcolormesh(lon, lat, np.ma.masked_less(fld, lo), transform=pc,
                          cmap='YlGnBu', norm=LogNorm(vmin=lo, vmax=hi),
                          shading='nearest', zorder=2, rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#8f8a84', zorder=3)
        ax.contour(lon, lat, fast.astype(float), levels=[0.5], transform=pc,
                   colors=[F.ACC], linewidths=0.8, zorder=4)
        ax.text(0.0, 1.035, f'{"abcd"[k - 1]}   {lab}', transform=ax.transAxes,
                ha='left', va='bottom', fontweight='bold',
                fontsize=matplotlib.rcParams['font.size'] + 1.0)
        ax.text(1.0, 1.035, f'{100 * share:.0f}% of the water', transform=ax.transAxes,
                ha='right', va='bottom', color=F.GRY,
                fontsize=matplotlib.rcParams['font.size'])
        ax.spines['geo'].set_edgecolor('#cfcac3'); ax.spines['geo'].set_linewidth(0.6)

    cax = fig.add_axes([0.20, 0.085, 0.33, 0.024])
    # plain decimal ticks: matplotlib sets a log exponent as a superscript about
    # three fifths the size of the mantissa, which prints below the legibility floor
    import matplotlib.ticker as mt
    cb = fig.colorbar(m, cax=cax, orientation='horizontal', extend='both')
    cb.set_ticks([0.1, 1, 10, 100])
    cb.ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v))
    cb.ax.xaxis.set_minor_formatter(mt.NullFormatter())
    cb.set_label('Water delivered to the transition zone (per cent of the maximum)')
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor('#cfcac3')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=F.ACC, lw=1.0,
                               label='observed fast transition zone')],
               loc='center left', bbox_to_anchor=(0.60, 0.097), frameon=False,
               handlelength=1.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.19,
                        wspace=0.03, hspace=0.20)
    F.check(fig)
    os.makedirs(FIG, exist_ok=True)
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'fig_water_history.{e}'), dpi=400,
                    bbox_inches='tight', facecolor='white')
    print(f'wrote {FIG}/fig_water_history  '
          + ', '.join(f'{l}: {100 * s:.0f}%' for l, _, s in labels))


if __name__ == '__main__':
    main()
