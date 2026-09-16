"""The go/no-go diagnostic figure: provinces against model and depth.

Recipe step 1. Rows are provinces sorted by class, columns are the three depth
intervals within each tomographic volume, and the colour is the standardised
score against the province's own continent-matched rotated null. If the pivot is
viable, blocks of one colour run across the whole width of a row; if it is not,
rows change colour between volumes.
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
try:
    import paths as P
    OUT, FIG = getattr(P, 'OUT', 'out'), getattr(P, 'FIG', 'figures')
except Exception:
    OUT, FIG = 'out', 'figures'
import figstyle as F

NULL = os.environ.get('PIVOT_NULL', 'continental')
try:
    from province_pivot import BANDS
except ImportError:
    BANDS = ['350-410', '410-520', '520-660']
F.apply(9.0)

t = pd.read_csv(os.path.join(OUT, f'province_scores_{NULL}.csv'))
c = pd.read_csv(os.path.join(OUT, f'province_classes_{NULL}.csv'))
models = [m for m in ('REVEAL', 'RevealLO', 'GLAD-M35', 'SPiRaL', 'SEMUCB-WM1')
          if m in set(t.model)]
ORDER = ['fast', 'slow', 'layered', 'near-neutral', 'unresolved']
c['k'] = c.cls.map({k: i for i, k in enumerate(ORDER)})
c = c.sort_values(['k', 'S_410-520'], ascending=[True, False])
rows = list(c.province)

M = np.full((len(rows), len(models) * len(BANDS)), np.nan)
for r, p in enumerate(rows):
    for mi, m in enumerate(models):
        for bi, b in enumerate(BANDS):
            v = t[(t.province == p) & (t.model == m) & (t.band == b)].S
            if len(v):
                M[r, mi * len(BANDS) + bi] = v.iloc[0]

fig, ax = plt.subplots(figsize=(17.5 * F.CM, (0.42 * len(rows) + 4.9) * F.CM),
                       constrained_layout=True)
im = ax.imshow(M, aspect='auto', cmap='RdBu_r',
               norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3))
ax.set_yticks(range(len(rows)))
short = {
    'Central Asian Foldbelt (+39, +117)': 'NE Central Asian Foldbelt',
    'Central Asian Foldbelt (+33, +86)': 'Central Asian Foldbelt',
    'Central Asian Foldbelt (+13, +108)': 'SE Central Asian Foldbelt',
    'Andean Basins (-33, -67)': 'S Andean Basins',
    'Andean Basins (-24, -66)': 'Central Andean Basins',
    'Australia (-18, +145)': 'E Australia (18°S)',
    'Australia (-18, +126)': 'NW Australia',
    'Australia (-35, +147)': 'SE Australia',
    'Australia (-34, +135)': 'S Australia',
    'European Orogenic Belt (+48, +9)': 'Central European Orogenic Belt',
    'European Orogenic Belt (+48, -4)': 'W European Orogenic Belt',
    'North American Cordillera (+38, -106)': 'S North American Cordillera',
    'North American Cordillera (+54, -126)': 'Central North American Cordillera',
    'North American Cordillera (+75, -98)': 'N North American Cordillera',
    'Arabian-Nubian Shield (+31, +37)': 'N Arabian-Nubian Shield',
    'Arabian-Nubian Shield (+14, +45)': 'S Arabian-Nubian Shield',
    'Canadian Shield (+65, -112)': 'W Canadian Shield',
    'Canadian Shield (+73, -92)': 'N Canadian Shield',
}
ax.set_yticklabels([f'{short.get(p, p)}  ({int(c[c.province == p].n_fields.iloc[0])})'
                    for p in rows])
ax.set_xticks(range(M.shape[1]))
ax.set_xticklabels(BANDS * len(models), rotation=90)
# Five model names across fifteen columns do not fit on one line at this width,
# so they alternate between two heights. A tick mark under each keeps the name
# tied to its own three columns.
for mi, m in enumerate(models):
    y = -1.15 if mi % 2 == 0 else -2.45
    ax.text(mi * 3 + 1, y, m, ha='center', va='bottom', fontweight='bold')
    ax.plot([mi * 3 - 0.4, mi * 3 + 2.4], [y + 0.05] * 2, '-', color=F.GRY,
            lw=0.9, clip_on=False)
    if mi:
        ax.axvline(mi * 3 - 0.5, color='white', lw=2.4)
# class boundaries
prev, seen = None, []
for r, p in enumerate(rows):
    k = c[c.province == p].cls.iloc[0]
    if prev is not None and k != prev:
        ax.axhline(r - 0.5, color=F.INK, lw=1.0)
    if k not in seen:
        seen.append(k)
        ax.text(M.shape[1] - 0.35, r, k, ha='left', va='top', color=F.INK,
                fontweight='bold', rotation=0)
    prev = k
ax.set_xlim(-0.5, M.shape[1] - 0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.16, extend='both')
cb.set_label('Standardised anomaly against the matched null (S)')
ax.tick_params(length=2)
for s in ('top', 'right', 'left', 'bottom'):
    ax.spines[s].set_visible(False)
F.check(fig)
os.makedirs(FIG, exist_ok=True)
for e in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_province_heatmap.{e}'), bbox_inches='tight',
                facecolor='white')
print(f'wrote {FIG}/fig_province_heatmap  ({len(rows)} provinces x {M.shape[1]} columns)')
