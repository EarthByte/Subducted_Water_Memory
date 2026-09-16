"""The graphical abstract: where the subducted water goes, and for how long.

Elsevier displays the graphical abstract at 13 x 5 cm, so it is drawn at that
width and every label is measured against the 8 pt floor at that size by
figstyle.check(). Two halves:

  left    a cross-section of the argument. A slab carries water down; three
          quarters leaves the slab above 125 km into the asthenosphere, where
          a cratonic keel later passes over it without removing it and a ridge
          melting column reaches it; the remaining quarter stays chemically
          bound and continues to the transition zone, where tomography images
          the cold slab rather than the water.
  right   the three timescales on one logarithmic axis: the slab stops being
          identifiable after 37 Myr, the chemical signature is still there
          after 400 Myr, and ridge melting removes it in 30 Myr.

Every number is one the paper reports (Tables S4, S14, S15); nothing is read
from out/, so this runs anywhere.

    python3 src/fig_graphical_abstract.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import paths as P
import figstyle as F

F.apply(8.5)
INK, ACC, GRY = F.INK, F.ACC, F.GRY
WET = '#1f6f9f'          # water released above 125 km, and the mantle holding it
DEEP = '#7d5ba6'         # water still chemically bound in the slab
SLAB = '#9aa3ab'
KEEL = '#cbb9a6'
WIDE = 13.0              # cm: the width Elsevier displays it at

fig = plt.figure(figsize=(WIDE * F.CM, 6.7 * F.CM))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 2, height_ratios=[0.20, 1.0], width_ratios=[1.30, 1.0],
                      left=0.035, right=0.95, top=0.985, bottom=0.13,
                      hspace=0.06, wspace=0.13)

# ------------------------------------------------------------------ headline
ax_t = fig.add_subplot(gs[0, :]); ax_t.axis('off')
ax_t.text(0.5, 0.70, 'Subducted water outlives the slab that carried it',
          ha='center', va='center', fontsize=11.5, fontweight='bold', color=INK)
ax_t.text(0.5, 0.08, 'released above 125 km, it survives cratonic keels and is erased by ridge melting',
          ha='center', va='center', fontsize=8.8, color=INK)

# --------------------------------------------------------------- the section
# Depth is schematic: the top 125 km is drawn with vertical exaggeration so the
# shallow release, the cratonic keel and the melting column are all legible.
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 100); ax.set_ylim(700, -60); ax.axis('off')
ax.add_patch(Rectangle((0, 0), 100, 700, facecolor='#f5f3f0', edgecolor='none'))
ax.add_patch(Rectangle((0, 410), 100, 250, facecolor='#e2e6ea', edgecolor='none'))
ax.text(97, 628, 'transition zone', ha='right', va='center', fontsize=8.2, color='#70757b')

# the mantle that holds the released water
ax.add_patch(Rectangle((14, 140), 86, 110, facecolor=WET, alpha=0.26, edgecolor='none'))

# the surface, with the trench at the left and a ridge at the right
ax.add_patch(Polygon([(1, 0), (10, 0), (46, 480), (37, 480)], closed=True,
                     facecolor=SLAB, edgecolor='none', zorder=3))
ax.plot([0, 100], [0, 0], color=INK, lw=1.3, zorder=6)
ax.text(0.5, -30, 'trench', fontsize=8.4, color=INK, ha='left', va='center')
ax.plot([88, 93, 98], [0, -16, 0], color=INK, lw=1.3, zorder=6)
ax.text(85, -30, 'ridge', fontsize=8.4, color=INK, ha='right', va='center')
for x0, x1 in ((89.5, 91.0), (93.0, 93.0), (96.5, 95.0)):
    ax.annotate('', xy=(x0, 8), xytext=(x1, 148), zorder=6,
                arrowprops=dict(arrowstyle='-|>,head_width=0.17,head_length=0.4',
                                lw=1.2, color=ACC))

# three quarters released above 125 km
ax.plot([2, 46], [170, 170], color=INK, lw=0.7, ls=(0, (3.5, 3)), zorder=4)
ax.text(29, 182, '125 km', fontsize=8.2, color=INK, ha='left', va='top', zorder=6)
ax.annotate('', xy=(34, 148), xytext=(20, 200), zorder=5,
            arrowprops=dict(arrowstyle='-|>,head_width=0.30,head_length=0.55',
                            lw=3.4, color=WET, shrinkA=0, shrinkB=0))
ax.text(6, 72, '¾ released\nabove 125 km', fontsize=8.8, color=WET,
        fontweight='bold', ha='left', va='center', linespacing=1.25, zorder=6)

# the cratonic keel, passing over the hydrated mantle
ax.add_patch(Polygon([(50, 0), (74, 0), (69, 250), (55, 250)], closed=True,
                     facecolor=KEEL, edgecolor=INK, lw=0.7, zorder=5))
ax.text(62, 125, 'cratonic\nkeel', ha='center', va='center', fontsize=8.6,
        color='#544433', linespacing=1.25, zorder=6)
ax.annotate('', xy=(86, 298), xytext=(56, 298), zorder=6,
            arrowprops=dict(arrowstyle='-|>,head_width=0.2,head_length=0.45',
                            lw=1.3, color=INK))
ax.text(71, 348, 'signature survives', fontsize=8.6, color=INK,
        ha='center', va='center', zorder=6)

# the quarter that stays chemically bound
ax.annotate('', xy=(44, 470), xytext=(32, 310), zorder=5,
            arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.55',
                            lw=1.9, color=DEEP, shrinkA=0, shrinkB=0))
ax.text(46, 430, '¼ stays bound', fontsize=8.6, color=DEEP,
        ha='left', va='center', zorder=6)
ax.text(56, 540, 'tomography sees\nthe cold slab', fontsize=8.4, color='#5a6067',
        ha='left', va='center', linespacing=1.25)

# ------------------------------------------------------ the three timescales
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_xscale('log'); ax2.set_xlim(9, 1000); ax2.set_ylim(-0.55, 2.9)
ax2.set_yticks([]); ax2.spines['left'].set_visible(False)
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)
ax2.set_xticks([10, 30, 100, 300])
ax2.set_xticklabels(['10', '30', '100', '300'])
ax2.set_xlabel('time since delivery (Myr)', fontsize=8.8, labelpad=2)
ax2.tick_params(labelsize=8.5, pad=2)

for y, x1, c, label, tag, openend in (
        (2.25, 37, SLAB, 'the slab stays visible in tomography', '37 Myr', False),
        (1.15, 1000, WET, 'the chemical signature persists', '≥ 400 Myr', True),
        (0.05, 30, ACC, 'ridge melting removes it', '30 Myr', False)):
    ax2.add_patch(Rectangle((9, y - 0.17), x1 - 9, 0.34, facecolor=c,
                            alpha=0.9, edgecolor='none', clip_on=True))
    ax2.text(9.4, y + 0.36, label, fontsize=8.8, color=INK, ha='left', va='bottom')
    if openend:
        ax2.text(260, y, tag, fontsize=9.4, color='white', fontweight='bold',
                 ha='center', va='center')
    else:
        ax2.text(x1 * 1.15, y, tag, fontsize=9.4, fontweight='bold',
                 color='#6b747c' if c == SLAB else c, ha='left', va='center')

F.check(fig, placed_cm=WIDE)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_graphical_abstract.{ext}'),
                dpi=400, bbox_inches='tight', facecolor='white')
print(f'wrote figures/fig_graphical_abstract.pdf and .png  ({WIDE:.0f} cm wide)')
