"""Two drawings of the same argument: the graphical abstract, and Figure 10.

The paper's last figure is a section through the argument rather than a result:
a slab carries water down, three quarters of it leaves the slab above 125 km
into mantle that a cratonic keel later passes over without removing it and that
a ridge melting column reaches, and the remaining quarter stays chemically bound
and goes on to the transition zone, where tomography images the cold slab rather
than the water. Beside it, the three timescales on one logarithmic axis.

The graphical abstract is the same drawing at the 13 cm Elsevier display width,
with the headline Elsevier asks for; the figure is drawn at the manuscript width
without it, because a figure in this paper carries no title and its explanation
belongs in the caption. Both come from this file so that the two cannot drift
apart. Every number is one the paper reports (Tables S4, S14, S15); nothing is
read from out/, so this runs anywhere.

    python3 src/fig_schematic.py              # both files
    python3 src/fig_schematic.py --only figure
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import paths as P
import figstyle as F

INK, ACC, GRY = F.INK, F.ACC, F.GRY
WET = '#1f6f9f'          # water released above 125 km, and the mantle holding it
DEEP = '#7d5ba6'         # water still chemically bound in the slab
# One grey only, and it is the slab: the same grey carries the 37 Myr bar of
# panel (b), so the timescale and the object it belongs to cannot be confused.
# The keel and the transition zone take tints of their own, chosen to stay
# distinguishable in the common forms of colour blindness: a muted green and a
# pale yellow differ in hue from the blue of the water and in lightness from
# each other and from the slab.
SLAB = '#7f8c99'         # the cold slab, and the 37 Myr bar
KEEL = '#b7c4ad'         # the cratonic keel
BACK = '#f4f4f4'         # the mantle behind the section
TZONE = '#f4e3b2'        # the transition zone, as a background band
TZEDGE = '#c9932a'       # an outline and a label colour for it, the same hue


def over(colour, alpha, back):
    """The colour a translucent patch actually shows on the section, so that a
    bar in panel (b) can carry exactly the colour of the band in panel (a)."""
    import matplotlib.colors as mc
    c, b = mc.to_rgb(colour), mc.to_rgb(back)
    return mc.to_hex(tuple(alpha * x + (1 - alpha) * y for x, y in zip(c, b)))


def section(ax, fs, compact=False):
    """The cross-section. Depth is schematic: the top 125 km is drawn with
    vertical exaggeration so that the shallow release, the cratonic keel and the
    melting column are all legible."""
    ax.set_xlim(0, 100); ax.set_ylim(700, -60); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 100, 700, facecolor=BACK, edgecolor='none'))
    ax.add_patch(Rectangle((0, 410), 100, 250, facecolor=TZONE, edgecolor='none'))
    ax.text(97, 628, 'Transition zone', ha='right', va='center', fontsize=fs, color=TZEDGE)

    # The mantle that holds the released water: on this axis the dashed line at
    # 170 is 125 km, so the band lies mostly above it and its base is well above
    # the base of the keel, which is what the argument requires.
    ax.add_patch(Rectangle((14, 55), 86, 145, facecolor=WET, alpha=0.26, edgecolor='none'))

    # the surface, with the trench at the left and a ridge at the right
    ax.add_patch(Polygon([(1, 0), (10, 0), (46, 480), (37, 480)], closed=True,
                         facecolor=SLAB, edgecolor='none', zorder=3))
    ax.text(30, 340, 'Slab', fontsize=fs, color='white', fontweight='bold',
            ha='center', va='center', rotation=-52, zorder=4)
    ax.plot([0, 100], [0, 0], color=INK, lw=1.3, zorder=6)
    ax.text(0.5, -30, 'Trench', fontsize=fs, color=INK, ha='left', va='center')
    ax.plot([88, 93, 98], [0, -16, 0], color=INK, lw=1.3, zorder=6)
    ax.text(85, -30, 'Ridge', fontsize=fs, color=INK, ha='right', va='center')
    for x0, x1 in ((89.5, 91.0), (93.0, 93.0), (96.5, 95.0)):
        ax.annotate('', xy=(x0, 8), xytext=(x1, 148), zorder=6,
                    arrowprops=dict(arrowstyle='-|>,head_width=0.17,head_length=0.4',
                                    lw=1.2, color=ACC))

    # three quarters released above 125 km
    ax.plot([2, 46], [170, 170], color=INK, lw=0.7, ls=(0, (3.5, 3)), zorder=4)
    ax.text(29, 182, '125 km', fontsize=fs, color=INK, ha='left', va='top', zorder=6)
    ax.annotate('', xy=(38, 130), xytext=(22, 205), zorder=5,
                arrowprops=dict(arrowstyle='-|>,head_width=0.30,head_length=0.55',
                                lw=3.4, color=WET, shrinkA=0, shrinkB=0))
    released = '¾ released\nabove 125 km' if compact else 'Three quarters\nreleased above\n125 km'   # the blue band
    ax.text(2, 95, released, fontsize=fs, color=WET,
            fontweight='bold', ha='left', va='center', linespacing=1.25, zorder=6)

    # the cratonic keel, passing over the hydrated mantle
    # the keel reaches 200 km, which on this exaggerated depth axis is 270
    ax.add_patch(Polygon([(50, 0), (74, 0), (69, 270), (55, 270)], closed=True,
                         facecolor=KEEL, edgecolor=INK, lw=0.7, zorder=5))
    ax.text(62, 135, 'Cratonic\nkeel', ha='center', va='center', fontsize=fs,
            color='#3c4a36', linespacing=1.25, zorder=6)
    ax.annotate('', xy=(86, 298), xytext=(56, 298), zorder=6,
                arrowprops=dict(arrowstyle='-|>,head_width=0.2,head_length=0.45',
                                lw=1.3, color=INK))
    ax.text(71, 348, 'Signature survives', fontsize=fs, color=INK,
            ha='center', va='center', zorder=6)

    # the quarter that stays chemically bound
    ax.annotate('', xy=(44, 470), xytext=(32, 310), zorder=5,
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.55',
                                lw=1.9, color=DEEP, shrinkA=0, shrinkB=0))
    bound = '¼ stays bound' if compact else 'One quarter stays bound'
    ax.text(46, 430, bound, fontsize=fs, color=DEEP,
            ha='left', va='center', zorder=6)
    ax.text(56, 540, 'Tomography sees\nthe cold slab', fontsize=fs, color=TZEDGE,
            ha='left', va='center', linespacing=1.25)


def timescales(ax, fs, compact=False):
    """The three timescales on one logarithmic axis."""
    ax.set_xscale('log'); ax.set_xlim(9, 1000); ax.set_ylim(-0.55, 2.9)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels(['10', '30', '100', '300'])
    ax.set_xlabel('Time since delivery (Myr)', fontsize=fs + 0.4, labelpad=2)
    ax.tick_params(labelsize=fs, pad=2)
    # The 37 Myr is the visibility of the slab in the transition zone, so its bar
    # is filled with the colour of the transition zone itself, not a darker
    # version of it; an outline of the same hue keeps a pale fill legible on
    # white, and the number beside it carries that outline colour.
    for y, x1, c, edge, textc, label, tag, openend in (
            (2.25, 37, TZONE, TZEDGE, TZEDGE,
             'The slab stays visible in tomography' if compact
             else 'The slab stays visible in the\ntransition zone', '37 Myr', False),
            (1.15, 1000, over(WET, 0.26, BACK), WET, WET,
             'The chemical signature persists', '≥ 400 Myr', True),
            (0.05, 30, ACC, ACC, ACC, 'Ridge melting removes it', '30 Myr', False)):
        ax.add_patch(Rectangle((9, y - 0.17), x1 - 9, 0.34, facecolor=c,
                               alpha=0.9, edgecolor=edge, lw=0.8, clip_on=True))
        ax.text(9.4, y + 0.36, label, fontsize=fs, color=INK, ha='left', va='bottom',
                linespacing=1.2)
        if openend:
            ax.text(260, y, tag, fontsize=fs + 0.6, color=textc, fontweight='bold',
                    ha='center', va='center')
        else:
            ax.text(x1 * 1.15, y, tag, fontsize=fs + 0.6, fontweight='bold',
                    color=textc, ha='left', va='center')   # the tag matches its bar


def abstract(out):
    """The graphical abstract: 13 cm, with the headline Elsevier asks for."""
    F.apply(8.5)
    wide, fs = 13.0, 8.5
    fig = plt.figure(figsize=(wide * F.CM, 6.7 * F.CM))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, height_ratios=[0.20, 1.0], width_ratios=[1.30, 1.0],
                          left=0.035, right=0.95, top=0.985, bottom=0.13,
                          hspace=0.06, wspace=0.13)
    ax_t = fig.add_subplot(gs[0, :]); ax_t.axis('off')
    ax_t.text(0.5, 0.70, 'Subducted water outlives the slab that carried it',
              ha='center', va='center', fontsize=11.5, fontweight='bold', color=INK)
    ax_t.text(0.5, 0.08,
              'Released above 125 km, it survives cratonic keels and is erased by ridge melting',
              ha='center', va='center', fontsize=8.8, color=INK)
    section(fig.add_subplot(gs[1, 0]), fs, compact=True)
    timescales(fig.add_subplot(gs[1, 1]), fs, compact=True)
    F.check(fig, placed_cm=wide)
    save(fig, 'fig_graphical_abstract', wide, out)


def figure(out):
    """Figure 10: the same drawing at the manuscript width, no headline, panel
    letters, and the explanation left to the caption."""
    F.apply(11.0)
    wide, fs = 16.0, 10.0
    fig = plt.figure(figsize=(wide * F.CM, 8.4 * F.CM))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0],
                          left=0.035, right=0.96, top=0.93, bottom=0.14, wspace=0.13)
    a = fig.add_subplot(gs[0, 0]); section(a, fs)
    b = fig.add_subplot(gs[0, 1]); timescales(b, fs)
    for ax, letter in ((a, 'a'), (b, 'b')):
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=13,
                fontweight='bold', va='bottom', ha='right')
    F.check(fig, placed_cm=wide)
    save(fig, 'fig_schematic', wide, out)


def save(fig, stem, wide, out):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{stem}.{ext}'), dpi=400,
                    bbox_inches='tight', facecolor='white')
    print(f'wrote figures/{stem}.pdf and .png  ({wide:.0f} cm wide)')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('abstract', 'figure'), default=None)
    a = ap.parse_args()
    if a.only in (None, 'abstract'):
        abstract(P.FIG)
    if a.only in (None, 'figure'):
        figure(P.FIG)
