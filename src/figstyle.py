"""One place for the figure style, so that every figure is legible.

Journal figures get reduced, and text that looks adequate on screen at 100 per
cent is unreadable at column width. Nothing here goes below 10 point. Anything
that wants to be smaller than that belongs in the caption instead of on the
figure, which is also where a reader looks for it.
"""
import numpy as np
import matplotlib

CM = 1 / 2.54
INK, BLU, ACC, GRY = '#1a1a1a', '#1D6BAA', '#B4442E', '#9a9a9a'
LAND, GRID = '#e5e5e5', '#ebebeb'   # neutral light grey continents, no warm cast


def apply(base=11.0):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': base,
        'axes.labelsize': base + 1.5,
        'axes.titlesize': base + 1.0,
        'xtick.labelsize': base,
        'ytick.labelsize': base,
        'legend.fontsize': base,
        'figure.titlesize': base + 2.0,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'pdf.fonttype': 42, 'savefig.dpi': 400,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def decimal_ticks(ax, axis='y'):
    """Log-axis ticks written as decimals (0.001, 0.01, 0.1, 1) at full size,
    rather than as powers of ten whose exponent prints below the floor."""
    import matplotlib.ticker as mt
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_formatter(mt.FuncFormatter(lambda v, pos: f'{v:g}'))
    a.set_minor_formatter(mt.NullFormatter())


def _deg(v, pos, neg):
    v = ((v + 180) % 360) - 180 if pos == 'E' else v
    if abs(v) < 1e-9 or (pos == 'E' and abs(abs(v) - 180) < 1e-9):
        return '0°' if abs(v) < 1e-9 else '180°'
    return f'{abs(v):g}°{pos if v > 0 else neg}'


def map_grid(ax, left=True, bottom=True, right=False, top=False, dlon=90, dlat=30,
             pad_pt=3.0):
    """Latitude and longitude on a map: a faint graticule beneath the data and
    labels on the frame. Every map in the paper carries them, and every map
    caption names the projection.

    On a global map the frame is the projection's own outline, so each label is
    placed where its parallel or meridian meets that outline, as ordinary text
    that check() measures with everything else. On a map cut to a rectangle,
    cartopy's own frame labels are used and kept on the axes for check()."""
    import cartopy.crs as ccrs
    import matplotlib.transforms as mtr
    pc = ccrs.PlateCarree()
    ax.gridlines(crs=pc, draw_labels=False, xlocs=np.arange(-180, 181, dlon),
                 ylocs=np.arange(-90 + dlat, 90, dlat), linewidth=0.4,
                 color='#b5b0a9', alpha=0.8, linestyle=(0, (2, 2)), zorder=1.5)
    size = matplotlib.rcParams['xtick.labelsize']
    x0, x1, y0, y1 = ax.get_extent()
    gx0, gx1, gy0, gy1 = ax.projection.x_limits + ax.projection.y_limits
    cropped = (x1 - x0) < 0.98 * (gx1 - gx0) or (y1 - y0) < 0.98 * (gy1 - gy0)
    if cropped:
        gl = ax.gridlines(crs=pc, draw_labels=True, xlocs=np.arange(-180, 181, dlon),
                          ylocs=np.arange(-90 + dlat, 90, dlat), alpha=0.0,
                          x_inline=False, y_inline=False)
        gl.top_labels, gl.right_labels = top, right
        gl.left_labels, gl.bottom_labels = left, bottom
        gl.xlabel_style = gl.ylabel_style = dict(size=size, color=INK)
        gl.xpadding = gl.ypadding = pad_pt
        ax._paper_gridliners = getattr(ax, '_paper_gridliners', []) + [gl]
        return gl
    lon0 = ax.projection.proj4_params.get('lon_0', 0.0)
    fig = ax.figure
    kw = dict(fontsize=size, color=INK, clip_on=False)
    def at(lon, lat):
        return ax.projection.transform_point(lon, lat, pc)
    for lat in np.arange(-90 + dlat, 90, dlat):
        txt = _deg(lat, 'N', 'S')
        if left:
            x, y = at(lon0 - 179.999, lat)
            ax.text(x, y, txt, ha='right', va='center', transform=mtr.offset_copy(
                ax.transData, fig=fig, x=-pad_pt, units='points'), **kw)
        if right:
            x, y = at(lon0 + 179.999, lat)
            ax.text(x, y, txt, ha='left', va='center', transform=mtr.offset_copy(
                ax.transData, fig=fig, x=pad_pt, units='points'), **kw)
    # each meridian once, and none on the seam at the edge of the map
    lons = sorted({((v + 180) % 360) - 180 for v in np.arange(-180, 180, dlon)
                   if abs(abs(((v - lon0 + 180) % 360) - 180) - 180) > 1e-6})
    for lon in lons:
        txt = _deg(lon, 'E', 'W')
        if bottom:
            x, y = at(lon, -89.999)
            ax.text(x, y, txt, ha='center', va='top', transform=mtr.offset_copy(
                ax.transData, fig=fig, y=-pad_pt, units='points'), **kw)
        if top:
            x, y = at(lon, 89.999)
            ax.text(x, y, txt, ha='center', va='bottom', transform=mtr.offset_copy(
                ax.transData, fig=fig, y=pad_pt, units='points'), **kw)
    return None


PLACED_CM = 16.01      # the width the manuscript places a full-width figure at
FLOOR_PT = 8.0         # smallest readable size on the printed page


def _placed(fig):
    """Every visible piece of text with the box it actually occupies.

    This has to happen here rather than on the saved file, because a label drawn
    with a halo reaches the PDF as vector outlines rather than as text and is
    invisible to anything reading the file afterwards. Those are the small
    annotations most likely to be too small or to sit on top of something else,
    so they are exactly the ones that must be measured before saving.
    """
    fig.canvas.draw()                       # extents need a renderer
    r = fig.canvas.get_renderer()
    items = list(fig.texts)
    for lg in getattr(fig, 'legends', []):
        items += lg.get_texts() + [lg.get_title()]
    for ax in fig.get_axes():
        items += [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        # Tick labels are taken by location rather than from get_xticklabels: an
        # axis keeps a label artist for every tick the locator proposed, marked
        # visible, including ones outside the view that are never drawn, and
        # measuring those reports collisions that are not on the figure.
        for _ax, _lim, _loc in ((ax.xaxis, ax.get_xlim(), ax.get_xticks()),
                                (ax.yaxis, ax.get_ylim(), ax.get_yticks())):
            _a, _b = sorted(_lim)
            items += [tk.label1 for tk, v in zip(_ax.get_major_ticks(), _loc)
                      if _a <= v <= _b]
        lg = ax.get_legend()
        if lg is not None:
            items += lg.get_texts() + [lg.get_title()]
        # latitude and longitude labels, which cartopy draws outside ax.texts
        for gl in getattr(ax, '_paper_gridliners', []):
            items += [t for t in gl.label_artists if t.get_visible()]
    out = []
    for t in items:
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
            sz = t.get_fontsize()
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            out.append((t, bb, sz))
    return out


# Text that may legitimately begin in lower case: a panel letter on its own or
# before its label, a quantity written as a symbol, a number or a sign.
_LOWER_OK = ('dVs', 'dVp', 'dln', 'log', 'km', 'p ', 'n ', 'et al')


def _uncapitalised(items):
    """Labels whose first letter is lower case. Every axis label, colour-bar
    label, legend entry and annotation starts with a capital; a panel letter is
    exempt, and so is the label after it only if it too starts with a capital."""
    bad = []
    for t, _, _ in items:
        s = t.get_text().strip()
        if not s or s.startswith('$'):
            continue
        if len(s) == 1 or (len(s) > 2 and s[0].isalpha() and s[1].isspace()
                           and s[0].islower() and t.get_fontweight() in ('bold', 700)):
            s = s[1:].strip()                     # a panel letter, then its label
            if not s:
                continue
        first = next((c for c in s if c.isalpha()), None)
        if first is None or s[0] != first:        # starts with a number or symbol
            continue
        if first.islower() and not s.startswith(_LOWER_OK):
            bad.append(t.get_text()[:48])
    return bad


def _letters_on_lines(fig):
    """Panel letters that a line of the same panel runs through: an axhline at
    zero under the letter is the usual case, and a text-to-text test cannot
    see it."""
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.get_axes():
        for t in ax.texts:
            s = t.get_text().strip()
            if not (len(s) == 1 and s.isalpha() and s.islower()):
                continue
            bb = t.get_window_extent(renderer=r)
            for ln in ax.lines:
                if not ln.get_visible():
                    continue
                xy = ln.get_transform().transform(ln.get_xydata())
                xy = xy[np.isfinite(xy).all(axis=1)]
                if len(xy) < 2:
                    continue
                # densify, so that a segment crossing the letter between two
                # vertices is seen as well as a vertex inside it
                f = np.linspace(0, 1, 60)[:, None]
                xy = np.concatenate([a + (b - a) * f for a, b in zip(xy[:-1], xy[1:])])
                if (((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1)
                     & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1)).any()):
                    bad.append(s)
                    break
    return bad


def check(fig, placed_cm=PLACED_CM, floor=FLOOR_PT, gap=1.5):
    """Fail loudly rather than shipping a figure with unreadable or piled-up text.

    A point size is judged at the width the figure is printed, not the width it
    is drawn. Ten point on a figure 18 cm wide is under nine point in a 16 cm
    column, while the same ten point on a 12 cm figure is over thirteen, so a
    fixed floor on the drawn size passes and fails the wrong figures.
    """
    k = placed_cm / (fig.get_size_inches()[0] * 2.54)
    px = 72.0 / fig.dpi * k                 # display pixels to printed points
    items = _placed(fig)
    small = [(t.get_text()[:34], sz) for t, _, sz in items if sz * k < floor]
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ta, ba, _), (tb, bb, _) = items[i], items[j]
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w * px > gap and h * px > gap:
                hits.append((ta.get_text()[:28], tb.get_text()[:28],
                             min(w, h) * px))
    # Text that runs off the canvas. bbox_inches='tight' usually rescues an
    # overhanging label, but not always, and a label clipped at the page edge is
    # the one defect a reader cannot work around. Require it to fit as drawn.
    fw, fh = fig.canvas.get_width_height()
    over = []
    for t, bb, _ in items:
        d = max(-bb.x0, bb.x1 - fw, -bb.y0, bb.y1 - fh)
        if d * px > 1.0:
            over.append((t.get_text()[:40], d * px))
    if over:
        raise SystemExit(
            '\ntext runs off the canvas of this figure, printed at %.1f cm:\n'
            % placed_cm
            + '\n'.join(f'  {d:4.1f} pt past the edge  {txt!r}'
                         for txt, d in over))
    if hits:
        raise SystemExit(
            '\ntext collides on this figure, printed at %.1f cm:\n' % placed_cm
            + '\n'.join(f'  {d:4.1f} pt  {a!r} / {b!r}' for a, b, d in hits))
    lower = _uncapitalised(items)
    if lower:
        raise SystemExit('\nlabels on this figure start in lower case:\n'
                         + '\n'.join(f'  {txt!r}' for txt in lower))
    crossed = _letters_on_lines(fig)
    if crossed:
        raise SystemExit('\npanel letters with a line through them: '
                         + ', '.join(crossed))
    if small:
        raise SystemExit(
            f'\ntext under {floor:.0f} pt once this figure is printed at '
            f'{placed_cm:.1f} cm (drawn at '
            f'{fig.get_size_inches()[0] * 2.54:.1f} cm, so x{k:.2f}):\n'
            + '\n'.join(f'  {sz * k:4.1f} pt printed ({sz:4.1f} drawn)  {txt!r}'
                         for txt, sz in small))
    return True
