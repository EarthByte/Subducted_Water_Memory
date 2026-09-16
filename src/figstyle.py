"""One place for the figure style, so that every figure is legible.

Journal figures get reduced, and text that looks adequate on screen at 100 per
cent is unreadable at column width. Nothing here goes below 10 point. Anything
that wants to be smaller than that belongs in the caption instead of on the
figure, which is also where a reader looks for it.
"""
import matplotlib

CM = 1 / 2.54
INK, BLU, ACC, GRY = '#1a1a1a', '#1D6BAA', '#B4442E', '#9a9a9a'
LAND, GRID = '#e9e6e1', '#ebebeb'


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
        items += lg.get_texts()
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
            items += lg.get_texts()
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
    if small:
        raise SystemExit(
            f'\ntext under {floor:.0f} pt once this figure is printed at '
            f'{placed_cm:.1f} cm (drawn at '
            f'{fig.get_size_inches()[0] * 2.54:.1f} cm, so x{k:.2f}):\n'
            + '\n'.join(f'  {sz * k:4.1f} pt printed ({sz:4.1f} drawn)  {txt!r}'
                         for txt, sz in small))
    return True
