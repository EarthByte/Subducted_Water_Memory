"""Check figures for text that will be unreadable in print, and for labels that collide.

A point size in a PDF means nothing on its own. A panel label set at 11 point in
a figure 24 cm wide is 8.6 point once that figure is placed in a 19 cm column, so
every size here is scaled by the ratio of the printed width to the figure's own
width before it is judged. Give the width the figure is actually placed at:

    python3 figcheck.py figures/*.pdf

The default width is the one the manuscript places a full-width figure at. Text
drawn with a halo does not reach the PDF as text at all, and nothing reading the
file afterwards can see it, so matplotlib figures are checked before saving as
well, by figstyle.check.

Collisions are found by reconstructing the text-drawing operations rather than by
grouping characters that happen to sit near each other, because two tick labels
printed on top of one another are a single cluster to any proximity rule, and
that is the case most worth catching. Every box is built from the glyph origin
and its advance width in the frame the text was set in, so labels around an arc
are compared as the slanted rectangles they are; comparing the upright boxes
drawn around them reports a whole arc of tick labels as overlapping when they are
merely close.
"""
import argparse, glob, math, os, sys
try:
    import pdfplumber
except ModuleNotFoundError:                 # the figures are drawn without it
    raise SystemExit('\nfigcheck needs pdfplumber, which is not installed here.\n'
                     'The figures were written; they have not been checked.\n\n'
                     '    python3 -m pip install pdfplumber\n')

PLACED = 16.01  # centimetres a full-width figure occupies in the manuscript
FLOOR = 8.0     # below this, printed, a reader needs a magnifier
ASC, DESC = 0.79, 0.21          # em fractions bounding a line of type


def _geom(page):
    """Every character as an origin, a direction and an advance, in page space.

    pdfplumber reports the upright box drawn around a character, which for a
    rotated one is larger than the character and in a different place. The text
    matrix carries where the glyph actually sits, so it is used for all of them.

    Its reported size is an upright box dimension too, so it is only the point
    size when the text is level. A label turned on its side reads as its own
    advance width, and one set around an arc reads about a seventh too large.
    Both are recovered from the box, the advance and the angle, which is what
    makes a size comparable between a figure drawn by GMT and one by matplotlib:

        w = adv |cos t| + S |sin t|,   h = adv |sin t| + S |cos t|
    """
    H = page.height + page.bbox[1]
    out = []
    for c in page.chars:
        if c['text'].strip() == '':
            out.append(None)
            continue
        m = c['matrix']
        th = math.atan2(m[1], m[0])
        ct, st = abs(math.cos(th)), abs(math.sin(th))
        if ct >= st:
            size = (c['height'] - c['adv'] * st) / ct
        else:
            size = (c['width'] - c['adv'] * ct) / st
        if not (0.1 < size < 400):          # a glyph box that makes no sense
            size = c['size']
        out.append(dict(text=c['text'], size=size, adv=c['adv'], th=th,
                        ox=m[4] - page.bbox[0], oy=H - m[5],
                        font=c['fontname'],
                        u=(math.cos(th), -math.sin(th)),
                        v=(-math.sin(th), -math.cos(th))))
    return out


def _box(chars):
    """One oriented rectangle around a sequence of characters on a line."""
    a, z = chars[0], chars[-1]
    L = math.hypot(z['ox'] - a['ox'], z['oy'] - a['oy']) + z['adv']
    size = max(c['size'] for c in chars)
    u, v = a['u'], a['v']
    return dict(text=''.join(c['text'] for c in chars), size=size,
                sizes=[c['size'] for c in chars], th=a['th'],
                ox=a['ox'], oy=a['oy'], L=L, u=u, v=v,
                lo=-DESC * size, hi=ASC * size)


def _corners(b):
    u, v = b['u'], b['v']
    return [(b['ox'] + t * u[0] + s * v[0], b['oy'] + t * u[1] + s * v[1])
            for t, s in ((0, b['lo']), (b['L'], b['lo']),
                         (b['L'], b['hi']), (0, b['hi']))]


def _overlap(a, b):
    """How deeply two oriented rectangles interpenetrate, in points."""
    pa, pb = _corners(a), _corners(b)
    best = 1e9
    for poly in (pa, pb):
        for i in range(4):
            ex = poly[(i + 1) % 4][0] - poly[i][0]
            ey = poly[(i + 1) % 4][1] - poly[i][1]
            n = math.hypot(ex, ey)
            if n == 0:
                continue
            ax, ay = -ey / n, ex / n
            sa = [p[0] * ax + p[1] * ay for p in pa]
            sb = [p[0] * ax + p[1] * ay for p in pb]
            d = min(max(sa), max(sb)) - max(min(sa), min(sb))
            if d <= 0:
                return 0.0
            best = min(best, d)
    return best


def _runs(chars):
    """Split the characters into the operations they were drawn as."""
    out, cur = [], []
    for c in chars:
        if c is None:
            if cur:
                out.append(_box(cur)); cur = []
            continue
        if cur:
            p = cur[-1]
            dx, dy = c['ox'] - p['ox'], c['oy'] - p['oy']
            along = dx * p['u'][0] + dy * p['u'][1]
            across = dx * p['v'][0] + dy * p['v'][1]
            same = (c['font'] == p['font'] and abs(c['size'] - p['size']) < 0.01
                    and abs(c['th'] - p['th']) < 0.01)
            # A step across the line starts a new label. Distance alone does not
            # say so: three depth labels stacked down the slanted edge of a wedge
            # sit closer together than the width of one of them, and reading them
            # as a single run is how a collision between them goes unreported.
            if (not same or abs(across) > 0.30 * p['size']
                    or not -0.30 * p['size'] <= along <= 1.9 * p['size']):
                out.append(_box(cur)); cur = []
        cur.append(c)
    if cur:
        out.append(_box(cur))
    return _stitch(out)


def _stitch(rs):
    """Rejoin the fragments of one label.

    A minus sign, a degree sign and a subscript come from a different font than
    the digits beside them, so a single tick label arrives as three drawing
    operations. Left apart they are reported as colliding with each other, which
    is noise. Fragments are rejoined where they sit on one line with a gap no
    wider than a space; two labels that genuinely collide have a negative gap and
    are never merged.
    """
    out = []
    for r in rs:
        if out:
            p = out[-1]
            if abs(p['th'] - r['th']) < 0.01:
                dx, dy = r['ox'] - p['ox'], r['oy'] - p['oy']
                t = dx * p['u'][0] + dy * p['u'][1]      # along the line
                s = dx * p['v'][0] + dy * p['v'][1]      # across it
                sz = max(p['size'], r['size'])
                if -0.05 * sz <= t - p['L'] <= 0.95 * sz and abs(s) < 0.42 * sz:
                    p['L'] = max(p['L'], t + r['L'])
                    p['lo'] = min(p['lo'], s - DESC * r['size'])
                    p['hi'] = max(p['hi'], s + ASC * r['size'])
                    p['text'] += r['text']
                    p['sizes'] += r['sizes']
                    p['size'] = max(p['size'], r['size'])
                    continue
        out.append(r)
    return out


def check(path, target_cm, floor):
    res = []
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            native = pg.width * 2.54 / 72.0
            k = target_cm / native
            rs = _runs(_geom(pg))
            small = [r for r in rs if r['size'] * k < floor]
            clash = []
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    a, b = rs[i], rs[j]
                    if abs(a['ox'] - b['ox']) > 80 or abs(a['oy'] - b['oy']) > 80:
                        continue
                    if (a['text'] == b['text']
                            and math.hypot(a['ox'] - b['ox'], a['oy'] - b['oy']) < 2):
                        continue            # the same label drawn twice, a halo
                    d = _overlap(a, b)
                    if d > 1.0:
                        clash.append((a, b, d))
            res.append(dict(native=native, k=k, runs=rs, small=small, clash=clash))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--width', type=float, default=PLACED,
                    help='centimetres the figure is printed at')
    ap.add_argument('--floor', type=float, default=FLOOR)
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    files = [f for pat in a.files for f in sorted(glob.glob(pat))]
    if not files:
        raise SystemExit('no figures matched')
    bad = 0
    for f in files:
        for n, r in enumerate(check(f, a.width, a.floor), 1):
            name = os.path.basename(f) + (f' p{n}' if n > 1 else '')
            sizes = [x['size'] * r['k'] for x in r['runs']]
            tiny = min((min(x['sizes']) for x in r['runs']), default=0) * r['k']
            ok = not (r['small'] or r['clash'])
            bad += 0 if ok else 1
            print(f'{"ok  " if ok else "FAIL"} {name:<40} {r["native"]:5.1f} cm, '
                  f'x{r["k"]:.2f} -> {min(sizes):.1f}-{max(sizes):.1f} pt, '
                  f'{len(r["runs"])} labels'
                  + (f', smallest glyph {tiny:.1f}' if tiny < a.floor - 0.05 else ''))
            for w in sorted(r['small'], key=lambda x: x['size'])[:6]:
                print(f'       small {w["size"] * r["k"]:4.1f} pt  {w["text"][:54]!r}')
            if len(r['small']) > 6:
                print(f'       ... {len(r["small"]) - 6} more under {a.floor} pt')
            for x, y, d in sorted(r['clash'], key=lambda v: -v[2])[:8]:
                print(f'       clash {d:4.1f} pt  {x["text"][:30]!r} / {y["text"][:30]!r}')
            if len(r['clash']) > 8:
                print(f'       ... {len(r["clash"]) - 8} more collisions')
            if a.verbose:
                for x in sorted(r['runs'], key=lambda v: v['size'])[:14]:
                    print(f'         {x["size"] * r["k"]:5.1f} pt {x["text"][:52]!r}')
    print(f'\n{len(files) - bad} of {len(files)} pass at {a.width} cm '
          f'with a {a.floor} pt floor')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
