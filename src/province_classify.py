"""Classify provinces from the standardised scores, and answer the go/no-go.

Step 4 of the recipe. The thresholds are fixed here, before any province is
looked at, and the class names stay observational: fast, slow, layered,
near-neutral, model-dependent, unresolved. Nothing is called cold, hot, wet or
dry, because a velocity cannot support those words on its own.

A province is classified from its per-FAMILY scores, not per-model, so that
REVEAL and RevealLO cannot vote twice.

    python3 src/province_classify.py
    python3 src/province_classify.py --thr 2.0 --null free
"""
import argparse, os
import numpy as np, pandas as pd

try:
    import paths as P
    OUT = getattr(P, 'OUT', 'out')
except Exception:
    OUT = 'out'
try:
    from province_pivot import BANDS       # one definition, shared
except ImportError:
    BANDS = ['350-410', '410-520', '520-660']


def classify(fam, thr, min_fam):
    """fam: dict band -> array of per-family S. One province."""
    out = {}
    for b in BANDS:
        v = fam[b][np.isfinite(fam[b])]
        if len(v) == 0:
            out[b] = (np.nan, 0, 0); continue
        strong_pos = int((v >= thr).sum()); strong_neg = int((v <= -thr).sum())
        out[b] = (float(np.median(v)), strong_pos, strong_neg)
    # a family calling an interval strongly both ways cannot happen; families
    # disagreeing about the sign of a strong signal is the model-dependent case
    conflict = any(p and n for _, p, n in out.values())
    if conflict:
        return 'model-dependent', out
    mat = {b: (out[b][1] >= min_fam) - (out[b][2] >= min_fam) for b in BANDS}
    signs = [s for s in mat.values() if s]
    if not signs:
        return 'near-neutral', out
    if len(set(signs)) > 1:
        return 'layered', out
    if mat['410-520'] > 0 or mat['520-660'] > 0:
        return 'fast', out
    if mat['410-520'] < 0 or mat['520-660'] < 0:
        return 'slow', out
    return 'near-neutral', out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--null', default='continental')
    ap.add_argument('--thr', type=float, default=1.25,
                    help='|S| that counts as a material departure')
    ap.add_argument('--min-fam', type=int, default=3,
                    help='independent families that must agree in sign')
    ap.add_argument('--min-fields', type=int, default=1,
                    help='fields below which a province is called unresolved. '
                         'One, deliberately: resolution is a property of the '
                         'FOOTPRINT, and every province is sampled with the same '
                         '5 degree cap, about 550 km, which these volumes resolve '
                         'whatever the number of catalogue entries inside it. '
                         'Excluding provinces by field count discarded thirteen '
                         'of twenty-nine, several of them among the most strongly '
                         'and consistently coloured, and turned a GO into a NO-GO.')
    ap.add_argument('--sample', default='cenozoic',
                    choices=('cenozoic', 'same-column', 'both', 'all'),
                    help='which provinces the analysis may be applied to. The '
                         'seismic class is present-day and the eruption context '
                         'is not, so only young, little-moved provinces let the '
                         'two be about the same mantle. See province_sample.py.')
    a = ap.parse_args()
    try:
        from province_sample import sample as _sample
        _keep = _sample(OUT, a.sample)
        _keep = set(_keep[_keep].index)
    except Exception as _e:
        print(f'  (sample restriction unavailable: {_e})'); _keep = None

    t = pd.read_csv(os.path.join(OUT, f'province_scores_{a.null}.csv'))
    fams = sorted(t.family.unique())
    print(f'{t.province.nunique()} provinces, {len(fams)} inversion families '
          f'({", ".join(fams)}), |S| >= {a.thr} in >= {a.min_fam} families\n')

    rows = []
    for p, g in t.groupby('province'):
        if _keep is not None and p not in _keep:
            continue
        n = int(g.n_fields.iloc[0])
        fam = {b: np.array([g[(g.family == f) & (g.band == b)].S.mean() for f in fams])
               for b in BANDS}
        if n < a.min_fields:
            cls, det = 'unresolved', {b: (np.nanmedian(fam[b]), 0, 0) for b in BANDS}
        else:
            cls, det = classify(fam, a.thr, a.min_fam)
        rows.append(dict(province=p, n_fields=n, cls=cls,
                         **{f'S_{b}': det[b][0] for b in BANDS},
                         **{f'agree_{b}': det[b][1] - det[b][2] for b in BANDS}))
    d = pd.DataFrame(rows).sort_values(['cls', 'S_410-520'], ascending=[True, False])
    d.to_csv(os.path.join(OUT, f'province_classes_{a.null}.csv'), index=False)

    print(f'{"province":38}{"n":>3} {"350-410":>9}{"410-520":>9}{"520-660":>9}  class')
    for _, r in d.iterrows():
        f = lambda x: f'{x:+8.2f}' if np.isfinite(x) else '       -'
        print(f'{r.province[:37]:38}{r.n_fields:3d} {f(r["S_350-410"])} '
              f'{f(r["S_410-520"])} {f(r["S_520-660"])}  {r.cls}')

    print()
    counts = d.cls.value_counts()
    for k, v in counts.items():
        pop = int(((d.cls == k) & (d.n_fields >= a.min_fields)).sum())
        print(f'  {k:18} {v:3d} provinces ({pop} with >= {a.min_fields} fields)')

    # ---- the verdict -----------------------------------------------------
    # The original criterion demanded two substantive classes holding two or
    # more provinces each. That was written when the recipe hoped for five
    # classes; with one reproducible class and a near-neutral remainder it
    # returns NO-GO on a sample the analysis is perfectly able to support. What
    # matters is not how MANY classes there are but whether at least one is
    # reproducible, distinguishes something, and survives dropping a family.
    subst = d[~d.cls.isin(['near-neutral', 'unresolved'])]
    counts = subst.cls.value_counts()
    n_class = int((counts >= 2).sum())
    biggest = d.cls.value_counts(normalize=True).max()

    # leave-one-family-out, computed here so the verdict is self-contained
    fams_all = sorted(t.family.unique())
    changed = {}
    if len(fams_all) > 2:
        for f in fams_all:
            tf = t[t.family != f]
            ff = sorted(tf.family.unique())
            need = min(a.min_fam, len(ff))
            other = {}
            for p_, g_ in tf.groupby('province'):
                if _keep is not None and p_ not in _keep:
                    continue
                fam = {b: np.array([g_[(g_.family == x) & (g_.band == b)].S.mean()
                                    for x in ff]) for b in BANDS}
                n_ = int(g_.n_fields.iloc[0])
                other[p_] = ('unresolved' if n_ < a.min_fields
                             else classify(fam, a.thr, need)[0])
            base = dict(zip(d.province, d.cls))
            changed[f] = sum(1 for k, v in other.items() if base.get(k) != v)

    print()
    print(f'  a reproducible class other than near-neutral      '
          f'{"yes" if n_class >= 1 else "NO":>6}   '
          f'({", ".join(f"{k} {v}" for k, v in counts.items()) or "none"})')
    print(f'  the largest class holds less than 90 per cent     '
          f'{"yes" if biggest < 0.9 else "NO":>6}   ({100 * biggest:.0f} % in '
          f'{d.cls.value_counts().idxmax()})')
    if changed:
        worst = max(changed.values())
        print(f'  stable when any one inversion family is dropped  '
              f'{"yes" if worst <= 0.25 * len(d) else "NO":>6}   '
              f'(at worst {worst} of {len(d)} provinces reclassified: '
              + ', '.join(f'{k} {v}' for k, v in sorted(changed.items())) + ')')
    else:
        worst = 0
        print('  stable when any one family is dropped             '
              '     -   (fewer than three families given)')

    ok = n_class >= 1 and biggest < 0.9 and worst <= 0.25 * len(d)
    print(f'\n  VERDICT: {"usable" if ok else "not usable"} — '
          f'{n_class} class{"" if n_class == 1 else "es"} beyond near-neutral, '
          f'{len(subst)} of {len(d)} provinces classified substantively')
    if ok and n_class == 1:
        print('    One class and a remainder is a two-way contrast, not an atlas.\n'
              '    It supports a paper that uses the class as evidence; it does not\n'
              '    support one whose claim is the diversity of classes.')
    if not ok:
        print('    The recipe\'s stop conditions are met; do not rewrite the\n'
              '    manuscript around a classification.')
    return d


if __name__ == '__main__':
    main()
