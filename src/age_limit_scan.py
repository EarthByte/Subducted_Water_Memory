"""How far back can the volcanic-field component be pushed?

Rather than imposing a Cenozoic cut, scan it. The seismic class is present-day
and the eruption is not, so the two are only about the same mantle while the
province is young enough and near enough to where it was. Where that stops being
true should be measured, not assumed.

Two axes, because age is only a proxy for the thing that matters: an eruption is
comparable with present tomography if the province has not moved off the mantle
it erupted over, so reconstruction displacement is the physical variable and age
is what stands in for it.

    python3 src/age_limit_scan.py
    python3 src/age_limit_scan.py --band 520-660

Reported at each cut: the number of provinces, how many are classified fast, the
gap in timed delivery between fast provinces and the rest with its permutation
probability, and the water-velocity correlation.
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import paths as P
    OUT = getattr(P, 'OUT', 'out')
except Exception:
    OUT = 'out'
from province_pivot import provinces, CAT, RADIUS_DEG


def footprint(field, lon, lat, d):
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    near = lambda ax, v: np.abs(ax[None, :] - np.asarray(v, float)[:, None]).argmin(1)
    jl, il = near(lat, d.lat.values), near(lon, d.lon_180.values)
    dj = max(1, int(round(RADIUS_DEG / dlat)))
    cl = np.maximum(np.cos(np.radians(d.lat.values)), 0.05)
    di = np.maximum(1, np.round(RADIUS_DEG / dlon / cl).astype(int))
    o = np.empty(len(d))
    for k in range(len(d)):
        jj = np.arange(max(0, jl[k] - dj), min(len(lat), jl[k] + dj + 1))
        ii = np.arange(il[k] - di[k], il[k] + di[k] + 1) % len(lon)
        o[k] = np.nanmean(field[np.ix_(jj, ii)])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', default='410-520')
    ap.add_argument('--null', default='continental')
    ap.add_argument('--nperm', type=int, default=10000)
    a = ap.parse_args()

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    pn = provinces(d)
    e = pd.read_csv(os.path.join(OUT, 'province_eruption_context.csv'))
    e['province'] = pn
    c = pd.read_csv(os.path.join(OUT, f'province_classes_{a.null}.csv'))
    t = pd.read_csv(os.path.join(OUT, f'province_scores_{a.null}.csv'))
    S = t[t.band == a.band].groupby('province').S.mean()
    z = np.load(os.path.join(OUT, 'water_forward.npz'))

    g = e.groupby('province').agg(n=('age_Ma', 'size'), age=('age_Ma', 'median'),
                                  moved=('moved_km', 'median'),
                                  tz=('tz_timed', 'mean')).reset_index()
    g['water'] = pd.Series(footprint(z['by_age'][0], z['lon'], z['lat'], d)
                           ).groupby(pd.Series(pn)).mean().reindex(g.province).values
    s = g.merge(c[['province', 'cls']], on='province')
    s['S'] = s.province.map(S)
    rng = np.random.default_rng(3)

    def stats(tt):
        gg = (tt.cls == 'fast').values
        x = tt.tz.values
        if gg.sum() < 2 or (~gg).sum() < 2:
            gap, p = np.nan, np.nan
        else:
            obs = np.median(x[gg]) - np.median(x[~gg])
            nul = np.array([np.median(x[q]) - np.median(x[~q])
                            for q in (rng.permutation(gg) for _ in range(a.nperm))])
            gap, p = obs, ((nul >= obs).sum() + 1) / (a.nperm + 1)
        j = tt[['water', 'S']].dropna(); j = j[j.water > 0]
        r, pw = spearmanr(j.water, j.S) if len(j) >= 8 else (np.nan, np.nan)
        return len(tt), int(gg.sum()), gap, p, r, pw, len(j)

    fmt = lambda v, f: (f % v) if np.isfinite(v) else '-'
    print(f'\n{"cut":>16}{"prov":>5}{"fast":>5}{"timed gap":>11}{"p":>8}'
          f'{"rho":>8}{"p":>8}{"n":>4}')
    rows = []
    for kind, cuts, col in (('eruption age (Ma)', (25, 50, 66, 100, 150, 250), 'age'),
                            ('displacement (km)', (300, 550, 800, 1100, 1500, 2500, 5000), 'moved')):
        print(f'  -- by {kind} --')
        for cut in cuts:
            tt = s[s[col] <= cut]
            n, nf, gap, p, r, pw, nw = stats(tt)
            rows.append(dict(axis=col, cut=cut, n=n, n_fast=nf, timed_gap=gap,
                             timed_p=p, rho=r, rho_p=pw, n_water=nw))
            print(f'{f"<= {cut}":>16}{n:5d}{nf:5d}{fmt(100 * gap if np.isfinite(gap) else gap, "%.0f%%"):>11}'
                  f'{fmt(p, "%.3f"):>8}{fmt(r, "%+.2f"):>8}{fmt(pw, "%.3f"):>8}{nw:4d}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'age_limit_scan.csv'), index=False)

    print('\n  A flat row means the provinces added by loosening the cut carry no')
    print('  signal either way: beyond the memory window there is nothing to see,')
    print('  so old provinces land in the comparison group and move no median.')
    print('  The cut only matters once an old province enters the fast class.')
    old_fast = s[(s.cls == 'fast') & (s.age > 100)]
    if len(old_fast):
        print(f'\n  fast provinces older than 100 Ma, which is where dilution begins:')
        for _, r in old_fast.iterrows():
            print(f'    {r.province[:34]:36} age {r.age:5.0f} Ma, moved {r.moved:5.0f} km, '
                  f'timed {100 * r.tz:3.0f}%')
        print('    For these the eruption-time delivery and the present anomaly are')
        print('    produced independently by a long-lived convergent margin, so the')
        print('    two facts coincide without being causally linked. That, and not')
        print('    a date, is the reason to exclude them.')


if __name__ == '__main__':
    main()
