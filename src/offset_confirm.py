"""Repeat the headline results at a prescribed down-dip offset.

The 300 km offset used throughout is the value the observed fast transition zone
preferred in the sweep in slab_forward.py. That makes it a parameter fitted on
the target, which is the most obvious thing a reviewer will object to. Slab
geometry prescribes about 200 km independently of any velocity field, so this
repeats every headline number there and reports the two side by side.

It reads what the other scripts wrote and computes nothing twice, so the inputs
must exist first:

    python3 src/occupancy_sweep.py --axes offset          # hydration_age_map_offset200.npz
    python3 src/water_forward.py --offset 200 --suffix _off200
    python3 src/depth_fwer.py --occupancy hydration_age_map_offset200.npz --suffix _off200
    python3 src/province_eruption_context.py --offset 200 --suffix _off200
    python3 src/offset_confirm.py --offset 200

Anything missing is named rather than guessed at.
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import paths as P
    OUT = getattr(P, 'OUT', 'out')
except Exception:
    OUT = 'out'

BAND = '410-520'
MAX_AGE = 66.0


def fit_tau(t, E, sigma):
    """The convention of persistence_decay.py: weighted, unbounded, with a floor.

    occupancy_sweep.py fits the same curve unweighted and with bounds, which on
    the reference curve gives 33.1 rather than 36.5, and rebuilds the velocity
    field with its own depth average, which takes it to 28.4. The three are the
    same statistic fitted differently; only ratios within one convention are
    comparable, which is why this script refits rather than reading a tau.
    """
    f = lambda t, a, tau, c: a * np.exp(-t / tau) + c
    q, _ = curve_fit(f, t, E, p0=[E.max(), 40.0, 0.05], sigma=sigma,
                     absolute_sigma=True, maxfev=400000)
    r = E - f(t, *q)
    return float(q[1]), 1 - np.sum(r ** 2) / np.sum((E - E.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=float, default=200.0)
    ap.add_argument('--band', default=BAND)
    ap.add_argument('--max-age', type=float, default=MAX_AGE)
    a = ap.parse_args()
    tag = f'_off{int(a.offset)}'
    miss, rows = [], []

    def need(name):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            miss.append(name)
            return None
        return p

    # ---- 1. the decay constant, refitted from the occupancy map at this offset
    for label, npz in (('preferred (300 km)', 'hydration_age_map.npz'),
                       (f'prescribed ({a.offset:.0f} km)',
                        f'hydration_age_map_offset{int(a.offset)}.npz')):
        p = need(npz)
        if p is None:
            continue
        z = np.load(p)
        occ, ab = z['occupancy'], z['bands']
        w = np.cos(np.radians(np.meshgrid(z['lon'], z['lat'])[1]))
        # the band file the paper's curve uses, so the field is not rebuilt here
        bf = need('tz_REVEAL.npz')
        if bf is None:
            break
        b = np.load(bf)
        # tz_REVEAL.npz stores the 410-660 km anomaly, not a mask; the fast set
        # is its upper decile, formed here exactly as elsewhere in the paper and
        # sampled onto the occupancy grid by nearest cell, which on a regular
        # grid is rounding rather than a search
        g = b['dvs']
        jj = np.abs(b['lat'][None, :] - z['lat'][:, None]).argmin(1)
        ii = np.abs(((b['lon'][None, :] - z['lon'][:, None] + 180) % 360) - 180
                    ).argmin(1)
        g = g[np.ix_(jj, ii)].astype(float)
        ok = np.isfinite(g)
        fast = ok & (g >= np.percentile(g[ok], 90.0))
        base = w[fast].sum() / w.sum()
        t, E, sg = [], [], []
        for k in range(len(ab)):
            younger = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
            m = occ[k] & ~younger
            n = int(m.sum())
            if n < 300:
                continue
            wm = w[m]
            fr = wm[fast[m]].sum() / wm.sum()
            se = np.sqrt(max(fr * (1 - fr), 1e-12) / n)
            t.append(0.5 * (ab[k, 0] + ab[k, 1])); E.append(fr / base)
            sg.append(1.96 * se / base)
        tau, r2 = fit_tau(np.array(t), np.array(E), np.array(sg))
        rows.append(dict(quantity='decay constant tau (Myr)', case=label,
                         value=f'{tau:.1f}', detail=f'r2 = {r2:.3f}'))

    # ---- 2. the visibility budget at this offset
    for label, npz, key in (('preferred (300 km)', 'water_forward.npz', ''),
                            (f'prescribed ({a.offset:.0f} km)',
                             f'water_forward{tag}.npz', '')):
        p = need(npz)
        if p is None:
            continue
        z = np.load(p)
        sh = np.array([b.sum() for b in z['by_age']])
        sh = sh / sh.sum()
        ab = z['abins']
        taus = [t for t in (25, 37, 50, 100)]
        det = '  '.join(f'{t}: {100 * sh[ab[:, 1] <= t].sum():.1f}%' for t in taus)
        rows.append(dict(quantity='water inside the window (%)', case=label,
                         value=f'{100 * sh[ab[:, 1] <= 37].sum():.1f}',
                         detail=det))

    # ---- 3. the depth scan
    for label, csv in (('preferred (300 km)', 'depth_fwer.csv'),
                       (f'prescribed ({a.offset:.0f} km)', f'depth_fwer{tag}.csv')):
        p = need(csv)
        if p is None:
            continue
        d = pd.read_csv(p)
        s = d[(d.z0 == 410) & (d.z1 == 520)]
        surv = ', '.join(f'{r.model} {r.p_adj:.3f}' for _, r in s.iterrows()
                         if r.p_adj <= 0.05)
        rows.append(dict(quantity='410-520 km survives maxT (of 5)',
                         case=label, value=f'{int((s.p_adj <= 0.05).sum())}',
                         detail=surv or 'none'))

    # ---- 4. the water-velocity correlation, from the sweep
    p = need('water_offset_sweep.csv')
    if p is not None:
        w = pd.read_csv(p)
        for label, off in (('preferred (300 km)', 300.0),
                           (f'prescribed ({a.offset:.0f} km)', a.offset)):
            r = w[np.isclose(w.offset, off)]
            if len(r):
                r = r.iloc[0]
                n = int(r.n_young) if 'n_young' in w.columns else -1
                rows.append(dict(quantity='rho, water 8-25 Ma vs S', case=label,
                                 value=f'{r.rho_young:+.3f}',
                                 detail=f'p = {r.p_young:.4f}, n = {n}'))

    # ---- 5. the eruption-timing comparison
    from province_pivot import provinces, CAT
    if os.path.exists(CAT):
        d0 = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
        pn = pd.Series(provinces(d0))
        cls = need('province_classes_continental.csv')
        for label, csv in (('preferred (300 km)', 'province_eruption_context.csv'),
                           (f'prescribed ({a.offset:.0f} km)',
                            f'province_eruption_context{tag}.csv')):
            p = need(csv)
            if p is None or cls is None:
                continue
            e = pd.read_csv(p); e['province'] = pn.values
            g = e.groupby('province').agg(age=('age_Ma', 'median'),
                                          tz=('tz_timed', 'mean')).reset_index()
            g = g.merge(pd.read_csv(cls)[['province', 'cls']], on='province')
            g = g[g.age <= a.max_age].reset_index(drop=True)
            y, fa = g.tz.values * 100, (g.cls == 'fast').values
            if fa.sum() == 0:
                continue
            obs = np.median(y[fa]) - np.median(y[~fa])
            n, k = len(y), int(fa.sum())
            hit = tot = 0
            for comb in itertools.combinations(range(n), k):
                m = np.zeros(n, bool); m[list(comb)] = True
                tot += 1
                if np.median(y[m]) - np.median(y[~m]) >= obs - 1e-9:
                    hit += 1
            rows.append(dict(quantity='timing gap, fast vs rest (pp)', case=label,
                             value=f'{obs:.1f}',
                             detail=f'exact p = {hit / tot:.3f} over {tot}, n = {n}'))

    if rows:
        t = pd.DataFrame(rows)
        f = os.path.join(OUT, f'offset_confirm{tag}.csv')
        t.to_csv(f, index=False)
        wq = max(len(x) for x in t.quantity); wc = max(len(x) for x in t.case)
        print()
        for q, grp in t.groupby('quantity', sort=False):
            for _, r in grp.iterrows():
                print(f'  {r.quantity:<{wq}}  {r.case:<{wc}}  {r.value:>7}   {r.detail}')
            print()
        print(f'wrote {f}')
    if miss:
        print('\n  no input for: ' + ', '.join(dict.fromkeys(miss)))
        print('  the header of this file lists the commands that write them.')


if __name__ == '__main__':
    main()
