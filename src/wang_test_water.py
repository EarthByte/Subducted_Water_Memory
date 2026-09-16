"""The Wang et al. (2025) test, using this project's own subducted-water model.

`wang_test.py` stands in for their hydration field with trench proximity, which
is geometric and ignores how much water each slab actually carried. This project
has the real thing: the Z22 stored-water grids, differenced per epoch and moved
down dip by `water_forward.py`, which writes a present-day map of
transition-zone water with the delivery age attached — the same kind of object
Wang et al. build, on our reconstruction and with a slab water budget.

`water_forward.npz` carries `by_age`, the water delivered in each of eight age
bands. Wang et al.'s residence condition keeps only water young enough still to
be there, so summing `by_age` over bands younger than the residence time R gives
the predicted wet field for that R. Sweeping R over their stated 30-100 Myr
gives the field they would call wet today.

The seismic anomaly is present-day, so a present-day water field is the right
comparison; no eruption-age reconstruction enters here.

    python3 src/water_forward.py          # writes out/water_forward.npz
    python3 src/wang_test_water.py

If the provinces the model calls wetter are not slower, shear velocity carries
no information about this prediction. That is the claim being tested, and it is
tested here against a real water budget rather than a geometric proxy.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', default='410-520')
    ap.add_argument('--null', default='continental')
    ap.add_argument('--residence', type=float, nargs='+', default=[30, 50, 75, 100],
                    help='Myr; Wang et al. give 30-100')
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

    f = os.path.join(OUT, 'water_forward.npz')
    if not os.path.exists(f):
        raise SystemExit(f'\n{f} missing. It needs the Z22 stored-water series:\n'
                         '    python3 src/water_forward.py\n')
    z = np.load(f)
    lon, lat, by_age, abins = z['lon'], z['lat'], z['by_age'], z['abins']
    print(f'water_forward: {by_age.shape[0]} age bands on {len(lat)}x{len(lon)}, '
          f'{abins[0][0]:.0f} to {abins[-1][1]:.0f} Ma')

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    pn = provinces(d)
    t = pd.read_csv(os.path.join(OUT, f'province_scores_{a.null}.csv'))
    S = t[t.band == a.band].groupby('province').S.mean()

    # each field's water is the mean over the same footprint the anomaly uses
    dlat = abs(lat[1] - lat[0]); dlon = abs(lon[1] - lon[0])
    near = lambda ax, v: np.abs(ax[None, :] - np.asarray(v, float)[:, None]).argmin(1)
    jl = near(lat, d.lat.values); il = near(lon, d.lon_180.values)
    dj = max(1, int(round(RADIUS_DEG / dlat)))
    cl = np.maximum(np.cos(np.radians(d.lat.values)), 0.05)
    di = np.maximum(1, np.round(RADIUS_DEG / dlon / cl).astype(int))

    def footprint_mean(field):
        out = np.empty(len(d))
        for k in range(len(d)):
            jj = np.arange(max(0, jl[k] - dj), min(len(lat), jl[k] + dj + 1))
            ii = np.arange(il[k] - di[k], il[k] + di[k] + 1) % len(lon)
            out[k] = np.nanmean(field[np.ix_(jj, ii)])
        return out

    print(f'\npredicted-wet water against the seismic anomaly at {a.band} km\n')
    print(f'{"residence":>11}{"bands kept":>12}{"provinces":>11}{"rho":>9}{"p":>8}')
    rows = []
    for R in a.residence:
        keep = abins[:, 1] <= R
        if not keep.any():
            continue
        w = footprint_mean(by_age[keep].sum(axis=0))
        pw = pd.Series(w).groupby(pd.Series(pn)).mean()
        j = pd.concat([pw.rename('wet'), S.rename('S')], axis=1).dropna()
        if _keep is not None:
            j = j[j.index.isin(_keep)]
        j = j[j.wet > 0] if (j.wet > 0).sum() >= 8 else j
        r, p = spearmanr(j.wet, j.S)
        rows.append(dict(residence_Myr=R, n_bands=int(keep.sum()),
                         n_prov=len(j), rho=r, p=p))
        print(f'{R:11.0f}{int(keep.sum()):12d}{len(j):11d}{r:9.3f}{p:8.3f}')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f'wang_test_water_{a.band.replace("-", "_")}.csv'),
               index=False)
    sig = int((out.p <= 0.05).sum())
    print(f'\n  median rho {out.rho.median():+.3f}; {sig} of {len(out)} residence '
          f'choices reach p <= 0.05')
    print('  negative rho would mean wetter provinces are SLOWER, as hydrous '
          'upwelling requires.')


if __name__ == '__main__':
    main()
