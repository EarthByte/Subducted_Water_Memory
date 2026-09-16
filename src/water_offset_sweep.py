"""Does the water-velocity correlation depend on the down-dip offset?

The subducted-water grids leave every parcel at the trench that delivered it, as
though slabs sank vertically. `water_forward.py` corrects that: each parcel takes
the trench-normal azimuth of the nearest segment at its own delivery epoch and is
displaced down dip by a fixed offset, 300 km by default, the value the observed
fast transition zone preferred in the offset sweep in slab_forward.py.

The correction is not cosmetic. On the total-water field the province-level
correlation with the seismic anomaly is -0.06 uncorrected and +0.28 corrected, so
a 300 km shift is enough to turn no relationship into a weak one. The offset was
also the term that dominated the uncertainty in the decay constant, moving it
from 36 to 20 Myr between 150 and 500 km. Neither fact was established for the
headline correlation, which uses the age-resolved corrected field, and it should
not be reported without this sweep.

    python3 src/water_offset_sweep.py                 # 150 to 500 km
    python3 src/water_offset_sweep.py --offsets 200 300 400

Each offset needs its own water_forward run, which is the slow part; runs whose
npz already exists are skipped, so this can be interrupted and restarted.
"""
import argparse, os, subprocess, sys
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
    ap.add_argument('--offsets', type=float, nargs='+',
                    default=[150, 200, 300, 400, 500])
    ap.add_argument('--band', default='410-520')
    ap.add_argument('--null', default='continental')
    ap.add_argument('--max-age', type=float, default=150.0,
                    help='provinces older than this are excluded; see age_limit_scan.py')
    a = ap.parse_args()

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    pn = provinces(d)
    t = pd.read_csv(os.path.join(OUT, f'province_scores_{a.null}.csv'))
    S = t[t.band == a.band].groupby('province').S.mean()
    e = pd.read_csv(os.path.join(OUT, 'province_eruption_context.csv'))
    e['province'] = pn
    age = e.groupby('province').age_Ma.median()

    print(f'\n{"offset":>8}{"band 8-25 Ma":>15}{"p":>8}{"n":>4}'
          f'{"total water":>14}{"p":>8}{"n":>4}')
    rows = []
    for off in a.offsets:
        tag = '' if abs(off - 300) < 1e-9 else f'_off{int(off)}'
        f = os.path.join(OUT, f'water_forward{tag}.npz')
        if not os.path.exists(f):
            cmd = [sys.executable, os.path.join('src', 'water_forward.py'),
                   '--offset', str(off), '--suffix', tag]
            print(f'  running water_forward at {off:.0f} km ...', flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(f):
                print(f'{off:8.0f}   FAILED: ' +
                      (r.stderr or r.stdout).strip().splitlines()[-1][:70])
                continue
        z = np.load(f)
        out = {}
        for lab, field in (('young', z['by_age'][0]), ('total', z['water'])):
            w = pd.Series(footprint(field, z['lon'], z['lat'], d)
                          ).groupby(pd.Series(pn)).mean()
            j = pd.concat([w.rename('w'), S.rename('S'), age.rename('a')],
                          axis=1).dropna()
            j = j[(j.a <= a.max_age) & (j.w > 0)]
            out[lab] = spearmanr(j.w, j.S) if len(j) >= 8 else (np.nan, np.nan)
            # one n per field: the young band is zero in more provinces than the
            # total is, so a single n column reported whichever ran last
            out['n_' + lab] = len(j)
        rows.append(dict(offset=off, rho_young=out['young'][0], p_young=out['young'][1],
                         n_young=out['n_young'], rho_total=out['total'][0],
                         p_total=out['total'][1], n_total=out['n_total']))
        print(f'{off:8.0f}{out["young"][0]:15.3f}{out["young"][1]:8.4f}'
              f'{out["n_young"]:4d}{out["total"][0]:14.3f}{out["total"][1]:8.4f}'
              f'{out["n_total"]:4d}')
    if rows:
        dd = pd.DataFrame(rows)
        dd.to_csv(os.path.join(OUT, 'water_offset_sweep.csv'), index=False)
        print(f'\n  rho for the young band spans {dd.rho_young.min():+.3f} to '
              f'{dd.rho_young.max():+.3f} across {dd.offset.min():.0f}-'
              f'{dd.offset.max():.0f} km.')
        print('  If it changes sign or loses significance anywhere in that range the')
        print('  correlation is a property of the chosen offset, not of the Earth.')


if __name__ == '__main__':
    main()
