"""The decay constant under a different tomographic model.

Every result in this paper rests on one tomographic model, which is the only
input that has not been replicated: the plate model has been varied three ways,
the continental mask three ways, the volcanism catalogue three ways and the water
product two. The decay constant is the newest measurement and the one most worth
checking.

Only the observed fast transition zone changes with the model. Trench positions,
down-dip displacement and the epoch occupancy come from the reconstruction and are
untouched, so the occupancy computed once is reused and only the mask is swapped.
That makes a second or third model cheap.

Bands come from extract_bands.py, which reduces a model of any size to the depth
averages this analysis wants.
"""
import os, argparse, numpy as np, pandas as pd, dataclasses
from scipy.optimize import curve_fit
import paths as P


def band_from_npz(path, b0, b1):
    z = np.load(path, allow_pickle=True)
    key = f'{b0}_{b1}'
    if key not in z.files:
        raise SystemExit(f'{path} has no {key} band; it holds {sorted(z.files)}')
    return z['lat'], z['lon'], z[key].astype(float)


def band_from_reveal(b0, b1):
    import plume_classifier as pc
    SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
    depth, lat, lon, arr = pc.load_anomaly(SPEC, P.REVEAL)
    k = (depth >= b0) & (depth < b1)
    return lat, lon, np.nanmean(arr[k], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', default=None, help='npz from extract_bands.py')
    ap.add_argument('--label', default='REVEAL')
    ap.add_argument('--occupancy', default=None,
                    help='hydration_age_map npz; the reconstruction side')
    a = ap.parse_args()

    occ_path = a.occupancy or os.path.join(P.OUT, 'hydration_age_map.npz')
    z = np.load(P.need(occ_path, 'PAPER_OUT', 'hydration_age_map.npz'))
    occ = z['occupancy']
    ABINS = z['bands']
    olon, olat = z['lon'], z['lat']

    if a.bands:
        lat, lon, tz = band_from_npz(a.bands, 410, 520)
        _, _, tz2 = band_from_npz(a.bands, 520, 660)
        tz = np.nanmean(np.stack([tz, tz2]), axis=0)
    else:
        lat, lon, tz = band_from_reveal(410, 660)
    # onto the grid the occupancy uses
    j = np.abs(lat[None, :] - olat[:, None]).argmin(1)
    i = np.abs(((lon[None, :] - olon[:, None] + 180) % 360) - 180).argmin(1)
    tz = tz[np.ix_(j, i)]
    f = np.isfinite(tz)
    fast = f & (tz >= np.percentile(tz[f], 90.0))
    LO, LA = np.meshgrid(olon, olat)
    w = np.cos(np.radians(LA))
    base = w[fast].sum() / w.sum()
    print(f'{a.label}: fast transition zone is {100 * base:.1f} per cent of the '
          f'surface, {int(fast.sum())} cells')

    print(f'\n  {"subduction age":>16s} {"cells":>9s} {"fast TZ":>9s} {"enrichment":>11s}')
    t, e, sg = [], [], []
    for k in range(len(ABINS)):
        younger = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
        m = occ[k] & ~younger
        n = int(m.sum())
        if n < 300:
            continue
        wm = w[m]
        frac = wm[fast[m]].sum() / wm.sum()
        se = np.sqrt(max(frac * (1 - frac), 1e-12) / n)
        print(f'  {ABINS[k,0]:6.0f}-{ABINS[k,1]:<9.0f} {n:9d} {100 * frac:8.1f}% '
              f'{frac / base:11.2f}')
        t.append(0.5 * (ABINS[k, 0] + ABINS[k, 1]))
        e.append(frac / base)
        sg.append(1.96 * se / base)
    t, e, sg = np.array(t), np.array(e), np.array(sg)
    popt, _ = curve_fit(lambda x, e0, ei, ta: ei + (e0 - ei) * np.exp(-x / ta),
                        t, e, p0=[e[0], 0.1, 40], sigma=np.maximum(sg, 1e-3),
                        absolute_sigma=True, bounds=([0, -1, 3], [60, 5, 600]),
                        maxfev=40000)
    print(f'\n  E_0 = {popt[0]:.2f}, E_inf = {popt[1]:.2f}, '
          f'tau = {popt[2]:.1f} Myr')
    pd.DataFrame(dict(t_mid=t, enrichment=e, sigma=sg)).to_csv(
        os.path.join(P.OUT, f'persistence_{a.label}.csv'), index=False)
    print(f'  wrote persistence_{a.label}.csv')


if __name__ == '__main__':
    main()
