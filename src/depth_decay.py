"""How deep can the transition zone's memory of subduction be followed?

The persistence measurement fixes the depth window at 410-660 km and varies the
age of subduction, giving an e-folding of 36 Myr. This asks the same question the
other way round: fix nothing, and for every depth band ask which age of subduction
best explains a fast anomaly there, and how strong that association still is.

Two numbers come out of one matrix.

The age that maximises the enrichment at depth z is the transit time to z, so
t*(z) is a sinking rate measured from the structure rather than prescribed.

The enrichment at that maximum, E*(z), is the test of the corollary. If the memory
of subduction fades thermally while a slab sinks, E* should decay with depth on a
length of v times tau, and quoting that as a depth needs a rate: the 1800 km that
falls out of 50 km/Myr is an upper-mantle number and slabs sink several times more
slowly below 660 km. If E* does not decay, the association is not being lost with
depth, it is moving to older ages, and the 36 Myr belongs to the transition zone
rather than to slabs in general.

E* is the largest of ten ratios, so it clears one without any signal at all. It is
reported against a null of the same fast set rotated, which is the null the rest of
this paper uses.

    python3 src/depth_decay.py --bands ../REVEAL_mantle_tomography/bands_RevealLO.npz
    python3 src/depth_decay.py                      # REVEAL, read directly

Nothing here uses the coherent-body decomposition, which percolates once the model
is better resolved and is not safe below the transition zone.
"""
import argparse
import dataclasses
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import paths as PATHS
import plume_classifier as pc
from s16_plume_null import random_rotation, to_xyz

# Bands as extract_bands.py writes them, from the transition zone down. The
# shallow ones are lithosphere and cannot be a sunk slab at any age.
DEPTHS = [(410, 520), (520, 660), (660, 720), (720, 780), (780, 880),
          (880, 1010), (1000, 1300), (1300, 1600), (1600, 2000), (2000, 2500)]


def band_from_npz(z, b0, b1):
    key = f'{b0}_{b1}'
    return z[key].astype(float) if key in z.files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', default=None,
                    help='npz from extract_bands.py; default reads REVEAL')
    ap.add_argument('--occupancy', default=None)
    ap.add_argument('--tau', type=float, default=36.5,
                    help='the age-domain e-folding, for the consistency check')
    ap.add_argument('--pctl', type=float, default=90.0)
    ap.add_argument('--label', default=None)
    ap.add_argument('--n-null', type=int, default=2000,
                    help='rotations of the fast set; E* is a maximum over ten '
                         'ages and exceeds one by chance. 200 is not enough: the '
                         '95th percentile moved by 0.1 between runs, which is '
                         'wider than most of the margins being judged')
    a = ap.parse_args()
    rng = np.random.default_rng(20260831)

    occ_path = a.occupancy or os.path.join(PATHS.OUT, 'hydration_age_map.npz')
    z = np.load(PATHS.need(occ_path, 'PAPER_OUT', 'hydration_age_map.npz'))
    occ, ABINS = z['occupancy'], z['bands']
    olat, olon = z['lat'], z['lon']
    label = a.label or (os.path.basename(a.bands).replace('bands_', '').replace('.npz', '')
                        if a.bands else 'REVEAL')

    if a.bands:
        bz = np.load(a.bands, allow_pickle=True)
        blat, blon = bz['lat'], bz['lon']
        get = lambda b0, b1: band_from_npz(bz, b0, b1)
    else:
        SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
        depth, blat, blon, arr = pc.load_anomaly(SPEC, PATHS.REVEAL)
        def get(b0, b1):
            k = (depth >= b0) & (depth < b1)
            return np.nanmean(arr[k], axis=0) if k.any() else None
    print(f'{label}: occupancy on {len(olat)}x{len(olon)}, '
          f'bands on {len(blat)}x{len(blon)}')

    j = np.abs(blat[None, :] - olat[:, None]).argmin(1)
    i = np.abs(((blon[None, :] - olon[:, None] + 180) % 360) - 180).argmin(1)
    LO, LA = np.meshgrid(olon, olat)
    w = np.cos(np.radians(LA))
    dlat, dlon = olat[1] - olat[0], olon[1] - olon[0]
    nlat, nlon = len(olat), len(olon)
    wflat, wtot = w.ravel(), w.sum()

    # given subduction in this band and none younger, which is the quantity the
    # persistence measurement uses and the only one that is not leaky
    masks = []
    for k in range(len(ABINS)):
        younger = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
        m = occ[k] & ~younger
        masks.append(m if m.sum() >= 300 else None)
    amid = 0.5 * (ABINS[:, 0] + ABINS[:, 1])
    # occ[k] & ~younger are mutually exclusive, so one label array holds them all
    lab = np.full(LO.size, -1, np.int64)
    for k, m in enumerate(masks):
        if m is not None:
            lab[m.ravel()] = k
    keepk = np.array([k for k, m in enumerate(masks) if m is not None])
    mwsum = np.array([w[masks[k]].sum() for k in keepk])

    rows, hdr = [], '  depth band   base'
    for k in range(len(ABINS)):
        if masks[k] is not None:
            hdr += f'{amid[k]:7.0f}'
    print('\nenrichment: P(fast at this depth | subduction at this age, none since)'
          ' / base rate')
    print(hdr + '     t*    E*  null95')
    for (b0, b1) in DEPTHS:
        band = get(b0, b1)
        if band is None:
            continue
        tz = band[np.ix_(j, i)]
        f = np.isfinite(tz)
        if f.sum() < 1000:
            continue
        fast = f & (tz >= np.percentile(tz[f], a.pctl))
        base = w[fast].sum() / w.sum()
        line, es, ts = f'  {b0:5d}-{b1:<5d} {100 * base:5.1f}%', [], []
        for k in range(len(ABINS)):
            if masks[k] is None:
                continue
            m = masks[k]
            e = (w[m & fast].sum() / w[m].sum()) / base
            line += f'{e:7.2f}'
            es.append(e); ts.append(amid[k])
        es, ts = np.array(es), np.array(ts)
        kbest = int(np.argmax(es))
        # E* is the largest of ten ratios, so it clears one without any signal.
        # The null is the paper's own: rotate the fast set and take the maximum
        # again, which keeps its size and its patchiness and destroys only where
        # it sits relative to the subduction history.
        cx = to_xyz(LO[fast], LA[fast])
        nulls = []
        for _ in range(a.n_null):
            q = cx @ random_rotation(rng).T
            # Nearest cell on a regular grid is rounding, not a tree search: the
            # tree cost 27 ms a rotation against 1. It disagrees for about one
            # point in 1600, always by one cell, which moves a percentile in the
            # third decimal. And the age masks are disjoint by construction, so
            # all ten reductions are one pass of bincount rather than ten passes
            # over the globe. Together, 39 ms a rotation down to 7.
            qa = np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1)))
            qo = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
            jj = np.clip(np.rint((qa - olat[0]) / dlat).astype(int), 0, nlat - 1)
            ii = np.rint((qo - olon[0]) / dlon).astype(int) % (nlon - 1)
            u = np.unique(jj * nlon + ii)
            wu = wflat[u]
            nb = wu.sum() / wtot
            lu = lab[u]
            good = lu >= 0
            num = np.bincount(lu[good], weights=wu[good], minlength=len(masks))
            nulls.append(float(np.max((num[keepk] / mwsum) / nb)))
        n50, n95 = float(np.median(nulls)), float(np.percentile(nulls, 95))
        line += f'{ts[kbest]:7.0f}{es[kbest]:6.2f}{n50:7.2f}{n95:6.2f}'
        print(line)
        rows.append(dict(z0=b0, z1=b1, z_mid=0.5 * (b0 + b1), base=base,
                         t_best=ts[kbest], e_best=es[kbest],
                         null_median=n50, null_p95=n95))

    d = pd.DataFrame(rows)
    if len(d) < 4:
        raise SystemExit('\ntoo few depth bands to fit anything\n')

    def decay(x, e0, ei, L):
        return ei + (e0 - ei) * np.exp(-x / L)

    # The corollary this was written to test: if the memory of subduction fades
    # thermally while a slab sinks, E* should decay with depth on a length of
    # v * tau. It is only worth quoting if the fit is actually constrained.
    try:
        popt, pcov = curve_fit(decay, d.z_mid.values, d.e_best.values,
                               p0=[d.e_best.iloc[0], 1.0, 1500.0],
                               bounds=([0, 0, 100], [50, 5, 20000]), maxfev=40000)
        L, dL = popt[2], float(np.sqrt(np.diag(pcov))[2])
        # Constrained means more than a finite error bar. A length of 122 +/- 121
        # km with an extrapolated E_0 of 48, against a largest observed E* of 3,
        # passed a bare dL < L and is not a measurement of anything.
        ok = (np.isfinite(dL) and dL < 0.5 * L
              and L > (d.z_mid.iloc[1] - d.z_mid.iloc[0])
              and popt[0] < 2.0 * d.e_best.max())
    except Exception:
        L, dL, ok = np.nan, np.nan, False
    if ok:
        print(f'\nE*(z) decays with a length of {L:.0f} +/- {dL:.0f} km, '
              f'a sinking rate of {L / a.tau:.1f} km/Myr against tau = {a.tau:.1f}')
    else:
        print(f'\nE* does not decay with depth: the fit is unconstrained '
              f'(L = {L:.0f} +/- {dL:.0f} km) and E* stays between '
              f'{d.e_best.min():.1f} and {d.e_best.max():.1f} from '
              f'{d.z0.min():.0f} to {d.z1.max():.0f} km. The association is not '
              f'lost with depth, it moves to older ages.')

    # Where t* moves, its slope is a sinking rate. The deepest band is reported
    # apart: it is the worst resolved and one point there swings the regression.
    # It is only worth reading at all where E* clears the null; below the
    # transition zone it does not, and the age of a maximum that is itself within
    # chance is not a transit time.
    sig = d.e_best > d.null_p95
    print(f'  E* clears the 95th percentile of the null in '
          f'{int(sig.sum())} of {len(d)} bands: '
          + (', '.join(f'{int(r.z0)}-{int(r.z1)}' for _, r in d[sig].iterrows())
             or 'none'))
    k = d.t_best.diff().fillna(0).ne(0).cumsum().gt(0) & (d.z_mid < 2000)
    if k.sum() >= 3:
        v = np.polyfit(d.t_best[k], d.z_mid[k], 1)[0]
        print(f'  t* advances at {v:.1f} km/Myr between {d.z_mid[k].min():.0f} '
              f'and {d.z_mid[k].max():.0f} km')
        vd = np.polyfit(d.t_best, d.z_mid, 1)[0]
        print(f'  including the deepest band it reads {vd:.1f} km/Myr, which is '
              f'that one point rather than a trend')
    else:
        print('  t* does not advance with depth')

    out = os.path.join(PATHS.OUT, f'depth_decay_{label}.csv')
    d.to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
