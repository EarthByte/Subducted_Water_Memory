"""The depth scan, corrected for the fact that it is thirty tests, not one.

depth_decay.py reports E*(z), the enrichment at the best-explaining age, against
the 95th percentile of a rotated null, band by band. Ten depth bands in three
volumes is thirty tests, so about 1.5 exceedances are expected with no signal
anywhere, and the manuscript says as much in words. This makes it a number.

The correction is Westfall and Young's maxT, done on the rotations themselves:
the SAME rotation is applied at every depth, so the strong correlation between
neighbouring bands is carried into the null rather than assumed away. For each
rotation the per-band probability is computed against the other rotations, the
smallest of those is kept, and a band's adjusted probability is the share of
rotations whose smallest beats the band's observed one. A Bonferroni factor of
ten would be far more conservative, because the bands are not independent.

    python3 src/depth_fwer.py                 # all three volumes
    python3 src/depth_fwer.py --n-null 500    # quicker, for a check

Everything is read from what earlier stages wrote; nothing is reconstructed.
"""
import argparse, os, time
import numpy as np, pandas as pd

try:
    import paths as _P
    U = getattr(_P, 'TOMO', '..')
    OUTDIR = getattr(_P, 'OUT', 'out')
except Exception:
    U, OUTDIR = os.environ.get('HYD_DATA', '..'), 'out'
DEPTHS = [(410, 520), (520, 660), (660, 720), (720, 780), (780, 880),
          (880, 1010), (1000, 1300), (1300, 1600), (1600, 2000), (2000, 2500)]
PCTL = 90.0


def rotation(rng, kind='haar'):
    """A rotation drawn without regard to where the observed set sits.

    'haar' is the uniform distribution on SO(3), from the QR of a Gaussian
    matrix. 'axis-angle' reproduces s16_plume_null.random_rotation, which draws
    a uniform axis and a uniform angle; that is not uniform on SO(3) and puts
    about an eighth of its draws within 20 degrees of the identity, so an eighth
    of the null barely moves the set and the null is inflated.
    """
    if kind == 'axis-angle':
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        an = rng.uniform(0, 2 * np.pi)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(an) * K + (1 - np.cos(an)) * (K @ K)
    # The QR of a Gaussian matrix is Haar-distributed only after the signs of
    # the diagonal of R are folded into Q (Mezzadri 2007). Without that step the
    # angle distribution is wrong in the other direction from the axis-angle
    # draw: it starves the small rotations, its mean angle is 144 degrees against
    # a true 126.5, and the null it produces is too easy to beat.
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    q = q * (d / np.abs(d))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-null', type=int, default=2000)
    ap.add_argument('--out', default=OUTDIR)
    ap.add_argument('--models', nargs='+',
                    default=['REVEAL', 'RevealLO', 'GLADM35', 'SPiRaL', 'SEMUCBWM1'],
                    help='volumes to scan; each except REVEAL needs bands_<tag>.npz '
                         'beside the tomography, so adding a sixth needs no code edit')
    ap.add_argument('--rotation', choices=('haar', 'axis-angle'),
                    default='haar',
                    help="'axis-angle' reproduces s16_plume_null.random_rotation, "
                         'which is not uniform; see rotation_check.py')
    # The occupancy field carries the down-dip offset, so scanning at a different
    # offset is a matter of pointing at the map built with it. --suffix keeps the
    # result beside the reference one instead of overwriting it.
    ap.add_argument('--occupancy', default='hydration_age_map.npz',
                    help='occupancy map in --out, e.g. hydration_age_map_offset200.npz')
    ap.add_argument('--suffix', default='',
                    help="appended to the output name, e.g. '_off200'")
    a = ap.parse_args()

    zf = os.path.join(a.out, a.occupancy)
    if not os.path.exists(zf):
        raise SystemExit(
            f'\nno occupancy map at {zf}\n'
            '  the offset variants are written by occupancy_sweep.py; run that '
            'first, or name one that exists.\n')
    z = np.load(zf)
    print(f'occupancy: {a.occupancy}')
    occ, ABINS, olat, olon = z['occupancy'], z['bands'], z['lat'], z['lon']
    NLAT, NLON = len(olat), len(olon)
    LO, LA = np.meshgrid(olon, olat)
    w = np.cos(np.radians(LA)).ravel()
    WT = w.sum()

    # "subduction in this band and none younger" -- disjoint by construction, so
    # the ten enrichments come from one bincount rather than ten masked sums
    age_id = np.full(occ.shape[1:], -1, np.int8)
    for k in range(len(ABINS) - 1, -1, -1):
        younger = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
        m = occ[k] & ~younger
        if m.sum() >= 300:
            age_id[m] = k
    age_id = age_id.ravel()
    nage = int(age_id.max()) + 1
    denom = np.bincount(age_id[age_id >= 0], weights=w[age_id >= 0], minlength=nage)
    print(f'{nage} age bands usable, {int((age_id >= 0).sum())} cells assigned')

    import xarray as xr
    def bands_for(tag):
        if tag == 'REVEAL':
            ds = xr.open_dataset(f'{U}/REVEAL_vs_full.nc')
            v = np.sqrt((2.0 * ds['vsv']**2 + ds['vsh']**2) / 3.0)
            mu = v.mean(dim=['latitude', 'longitude'])
            arr = ((v - mu) * 100.0 / mu).clip(min=-100, max=100).transpose(
                'depth', 'latitude', 'longitude').values
            dep = np.asarray(ds['depth'].values, float)
            if dep[0] > dep[-1]:
                o = np.argsort(dep); dep, arr = dep[o], arr[o]
            blat = np.asarray(ds['latitude'].values, float)
            blon = np.asarray(ds['longitude'].values, float)
            out = {}
            for b0, b1 in DEPTHS:
                k = (dep >= b0) & (dep < b1)
                if k.any():
                    out[(b0, b1)] = np.nanmean(arr[k], axis=0)
            return blat, blon, out
        bp = f'{U}/REVEAL_mantle_tomography/bands_{tag}.npz'
        if not os.path.exists(bp):
            # name the file rather than raising: one missing volume should not
            # lose the scan for the others
            print(f'  {tag}: no {os.path.basename(bp)}, skipped')
            return None, None, None
        f = np.load(bp, allow_pickle=True)
        out = {}
        for b0, b1 in DEPTHS:
            key = f'{b0}_{b1}'
            if key in f.files:
                out[(b0, b1)] = f[key].astype(float)
        return f['lat'], f['lon'], out

    rng = np.random.default_rng(20260902)
    ROT = [rotation(rng, a.rotation) for _ in range(a.n_null)]   # shared across depths
    rows = []

    for tag in a.models:
        t0 = time.time()
        blat, blon, bands = bands_for(tag)
        if blat is None:
            continue
        j = np.abs(blat[None, :] - olat[:, None]).argmin(1)
        i = np.abs(((blon[None, :] - olon[:, None] + 180) % 360) - 180).argmin(1)
        obs, null = {}, {}
        H = np.empty(NLAT * NLON, bool)
        for (b0, b1), band in bands.items():
            g = band[np.ix_(j, i)].ravel()
            f = np.isfinite(g)
            fast = f & (g >= np.percentile(g[f], PCTL))
            base = w[fast].sum() / WT
            hit_age = age_id[fast]
            num = np.bincount(hit_age[hit_age >= 0], weights=w[fast][hit_age >= 0],
                              minlength=nage)
            obs[(b0, b1)] = float(np.max(num / denom) / base)

            # the rotated set, at the same depth, under every shared rotation
            # No cache here. Every depth band holds its own top decile, and by
            # construction they are all the same size, so a cache keyed on size
            # silently hands one band another band's geometry and makes every
            # null identical.
            la = LA.ravel()[fast]; lo = LO.ravel()[fast]
            rr = np.radians(la); ss = np.radians(lo)
            X = np.column_stack(
                [np.cos(rr) * np.cos(ss), np.cos(rr) * np.sin(ss), np.sin(rr)])
            es = np.empty(a.n_null)
            for r_, Q in enumerate(ROT):
                P_ = X @ Q.T
                sla = np.degrees(np.arcsin(np.clip(P_[:, 2], -1, 1)))
                slo = np.degrees(np.arctan2(P_[:, 1], P_[:, 0]))
                jj = np.clip(np.rint((sla + 90.0) * 2.0).astype(np.int32), 0, NLAT - 1)
                ii = np.clip(np.rint((slo + 180.0) * 2.0).astype(np.int32), 0, NLON - 1)
                H.fill(False)
                H[jj * NLON + ii] = True
                idx = np.flatnonzero(H)
                nb = w[idx].sum() / WT
                aa = age_id[idx]
                k2 = aa >= 0
                n2 = np.bincount(aa[k2], weights=w[idx][k2], minlength=nage)
                es[r_] = np.max(n2 / denom) / nb
            null[(b0, b1)] = es
        keys = list(obs)
        E = np.array([obs[k] for k in keys])
        N = np.array([null[k] for k in keys])                    # bands x rotations

        # raw probability per band, then Westfall-Young across bands
        praw = np.array([((N[b] >= E[b]).sum() + 1) / (a.n_null + 1)
                         for b in range(len(keys))])
        R = a.n_null
        # each rotation's own per-band probability, against the other rotations
        order = np.argsort(np.argsort(-N, axis=1), axis=1)       # rank, 0 = largest
        pr = (order + 1) / R
        minp = pr.min(axis=0)                                    # per rotation
        padj = np.array([((minp <= praw[b]).sum() + 1) / (R + 1)
                         for b in range(len(keys))])
        for b, (b0, b1) in enumerate(keys):
            rows.append(dict(model=tag, z0=b0, z1=b1, e_best=E[b],
                             null_p95=float(np.percentile(N[b], 95)),
                             p_raw=praw[b], p_adj=padj[b]))
        print(f'{tag}: {time.time() - t0:.0f} s')

    d = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, f'depth_fwer{a.suffix}.csv'), index=False)
    print(f'\n{"model":10}{"band":>12}{"E*":>7}{"null95":>8}{"p":>8}{"p adj":>8}')
    for _, r in d.iterrows():
        mark = '  <-- survives' if r.p_adj <= 0.05 else ''
        print(f'{r.model:10}{int(r.z0):5d}-{int(r.z1):<6d}{r.e_best:7.2f}'
              f'{r.null_p95:8.2f}{r.p_raw:8.4f}{r.p_adj:8.4f}{mark}')
    n_raw = int((d.p_raw <= 0.05).sum()); n_adj = int((d.p_adj <= 0.05).sum())
    print(f'\n{n_raw} of {len(d)} bands clear at p <= 0.05 uncorrected, '
          f'{n_adj} after correcting for the {len(d) // 3} depths tested in each volume')
    print(f'wrote {a.out}/depth_fwer{a.suffix}.csv')


if __name__ == '__main__':
    main()
