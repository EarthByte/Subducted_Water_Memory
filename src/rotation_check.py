"""Is the rotation null actually uniform?

`s16_plume_null.random_rotation` is documented as a uniform random rotation. It
draws a uniform axis and an angle uniform on [0, 2*pi), which is not the uniform
(Haar) distribution on SO(3): the Haar angle density goes as 1 - cos(theta), so
a uniform angle over-weights small rotations. A small rotation leaves the
observed set almost where it was, and a null draw that barely moves the set
carries the observed association into the null, inflating it.

`table1.py` is protected from this by rejecting any draw whose median
displacement is under s17.MIN_DISPLACE_DEG. `depth_decay.py` has no such guard,
so its nulls are inflated and its depth test is too conservative.

This measures the size of the effect on the paper's own fast set rather than
arguing it, and prints what the 410-520 km null becomes under each choice.

    python3 src/rotation_check.py
    python3 src/rotation_check.py --n 4000 --band 520 660
"""
import argparse, os
import numpy as np

try:
    import paths as P
    OUT, TOMO = P.OUT, getattr(P, 'TOMO', '..')
except Exception:
    OUT, TOMO = 'out', '..'

GUARD = 20.0          # degrees, s17.MIN_DISPLACE_DEG


def axis_angle(rng):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    an = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(an) * K + (1 - np.cos(an)) * (K @ K)


def haar(rng):
    """Uniform on SO(3). numpy's qr does not give a Haar Q on its own: the
    signs of the diagonal of R have to be folded in first (Mezzadri 2007).
    Skipping that starves the small rotations and gives a mean rotation angle of
    144 degrees against the correct 126.5."""
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    q = q * (d / np.abs(d))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--band', type=float, nargs=2, default=[410.0, 520.0])
    ap.add_argument('--bands', default=None,
                    help='npz from extract_bands.py; default reads REVEAL')
    a = ap.parse_args()

    z = np.load(os.path.join(OUT, 'hydration_age_map.npz'))
    occ, ABINS, olat, olon = z['occupancy'], z['bands'], z['lat'], z['lon']
    NLAT, NLON = len(olat), len(olon)
    LO, LA = np.meshgrid(olon, olat)
    w = np.cos(np.radians(LA)).ravel(); WT = w.sum()

    age = np.full(occ.shape[1:], -1, np.int8)
    for k in range(len(ABINS) - 1, -1, -1):
        y = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
        m = occ[k] & ~y
        if m.sum() >= 300:
            age[m] = k
    age = age.ravel(); nage = int(age.max()) + 1
    den = np.bincount(age[age >= 0], weights=w[age >= 0], minlength=nage)

    b0, b1 = a.band
    if a.bands:
        f = np.load(a.bands, allow_pickle=True)
        blat, blon = f['lat'], f['lon']
        band = f[f'{int(b0)}_{int(b1)}'].astype(float)
    else:
        import dataclasses, plume_classifier as pc
        spec = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
        dep, blat, blon, arr = pc.load_anomaly(spec, P.REVEAL)
        band = np.nanmean(arr[(dep >= b0) & (dep < b1)], axis=0)

    j = np.abs(blat[None, :] - olat[:, None]).argmin(1)
    i = np.abs(((blon[None, :] - olon[:, None] + 180) % 360) - 180).argmin(1)
    g = band[np.ix_(j, i)].ravel(); fin = np.isfinite(g)
    fast = fin & (g >= np.percentile(g[fin], 90.0))
    la, lo = LA.ravel()[fast], LO.ravel()[fast]
    r_, s_ = np.radians(la), np.radians(lo)
    X = np.column_stack([np.cos(r_) * np.cos(s_), np.cos(r_) * np.sin(s_), np.sin(r_)])
    H = np.empty(NLAT * NLON, bool)

    def enr(Pt):
        sla = np.degrees(np.arcsin(np.clip(Pt[:, 2], -1, 1)))
        slo = np.degrees(np.arctan2(Pt[:, 1], Pt[:, 0]))
        jj = np.clip(np.rint((sla + 90.0) * 2.0).astype(int), 0, NLAT - 1)
        ii = np.clip(np.rint((slo + 180.0) * 2.0).astype(int), 0, NLON - 1)
        H.fill(False); H[jj * NLON + ii] = True
        idx = np.flatnonzero(H)
        nb = w[idx].sum() / WT
        a2 = age[idx]; k2 = a2 >= 0
        return np.max(np.bincount(a2[k2], weights=w[idx][k2], minlength=nage) / den) / nb

    obs = enr(X)
    print(f'\n{int(b0)}-{int(b1)} km, top decile: {int(fast.sum())} cells, '
          f'observed E* = {obs:.2f}\n')
    print(f'{"rotation":34}{"med disp":>10}{"< 20 deg":>10}'
          f'{"p95":>8}{"p95 guarded":>13}{"p":>8}')
    for name, gen in (('axis + uniform angle (as used now)', axis_angle),
                      ('Haar, uniform on SO(3)', haar)):
        rng = np.random.default_rng(7)
        es, esg, disp = [], [], []
        for _ in range(a.n):
            Pt = X @ gen(rng).T
            d = np.degrees(np.arccos(np.clip((Pt * X).sum(1), -1, 1)))
            md = float(np.median(d)); disp.append(md)
            e = enr(Pt); es.append(e)
            if md >= GUARD:
                esg.append(e)
        es, esg, disp = np.array(es), np.array(esg), np.array(disp)
        p = ((es >= obs).sum() + 1) / (len(es) + 1)
        print(f'{name:34}{np.median(disp):9.0f} {100 * np.mean(disp < GUARD):9.1f}%'
              f'{np.percentile(es, 95):8.2f}{np.percentile(esg, 95):13.2f}{p:8.4f}')
    print(f'\n  A draw that moves the set less than {GUARD:.0f} degrees is not a null; it is\n'
          f'  the observation with a nudge. table1.py rejects those, depth_decay.py\n'
          f'  does not. Under a genuinely uniform rotation the question does not arise.')


if __name__ == '__main__':
    main()
