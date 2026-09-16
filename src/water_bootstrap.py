"""Spatial block bootstrap for the water-weighted decay constant.

The same treatment the trench-based constant received. Blocks are spherical
Voronoi cells around quasi-uniform seeds, resampled whole, each replicate
rebuilding the water curve and refitting, because the water field is at least as
smooth as the tomography and a cell-wise interval would be meaningless.

The dominance criterion is swept alongside, since a vintage has to be scored
somewhere and the choice of where should be shown rather than settled by fiat.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import paths as P

rng = np.random.default_rng(20260829)
z = np.load(os.path.join(P.OUT, 'water_forward.npz'))
by_age = z['by_age'].reshape(len(z['abins']), -1)
abins = z['abins']
fast = z['fast'].ravel()
lon, lat = z['lon'], z['lat']
LO, LA = np.meshgrid(lon, lat)
w = np.cos(np.radians(LA)).ravel()
R = 6371.0088
tmid = 0.5 * (abins[:, 0] + abins[:, 1])


def to_xyz(lo, la):
    lo, la = np.radians(lo), np.radians(la)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return (np.degrees(th % (2 * np.pi)) - 180.0, 90.0 - np.degrees(phi))


def model(t, e0, einf, tau):
    return einf + (e0 - einf) * np.exp(-t / tau)


def curve(sel, Rdom):
    ww, ff = w[sel], fast[sel]
    base = ww[ff].sum() / ww.sum()
    if not np.isfinite(base) or base <= 0:
        return None
    t, e = [], []
    for bi in range(len(abins)):
        younger = by_age[:bi, sel].sum(axis=0) if bi else np.zeros(int(sel.sum()))
        m = (by_age[bi, sel] > 0) & (by_age[bi, sel] > Rdom * younger)
        tot = by_age[bi, sel][m].sum()
        if tot <= 0:
            continue
        t.append(tmid[bi])
        e.append((by_age[bi, sel][m & ff[m] if False else m & ff].sum() / tot) / base)
    return (np.array(t), np.array(e)) if len(t) >= 4 else None


def fit(t, e):
    try:
        p, _ = curve_fit(model, t, e, p0=[e[0], 0.1, 40],
                         bounds=([0, -1, 3], [60, 5, 600]), maxfev=40000)
        return p[2]
    except Exception:
        return None


cells = to_xyz(LO.ravel(), LA.ravel())
N_BOOT = 600
print(f'{"R":>5s} {"tau, all cells":>15s} {"blocks":>8s} {"tau median":>11s} '
      f'{"95 % interval":>18s}')
rows = []
for Rdom in (0.5, 1.0, 2.0, 4.0):
    c = curve(np.ones(len(w), bool), Rdom)
    t_full = fit(*c) if c else np.nan
    for nblk in (60, 240):
        slon, slat = fib(nblk)
        lab = cKDTree(to_xyz(slon, slat)).query(cells)[1]
        groups = [np.where(lab == b)[0] for b in range(nblk)]
        groups = [g for g in groups if len(g) > 50]
        taus = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(groups), len(groups))
            sel = np.zeros(len(w), bool)
            for gi in pick:
                sel[groups[gi]] = True
            cc = curve(sel, Rdom)
            if cc is None:
                continue
            tt = fit(*cc)
            if tt is not None and 3 < tt < 500:
                taus.append(tt)
        taus = np.array(taus)
        lo, hi = np.percentile(taus, [2.5, 97.5])
        across = R * np.sqrt(4 * np.pi / len(groups))
        print(f'{Rdom:5.1f} {t_full:15.1f} {across:6.0f} km {np.median(taus):11.1f} '
              f'{lo:8.1f} {hi:8.1f}')
        rows.append(dict(R=Rdom, tau_all=t_full, n_blocks=len(groups),
                         block_km=across, tau_median=np.median(taus),
                         lo=lo, hi=hi, n_ok=len(taus)))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(P.OUT, 'water_bootstrap.csv'), index=False)
big = d[d.block_km > 2000]
print(f'\nacross dominance criteria and the larger blocks, tau spans '
      f'{big.lo.min():.0f} to {big.hi.max():.0f} Myr')
print('wrote water_bootstrap.csv')
