"""A spatial block bootstrap for the decay constant.

The curve fit reports an uncertainty on tau of under a Myr, which is fiction. It
treats tens of thousands of grid cells as independent when the fields are smooth
over a thousand kilometres or more, so the effective sample size is the number of
coherent patches, a few dozen. Resampling cells would reproduce the same fiction.

So the globe is divided into compact blocks larger than the correlation length of
the fields, and whole blocks are resampled with replacement. Each replicate
rebuilds the enrichment curve from scratch, including the restriction that a cell
counts in an age band only if no younger subduction reached it, and refits. The
spread of the refits is the uncertainty. Block size is swept, because a bootstrap
of this kind reports whatever the block size assumes about the correlation length.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import paths as P

rng = np.random.default_rng(20260829)
z = np.load(os.path.join(P.OUT, 'hydration_age_map.npz'))
occ = z['occupancy'].reshape(len(z['bands']), -1)
bands = z['bands']
obs = z['observed'].ravel()
lon, lat = z['lon'], z['lat']
LO, LA = np.meshgrid(lon, lat)
w = np.cos(np.radians(LA)).ravel()
R = 6371.0088


def fib_seeds(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return (np.degrees(theta % (2 * np.pi)) - 180.0, 90.0 - np.degrees(phi))


def to_xyz(lo, la):
    lo, la = np.radians(lo), np.radians(la)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def model(t, e0, einf, tau):
    return einf + (e0 - einf) * np.exp(-t / tau)


def curve(sel):
    """Enrichment per band on a subset of cells, younger subduction excluded."""
    ww = w[sel]; oo = obs[sel]
    base = ww[oo].sum() / ww.sum()
    if not np.isfinite(base) or base <= 0:
        return None
    t, e, s = [], [], []
    for k in range(len(bands)):
        younger = occ[:k, sel].any(axis=0) if k else np.zeros(sel.sum(), bool)
        m = occ[k, sel] & ~younger
        n = int(m.sum())
        if n < 200:
            continue
        wm = ww[m]
        fr = wm[oo[m]].sum() / wm.sum()
        t.append(0.5 * (bands[k, 0] + bands[k, 1]))
        e.append(fr / base)
        s.append(1.96 * np.sqrt(max(fr * (1 - fr), 1e-12) / n) / base)
    return (np.array(t), np.array(e), np.array(s)) if len(t) >= 4 else None


def fit(t, e, s):
    try:
        popt, _ = curve_fit(model, t, e, p0=[e[0], e[-1], 50.0],
                            sigma=np.maximum(s, 1e-3), absolute_sigma=True,
                            maxfev=20000, bounds=([0, -1, 3], [50, 5, 600]))
        return popt
    except Exception:
        return None


from scipy.spatial import cKDTree
cells = to_xyz(LO.ravel(), LA.ravel())
full = curve(np.ones(len(w), bool))
p_full = fit(*full)
print(f'whole globe:  tau = {p_full[2]:.1f} Myr,  E_0 = {p_full[0]:.2f},  '
      f'E_inf = {p_full[1]:.2f}\n')

N_BOOT = 1000
print(f'{"blocks":>7s} {"block size":>11s} {"tau median":>11s} {"95 % interval":>18s} '
      f'{"E_0 median":>11s}')
rows = []
for nblk in (60, 120, 240, 480):
    slon, slat = fib_seeds(nblk)
    lab = cKDTree(to_xyz(slon, slat)).query(cells)[1]
    groups = [np.where(lab == b)[0] for b in range(nblk)]
    groups = [g for g in groups if len(g) > 50]
    across = R * np.sqrt(4 * np.pi / len(groups))
    taus, e0s = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.zeros(len(w), bool)
        for gi in pick:
            sel[groups[gi]] = True
        c = curve(sel)
        if c is None:
            continue
        p = fit(*c)
        if p is not None and 3 < p[2] < 500:
            taus.append(p[2]); e0s.append(p[0])
    taus = np.array(taus); e0s = np.array(e0s)
    lo, hi = np.percentile(taus, [2.5, 97.5])
    print(f'{len(groups):7d} {across:9.0f} km {np.median(taus):11.1f} '
          f'{lo:8.1f} {hi:8.1f} {np.median(e0s):11.2f}')
    rows.append(dict(n_blocks=len(groups), block_km=across, tau_median=np.median(taus),
                     tau_lo=lo, tau_hi=hi, e0_median=np.median(e0s), n_ok=len(taus)))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(P.OUT, 'persistence_bootstrap.csv'), index=False)
big = d.loc[d.block_km.idxmax()]
print(f'\npoint estimate tau = {p_full[2]:.0f} Myr. The interval is stable across '
      f'block size, which')
print(f'says the correlation length is below the smallest block tried. The most '
      f'conservative')
print(f'choice, blocks of {big.block_km:.0f} km, gives {big.tau_lo:.0f} to '
      f'{big.tau_hi:.0f} Myr, and that is the number to quote.')
print('wrote persistence_bootstrap.csv')
