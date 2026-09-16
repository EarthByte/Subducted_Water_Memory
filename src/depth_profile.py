"""Anomaly beneath continental intraplate volcanism, as a function of depth.

The hydration test is confined to 250-660 km because that is where the model
makes its predictions. Two things outside that range bear on the interpretation
and are not otherwise measured.

ABOVE. Slab-window opening, edge-driven convection and small-scale instability all
place volcanism near slabs without hydration, and all produce shallow low-velocity
structure. Whether the uppermost mantle beneath these fields is slow is positive
evidence about a shallow source, rather than absence of evidence about a deep one.

BELOW. The paper's reconciliation is that the fast transition-zone signal is the
stagnant slab itself. A slab passing through, or one that has begun to sink,
continues below 660 km; a feature confined to the transition zone does not. The
depth at which the fast anomaly stops therefore discriminates between them.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, dataclasses
import plume_classifier as pc
import s17_wet_plume as s17
import contmask
from s16_plume_null import random_rotation, to_xyz, to_lonlat

N_SPIN = int(os.environ.get('N_SPIN', 1000))
import paths as P
DATA = P.TOMO
CAT = os.environ.get('CATALOGUE', P.IPV_V3)
SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
depth, lat, lon, arr = pc.load_anomaly(SPEC, f'{DATA}/REVEAL_vs_full.nc')

BANDS = [(0, 100), (100, 200), (200, 300), (300, 410), (410, 520), (520, 660),
         (660, 800), (800, 1000), (1000, 1300), (1300, 1600), (1600, 2000),
         (2000, 2500)]
NLON, DLAT, DLON = len(lon), abs(lat[1] - lat[0]), abs(lon[1] - lon[0])

def integral(m):
    t = np.concatenate([m, m], axis=1).astype(np.float64)
    return np.pad(t.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

MAPS = {}
for lo_, hi_ in BANDS:
    k = (depth >= lo_) & (depth < hi_)
    MAPS[(lo_, hi_)] = integral(arr[k].mean(axis=0)) if k.any() else None

def nearest(axis, values):
    return np.abs(np.asarray(axis)[None, :]
                  - np.asarray(values, float)[:, None]).argmin(axis=1)

def boxes(la, lo, r=s17.SAMPLE_RADIUS_DEG):
    lo = ((np.asarray(lo, float) + 180.0) % 360.0) - 180.0
    jl = nearest(lat, la); dj = max(1, int(round(r / DLAT)))
    j0 = np.clip(jl - dj, 0, len(lat)); j1 = np.clip(jl + dj + 1, 0, len(lat))
    cl = np.maximum(np.cos(np.radians(np.asarray(la, float))), 0.05)
    di = np.maximum(1, np.round(r / DLON / cl).astype(int))
    i0 = (nearest(lon, lo) - di) % NLON
    return j0, j1, i0, i0 + 2 * di + 1

def profile(la, lo):
    j0, j1, i0, i1 = boxes(la, lo)
    out = {}
    for b, img in MAPS.items():
        if img is None:
            out[b] = np.nan; continue
        tot = img[j1, i1] - img[j0, i1] - img[j1, i0] + img[j0, i0]
        out[b] = float(np.nanmean(tot / ((j1 - j0) * (i1 - i0))))
    return out

d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180'])
la0, lo0 = d.lat.values.astype(float), d.lon_180.values.astype(float)
obs = profile(la0, lo0)
xyz0 = to_xyz(lo0, la0)
frac = float(contmask.is_continental(lo0, la0).mean())
print(f'{len(d)} fields, REVEAL Voigt, {N_SPIN} rotations\n')

def spin(cont):
    rng = np.random.default_rng(17); out = []
    while len(out) < N_SPIN:
        xyz = xyz0 @ random_rotation(rng).T
        disp = np.degrees(np.arccos(np.clip((xyz * xyz0).sum(1), -1, 1)))
        if np.median(disp) < s17.MIN_DISPLACE_DEG:
            continue
        slo, sla = to_lonlat(xyz)
        if cont:
            k = contmask.is_continental(slo, sla)
            if k.sum() < max(50, 0.25 * len(slo)):
                continue
            slo, sla = slo[k], sla[k]
        out.append(profile(sla, slo))
    return pd.DataFrame(out)

free, cont = spin(False), spin(True)
def tail(o, n):
    n = np.asarray(n, float)
    return (((n >= o).sum() + 1) / (len(n) + 1) if o >= np.median(n)
            else ((n <= o).sum() + 1) / (len(n) + 1))

print(f"{'band (km)':>14s} {'observed':>9s} | {'free med':>9s} {'p':>7s} | "
      f"{'cont med':>9s} {'p':>7s}   sign")
rows = []
for b in BANDS:
    o = obs[b]
    if not np.isfinite(o):
        continue
    a, c = free[b].values, cont[b].values
    pf, pc_ = tail(o, a), tail(o, c)
    s = 'FAST' if o > 0 else 'slow'
    star = ' *' if pc_ < 0.05 else ''
    print(f'{b[0]:6d}-{b[1]:<7d} {o:+9.3f} | {np.median(a):+9.3f} {pf:7.3f} | '
          f'{np.median(c):+9.3f} {pc_:7.3f}   {s}{star}')
    rows.append(dict(lo=b[0], hi=b[1], obs=o, free_med=np.median(a), p_free=pf,
                     cont_med=np.median(c), p_cont=pc_))
pd.DataFrame(rows).to_csv(os.path.join(P.OUT, 'depth_profile.csv'), index=False)

# The figure needs the null distributions, not just their medians, and it must not
# recompute them: two scripts drawing their own rotations is how a figure and a
# table come to disagree. Everything downstream reads this file.
np.savez_compressed(os.path.join(P.OUT, 'depth_profile_nulls.npz'),
                    bands=np.array(BANDS, float),
                    obs=np.array([obs[b] for b in BANDS], float),
                    free=free[list(BANDS)].values,
                    cont=cont[list(BANDS)].values,
                    n_spin=N_SPIN)
print('\n* continental-null p < 0.05.')
print('wrote out/depth_profile.csv and out/depth_profile_nulls.npz')
