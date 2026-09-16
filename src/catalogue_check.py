"""Does the headline transition-zone result depend on the catalogue version?

The manuscript uses a 539-field GEOROC catalogue built by single-linkage
clustering. The current builder grid-bins above 2000 points, because linkage
chains at GEOROC density, and re-joins cells into named provinces; its output is
385 fields. The two share no field names, so this is a rebuild.

The claim under test is that the transition zone beneath continental intraplate
volcanism is FAST, the opposite of what the hydration model predicts. If that
holds on both catalogues the choice between them is presentational.

Free rotation null only: the continental-restricted null needs a plate model that
is not on disk, and it is not required to compare two catalogues with each other.
"""
import warnings, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import plume_classifier as pc
import s17_wet_plume as s17
from s16_plume_null import random_rotation, to_xyz, to_lonlat

N_SPIN = 300
import paths as P
FILE = P.REVEAL

spec = pc.MODELS['REVEAL']
depth, lat, lon, arr = pc.load_anomaly(spec, FILE)
print(f'REVEAL: {arr.shape[0]} shells {depth.min():.0f}-{depth.max():.0f} km, '
      f'{arr.shape[1]}x{arr.shape[2]}\n')

# The cube has no NaNs, so averaging the shells of a band first and then taking
# the box mean is exactly band_mean, and turns 300 rotations from minutes into
# seconds. Verified against band_mean below before use.
def band_map(d0, d1):
    k = (depth >= d0) & (depth < d1)
    return arr[k].mean(axis=0)

MAP_TZ = band_map(*s17.TZ)
MAP_MELT = band_map(*s17.MELT_LAYER)
MAP_REF = band_map(250.0, 350.0)

def sample(m, hlat, hlon, radius=s17.SAMPLE_RADIUS_DEG):
    out = np.empty(len(hlat))
    for i, (a, o) in enumerate(zip(hlat, hlon)):
        jj, ii = pc.box_indices(lat, lon, a, o, radius)
        out[i] = m[np.ix_(jj, ii)].mean()
    return out

def mtz_mean(hlat, hlon):
    return sample(MAP_TZ, hlat, hlon)

def melt_contrast(hlat, hlon):
    return sample(MAP_MELT, hlat, hlon) - sample(MAP_REF, hlat, hlon)

import os
CATS = [('v2  (539, single-linkage)', os.path.join(P.TOMO, 'ipv_catalogue_georoc_v2.csv')),
        ('v3  (385, grid + rejoin)', P.IPV_V3),
        ('GVP (91, Holocene)',       P.IPV_GVP)]

# equivalence check against the original before anything is quoted
_d = pd.read_csv(os.path.join(P.TOMO, 'ipv_catalogue_georoc_v2.csv')).head(40)
_a, _o = _d.lat.values.astype(float), _d.lon_180.values.astype(float)
_slow = np.array([s17.band_mean(arr, depth, lat, lon, a, o, *s17.TZ)
                  for a, o in zip(_a, _o)])
_fast = mtz_mean(_a, _o)
print(f'precomputed band map vs band_mean on 40 fields: '
      f'max |diff| = {np.nanmax(np.abs(_slow - _fast)):.2e}\n')
assert np.nanmax(np.abs(_slow - _fast)) < 1e-5   # float32 accumulation order, not a logic difference

rng = np.random.default_rng(17)
print(f"{'catalogue':28s} {'n':>5s} {'MTZ mean':>10s} {'null med':>9s} {'p':>7s}   "
      f"{'melt contrast':>14s} {'null med':>9s} {'p':>7s}")
for label, path in CATS:
    d = pd.read_csv(path)
    la, lo = d.lat.values.astype(float), d.lon_180.values.astype(float)
    obs_mtz  = float(np.nanmean(mtz_mean(la, lo)))
    obs_melt = float(np.nanmean(melt_contrast(la, lo)))
    xyz = to_xyz(lo, la)
    nm, nc = [], []
    for _ in range(N_SPIN):
        R = random_rotation(rng)
        blo, bla = to_lonlat(xyz @ R.T)
        nm.append(float(np.nanmean(mtz_mean(bla, blo))))
        nc.append(float(np.nanmean(melt_contrast(bla, blo))))
    nm, nc = np.array(nm), np.array(nc)
    tail = lambda o, n: (((n >= o).sum() + 1) / (len(n) + 1) if o >= np.median(n)
                         else ((n <= o).sum() + 1) / (len(n) + 1))
    print(f'{label:28s} {len(d):5d} {obs_mtz:+10.3f} {np.median(nm):+9.3f} '
          f'{tail(obs_mtz, nm):7.3f}   {obs_melt:+14.3f} {np.median(nc):+9.3f} '
          f'{tail(obs_melt, nc):7.3f}')
print()
print('MTZ mean: positive = FAST transition zone = opposite to the hydration prediction.')
print('melt contrast: negative would be a low-velocity layer above the 410.')
