"""Table 1 — what the spin null gives, and what it gives when its domain is right.

The script that produced the draft's Table 1 was lost with a cloud session, so
this rebuilds it. Two things are done differently and deliberately:

  * The tail is one-sided on the side of the null median the observation falls on,
    which is what the manuscript's methods section specifies. `s17.report` instead
    compares absolute values, which the same paragraph says is valid only for a
    statistic whose null is centred on zero and "inverts the result for a
    distance" - and the first row of this table is a distance.
  * 5000 rotations rather than 300, so that a reported p means what it says. At
    300 the finest resolvable value is 1/301 = 0.0033, and quoting "p = 0.003"
    when the truth is "no rotation out of 300 exceeded" is the kind of thing this
    paper is about.

The continental null accepts a rotation only if the rotated constellation sits on
continental lithosphere in the same proportion as the observed one does, so that
the null population resembles the cases in the one respect that matters.
"""
import warnings, sys, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

import plume_classifier as pc
import s17_wet_plume as s17
import contmask
from s16_plume_null import random_rotation, to_xyz, to_lonlat

R_EARTH = 6371.0
TZ, MELT, REF = (410.0, 660.0), (350.0, 410.0), (250.0, 350.0)
SLAB_PCTL, MIN_BODY_KM2 = 90.0, 4.0e5
N_SPIN = int(os.environ.get('N_SPIN', 5000))
import paths as P
DATA = P.TOMO
CAT = os.environ.get('CATALOGUE', P.IPV_V3)

# The default ModelSpec for REVEAL carries velocity_var='vsv'. The manuscript
# uses the Voigt average of the vertically and horizontally polarised shear
# velocities, and the two give materially different fast-body geometry.
import dataclasses
SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
depth, lat, lon, arr = pc.load_anomaly(SPEC, f'{DATA}/REVEAL_vs_full.nc')
print(f"velocity variable: {SPEC.velocity_var}")
print(f'REVEAL {arr.shape[0]} shells, {arr.shape[1]}x{arr.shape[2]}; '
      f'catalogue {os.path.basename(CAT)}; {N_SPIN} rotations\n')

def band_map(d0, d1):
    k = (depth >= d0) & (depth < d1)
    return arr[k].mean(axis=0)

MAP_TZ, MAP_MELT, MAP_REF = band_map(*TZ), band_map(*MELT), band_map(*REF)

# ---- fast transition-zone bodies, as s19 defines them
f = np.isfinite(MAP_TZ)
mask = f & (MAP_TZ >= np.percentile(MAP_TZ[f], SLAB_PCTL))
labels, n0 = ndimage.label(mask, structure=np.ones((3, 3)))
left, right = labels[:, 0], labels[:, -1]                     # stitch the seam
both = (left > 0) & (right > 0)
pairs = {(min(a, b), max(a, b)) for a, b in zip(left[both], right[both]) if a != b}
if pairs:
    parent = np.arange(n0 + 1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    labels = np.array([find(i) for i in range(n0 + 1)])[labels]
dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
area = np.repeat((((np.radians(dlat) * R_EARTH) *
                   (np.radians(dlon) * R_EARTH * np.cos(np.radians(lat)))))[:, None],
                 len(lon), axis=1)
keep = [i for i in np.unique(labels) if i > 0 and area[labels == i].sum() >= MIN_BODY_KM2]
# TARGET selects what a distance is measured to. 'bodies' is what the methods
# section describes - the coherent bodies above the area threshold, small patches
# having been discarded as noise. 'cells' is every cell of the top-decile mask,
# which is what the draft's distance numbers were actually computed against; it
# reproduces its 115 km overall and 65 km for northeast Asia exactly.
TARGET = os.environ.get('TARGET', 'bodies')
body = np.isin(labels, keep) if TARGET == 'bodies' else mask
LO, LA = np.meshgrid(lon, lat)
print(f'{len(keep)} coherent bodies above {MIN_BODY_KM2:.0e} km2 = '
      f'{100 * area[np.isin(labels, keep)].sum() / area.sum():.1f} per cent of the surface; '
      f'the full top-decile mask = {100 * area[mask].sum() / area.sum():.1f} per cent')
print(f'distances measured to: {TARGET}')
cont = contmask.is_continental(LO.ravel(), LA.ravel()).reshape(LO.shape)
w = np.cos(np.radians(LA))
print(f'continental fraction of the surface {100 * (cont * w).sum() / w.sum():.1f} '
      f'per cent; of the fast-TZ area {100 * area[body & cont].sum() / area[body].sum():.1f} '
      f'per cent\n')
tree = cKDTree(to_xyz(LO[body], LA[body]))

# ---- the four statistics
# A box mean is four lookups on an integral image. The longitude index wraps, so
# the image is tiled once in longitude and the wrapped box read as one rectangle.
# Verified against pc.box_indices below before use.
NLON = len(lon)
def integral(m):
    # float64: a float32 cumulative sum over a 361 x 1442 tile accumulates enough
    # error to shift a box mean in the fifth decimal.
    t = np.concatenate([m, m], axis=1).astype(np.float64)
    return np.pad(t.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

II_TZ, II_MELT, II_REF = integral(MAP_TZ), integral(MAP_MELT), integral(MAP_REF)
DLAT, DLON = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])

def nearest(axis, values):
    """Vectorised pc.nearest_index. It is argmin of |axis - value|, so a value
    landing exactly on a cell boundary takes the LOWER index; np.round would take
    the even one and shift the box by a column, which moves a box mean by five
    per cent at points like lon = -3.25 on a 0.5 degree axis."""
    return np.abs(np.asarray(axis)[None, :]
                  - np.asarray(values, float)[:, None]).argmin(axis=1)

def boxes(la, lo, r=s17.SAMPLE_RADIUS_DEG):
    lo = ((np.asarray(lo, float) + 180.0) % 360.0) - 180.0
    jl = nearest(lat, la)
    dj = max(1, int(round(r / DLAT)))
    j0 = np.clip(jl - dj, 0, len(lat)); j1 = np.clip(jl + dj + 1, 0, len(lat))
    coslat = np.maximum(np.cos(np.radians(np.asarray(la, float))), 0.05)
    di = np.maximum(1, np.round(r / DLON / coslat).astype(int))
    il = nearest(lon, lo)
    i0 = (il - di) % NLON; i1 = i0 + 2 * di + 1
    return j0, j1, i0, i1

def box_mean(ii_img, j0, j1, i0, i1):
    tot = (ii_img[j1, i1] - ii_img[j0, i1] - ii_img[j1, i0] + ii_img[j0, i0])
    return tot / ((j1 - j0) * (i1 - i0))

def sample(m, la, lo, r=s17.SAMPLE_RADIUS_DEG):
    out = np.empty(len(la))
    for i, (a, o) in enumerate(zip(la, lo)):
        jj, ii = pc.box_indices(lat, lon, a, o, r)
        out[i] = m[np.ix_(jj, ii)].mean()
    return out

def stats(la, lo):
    d, _ = tree.query(to_xyz(lo, la))
    dist = R_EARTH * 2.0 * np.arcsin(np.clip(d / 2.0, 0, 1))
    j0, j1, i0, i1 = boxes(la, lo)
    mtz = box_mean(II_TZ, j0, j1, i0, i1)
    contrast = (box_mean(II_MELT, j0, j1, i0, i1)
                - box_mean(II_REF, j0, j1, i0, i1))
    ok = np.isfinite(contrast) & np.isfinite(dist)
    r = (np.corrcoef(pd.Series(contrast[ok]).rank(), pd.Series(dist[ok]).rank())[0, 1]
         if ok.sum() > 5 else np.nan)
    return dict(dist=float(np.median(dist)), mtz=float(np.nanmean(mtz)),
                rho=float(r), contrast=float(np.nanmean(contrast)))

_chk = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).head(60)
_a, _o = _chk.lat.values.astype(float), _chk.lon_180.values.astype(float)
_slow = sample(MAP_TZ, _a, _o)
_j0, _j1, _i0, _i1 = boxes(_a, _o)
_fast = box_mean(II_TZ, _j0, _j1, _i0, _i1)
print(f'integral image vs box_indices on 60 fields: '
      f'max |diff| = {np.nanmax(np.abs(_slow - _fast)):.2e}')
assert np.nanmax(np.abs(_slow - _fast)) < 1e-5

d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180'])
la0, lo0 = d.lat.values.astype(float), d.lon_180.values.astype(float)
obs = stats(la0, lo0)
xyz0 = to_xyz(lo0, la0)
frac_cont_obs = float(contmask.is_continental(lo0, la0).mean())
print(f'{len(d)} fields, {100 * frac_cont_obs:.0f} per cent on continental lithosphere\n')

def spin(require_continental):
    rng = np.random.default_rng(17)
    out, tries = [], 0
    while len(out) < N_SPIN and tries < N_SPIN * 400:
        tries += 1
        xyz = xyz0 @ random_rotation(rng).T
        disp = np.degrees(np.arccos(np.clip((xyz * xyz0).sum(1), -1, 1)))
        if np.median(disp) < s17.MIN_DISPLACE_DEG:
            continue
        slo, sla = to_lonlat(xyz)
        if require_continental:
            # A rigid rotation of a continental constellation lands most of it in
            # ocean, so demanding the whole set stay continental accepts nothing.
            # Keep the points that do land on continental lithosphere, which
            # preserves the constellation's geometry while restricting the null
            # to the domain the cases were drawn from, and require enough of them
            # for the statistics to mean anything.
            k = contmask.is_continental(slo, sla)
            if k.sum() < max(50, 0.25 * len(slo)):
                continue
            slo, sla = slo[k], sla[k]
        out.append(stats(sla, slo))
    return pd.DataFrame(out), tries

def tail(o, n):
    n = np.asarray([x for x in n if np.isfinite(x)], float)
    return (((n >= o).sum() + 1) / (len(n) + 1) if o >= np.median(n)
            else ((n <= o).sum() + 1) / (len(n) + 1))

ROWS = [('median distance to fast TZ anomaly (km)', 'dist', '{:.0f}'),
        ('mean transition-zone anomaly (per cent)', 'mtz', '{:+.3f}'),
        ('rho(melt contrast, distance to slab)', 'rho', '{:+.3f}'),
        ('mean melt-layer contrast (per cent)', 'contrast', '{:+.3f}')]
# A constellation that is not continental cannot supply a continental null: the
# acceptance rule would reject every rotation. NULLS=free reports the free null
# alone, which is what the oceanic hotspot population needs.
WANT = os.environ.get('NULLS', 'free,continental').split(',')
res = {}
for lab, req in (('free', False), ('continental', True)):
    if lab not in WANT:
        res[lab] = None
        continue
    nul, tries = spin(req)
    res[lab] = nul
    print(f'{lab} null: {len(nul)} rotations accepted from {tries} tried')
print()
print(f"{'test':42s} {'observed':>10s} | {'free null':>10s} {'p':>7s} | "
      f"{'cont null':>10s} {'p':>7s}")
print('-' * 96)
def col(nul, key, fmt, o):
    if nul is None:
        return f'{"-":>10s} {"-":>7s}'
    v = nul[key].values
    return f'{fmt.format(np.median(v)):>10s} {tail(o, v):7.4f}'

for lab, key, fmt in ROWS:
    o = obs[key]
    print(f'{lab:42s} {fmt.format(o):>10s} | {col(res["free"], key, fmt, o)} '
          f'| {col(res["continental"], key, fmt, o)}')
print()
print('One-sided, on the side of the null median the observation falls on.')

# Written as well as printed. Every other stage leaves its numbers on disk, and
# a table that exists only in a terminal cannot be checked against the
# manuscript later, nor re-read after the null generator changes.
_rows = []
for lab, key, fmt in ROWS:
    r = dict(test=lab, observed=obs[key])
    for nm in ('free', 'continental'):
        nul = res.get(nm)
        r[f'{nm}_null'] = (float(np.median(nul[key].values))
                           if nul is not None else float('nan'))
        r[f'{nm}_p'] = (float(tail(obs[key], nul[key].values))
                        if nul is not None else float('nan'))
        r[f'{nm}_n'] = int(len(nul)) if nul is not None else 0
    _rows.append(r)
_out = os.path.join(P.OUT, 'table1.csv')
pd.DataFrame(_rows).to_csv(_out, index=False)
print(f'wrote {_out}')
