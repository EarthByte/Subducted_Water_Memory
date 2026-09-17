"""Stage 17 — can we identify upper-mantle upwellings fed by a transition-zone water reservoir?

The hypothesis has a seismic signature, and it is NOT the deep-plume signature.
If hydrous material stored in the transition zone upwells across the 410 km
discontinuity, it crosses into a phase assemblage with far lower water capacity
and undergoes dehydration melting (Bercovici & Karato's water filter; the
big-mantle-wedge picture of Zhao and co-workers). That predicts three things that
a lower-mantle plume does not:

  1. Structure ROOTED IN THE TRANSITION ZONE and not continuing below 660 km. A
     deep plume continues; a TZ-sourced upwelling starts there.
  2. A low-velocity layer immediately ABOVE the 410, where the melt sits. The
     discriminant is a CONTRAST — 350-410 km more negative than 250-350 km —
     not an absolute anomaly, so it is insensitive to the overall amplitude of a
     hotspot's signature.
  3. Proximity to a STAGNANT SLAB in the transition zone, which is the water
     source. Slabs are cold and therefore fast, so REVEAL locates them directly
     and no reconstruction is needed for this part.

Predictions 1-3 are computed here per hotspot, and each is tested against a spin
null: the constellation is rotated rigidly on the sphere carrying its labels, so
the autocorrelation of both fields is preserved and only their alignment is
destroyed.

The reconstructed cumulative stored-water field enters as a fourth, weaker test.
It is weak for a reason recorded in RESULTS_tomography.md: that field correlates
POSITIVELY with shear velocity in the transition zone, i.e. it marks where slabs
went rather than where the mantle is hydrated, and it has no depth resolution at
all. It is included so the comparison is on the record, not because I expect it
to carry the argument.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
import plume_classifier as pc
from s16_plume_null import random_rotation, to_xyz, to_lonlat, rho

warnings.filterwarnings('ignore')

N_SPIN = 300
MIN_DISPLACE_DEG = 20.0
SEED = 20260804
R_EARTH = 6371.0

MELT_LAYER = (350.0, 410.0)     # where dehydration melt should sit
REFERENCE_LAYER = (250.0, 350.0)  # the column just above it, as the baseline
TZ = (410.0, 660.0)
SLAB_PCTL = 90.0                # per-shell percentile defining "fast" in the TZ
SAMPLE_RADIUS_DEG = 5.0


# --------------------------------------------------------------------------
def band_mean(arr, depth, lat, lon, hlat, hlon, d0, d1, radius=SAMPLE_RADIUS_DEG):
    jj, ii = pc.box_indices(lat, lon, hlat, hlon, radius)
    k = (depth >= d0) & (depth < d1)
    if not k.any():
        return np.nan
    b = arr[k][:, jj, :][:, :, ii]
    return float(np.nanmean(b)) if np.isfinite(b).any() else np.nan


def slab_tree(arr, depth, lat, lon):
    """Unit-sphere KD-tree over fast transition-zone cells: a stagnant-slab proxy.

    The anomaly is averaged over the whole 410-660 km band FIRST and thresholded
    once, so a cell qualifies only if it is fast through the transition zone.
    Thresholding shell by shell and taking the union would let a single-shell
    blip in any one of ~10 shells qualify, which flags a quarter of the globe.
    """
    k = (depth >= TZ[0]) & (depth < TZ[1])
    band = np.nanmean(arr[k], axis=0)
    f = np.isfinite(band)
    cut = np.percentile(band[f], SLAB_PCTL)
    m = f & (band >= cut)
    LO, LA = np.meshgrid(lon, lat)
    return cKDTree(to_xyz(LO[m], LA[m])), int(m.sum())


def dist_to_slab_km(tree, lons, lats):
    d, _ = tree.query(to_xyz(np.asarray(lons, float), np.asarray(lats, float)))
    return R_EARTH * 2.0 * np.arcsin(np.clip(d / 2.0, 0, 1))


def load_water(folder):
    f = sorted(glob.glob(os.path.join(folder, '*_0.nc')))
    if not f:
        return None
    ds = xr.open_dataset(f[0])
    var = 'z' if 'z' in ds.data_vars else [v for v in ds.data_vars
                                           if ds[v].ndim == 2][0]
    lonn = 'lon' if 'lon' in ds.coords else 'x'
    latn = 'lat' if 'lat' in ds.coords else 'y'
    g = (np.asarray(ds[lonn].values, float), np.asarray(ds[latn].values, float),
         np.asarray(ds[var].values, float))
    ds.close()
    return g


def sample_water(g, lons, lats):
    glon, glat, gz = g
    lo = ((np.asarray(lons, float) - glon[0]) / (glon[1] - glon[0]))
    la = ((np.asarray(lats, float) - glat[0]) / (glat[1] - glat[0]))
    i = np.clip(np.round(lo).astype(int), 0, len(glon) - 1)
    j = np.clip(np.round(la).astype(int), 0, len(glat) - 1)
    return gz[j, i]


def root_class(row):
    d = row.max_depth_premerge
    if not np.isfinite(d):
        return 'unclassified'
    if d < TZ[0]:
        return 'shallow (<410)'
    if d < TZ[1]:
        return 'TZ-rooted (410-660)'
    return 'sub-660'


# --------------------------------------------------------------------------
def descriptors(lons, lats, depth, lat, lon, arr, tree, water):
    melt = np.array([band_mean(arr, depth, lat, lon, a, o, *MELT_LAYER)
                     for o, a in zip(lons, lats)])
    ref = np.array([band_mean(arr, depth, lat, lon, a, o, *REFERENCE_LAYER)
                    for o, a in zip(lons, lats)])
    contrast = melt - ref                       # negative = low-V layer atop 410
    # mean anomaly through the transition zone itself: the direct analogue of
    # Wang et al. (2025), who report hydrous MTZ beneath continental intraplate
    # volcanism.  Negative = slow = the sign hydration predicts.
    mtz = np.array([band_mean(arr, depth, lat, lon, a, o, *TZ)
                    for o, a in zip(lons, lats)])
    dslab = dist_to_slab_km(tree, lons, lats)
    w = sample_water(water, lons, lats) if water else np.full(len(lons), np.nan)
    return contrast, dslab, w, mtz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='REVEAL', choices=list(pc.MODELS))
    ap.add_argument('--file', default=None)
    ap.add_argument('--var', default=None)
    ap.add_argument('--src', default=pc.DATA)
    ap.add_argument('--hotspots', default=pc.HOTSPOTS)
    ap.add_argument('--summary', default=None,
                    help='plume_summary_<tag>.csv from plume_classifier.py')
    ap.add_argument('--water', default=None, help='folder of cumulative water grids')
    ap.add_argument('--out', default='.')
    ap.add_argument('--spins', type=int, default=N_SPIN)
    a = ap.parse_args()

    spec = pc.MODELS[a.model]
    if a.var:
        spec = pc.ModelSpec(spec.name, spec.file, a.var, spec.anomaly_label)
    path = a.file if a.file and os.path.isabs(a.file) else os.path.join(
        a.src, a.file or spec.file)
    depth, lat, lon, arr = pc.load_anomaly(spec, path)
    tag = f'{spec.name}_{spec.velocity_var}_{int(round(depth.max()))}km'
    print(f'{tag}: {len(depth)} shells to {depth.max():.0f} km', flush=True)

    hs = pd.read_csv(a.hotspots).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    lon0 = hs.lon_180.values.astype(float)
    lat0 = hs.lat.values.astype(float)

    tree, nslab = slab_tree(arr, depth, lat, lon)
    print(f'stagnant-slab proxy: {nslab} surface cells with a fast '
          f'({SLAB_PCTL:.0f}th pctl) transition-zone anomaly', flush=True)

    water = load_water(a.water) if a.water else None
    print(f'reconstructed stored water at 0 Ma: '
          f'{"loaded" if water else "not supplied"}\n', flush=True)

    contrast, dslab, w, mtz = descriptors(lon0, lat0, depth, lat, lon, arr, tree, water)
    hs['melt_layer_contrast'] = contrast
    hs['dist_to_TZ_slab_km'] = dslab
    hs['stored_water_0Ma'] = w
    hs['mtz_mean_anomaly'] = mtz

    if a.summary and os.path.exists(a.summary):
        s = pd.read_csv(a.summary)
        hs = hs.merge(s[['hotspot', 'max_depth_premerge', 'verdict']],
                      on='hotspot', how='left')
        hs['root_class'] = hs.apply(root_class, axis=1)
    else:
        hs['root_class'] = 'unclassified'

    # ------------------------------------------------------------------ W1
    print('=' * 84)
    print('W1  where does the structure bottom out?')
    print('    a TZ-sourced upwelling should root in 410-660 and NOT continue below')
    print('=' * 84)
    for k, v in hs.root_class.value_counts().items():
        print(f'  {k:24s} {v:3d}')
    tzr = hs[hs.root_class == 'TZ-rooted (410-660)']
    if len(tzr):
        print(f'\n  TZ-rooted hotspots: {", ".join(sorted(tzr.hotspot))}')

    # ------------------------------------------------------------------ W2
    print('\n' + '=' * 84)
    print('W2  is there a low-velocity layer immediately above the 410?')
    print(f'    contrast = mean dv({MELT_LAYER[0]:.0f}-{MELT_LAYER[1]:.0f}) - '
          f'mean dv({REFERENCE_LAYER[0]:.0f}-{REFERENCE_LAYER[1]:.0f}); negative = melt layer')
    print('=' * 84)
    print(f'  all hotspots      median {np.nanmedian(contrast):+.3f} %, '
          f'{int(np.nansum(contrast < 0))} of {np.isfinite(contrast).sum()} negative')
    for k, g in hs.groupby('root_class'):
        if len(g) >= 4:
            print(f'  {k:24s} median {g.melt_layer_contrast.median():+.3f} %  (n={len(g)})')

    # ------------------------------------------------------------------ spins
    rng = np.random.default_rng(SEED)
    xyz0 = to_xyz(lon0, lat0)
    null = []
    done = 0
    while done < a.spins:
        R = random_rotation(rng)
        xyz = xyz0 @ R.T
        disp = np.degrees(np.arccos(np.clip((xyz * xyz0).sum(1), -1, 1)))
        if np.median(disp) < MIN_DISPLACE_DEG:
            continue
        slon, slat = to_lonlat(xyz)
        null.append(descriptors(slon, slat, depth, lat, lon, arr, tree, water))
        done += 1
        if done % 50 == 0:
            print(f'  spin {done}/{a.spins}', flush=True)

    print('\n' + '=' * 92)
    print(f'W3  spin test, {a.spins} rigid rotations, median displacement >= '
          f'{MIN_DISPLACE_DEG:.0f} deg')
    print('=' * 92)
    print(f'{"comparison":52s} {"stat":>8s} {"null 2.5-97.5%":>20s} {"p":>7s}')
    res = []

    def report(label, obs, nullvals):
        nullvals = np.asarray([x for x in nullvals if np.isfinite(x)], float)
        if not np.isfinite(obs) or len(nullvals) < 30:
            print(f'{label:52s}   degenerate')
            return
        p = ((np.abs(nullvals) >= abs(obs)).sum() + 1) / (len(nullvals) + 1)
        print(f'{label:52s} {obs:+8.3f} '
              f'[{np.percentile(nullvals, 2.5):+6.3f},{np.percentile(nullvals, 97.5):+6.3f}] '
              f'{p:7.3f}{"  SURVIVES" if p < 0.05 else ""}')
        res.append(dict(comparison=label, stat=obs,
                        null_lo=np.percentile(nullvals, 2.5),
                        null_hi=np.percentile(nullvals, 97.5), p=p, n_null=len(nullvals)))

    # is the melt-layer contrast at hotspots more negative than at rotated positions?
    report('melt-layer contrast, mean over hotspots (% )',
           float(np.nanmean(contrast)),
           [np.nanmean(c) for c, _, _, _ in null])

    # do hotspots sit closer to transition-zone slabs than chance?
    report('median distance to TZ slab (km, / 1000)',
           float(np.nanmedian(dslab) / 1000.0),
           [np.nanmedian(d) / 1000.0 for _, d, _, _ in null])

    # Wang et al. (2025) analogue: is the transition zone slow beneath hotspots?
    report('mean transition-zone anomaly at hotspots (%)',
           float(np.nanmean(mtz)), [np.nanmean(m) for _, _, _, m in null])

    # does the melt layer strengthen near slabs?  (negative rho = yes)
    report('rho(melt-layer contrast, distance to TZ slab)',
           rho(contrast, dslab),
           [rho(c, d) for c, d, _, _ in null])

    if water:
        report('rho(melt-layer contrast, reconstructed stored water)',
               rho(contrast, w), [rho(c, ww) for c, _, ww, _ in null])
        report('rho(distance to TZ slab, reconstructed stored water)',
               rho(dslab, w), [rho(d, ww) for _, d, ww, _ in null])
        report('rho(transition-zone anomaly, reconstructed stored water)',
               rho(mtz, w), [rho(m, ww) for _, _, ww, m in null])

    os.makedirs(a.out, exist_ok=True)
    hs.to_csv(os.path.join(a.out, f'wet_plume_descriptors_{tag}.csv'), index=False)
    pd.DataFrame(res).to_csv(os.path.join(a.out, f'wet_plume_spin_{tag}.csv'), index=False)
    print(f'\nwrote wet_plume_descriptors_{tag}.csv and wet_plume_spin_{tag}.csv')


if __name__ == '__main__':
    main()
