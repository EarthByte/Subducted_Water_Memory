"""Automated characterisation of mantle structure beneath hotspots.

Model-agnostic by construction. It consumes the same ModelSpec abstraction and
the same lateral-mean-removed percent anomaly as `generate_cross_sections.py`
from the GPlately-pyGMT tutorials, so the numbers here and the cross-sections
you inspect visually are identical. Swapping REVEAL for its successor, or for
SPiRaL / GLADM35 / SEMUCB-WM1, is a command-line argument.

    python plume_classifier.py --model REVEAL
    python plume_classifier.py --model REVEAL --file REVEAL_vs_full.nc --var voigt
    python plume_classifier.py --model SPiRaL --file SPiRaL.nc --var vsv

WHY THIS IS NOT A "PLUME DEPTH" CODE
------------------------------------
A first version thresholded the slow field and followed the connected component
containing each hotspot downward. It reported 31 of 49 hotspots as continuously
slow to 1000 km. That number was an artefact, and the diagnostic that catches it
is now stage A of the run: the slow field PERCOLATES. Between the 5th and 10th
percentile the largest connected component jumps from 16% to 68% of the mask and
spans the entire depth range. Above the percolation threshold, "connected to the
hotspot" means "connected to everything", and any depth extent read off it is a
property of the threshold, not of the Earth.

So connectivity is used only below the measured percolation threshold, and the
structural quantity reported is not a depth but a PERSISTENCE: the threshold at
which the hotspot's own structure loses its identity by merging into the global
network. That is a merge-tree quantity, in the spirit of Kamakshidasan et al.
(2026), and it degrades gracefully where a depth does not.

Three further constraints come from the plume-complexity literature and are built
in rather than bolted on:

  * Lin & Van Keken (2006) show vertical continuity of a low-velocity signature
    is not a universally valid mapping criterion for thermochemical plumes. So
    continuity is reported as one descriptor among several, never as the class.
  * Poletto (2015) and Puthenveettil et al. (2004) find plume structure to be
    fractal/multifractal, with no unique characteristic diameter. So the lateral
    descriptors are computed at several radii and reported as a profile, and no
    single width is claimed.
  * Koppers et al. (2021) and Lu & Rudolph (2024) describe conduits that branch,
    tilt and are entrained rather than rising as cylinders. So branching and
    interruption are recorded, not resolved away.

The threshold-free descriptors in stage B do not depend on connectivity at all
and are therefore immune to the percolation failure. They are the part of this
that is safe to trust.

Outputs
  plume_columns_<MODEL>.csv     threshold-free amplitude/coherence profiles
  plume_persistence_<MODEL>.csv connectivity descriptors, below p_c only
  plume_summary_<MODEL>.csv     one row per hotspot with an explicit verdict
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, stats

warnings.filterwarnings('ignore')

# Defaults are relative to the current directory so the scripts run from your
# project folder unmodified. Override any of them with --src / --hotspots / --out,
# or with the PLUME_SRC / PLUME_HOTSPOTS / PLUME_OUT environment variables.
DATA = os.environ.get('PLUME_SRC', '.')
HOTSPOTS = os.environ.get(
    'PLUME_HOTSPOTS',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'data', 'hotspots_courtillot2003.csv'))
OUT = os.environ.get('PLUME_OUT', '.')


# --------------------------------------------------------------------------
# model registry — mirrors ModelSpec in generate_cross_sections.py
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str
    velocity_var: str
    anomaly_label: str


MODELS = {
    'REVEAL':     ModelSpec('REVEAL', 'REVEAL_0_1000km.nc', 'vsv', 'dvsv/vsv in %'),
    'SPiRaL':     ModelSpec('SPiRaL', 'SPiRaL.nc', 'vsv', 'dvsv/vsv in %'),
    'GLADM35':    ModelSpec('GLADM35', 'GLADM35.nc', 'vsv', 'dvsv/vsv in %'),
    'SEMUCB-WM1': ModelSpec('SEMUCB-WM1', 'SEMUCB-WM1.nc', 'vs', 'dvs/vs in %'),
}

# thresholds are percentiles of the anomaly WITHIN each depth shell, because
# anomaly amplitude falls with depth and a fixed -1% cut silently becomes far
# stricter at 900 km than at 200 km
LADDER = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
GIANT_FRAC = 0.30          # a component holding this share of the mask has percolated
SEED_RADII = [2.0, 5.0, 10.0]   # degrees; no single one is privileged (Poletto 2015)
SEED_DEPTH = (100.0, 300.0)
MIN_DEPTH = 80.0
BANDS = [('lith_80_200', 80, 200), ('um_200_410', 200, 410),
         ('mtz_410_660', 410, 660), ('lm_660_1000', 660, 1000),
         ('deep_1000_2000', 1000, 2000), ('lowermost_2000+', 2000, 1e9)]


# --------------------------------------------------------------------------
def load_anomaly(spec: ModelSpec, path: str):
    """Lateral-mean-removed percent anomaly, exactly as generate_cross_sections does."""
    ds = xr.open_dataset(path)

    latn = 'latitude' if 'latitude' in ds.coords else 'lat'
    lonn = 'longitude' if 'longitude' in ds.coords else 'lon'

    if spec.velocity_var == 'voigt':
        v = np.sqrt((2.0 * ds['vsv'] ** 2 + ds['vsh'] ** 2) / 3.0)
    else:
        if spec.velocity_var not in ds:
            raise SystemExit(
                f"variable '{spec.velocity_var}' not in {os.path.basename(path)}; "
                f"present: {list(ds.data_vars)}")
        v = ds[spec.velocity_var]

    mean_layer = v.mean(dim=[latn, lonn])
    a = ((v - mean_layer) * 100.0 / mean_layer).clip(min=-100, max=100)

    # depth axis may be a dimension or a coordinate riding on one
    dname = None
    for cand in ('depth', 'depth_km', 'radius_nondim'):
        if cand in a.dims:
            dname = cand
            break
    if dname is None:
        dname = [d for d in a.dims if d not in (latn, lonn)][0]
    if 'depth_km' in ds.coords:
        depth = np.asarray(ds['depth_km'].values, float)
    elif 'depth' in ds.coords:
        depth = np.asarray(ds['depth'].values, float)
    else:
        depth = np.asarray(ds[dname].values, float)

    lat = np.asarray(ds[latn].values, float)
    lon = np.asarray(ds[lonn].values, float)
    arr = a.transpose(dname, latn, lonn).values.astype(np.float32)
    ds.close()

    if depth[0] > depth[-1]:                       # force increasing depth
        order = np.argsort(depth)
        depth, arr = depth[order], arr[order]
    return depth, lat, lon, arr


def shell_threshold_mask(arr, pct):
    m = np.empty(arr.shape, bool)
    for k in range(arr.shape[0]):
        s = arr[k]
        f = np.isfinite(s)
        if not f.any():
            m[k] = False
            continue
        m[k] = f & (s <= np.percentile(s[f], pct))
    return m


def label_3d(mask):
    """26-connected labelling with the +/-180 seam stitched."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return lab, 0
    left, right = lab[:, :, 0], lab[:, :, -1]
    both = (left > 0) & (right > 0)
    pairs = {(min(a, b), max(a, b))
             for a, b in zip(left[both].ravel(), right[both].ravel()) if a != b}
    if pairs:
        parent = np.arange(n + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        lab = np.array([find(i) for i in range(n + 1)])[lab]
    return lab, n


def nearest_index(axis, value):
    return int(np.argmin(np.abs(axis - value)))


def box_indices(lat, lon, hlat, hlon, radius_deg):
    """Index arrays for a great-circle cap of the given radius, seam-safe."""
    hlon = ((hlon + 180.0) % 360.0) - 180.0
    jlat = nearest_index(lat, hlat)
    dj = max(1, int(round(radius_deg / abs(lat[1] - lat[0]))))
    j0, j1 = max(0, jlat - dj), min(len(lat), jlat + dj + 1)
    coslat = max(np.cos(np.radians(hlat)), 0.05)
    di = max(1, int(round(radius_deg / abs(lon[1] - lon[0]) / coslat)))
    ilon = nearest_index(lon, hlon)
    ii = np.arange(ilon - di, ilon + di + 1) % len(lon)
    return np.arange(j0, j1), ii


# --------------------------------------------------------------------------
# stage A — percolation
# --------------------------------------------------------------------------
def percolation_scan(arr, depth):
    """Where does the slow field stop being a set of objects and become a network?"""
    rows, labels = [], {}
    for pct in LADDER:
        lab, _ = label_3d(shell_threshold_mask(arr, pct))
        labels[pct] = lab
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        if len(ids) == 0:
            rows.append(dict(pct=pct, n_comp=0, giant_frac=np.nan,
                             giant_span_km=np.nan, percolated=False))
            continue
        o = np.argsort(cnt)[::-1]
        frac = cnt[o[0]] / cnt.sum()
        kz = np.where((lab == ids[o[0]]).any(axis=(1, 2)))[0]
        rows.append(dict(pct=pct, n_comp=len(ids), giant_frac=float(frac),
                         giant_span_km=float(depth[kz[-1]] - depth[kz[0]]),
                         giant_id=int(ids[o[0]]),
                         percolated=bool(frac >= GIANT_FRAC)))
    df = pd.DataFrame(rows)
    perc = df[df.percolated]
    p_c = float(perc.pct.min()) if len(perc) else float('inf')
    return df, labels, p_c


# --------------------------------------------------------------------------
# stage B — threshold-free column descriptors (immune to percolation)
# --------------------------------------------------------------------------
_BAND_CUT = {}


def band_cuts(depth, arr):
    """Global 10th-percentile anomaly per depth band. Computed once, not per hotspot."""
    key = (arr.shape, float(depth[-1]))
    if key in _BAND_CUT:
        return _BAND_CUT[key]
    cuts = {}
    for label, d0, d1 in BANDS:
        k = (depth >= d0) & (depth < d1) & (depth >= MIN_DEPTH)
        if not k.any():
            continue
        g = arr[k]
        g = g[np.isfinite(g)]
        cuts[label] = float(np.percentile(g, 10)) if g.size else np.nan
    _BAND_CUT[key] = cuts
    return cuts


def column_descriptors(name, hlat, hlon, depth, lat, lon, arr):
    cuts = band_cuts(depth, arr)
    rows = []
    for r in SEED_RADII:
        jj, ii = box_indices(lat, lon, hlat, hlon, r)
        sub = arr[:, jj, :][:, :, ii]
        rec = dict(hotspot=name, radius_deg=r)
        for label, d0, d1 in BANDS:
            k = (depth >= d0) & (depth < d1) & (depth >= MIN_DEPTH)
            if not k.any():
                continue
            b = sub[k]
            if not np.isfinite(b).any():
                continue
            rec[f'{label}_mean'] = float(np.nanmean(b))
            rec[f'{label}_min'] = float(np.nanmin(b))
            # fraction of the local cap slower than the global 10th pct for that band
            rec[f'{label}_slowfrac'] = float(np.nanmean(b <= cuts.get(label, np.nan)))
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# stage C — persistence: at what threshold does this structure lose its identity?
# --------------------------------------------------------------------------
def persistence(name, hlat, hlon, depth, lat, lon, labels, scan):
    giant = {r['pct']: r.get('giant_id', -1) for r in scan.to_dict('records')}
    perc = {r['pct']: r['percolated'] for r in scan.to_dict('records')}

    k0, k1 = nearest_index(depth, SEED_DEPTH[0]), nearest_index(depth, SEED_DEPTH[1])
    jj, ii = box_indices(lat, lon, hlat, hlon, 3.0)

    out = []
    for pct in LADDER:
        lab = labels[pct]
        seed = set(np.unique(lab[min(k0, k1):max(k0, k1) + 1][:, jj, :][:, :, ii]))
        seed.discard(0)
        if not seed:
            out.append(dict(hotspot=name, pct=pct, seeded=False, merged=np.nan,
                            max_depth=np.nan, continuous_to=np.nan, n_breaks=np.nan,
                            break_depths='', n_seed_comp=0, tilt_deg=np.nan,
                            below_pc=not perc[pct]))
            continue
        merged = giant.get(pct, -1) in seed and perc[pct]
        # membership judged in a generous 15 deg cap so a tilted conduit is followed
        jw, iw = box_indices(lat, lon, hlat, hlon, 15.0)
        hit3 = np.isin(lab[:, jw, :][:, :, iw], list(seed))
        present, cents = [], []
        for k in range(len(depth)):
            if depth[k] < MIN_DEPTH:
                present.append(False)
                cents.append((np.nan, np.nan))
                continue
            sl = hit3[k]
            present.append(bool(sl.any()))
            if sl.any():
                a, b = np.nonzero(sl)
                cents.append((float(lat[jw][a].mean()), float(lon[iw][b].mean())))
            else:
                cents.append((np.nan, np.nan))
        present = np.array(present)
        idx = np.where(present)[0]
        if len(idx) == 0:
            out.append(dict(hotspot=name, pct=pct, seeded=True, merged=merged,
                            max_depth=np.nan, continuous_to=np.nan, n_breaks=np.nan,
                            break_depths='', n_seed_comp=len(seed), tilt_deg=np.nan,
                            below_pc=not perc[pct]))
            continue
        cont = idx[0]
        for k in range(idx[0], len(depth)):
            if present[k]:
                cont = k
            else:
                break
        breaks, kk = [], cont
        while kk < len(depth) - 1:
            if present[kk] and not present[kk + 1]:
                nxt = np.where(present[kk + 1:])[0]
                if not len(nxt):
                    break
                breaks.append((depth[kk], depth[kk + 1 + nxt[0]]))
                kk = kk + nxt[0]
            kk += 1
        c = [x for x in cents if np.isfinite(x[0])]
        tilt = (float(np.hypot(c[-1][0] - c[0][0],
                               (c[-1][1] - c[0][1]) * np.cos(np.radians(hlat))))
                if len(c) > 2 else np.nan)
        out.append(dict(hotspot=name, pct=pct, seeded=True, merged=bool(merged),
                        max_depth=float(depth[idx[-1]]),
                        continuous_to=float(depth[cont]),
                        n_breaks=len(breaks),
                        break_depths=';'.join(f'{a:.0f}-{b:.0f}' for a, b in breaks),
                        n_seed_comp=len(seed), tilt_deg=tilt,
                        below_pc=not perc[pct]))
    return out


# --------------------------------------------------------------------------
def main():
    global HOTSPOTS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='REVEAL', choices=list(MODELS))
    ap.add_argument('--file', default=None, help='override the model file')
    ap.add_argument('--var', default=None,
                    help="velocity variable; 'voigt' computes sqrt((2 vsv^2 + vsh^2)/3)")
    ap.add_argument('--src', default=DATA,
                    help='folder holding the tomography file (default: cwd)')
    ap.add_argument('--hotspots', default=HOTSPOTS,
                    help='Courtillot table CSV')
    ap.add_argument('--out', default=OUT, help='where to write the CSVs')
    a = ap.parse_args()

    HOTSPOTS, OUT = a.hotspots, a.out
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(HOTSPOTS):
        raise SystemExit(f'hotspot table not found: {HOTSPOTS}\n'
                         '  pass --hotspots /path/to/courtillot_2003_table1.csv')

    spec = MODELS[a.model]
    if a.var:
        spec = ModelSpec(spec.name, spec.file, a.var, spec.anomaly_label)
    path = a.file if a.file and os.path.isabs(a.file) else os.path.join(
        a.src, a.file or spec.file)
    if not os.path.exists(path):
        raise SystemExit(f'model file not found: {path}')

    depth, lat, lon, arr = load_anomaly(spec, path)
    # tag output by model, variable and depth range, so a 0-1000 km run and a
    # full-depth run of the same model do not overwrite each other
    tag = f'{spec.name}_{spec.velocity_var}_{int(round(depth.max()))}km'
    print(f'{tag} from {os.path.basename(path)}: '
          f'{len(depth)} shells {depth.min():.0f}-{depth.max():.0f} km, '
          f'{len(lat)} x {len(lon)}', flush=True)

    hs = pd.read_csv(HOTSPOTS).dropna(subset=['lat', 'lon_180'])
    print(f'{len(hs)} hotspots from Courtillot (2003)\n', flush=True)

    # ---------------------------------------------------------------- stage A
    print('=' * 78)
    print('A  PERCOLATION SCAN — above p_c, "connected to the hotspot" means')
    print('   "connected to everything", and depth extent is meaningless')
    print('=' * 78)
    scan, labels, p_c = percolation_scan(arr, depth)
    for r in scan.to_dict('records'):
        flag = '  <-- PERCOLATED' if r['percolated'] else ''
        print(f"  p{r['pct']:<4g} {r['n_comp']:5d} components   largest = "
              f"{100 * r['giant_frac']:5.1f}% of mask, spans "
              f"{r['giant_span_km']:6.0f} km{flag}")
    print(f'\n  percolation threshold p_c = p{p_c:g}; connectivity descriptors are '
          f'reported\n  only for thresholds strictly below it', flush=True)

    # ---------------------------------------------------------------- stage B
    crows, prows = [], []
    for _, h in hs.iterrows():
        crows += column_descriptors(h.hotspot, float(h.lat), float(h.lon_180),
                                    depth, lat, lon, arr)
        prows += persistence(h.hotspot, float(h.lat), float(h.lon_180),
                             depth, lat, lon, labels, scan)
    cols = pd.DataFrame(crows)
    pers = pd.DataFrame(prows)
    cols.to_csv(f'{OUT}/plume_columns_{tag}.csv', index=False)
    pers.to_csv(f'{OUT}/plume_persistence_{tag}.csv', index=False)

    # ---------------------------------------------------------------- summary
    valid = pers[pers.below_pc]
    summ = []
    for name, g in pers.groupby('hotspot'):
        gv = g[g.below_pc]
        seeded = gv[gv.seeded]
        # persistence = strictest ladder step at which the hotspot has any structure,
        # and the step at which that structure merges into the network
        seed_pct = float(seeded.pct.min()) if len(seeded) else np.nan
        m = g[g.merged == True]
        merge_pct = float(m.pct.min()) if len(m) else np.nan
        sub = seeded[seeded.pct < (merge_pct if np.isfinite(merge_pct) else np.inf)]
        d = sub.max_depth.dropna()
        ct = sub.continuous_to.dropna()
        c5 = cols[(cols.hotspot == name) & (cols.radius_deg == 5.0)]
        rec = dict(
            hotspot=name,
            seed_pct=seed_pct,
            merge_pct=merge_pct,
            persistence=(merge_pct - seed_pct
                         if np.isfinite(seed_pct) and np.isfinite(merge_pct) else np.nan),
            n_usable_steps=int(len(sub)),
            max_depth_premerge=float(d.max()) if len(d) else np.nan,
            max_depth_range=float(d.max() - d.min()) if len(d) > 1 else np.nan,
            continuous_to_premerge=float(ct.max()) if len(ct) else np.nan,
            breaks_typical=(int(sub.n_breaks.median())
                            if sub.n_breaks.notna().any() else -1),
            branches_typical=(int(sub.n_seed_comp.median())
                              if sub.n_seed_comp.notna().any() else -1),
            tilt_deg=float(sub.tilt_deg.median()) if sub.tilt_deg.notna().any() else np.nan)
        for _, cr in c5.iterrows():
            for label, _, _ in BANDS:
                if f'{label}_mean' in cr:
                    rec[f'{label}_mean'] = cr[f'{label}_mean']
        summ.append(rec)
    s = pd.DataFrame(summ).merge(
        hs[['hotspot', 'count', 'he_ratio', 'tomo_anomaly']], on='hotspot', how='left')

    def verdict(r):
        if not np.isfinite(r.seed_pct):
            return 'no slow anomaly below p_c'
        if r.n_usable_steps < 2:
            return 'unresolved: merges immediately'
        if r.max_depth_range > 300:
            return 'threshold-sensitive'
        return 'resolved'

    s['verdict'] = s.apply(verdict, axis=1)
    s.to_csv(f'{OUT}/plume_summary_{tag}.csv', index=False)

    print('\n' + '=' * 78)
    print('B/C  per-hotspot structure, thresholds below p_c only')
    print('     persistence = ladder span over which the structure keeps its identity')
    print('=' * 78)
    show = ['hotspot', 'seed_pct', 'merge_pct', 'max_depth_premerge',
            'continuous_to_premerge', 'breaks_typical', 'branches_typical',
            'um_200_410_mean', 'mtz_410_660_mean', 'tomo_anomaly', 'verdict']
    show = [c for c in show if c in s.columns]
    print(s.sort_values('max_depth_premerge', ascending=False)[show]
          .to_string(index=False, max_colwidth=24, float_format=lambda x: f'{x:.1f}'))

    # ------------------------------------------------------------ validation
    print('\n' + '=' * 78)
    print("VALIDATION: does this reproduce Courtillot's manual tomographic call?")
    print('=' * 78)
    lab = s.dropna(subset=['tomo_anomaly'])
    print(f'  labelled hotspots: {len(lab)} '
          f'({(lab.tomo_anomaly == "slow").sum()} slow, '
          f'{(lab.tomo_anomaly == "fast").sum()} fast), '
          f'{len(s) - len(lab)} unassessed')
    for field, nice in (('um_200_410_mean', 'upper mantle 200-410 km anomaly'),
                        ('mtz_410_660_mean', 'transition zone 410-660 km anomaly'),
                        ('max_depth_premerge', 'pre-merge depth extent'),
                        ('merge_pct', 'merge threshold')):
        if field not in lab:
            continue
        gslow = lab.loc[lab.tomo_anomaly == 'slow', field].dropna()
        gother = lab.loc[lab.tomo_anomaly != 'slow', field].dropna()
        if len(gslow) < 3 or len(gother) < 1:
            continue
        line = (f'  {nice:36s} slow median {gslow.median():7.2f}  '
                f'other median {gother.median():7.2f}')
        if len(gother) >= 3:
            u, p = stats.mannwhitneyu(gslow, gother)
            line += f'   p={p:.3f}'
        print(line)

    u = s.dropna(subset=['count'])
    for field in ('um_200_410_mean', 'mtz_410_660_mean', 'max_depth_premerge'):
        if field not in u:
            continue
        m = u[field].notna()
        if m.sum() < 8:
            continue
        r, p = stats.spearmanr(u.loc[m, 'count'], u.loc[m, field])
        print(f'  Courtillot criteria count vs {field:22s} '
              f'rho={r:+.3f} p={p:.3f} (n={m.sum()})')

    print('\n  verdicts: ' + ', '.join(
        f'{k} {v}' for k, v in s.verdict.value_counts().items()))
    print(f'\nwrote plume_columns_{tag}.csv, plume_persistence_{tag}.csv, '
          f'plume_summary_{tag}.csv')


if __name__ == '__main__':
    main()
