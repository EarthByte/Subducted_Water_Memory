"""Stage 19 — are the fast transition-zone anomalies slabs, and if so, whose?

The claim that intraplate volcanism without a deep root sits above stagnant slabs
rests on a proxy: proximity to mantle that is fast when averaged through
410-660 km. That proxy is not a slab until it is tied to a subduction history,
and it can fail in a specific and damaging way. Continental lithosphere is cold
and its thermal signature can persist to transition-zone depths, so "continental
intraplate volcanism is near fast transition zone" risks restating "continental
volcanism is on continents". This stage tests that directly and then attributes
each anomaly to a subduction margin and an age.

WHAT IS COMPUTED

1. Composition of the fast set. What fraction of the fast transition-zone area
   lies on continental lithosphere, against the 40.2% continental fraction of
   the surface. A large excess means the proxy is partly continental structure.

2. Coherent bodies. The fast cells are grouped into connected bodies so that an
   anomaly is attributed as an object rather than cell by cell, and bodies below
   a minimum area are discarded as noise.

3. Provenance. Trenches are reconstructed at 2 Myr intervals in the plate
   model's mantle reference frame, using alfonso2024 rather than a
   Zahirovic-derived model because the latter does not carry a reliable
   subduction history west of North America, which is where this analysis
   concentrates. For each body we record every time
   at which a trench lay within a search radius of it, and report the youngest
   such time, the oldest, and the total duration. Material now in the transition
   zone beneath a point sank from a trench that stood above that mantle column,
   so a body with no trench overhead at any time in the window is not a slab that
   this plate model can account for, and is reported as unattributed.

   No sinking rate is assumed. The output is the interval during which
   subduction could have delivered the body, not a single arrival age; converting
   one to the other requires a rate model and is deliberately left out.

4. Assignment. Each rootless hotspot and each intraplate volcanic field is
   assigned to the nearest attributed body, giving a named margin and an age
   range rather than an anonymous distance.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plume_classifier as pc
import paths as P
import contmask
from s16_plume_null import to_xyz

warnings.filterwarnings('ignore')

R_EARTH = 6371.0
TZ = (410.0, 660.0)
SLAB_PCTL = 90.0
MIN_BODY_KM2 = 4.0e5          # discard fast patches smaller than this
# Transition-zone anomalies record RECENT subduction only. Hall and Spakman
# (2015) find upper-mantle anomalies mainly record the last 10-25 Myr depending
# on region, and Tao et al. (2018) attribute the stagnant Pacific slab beneath
# East Asia to roughly 25 Myr of subduction. A 250 Myr search window would
# therefore attribute nearly every body to some trench at some time and mean
# nothing. The primary window is 0-30 Ma; 0-50 and 0-80 Ma are run as
# sensitivity tests and reported, because the correct value is regional.
WINDOWS = [30.0, 50.0, 80.0]
DT = 2.0
# Horizontal extent of stagnation, not of deep lateral transport: Tao et al.
# (2018) image stagnation over 600 km or less at the top of the lower mantle.
TRENCH_RADII = [400.0, 600.0, 1000.0]
TIMES = np.arange(0.0, max(WINDOWS) + DT, DT)
TRENCH_RADIUS_KM = 600.0
MODEL_DEFAULT = 'alfonso2024'


def cell_area_km2(lat, lon):
    """Area of every cell, broadcast to the full (lat, lon) grid."""
    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])
    col = ((np.radians(dlat) * R_EARTH)
           * (np.radians(dlon) * R_EARTH * np.cos(np.radians(lat))))
    return np.repeat(col[:, None], len(lon), axis=1)


def fast_tz_mask(depth, arr):
    k = (depth >= TZ[0]) & (depth < TZ[1])
    band = np.nanmean(arr[k], axis=0)
    f = np.isfinite(band)
    return f & (band >= np.percentile(band[f], SLAB_PCTL)), band


def label_seam(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return lab, 0
    left, right = lab[:, 0], lab[:, -1]
    both = (left > 0) & (right > 0)
    pairs = {(min(a, b), max(a, b))
             for a, b in zip(left[both], right[both]) if a != b}
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


REGIONS = [
    ('NW Pacific / E Asia',      lambda o, a: 100 <= o <= 180 and 15 <= a <= 65),
    ('N America / E Pacific',    lambda o, a: -170 <= o <= -60 and 15 <= a <= 70),
    ('Central America',          lambda o, a: -115 <= o <= -60 and 0 <= a < 25),
    ('S America / SE Pacific',   lambda o, a: -110 <= o <= -50 and -60 <= a < 5),
    ('Mediterranean / Tethys',   lambda o, a: -15 <= o <= 70 and 25 <= a <= 60),
    ('SE Asia / Indonesia',      lambda o, a: 90 <= o <= 160 and -15 <= a < 20),
    ('SW Pacific / Tonga',       lambda o, a: 155 <= o <= 180 or -180 <= o <= -160),
    ('Africa / Arabia',          lambda o, a: -20 <= o <= 60 and -40 <= a < 25),
    ('Antarctic / S Ocean',      lambda o, a: a < -55),
]


def region_of(lon, lat):
    for name, test in REGIONS:
        try:
            if test(lon, lat):
                return name
        except Exception:
            pass
    return 'unassigned'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='REVEAL_vs_full.nc')
    ap.add_argument('--var', default='voigt')
    ap.add_argument('--src', default=P.TOMO)
    ap.add_argument('--hotspots', default=pc.HOTSPOTS)
    ap.add_argument('--ipv', default=P.IPV_V3, help='intraplate volcanism catalogue')
    ap.add_argument('--roots', default=None,
                    help='optional hotspot root classification; not used by this paper')
    ap.add_argument('--out', default=os.path.join(P.OUT, 's19'))
    ap.add_argument('--model', default='alfonso2024',
                    help='plate model for the trench history. The default is '
                         'alfonso2024: the Zahirovic-derived models do not have '
                         'a reliable subduction history west of North America, '
                         'which is exactly the region this analysis turns on.')
    ap.add_argument('--model-dir', default=P.MODELS)
    a = ap.parse_args()

    spec = pc.ModelSpec('REVEAL', a.file, a.var, 'dv/v in %')
    depth, lat, lon, arr = pc.load_anomaly(spec, os.path.join(a.src, a.file))
    mask, band = fast_tz_mask(depth, arr)
    LO, LA = np.meshgrid(lon, lat)
    area = cell_area_km2(lat, lon)

    # ------------------------------------------------------------------ (1)
    print('=' * 78)
    print('1  IS THE FAST TRANSITION ZONE JUST CONTINENTAL STRUCTURE?')
    print('=' * 78)
    cont_all = contmask.is_continental(LO.ravel(), LA.ravel()).reshape(LO.shape)
    w = np.cos(np.radians(LA))
    frac_surface = float((cont_all * w).sum() / w.sum())
    frac_fast = float((cont_all & mask)[mask].sum() / mask.sum()) if mask.any() else np.nan
    frac_fast_area = float(area[mask & cont_all].sum() / area[mask].sum())
    print(f'  continental fraction of the surface        {100 * frac_surface:5.1f}%')
    print(f'  continental fraction of fast-TZ area       {100 * frac_fast_area:5.1f}%')
    excess = frac_fast_area / frac_surface
    print(f'  enrichment                                 {excess:5.2f}x')
    if excess > 1.3:
        print('  ! the fast transition zone IS preferentially continental. Proximity of\n'
              '    continental volcanism to it is therefore partly a statement about\n'
              '    where continents are, and the slab interpretation needs the\n'
              '    attribution below to stand up.')
    else:
        print('  the fast transition zone is not strongly continental, so the\n'
              '  proximity result is not simply restating continental geography.')

    # ------------------------------------------------------------------ (2)
    lab, _ = label_seam(mask)
    bodies = []
    for bid in np.unique(lab[lab > 0]):
        sel = lab == bid
        km2 = float(area[sel].sum())
        if km2 < MIN_BODY_KM2:
            continue
        v = to_xyz(LO[sel], LA[sel]).mean(0)
        v /= np.linalg.norm(v)
        clon = float(np.degrees(np.arctan2(v[1], v[0])))
        clat = float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))
        bodies.append(dict(body=int(bid), area_km2=km2, lon=clon, lat=clat,
                           n_cells=int(sel.sum()),
                           cont_frac=float(area[sel & cont_all].sum() / km2),
                           region=region_of(clon, clat)))
    bodies = pd.DataFrame(bodies).sort_values('area_km2', ascending=False)
    mask_pct = 100 * area[lab > 0].sum() / area.sum()
    body_pct = 100 * bodies.area_km2.sum() / area.sum()
    print(f'\n  the top-decile fast band covers {mask_pct:.1f}% of the surface')
    print(f'  {len(bodies)} coherent bodies above {MIN_BODY_KM2:.0e} km2 survive the '
          f'size filter,')
    print(f'  and those bodies cover {body_pct:.1f}% of the surface. The two '
          f'percentages are not')
    print(f'  interchangeable: the difference is the patches discarded as noise.')

    # ------------------------------------------------------------------ (3)
    print('\n' + '=' * 78)
    print('3  TRENCH HISTORY ABOVE EACH BODY, MANTLE REFERENCE FRAME')
    print(f'   0-{max(TIMES):.0f} Ma at {DT:.0f} Myr steps; a trench within {TRENCH_RADIUS_KM:.0f} km counts')
    print('=' * 78)
    from gplately import PlateReconstruction
    from plate_model_manager import PlateModelManager
    from trenches import boundary_points
    pmm = PlateModelManager().get_model(a.model, data_dir=a.model_dir)
    recon = PlateReconstruction(pmm.get_rotation_model(),
                                topology_features=pmm.get_topologies(),
                                static_polygons=pmm.get_static_polygons())
    tmax = float(pmm.get_big_time())
    print(f'  plate model: {a.model}, valid to {tmax:.0f} Ma')
    if tmax < max(WINDOWS):
        print(f'  ! model does not reach {max(WINDOWS):.0f} Ma; '
              f'windows are truncated at {tmax:.0f} Ma')

    # nearest-trench distance to every body at every epoch, computed once
    body_xyz = to_xyz(bodies.lon.values, bodies.lat.values)
    dmat = np.full((len(TIMES), len(bodies)), np.inf)
    for i, t in enumerate(TIMES):
        bp = boundary_points(recon, t)
        slon, slat, _ = bp['subduction']
        if not len(slon):
            continue
        tree = cKDTree(to_xyz(np.asarray(slon), np.asarray(slat)))
        d, _ = tree.query(body_xyz)
        dmat[i] = R_EARTH * 2.0 * np.arcsin(np.clip(d / 2.0, 0, 1))
    print(f'  trenches resolved at {len(TIMES)} epochs, '
          f'{TIMES.min():.0f}-{TIMES.max():.0f} Ma')

    print('\n  sensitivity: fraction of bodies with a trench overhead')
    print(f'  {"window":>10s} ' + ' '.join(f'{r:.0f} km'.rjust(9)
                                           for r in TRENCH_RADII))
    for wdw in WINDOWS:
        sel = TIMES <= wdw
        row = [f'{100 * (dmat[sel] <= r).any(axis=0).mean():8.0f}%'
               for r in TRENCH_RADII]
        print(f'  {wdw:8.0f} Ma ' + ' '.join(row))

    within = TIMES <= WINDOWS[0]
    hits = {int(b): [float(t) for t, dd in zip(TIMES[within], dmat[within, j])
                     if dd <= TRENCH_RADIUS_KM]
            for j, b in enumerate(bodies.body.values)}
    bodies['min_trench_km'] = dmat[within].min(axis=0)
    bodies['trench_times'] = [';'.join(f'{x:.0f}' for x in hits[int(b)])
                              for b in bodies.body]
    bodies['t_young'] = [min(hits[int(b)]) if hits[int(b)] else np.nan
                         for b in bodies.body]
    bodies['t_old'] = [max(hits[int(b)]) if hits[int(b)] else np.nan
                       for b in bodies.body]
    bodies['n_epochs'] = [len(hits[int(b)]) for b in bodies.body]
    bodies['attributed'] = bodies.n_epochs > 0

    att = bodies[bodies.attributed]
    print(f'\n  primary attribution: window 0-{WINDOWS[0]:.0f} Ma, radius '
          f'{TRENCH_RADIUS_KM:.0f} km')
    print(f'  {len(att)} of {len(bodies)} bodies had a trench overhead')
    print(f'  attributed bodies hold '
          f'{100 * att.area_km2.sum() / bodies.area_km2.sum():.0f}% of the '
          f'fast-TZ area in bodies')
    show = bodies.head(16)[['region', 'lon', 'lat', 'area_km2', 'cont_frac',
                            'min_trench_km', 't_young', 't_old', 'n_epochs']]
    print('\n  largest bodies:')
    print(show.to_string(index=False, float_format=lambda x: f'{x:.1f}'))

    # ------------------------------------------------------------------ (4)
    print('\n' + '=' * 78)
    print('4  WHICH BODY IS EACH VOLCANIC SITE NEAREST?')
    print('=' * 78)
    cell_tree = {}
    for bid in bodies.body:
        sel = lab == bid
        cell_tree[int(bid)] = to_xyz(LO[sel], LA[sel])
    allpts = np.vstack([cell_tree[int(b)] for b in bodies.body])
    owner = np.concatenate([np.full(len(cell_tree[int(b)]), int(b))
                            for b in bodies.body])
    tree = cKDTree(allpts)

    def assign(df, label):
        d, i = tree.query(to_xyz(df.lon_180.values, df.lat.values))
        arc = R_EARTH * 2.0 * np.arcsin(np.clip(d / 2.0, 0, 1))
        bid = owner[i]
        info = bodies.set_index('body')
        out = df.copy()
        out['dist_body_km'] = arc
        out['body'] = bid
        out['body_region'] = info.loc[bid, 'region'].values
        out['body_t_young'] = info.loc[bid, 't_young'].values
        out['body_t_old'] = info.loc[bid, 't_old'].values
        out['body_attributed'] = info.loc[bid, 'attributed'].values
        print(f'\n  {label}: {len(out)} sites, median distance to nearest body '
              f'{np.median(arc):.0f} km')
        print(f'  fraction whose nearest body is attributed to subduction: '
              f'{100 * out.body_attributed.mean():.0f}%')
        g = (out.groupby('body_region')
             .agg(n=('body', 'size'), med_km=('dist_body_km', 'median'),
                  t_young=('body_t_young', 'median'),
                  t_old=('body_t_old', 'median'))
             .sort_values('n', ascending=False))
        print(g.to_string(float_format=lambda x: f'{x:.0f}'))
        return out

    # The hotspot root classification belongs to a separate paper and rests on a
    # superseded method. It is used here only to write rootless_slab_assignment,
    # so it is optional; the outputs this paper needs do not depend on it.
    os.makedirs(a.out, exist_ok=True)
    bodies.to_csv(os.path.join(a.out, 'slab_bodies.csv'), index=False)
    if a.roots and os.path.exists(a.roots):
        roots = pd.read_csv(a.roots)
        rootless = roots[roots.root_verdict.str.lower().str.startswith('rootless')]
        r_out = assign(rootless, 'rootless hotspots')
        print('\n  per hotspot:')
        print(r_out[['hotspot', 'dist_body_km', 'body_region', 'body_t_young',
                     'body_t_old']].sort_values('dist_body_km')
              .to_string(index=False, float_format=lambda x: f'{x:.0f}'))
        r_out.to_csv(os.path.join(a.out, 'rootless_slab_assignment.csv'), index=False)
    else:
        print('\n  no root classification supplied; skipping that assignment')

    if a.ipv and os.path.exists(a.ipv):
        ipv = pd.read_csv(a.ipv).dropna(subset=['lat', 'lon_180'])
        i_out = assign(ipv, 'continental intraplate fields')
        i_out.to_csv(os.path.join(a.out, 'ipv_slab_assignment.csv'), index=False)

    print('\nwrote slab_bodies.csv')


if __name__ == '__main__':
    main()
