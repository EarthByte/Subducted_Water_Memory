"""When was the transition zone hydrated, and how old can that hydration be?

Everything up to here asks whether a fast transition-zone anomaly can be tied to
subduction inside a short window. The more interesting question runs the other
way: for each anomaly, what is the youngest subduction that could have delivered
it, and how far back do we have to go before the reconstruction can account for
it at all? The oldest such age is a lower bound on how long the transition zone
can hold what subduction puts there, which does not appear to have been measured.

The construction is one global map. For every point on the globe and every epoch,
material entering a trench at that epoch is displaced down-dip by the offset a
30-to-90-degree slab gives, and it is in the transition zone now if it entered
early enough to have reached 410 km. Marking the globe epoch by epoch and keeping
the earliest mark gives, per point, the age of the youngest subduction that can
account for a slab beneath it. Reading that map at the observed fast bodies gives
each body a hydration age.

The map also makes the control cheap, and the control is what decides whether any
of this means anything. Over hundreds of millions of years trenches sweep across
most of the globe, so an old age is only evidence if it is older than a random
location's would be. Rotating the set of observed cells and reading the same map
gives that null directly.

Two assumptions are carried openly. Material that reaches the transition zone
stays there, which is what makes an old age possible at all and is the strong
claim in the whole exercise. And the offset is the one the dip prescription gives,
swept in slab_forward.py, where the observed transition zone prefers 300 km
against a prescription of 199.
"""
import os, argparse, numpy as np, pandas as pd, dataclasses
from scipy import ndimage
from scipy.spatial import cKDTree
import paths as P
import plume_classifier as pc
from s16_plume_null import random_rotation, to_xyz, to_lonlat

R_EARTH = 6371.0088
TZ = (410.0, 660.0)


def displace(lon, lat, az_deg, km):
    d = km / R_EARTH
    la1, lo1, th = np.radians(lat), np.radians(lon), np.radians(az_deg)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(th))
    lo2 = lo1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return (np.degrees(lo2) + 180) % 360 - 180, np.degrees(la2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='zahirovic2022',
                    help='the last 400 Myr are better constrained here than in a '
                         'model built for deep time on an older 400-0 Ma base')
    ap.add_argument('--offset', type=float, default=300.0)
    ap.add_argument('--radius', type=float, default=250.0)
    ap.add_argument('--v-sink', type=float, default=50.0)
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--tmax', type=float, default=400.0)
    ap.add_argument('--n-null', type=int, default=2000)
    ap.add_argument('--bands', default=None,
                    help='an npz from extract_bands.py to draw the fast '
                         'transition zone from, in place of REVEAL')
    ap.add_argument('--suffix', default='', help='tag the outputs')
    a = ap.parse_args()
    rng = np.random.default_rng(20260829)

    # The tomographic model enters here and nowhere else. Everything after this
    # -- trenches, down-dip displacement, occupancy, the null -- comes from the
    # reconstruction and does not know which model drew the mask, so swapping it
    # is the whole of what a model sweep needs. --bands takes an extract_bands
    # npz, which is how a model too large to move is used; the default path is
    # REVEAL read directly, byte for byte what Table 1 was built from.
    if a.bands:
        z = np.load(P.need(a.bands, 'PAPER_OUT',
                           'a bands npz from extract_bands.py'), allow_pickle=True)
        need = [f'{TZ[0]:.0f}_520', f'520_{TZ[1]:.0f}']
        miss = [k for k in need if k not in z.files]
        if miss:
            raise SystemExit(f"\n{a.bands} has no {', '.join(miss)} band; it "
                             f"holds {sorted(z.files)}\n")
        lat, lon = z['lat'], z['lon']
        band = np.nanmean(np.stack([z[k].astype(float) for k in need]), axis=0)
        print(f'mask from {os.path.basename(a.bands)}')
    else:
        SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
        depth, lat, lon, arr = pc.load_anomaly(SPEC, P.REVEAL)
        band = np.nanmean(arr[(depth >= TZ[0]) & (depth < TZ[1])], axis=0)
        print('mask from REVEAL')
    f = np.isfinite(band)
    mask = f & (band >= np.percentile(band[f], 90.0))
    LO, LA = np.meshgrid(lon, lat)
    lab, nlab = ndimage.label(mask, structure=np.ones((3, 3)))
    # Coherent bodies are recomputed from whichever mask is loaded, by s19's own
    # rule: a connected component above MIN_BODY_KM2 survives. They used to be
    # imported from s19's table as a list of label NUMBERS, which is correct only
    # while the mask is the one s19 labelled. Under --bands ndimage renumbers
    # everything, so body 83 becomes a different patch of the globe and the
    # observed set becomes an arbitrary subset picked by label collision.
    MIN_BODY_KM2 = 4.0e5
    cell_km2 = (np.cos(np.radians(LA)) * (111.195 ** 2)
                * abs(lat[1] - lat[0]) * abs(lon[1] - lon[0]))
    area_of = ndimage.sum_labels(cell_km2, lab, index=np.arange(1, nlab + 1))
    keep = 1 + np.flatnonzero(area_of >= MIN_BODY_KM2)
    big = np.isin(lab, keep) & mask
    print(f'{len(keep)} coherent bodies above {MIN_BODY_KM2:.0e} km2, '
          f'{int(big.sum())} cells')
    # s19's table still supplies the region names and the recent-subduction flag,
    # matched by position rather than by label number so it survives a new mask.
    bodies = pd.read_csv(P.need(os.path.join(P.OUT, 's19', 'slab_bodies.csv'),
                                'PAPER_OUT', 'slab_bodies.csv'))

    from gplately import PlateReconstruction
    if a.model == 'muller2025-1800':
        # The 1.8 Ga model as supplied, built from local files. Its pre-410 Ma
        # topologies only close once the Convergence, Divergence and Transform
        # files are loaded alongside the plate boundaries; without them the
        # reconstruction returns no trenches at all before 410 Ma, silently.
        U = os.path.join(P.ROOT, 'Muller_etal_2025_1.8_Ga_mantle_ref_frame')
        if not os.path.isdir(U):
            U = ('/mnt/user-data/uploads/Muller_mantle_tomography_subduction/'
                 'Paper_hydration/Muller_etal_2025_1.8_Ga_mantle_ref_frame')
        rot = [f'{U}/optimisation/1000_0_rotfile_20240725.rot',
               f'{U}/optimisation/1800_1000_rotfile_20240725.rot']
        rot = [r for r in rot if os.path.exists(r)]
        topo = [f'{U}/250-0_plate_boundaries.gpml',
                f'{U}/410-250_plate_boundaries.gpml',
                f'{U}/1000-410-plate-boundaries.gpml',
                f'{U}/1800-1000_plate_boundaries.gpml',
                f'{U}/1000-410-Convergence.gpml',
                f'{U}/1000-410-Divergence.gpml',
                f'{U}/1000-410-Transforms.gpml',
                f'{U}/TopologyBuildingBlocks.gpml']
        topo = [t for t in topo if os.path.exists(t)]
        P.need(topo[0] if topo else '', 'PAPER_ROOT', 'the 1.8 Ga plate model')
        recon = PlateReconstruction(rot, topology_features=topo,
                                    static_polygons=[f'{U}/static_polygons.gpmlz'])
        tmax = a.tmax
        print(f'muller2025 1.8 Ga, {len(topo)} topology files, from local paths')
    else:
        from plate_model_manager import PlateModelManager
        pmm = PlateModelManager().get_model(a.model, data_dir=P.MODELS)
        recon = PlateReconstruction(pmm.get_rotation_model(),
                                    topology_features=pmm.get_topologies(),
                                    static_polygons=pmm.get_static_polygons())
        tmax = min(a.tmax, float(pmm.get_big_time()))
    t_410 = TZ[0] / a.v_sink
    times = np.arange(0.0, tmax + a.dt, a.dt)
    times = times[times >= t_410]
    print(f'{a.model}: scanning {len(times)} epochs from {times.min():.1f} to '
          f'{times.max():.0f} Ma')
    print(f'offset {a.offset:.0f} km down-dip, {a.radius:.0f} km search radius, '
          f'sinking {a.v_sink:.0f} km/Myr\n')

    cell_tree = cKDTree(to_xyz(LO.ravel(), LA.ravel()))
    chord = 2.0 * np.sin(0.5 * a.radius / R_EARTH)
    t_min = np.full(LO.size, np.nan)
    n_marked = np.zeros(LO.size, int)
    # Occupancy per age band, not just the youngest epoch. t_min records only the
    # most recent explanation, so a cell subducted at 20 and again at 150 Ma is
    # counted at 20 and never at 150, and old episodes disappear wherever recent
    # ones exist. Persistence has to be measured the other way round: given
    # subduction at age t, how often is the transition zone fast there.
    BANDS = [(10, 25), (25, 50), (50, 75), (75, 100), (100, 150),
             (150, 200), (200, 250), (250, 300), (300, 350), (350, 400)]
    if a.tmax > 400:
        BANDS += [(400, 500), (500, 650), (650, 800), (800, 1000)]
    occ = np.zeros((len(BANDS), LO.size), bool)
    for t in times:
        sz = recon.tessellate_subduction_zones(
            float(t), tessellation_threshold_radians=0.01, ignore_warnings=True)
        if sz is None or not len(sz):
            continue
        plon, plat = displace(sz[:, 0], sz[:, 1], sz[:, 7], a.offset)
        hit = np.unique(np.concatenate(
            [np.asarray(h, int) for h in cell_tree.query_ball_point(
                to_xyz(plon, plat), chord) if len(h)] or [np.zeros(0, int)]))
        if not len(hit):
            continue
        n_marked[hit] += 1
        fresh = hit[np.isnan(t_min[hit])]
        t_min[fresh] = t
        for k, (b0, b1) in enumerate(BANDS):
            if b0 <= t < b1:
                occ[k, hit] = True
                break
    cov = np.isfinite(t_min)
    w = area_w = np.cos(np.radians(LA)).ravel()
    print(f'{100 * (w[cov].sum() / w.sum()):.1f} per cent of the surface has any '
          f'subduction explanation within {tmax:.0f} Myr')

    obs = big.ravel()
    o_tmin = t_min[obs]
    print(f'\nobserved fast transition zone: {obs.sum()} cells, '
          f'{100 * np.isfinite(o_tmin).mean():.0f} per cent with an explanation')
    qs = [10, 25, 50, 75, 90, 95, 99]
    print('  youngest explaining subduction age, percentiles (Ma):')
    print('   ' + '  '.join(f'p{q}={np.nanpercentile(o_tmin, q):.0f}' for q in qs))

    # the null: rotate the observed constellation and read the same map
    xyz0 = to_xyz(LO.ravel()[obs], LA.ravel()[obs])
    nulls = np.empty((a.n_null, len(qs)))
    keep = 0
    while keep < a.n_null:
        xyz = xyz0 @ random_rotation(rng).T
        slo, sla = to_lonlat(xyz)
        j = np.abs(lat[None, :] - sla[:, None]).argmin(1)
        i = np.abs(((lon[None, :] - slo[:, None] + 180) % 360) - 180).argmin(1)
        v = t_min.reshape(LO.shape)[j, i]
        if np.isfinite(v).sum() < 100:
            continue
        nulls[keep] = [np.nanpercentile(v, q) for q in qs]
        keep += 1
    print('  the same percentiles for randomly rotated positions (median of '
          f'{a.n_null}):')
    print('   ' + '  '.join(f'p{q}={np.median(nulls[:, k]):.0f}'
                            for k, q in enumerate(qs)))
    print('  one-sided p that the observed percentile is older than chance:')
    print('   ' + '  '.join(
        f'p{q}={((nulls[:, k] >= np.nanpercentile(o_tmin, q)).sum() + 1) / (a.n_null + 1):.3f}'
        for k, q in enumerate(qs)))

    # persistence: given subduction at age t, how often is the transition zone
    # fast there? The base rate is the top decile by construction.
    base = area_w[obs].sum() / area_w.sum()
    print(f'\npersistence. Base rate of a fast transition zone is '
          f'{100 * base:.1f} per cent of the surface by construction.')
    print(f'  {"subduction age":>16s} {"cells":>9s} {"fast TZ":>9s} {"enrichment":>11s}'
          f' {"95% band":>16s}')
    prows = []
    for k, (b0, b1) in enumerate(BANDS):
        # Every cell that saw subduction in this band, including those that saw it
        # again since. That leak inflates every band but the youngest, so this
        # curve is NOT the one the paper quotes: Table 1 and the decay fit come
        # from persistence_decay.py, which counts a cell in a band only where no
        # younger subduction reached it. The two agree on the first band, which
        # has no younger band to leak from, and disagree on every band after.
        m = occ[k]
        if not m.any():
            continue
        w_m = area_w[m]
        frac = w_m[obs[m]].sum() / w_m.sum()
        n = int(m.sum())
        se = np.sqrt(max(frac * (1 - frac), 1e-12) / max(n, 1))
        lo_, hi_ = (frac - 1.96 * se) / base, (frac + 1.96 * se) / base
        print(f'  {b0:6.0f}-{b1:<9.0f} {n:9d} {100 * frac:8.1f}% '
              f'{frac / base:11.2f} {lo_:7.2f} {hi_:7.2f}')
        prows.append(dict(t0=b0, t1=b1, n_cells=n, frac_fast=frac,
                          enrichment=frac / base, lo=lo_, hi=hi_))
    pd.DataFrame(prows).to_csv(os.path.join(P.OUT, f'persistence{a.suffix}.csv'), index=False)

    # per body
    body_of = lab.ravel()[obs]
    olon, olat = LO.ravel()[obs], LA.ravel()[obs]
    ow = area_w[obs]

    def nearest_named(clon, clat, limit_km=1500.0):
        """The s19 body nearest this centroid, or nothing if none is close.

        Matching by position rather than by label number, so a name still means
        the same place when the mask changes.
        """
        p1, p2 = np.radians(clat), np.radians(bodies.lat.values)
        dl = np.radians(bodies.lon.values - clon)
        d = 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2)
                               + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))
        k = int(np.argmin(d))
        return (bodies.iloc[k], float(d[k])) if d[k] <= limit_km else (None, float(d[k]))

    rows = []
    for b in np.unique(body_of):
        k = body_of == b
        v = o_tmin[k]
        clon = float(np.average(olon[k], weights=ow[k]))
        clat = float(np.average(olat[k], weights=ow[k]))
        named, dist = nearest_named(clon, clat)
        rows.append(dict(body=int(b),
                         region=str(named.region) if named is not None else 'unassigned',
                         lon=clon, lat=clat,
                         area_km2=float(cell_km2.ravel()[obs][k].sum()),
                         attributed_0_30=bool(named.attributed) if named is not None else False,
                         named_from_km=round(dist, 0),
                         n_cells=int(len(v)),
                         explained=float(np.isfinite(v).mean()),
                         t_min_p10=float(np.nanpercentile(v, 10)) if np.isfinite(v).any() else np.nan,
                         t_min_median=float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,
                         t_min_p90=float(np.nanpercentile(v, 90)) if np.isfinite(v).any() else np.nan))
    ob = pd.DataFrame(rows).sort_values('t_min_median', ascending=False)
    print('\nhydration age per body, oldest first')
    print(ob[['region', 'lon', 'lat', 'area_km2', 'attributed_0_30', 'explained',
              't_min_p10', 't_min_median', 't_min_p90']].to_string(
        index=False, float_format=lambda x: f'{x:.0f}'))
    ob.to_csv(os.path.join(P.OUT, f'hydration_age_bodies{a.suffix}.csv'), index=False)
    np.savez_compressed(os.path.join(P.OUT, f'hydration_age_map{a.suffix}.npz'),
                        lon=lon, lat=lat, t_min=t_min.reshape(LO.shape),
                        n_marked=n_marked.reshape(LO.shape), observed=big,
                        offset=a.offset, radius=a.radius, model=a.model,
                        occupancy=occ.reshape((len(BANDS),) + LO.shape),
                        bands=np.array(BANDS, float))
    print(f'\nwrote hydration_age_bodies{a.suffix}.csv, '
          f'hydration_age_map{a.suffix}.npz and persistence{a.suffix}.csv')


if __name__ == '__main__':
    main()
