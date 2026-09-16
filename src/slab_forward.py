"""Forward-model the transition-zone slab from the subduction history.

The attribution in stage 19 is a proximity test: a body counts as a slab if a
trench stood within 600 km of its centroid inside a window. That absorbs the
descent geometry by accident rather than modelling it, and it uses the epoch only
to check window membership, so it cannot say which subduction explains what.

This runs the chain forwards instead. Material entering a trench at time t sinks,
and by now it lies at some depth and some horizontal distance from where it went
in. The horizontal distance is the integral of dz/tan(dip). Because 1/tan is largest
where the dip is shallowest, that integral is dominated by the top few hundred
kilometres, and below about 300 km a steep slab adds almost nothing to it. The
prescription used here is a slab steepening from 30 degrees at the surface to
vertical by 300 km, which integrates to 199 km and is then the same at 410 as at
660. Hu and Gurnis (2020) show the shallow dip is set mainly by how long the
subduction zone has been running, but recovering that per segment from a
reconstruction would carry more uncertainty than it removes, so the offset is
swept around the prescription instead.

Direction comes from the reconstruction. Column 7 of the subduction tessellation
is the trench-normal azimuth pointing towards the overriding plate, checked
against the known polarity of Japan, Peru-Chile, Tonga, Cascadia, Sumatra and the
Aleutians.

Two nulls. Randomising the azimuths keeps the trench geometry and the timing and
destroys only the down-dip direction, which is the sharpest control available: if
the direction carries no information, the offset is not doing what it claims.
Rotating the whole predicted set rigidly is the usual domain null.
"""
import os, argparse, numpy as np, pandas as pd, dataclasses
from scipy import ndimage
from scipy.spatial import cKDTree
import paths as P
import plume_classifier as pc
from trenches import boundary_points
from s16_plume_null import random_rotation, to_xyz, to_lonlat

R_EARTH = 6371.0088
TZ = (410.0, 660.0)


def displace(lon, lat, az_deg, km):
    """Move each point km along the given azimuth, on the sphere."""
    d = km / R_EARTH
    la1 = np.radians(lat); lo1 = np.radians(lon); th = np.radians(az_deg)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(th))
    lo2 = lo1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return (np.degrees(lo2) + 180) % 360 - 180, np.degrees(la2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='alfonso2024')
    ap.add_argument('--v-sink', type=float, default=50.0, help='km/Myr in the upper mantle')
    ap.add_argument('--residence', type=float, default=60.0,
                    help='Myr a slab stays in the transition zone once it stagnates')
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--offsets', default='0,100,150,199,250,300,400,500,700')
    ap.add_argument('--radius', type=float, default=250.0,
                    help='km; slab half-width plus tomographic smearing')
    ap.add_argument('--n-null', type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(20260829)

    SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
    depth, lat, lon, arr = pc.load_anomaly(SPEC, P.REVEAL)
    band = np.nanmean(arr[(depth >= TZ[0]) & (depth < TZ[1])], axis=0)
    f = np.isfinite(band)
    mask = f & (band >= np.percentile(band[f], 90.0))
    LO, LA = np.meshgrid(lon, lat)
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    bodies = pd.read_csv(P.need(os.path.join(P.OUT, 's19', 'slab_bodies.csv'),
                                'PAPER_OUT', 'slab_bodies.csv'))
    big = np.isin(lab, bodies.body.astype(int).tolist()) & mask
    obs = to_xyz(LO[big], LA[big])
    print(f'observed fast transition zone: {big.sum()} cells in coherent bodies')

    from gplately import PlateReconstruction
    from plate_model_manager import PlateModelManager
    pmm = PlateModelManager().get_model(a.model, data_dir=P.MODELS)
    recon = PlateReconstruction(pmm.get_rotation_model(),
                                topology_features=pmm.get_topologies(),
                                static_polygons=pmm.get_static_polygons())
    tmax_model = float(pmm.get_big_time())

    # epochs whose material is in the transition zone now
    t_in = TZ[0] / a.v_sink
    t_out = t_in + a.residence
    times = np.arange(0.0, min(t_out, tmax_model) + a.dt, a.dt)
    times = times[times >= t_in]
    print(f'{a.model}: sinking {a.v_sink:.0f} km/Myr reaches {TZ[0]:.0f} km at '
          f'{t_in:.1f} Ma; with {a.residence:.0f} Myr of residence the transition '
          f'zone holds material subducted {t_in:.1f} to {times.max():.0f} Ma '
          f'({len(times)} epochs)')

    tr_lon, tr_lat, tr_az, tr_t = [], [], [], []
    for t in times:
        sz = recon.tessellate_subduction_zones(
            float(t), tessellation_threshold_radians=0.01, ignore_warnings=True)
        if sz is None or not len(sz):
            continue
        tr_lon.append(sz[:, 0]); tr_lat.append(sz[:, 1]); tr_az.append(sz[:, 7])
        tr_t.append(np.full(len(sz), t))
    tr_lon = np.concatenate(tr_lon); tr_lat = np.concatenate(tr_lat)
    tr_az = np.concatenate(tr_az); tr_t = np.concatenate(tr_t)
    print(f'{len(tr_lon)} trench samples across those epochs\n')

    # Score by overlap, not by distance to the nearest prediction. Distance to the
    # nearest rewards over-coverage: scattering the predicted points in random
    # directions puts something near everything and scores better than the truth.
    # Overlap penalises a miss and an over-prediction alike.
    cell_xyz = to_xyz(LO.ravel(), LA.ravel())
    cell_tree = cKDTree(cell_xyz)
    area = (R_EARTH ** 2 * np.radians(abs(lat[1] - lat[0]))
            * np.radians(abs(lon[1] - lon[0])) * np.cos(np.radians(LA))).ravel()
    obs_flat = big.ravel()
    chord = 2.0 * np.sin(0.5 * a.radius / R_EARTH)

    def predict(off, az):
        plon, plat = displace(tr_lon, tr_lat, az, off)
        hit = cell_tree.query_ball_point(to_xyz(plon, plat), chord)
        m = np.zeros(len(area), bool)
        for h in hit:
            m[h] = True
        return m

    def score(m):
        inter = area[m & obs_flat].sum()
        union = area[m | obs_flat].sum()
        return (inter / union, inter / area[obs_flat].sum(),
                inter / max(area[m].sum(), 1e-9))

    offs = [float(x) for x in a.offsets.split(',')]
    print(f'a slab within {a.radius:.0f} km of a predicted position counts as '
          f'predicted\n')
    print(f'{"offset km":>10s} {"overlap":>9s} {"of observed":>12s} '
          f'{"of predicted":>13s} | {"azimuth null":>13s} {"p":>7s}')
    rows = []
    for off in offs:
        m = predict(off, tr_az)
        j, rec, prec = score(m)
        null = np.empty(a.n_null)
        for k in range(a.n_null):
            null[k] = score(predict(off, rng.uniform(0, 360, len(tr_az))))[0]
        p = ((null >= j).sum() + 1) / (a.n_null + 1)
        print(f'{off:10.0f} {j:9.3f} {100 * rec:11.0f}% {100 * prec:12.0f}% '
              f'| {np.median(null):13.3f} {p:7.3f}')
        rows.append(dict(offset_km=off, overlap=j, recall=rec, precision=prec,
                         azimuth_null=float(np.median(null)), p=p))
    d = pd.DataFrame(rows)
    best = d.loc[d.overlap.idxmax()]
    print(f'\nbest overlap {best.overlap:.3f} at an offset of {best.offset_km:.0f} km; '
          f'the dip prescription gives 199 km')

    def median_km(off, az):
        plon, plat = displace(tr_lon, tr_lat, az, off)
        dd, ii = cKDTree(to_xyz(*displace(tr_lon, tr_lat, az, off))).query(obs)
        return R_EARTH * 2.0 * np.arcsin(np.clip(dd / 2.0, 0, 1)), ii

    # which subduction age explains each body, at the preferred offset
    dkm, idx = median_km(best.offset_km, tr_az)
    age = tr_t[idx]
    body_of = lab[big]
    info = bodies.set_index('body')
    out = []
    for b, grp in pd.DataFrame(dict(body=body_of, age=age, km=dkm)).groupby('body'):
        out.append(dict(body=int(b), region=info.loc[int(b), 'region'],
                        area_km2=float(info.loc[int(b), 'area_km2']),
                        n_cells=len(grp), median_km=float(grp.km.median()),
                        age_p10=float(grp.age.quantile(0.1)),
                        age_median=float(grp.age.median()),
                        age_p90=float(grp.age.quantile(0.9))))
    ob = pd.DataFrame(out).sort_values('area_km2', ascending=False)
    print('\nsubduction age that accounts for each body, at the preferred offset')
    print(ob[['region', 'area_km2', 'n_cells', 'median_km',
              'age_p10', 'age_median', 'age_p90']].to_string(
        index=False, float_format=lambda x: f'{x:.0f}'))
    d.to_csv(os.path.join(P.OUT, 'slab_forward_offsets.csv'), index=False)
    ob.to_csv(os.path.join(P.OUT, 'slab_forward_bodies.csv'), index=False)
    print(f'\nwrote slab_forward_offsets.csv and slab_forward_bodies.csv')


if __name__ == '__main__':
    main()
