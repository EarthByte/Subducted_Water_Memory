"""Move the subducted water down-dip, then ask when it was delivered.

The Z22 stored-water grids record water still bound in the slab below 125 km, the
mantle-wedge depth of the thermodynamic workflow (smoothed over 300 km), and so can
reach the transition zone, but they are built without a descent-angle correction:
each parcel is left at the trench that delivered it, as though slabs sank
vertically. That is convenient rather than a problem, because it means the water
is in mantle-frame coordinates at the point of entry and can be moved from there.

It cannot be moved as a block. The down-dip direction differs at every trench
segment, so each parcel is shifted along the normal azimuth of the trench nearest
to it at the epoch that delivered it, by the offset a 30-to-90-degree slab implies.
Water is deposited by accumulation, so the shift conserves it.

The shift is testable rather than assumed: the corrected field should overlie the
observed fast transition zone better than the uncorrected one, and by roughly the
margin the offset sweep in slab_forward.py found.

What comes out is a present-day map of transition-zone water with a delivery age
attached to every cell, which is what the persistence measurement wants in place
of binary trench presence.
"""
import os, argparse, glob, re, numpy as np, pandas as pd, xarray as xr, dataclasses
from scipy import ndimage
from scipy.spatial import cKDTree
import paths as P
import plume_classifier as pc
from s16_plume_null import to_xyz

R_EARTH = 6371.0088
WDIR = os.path.join(P.GRIDS, 'Z22_water_grids_300km',
                    'cumulative_stored_water_mantle_Z22')


def displace(lon, lat, az, km):
    d = km / R_EARTH
    la1, lo1, th = np.radians(lat), np.radians(lon), np.radians(az)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(th))
    lo2 = lo1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return (np.degrees(lo2) + 180) % 360 - 180, np.degrees(la2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='zahirovic2022')
    ap.add_argument('--offset', type=float, default=300.0)
    ap.add_argument('--v-sink', type=float, default=50.0)
    ap.add_argument('--suffix', default='',
                    help='tag the outputs, so an offset sweep does not \n                         overwrite the reference run')
    a = ap.parse_args()

    have = sorted(int(re.search(r'_(\d+)\.nc$', f).group(1))
                  for f in glob.glob(os.path.join(WDIR, '*.nc')))
    if len(have) < 3:
        raise SystemExit(f'\nfewer than three water grids under {WDIR}\n'
                         '  stage the Z22 stored-water series first\n')
    print(f'{len(have)} Z22 stored-water epochs: {have[0]} to {have[-1]} Ma')

    def load(t):
        d = xr.open_dataset(os.path.join(WDIR, f'cumulative_stored_water_mantle_{t}.nc'))
        return d.lat.values, d.lon.values, np.nan_to_num(d.z.values.astype(float))
    wlat, wlon, _ = load(have[0])

    SPEC = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
    depth, lat, lon, arr = pc.load_anomaly(SPEC, P.REVEAL)
    band = np.nanmean(arr[(depth >= 410) & (depth < 660)], axis=0)
    f = np.isfinite(band)
    fast = f & (band >= np.percentile(band[f], 90.0))
    LO, LA = np.meshgrid(lon, lat)
    dla, dlo = lat[1] - lat[0], lon[1] - lon[0]

    from gplately import PlateReconstruction
    from plate_model_manager import PlateModelManager
    pmm = PlateModelManager().get_model(a.model, data_dir=P.MODELS)
    recon = PlateReconstruction(pmm.get_rotation_model(),
                                topology_features=pmm.get_topologies(),
                                static_polygons=pmm.get_static_polygons())

    t_410 = 410.0 / a.v_sink
    shifted = np.zeros(LO.shape)          # water now in the transition zone
    plain = np.zeros(LO.shape)            # the same without the dip correction
    age_num = np.zeros(LO.shape)          # water-weighted delivery age
    # Water kept separately by delivery age, which is what the decay curve needs:
    # a weighted mean age per cell cannot say how much of each vintage survives.
    ABINS = [(8, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 200),
             (200, 300), (300, 400)]
    by_age = np.zeros((len(ABINS),) + LO.shape)
    print(f'\nreached the transition zone by {t_410:.1f} Ma after entry; '
          f'shifting {a.offset:.0f} km down-dip\n')
    print(f'{"window (Ma)":>14s} {"water added":>13s} {"cells":>8s} {"mean shift":>11s}')
    for k in range(len(have) - 1):
        t0, t1 = have[k], have[k + 1]
        if t0 < t_410:
            continue
        _, _, w0 = load(t0)
        _, _, w1 = load(t1)
        inc = w0 - w1                      # delivered between t1 and t0
        inc[inc < 0] = 0.0
        if inc.sum() <= 0:
            continue
        tmid = 0.5 * (t0 + t1)
        sz = recon.tessellate_subduction_zones(
            float(tmid), tessellation_threshold_radians=0.02, ignore_warnings=True)
        if sz is None or not len(sz):
            continue
        jj, ii = np.nonzero(inc > 0)
        clat, clon, cval = wlat[jj], wlon[ii], inc[jj, ii]
        # each parcel takes the azimuth of the trench nearest to it
        d, nb = cKDTree(to_xyz(sz[:, 0], sz[:, 1])).query(to_xyz(clon, clat))
        slon, slat = displace(clon, clat, sz[nb, 7], a.offset)
        def deposit(target, plon, plat, val):
            j = np.clip(np.rint((plat - lat[0]) / dla).astype(int), 0, len(lat) - 1)
            i = np.rint((plon - lon[0]) / dlo).astype(int) % len(lon)
            np.add.at(target, (j, i), val)
        deposit(shifted, slon, slat, cval)
        deposit(plain, clon, clat, cval)
        deposit(age_num, slon, slat, cval * tmid)
        for bi, (b0, b1) in enumerate(ABINS):
            if b0 <= tmid < b1:
                deposit(by_age[bi], slon, slat, cval)
                break
        print(f'{t1:6.0f}-{t0:<7.0f} {cval.sum():13.4g} {len(cval):8d} '
              f'{a.offset:10.0f} km')

    with np.errstate(invalid='ignore', divide='ignore'):
        age = np.where(shifted > 0, age_num / np.maximum(shifted, 1e-30), np.nan)
    w = np.cos(np.radians(LA))

    def carried(field):
        """Share of all the water that sits under the observed fast transition zone."""
        return field[fast].sum() / field.sum()
    base = (w * fast).sum() / w.sum()
    print(f'\nthe fast transition zone is {100 * base:.1f} per cent of the surface')
    print(f'  water landing on it, uncorrected : {100 * carried(plain):.1f} per cent '
          f'(enrichment {carried(plain) / base:.2f})')
    print(f'  water landing on it, dip-corrected: {100 * carried(shifted):.1f} per cent '
          f'(enrichment {carried(shifted) / base:.2f})')
    gain = carried(shifted) / carried(plain)
    print(f'  the correction moves {gain:.2f} times as much water onto it')

    ok = np.isfinite(age) & (shifted > 0)
    print(f'\nwater-weighted delivery age where water is predicted')
    for q in (10, 25, 50, 75, 90):
        print(f'  p{q:<3d} {np.percentile(age[ok], q):6.0f} Ma', end='')
    print()
    on = ok & fast
    print(f'  under the fast transition zone, median '
          f'{np.median(age[on]):.0f} Ma; elsewhere {np.median(age[ok & ~fast]):.0f} Ma')
    # A cell that received water at 80 Ma and again at 20 Ma would credit its
    # 80 Ma vintage with a fast transition zone that the 20 Ma delivery produced,
    # so a vintage cannot simply be scored everywhere it landed. Requiring that
    # nothing arrived since is the other extreme and is too strict: water spreads
    # over a wider footprint than a trench line, so a trace of young water
    # disqualifies a cell whose budget is overwhelmingly ancient. The test is
    # therefore dominance. A vintage is scored where it supplies at least a given
    # multiple of everything delivered since, and the multiple is swept, because
    # the answer should not depend on where an arbitrary line is drawn.
    print(f'\nwhat fraction of each vintage now sits under a fast transition zone')
    print(f'  a vintage counts in a cell where it supplies at least R times all '
          f'the water delivered since\n')
    rows = []
    for R in (0.0, 0.5, 1.0, 2.0, 4.0, np.inf):
        tag = ('everywhere it landed' if R == 0 else
               'nothing since' if not np.isfinite(R) else f'R = {R:g}')
        line, fitpts = [], []
        for bi, (b0, b1) in enumerate(ABINS):
            younger = by_age[:bi].sum(axis=0) if bi else np.zeros(LO.shape)
            if not np.isfinite(R):
                m = (by_age[bi] > 0) & (younger <= 0)
            else:
                m = by_age[bi] > np.maximum(R * younger, 0)
                m &= by_age[bi] > 0
            tot = by_age[bi][m].sum()
            if tot <= 0:
                line.append('     -'); continue
            on = by_age[bi][m & fast].sum() / tot
            line.append(f'{on / base:6.2f}')
            fitpts.append((0.5 * (b0 + b1), on / base))
            rows.append(dict(R=R, t0=b0, t1=b1, water=float(tot),
                             enrichment=float(on / base), n_cells=int(m.sum())))
        tau = np.nan
        if len(fitpts) >= 4:
            from scipy.optimize import curve_fit
            tt = np.array([x for x, _ in fitpts]); ee = np.array([y for _, y in fitpts])
            try:
                pp, _ = curve_fit(lambda t, e0, ei, ta: ei + (e0 - ei) * np.exp(-t / ta),
                                  tt, ee, p0=[ee[0], 0.1, 30],
                                  bounds=([0, -1, 3], [60, 5, 600]), maxfev=40000)
                tau = pp[2]
            except Exception:
                pass
        print(f'  {tag:22s} ' + ' '.join(line) + f'   tau {tau:5.1f} Myr')
    print(f'  {"":22s} ' + ' '.join(f'{int(b0):3d}-{int(b1):<2d}' for b0, b1 in ABINS))

    pd.DataFrame(rows).to_csv(os.path.join(P.OUT, f'water_persistence{a.suffix}.csv'), index=False)

    np.savez_compressed(os.path.join(P.OUT, f'water_forward{a.suffix}.npz'),
                        lon=lon, lat=lat, water=shifted, water_plain=plain,
                        age=age, fast=fast, offset=a.offset, model=a.model,
                        by_age=by_age, abins=np.array(ABINS, float))
    print(f'\nwrote {P.OUT}/water_forward{a.suffix}.npz')


if __name__ == '__main__':
    main()
