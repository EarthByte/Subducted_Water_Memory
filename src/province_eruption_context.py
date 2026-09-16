"""What was beneath each province WHEN IT ERUPTED, not where it sits today.

The first version of the geological test compared each province's present-day
position against a present-day occupancy map. That is wrong for anything old:
a province that erupted at 328 Ma was somewhere else, and the mantle beneath it
then is not the mantle beneath its present position now. Both the site and the
subduction have to be put back to the eruption age.

For every field with a date, this:

  1. reconstructs the field to its eruption age T in the plate model's mantle
     reference frame, so that the site and the trenches are in the same frame;
  2. walks epochs t > T, reconstructing trenches, displacing each trench sample
     down dip by the same offset the rest of the analysis uses, and asking
     whether any displaced sample lies within the search radius of the site AS
     IT WAS AT TIME T;
  3. reports the LEAD TIME, t - T, for the youngest such epoch: how long before
     the eruption subduction last delivered material beneath that spot.

A LIMIT ON HOW FAR BACK ANY OF THIS CAN BE READ. The seismic class is measured
now; the eruption context is measured then. For an old province the two cannot be
about the same mantle, because the persistence analysis puts the transition
zone's memory of subduction at a few tens of Myr: whatever fed a 176 Ma eruption
left the transition zone long ago, and if that province is fast today it is fast
for a different reason. The reconstruction runs to 400 Ma because the plate model
does, but the JOIN between reconstruction and tomography is only interpretable
where the eruption is young. Restrict on `age_Ma` before drawing a causal
inference; `province_subduction.py --max-age` does this.

It also flags whether any epoch falls in the window where the delivered parcel
would actually have been at transition-zone depth at the moment of eruption,

    T + 410/v_sink  <=  t  <=  T + 660/v_sink,

which at 50 km/Myr is 8 to 13 Myr before eruption. That is the condition the
hydrous-upwelling model needs, and it is much stricter than mere proximity.

    python3 src/province_eruption_context.py
    python3 src/province_eruption_context.py --dt 5 --tmax 400

Needs gplately and the plate model, so it runs where deep_time_hydration.py does.
Writes out/province_eruption_context.csv; province_subduction.py reads it if it
is present and falls back to the present-day map if not.
"""
import argparse, os
import numpy as np, pandas as pd

try:
    import paths as P
    CAT, OUT, MODELS = P.IPV_V3, getattr(P, 'OUT', 'out'), getattr(P, 'MODELS', None)
except Exception:
    _D = os.environ.get('HYD_DATA', '..')
    CAT, OUT, MODELS = os.path.join(_D, 'ipv_catalogue_georoc_v3.csv'), 'out', None

R_EARTH = 6371.0088


def displace(lon, lat, az_deg, km):
    """Down-dip displacement along the trench-normal azimuth, as in
    deep_time_hydration.py. Identical formula, deliberately."""
    d = km / R_EARTH
    la1, lo1, th = np.radians(lat), np.radians(lon), np.radians(az_deg)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(th))
    lo2 = lo1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return (np.degrees(lo2) + 180) % 360 - 180, np.degrees(la2)


def to_xyz(lon, lat):
    a, b = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='zahirovic2022')
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--tmax', type=float, default=400.0)
    ap.add_argument('--offset', type=float, default=300.0)
    ap.add_argument('--radius', type=float, default=250.0)
    ap.add_argument('--v-sink', type=float, default=50.0)
    # a run at a different offset must not overwrite the reference one
    ap.add_argument('--suffix', default='',
                    help="appended to the output name, e.g. '_off200'")
    a = ap.parse_args()

    from scipy.spatial import cKDTree
    from gplately import PlateReconstruction, Points
    from plate_model_manager import PlateModelManager
    pmm = PlateModelManager().get_model(a.model, data_dir=MODELS)
    recon = PlateReconstruction(pmm.get_rotation_model(),
                                topology_features=pmm.get_topologies(),
                                static_polygons=pmm.get_static_polygons())

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    age = pd.to_numeric(d.get('age_Ma'), errors='coerce').values
    dated = np.isfinite(age) & (age >= 0) & (age <= a.tmax)
    print(f'{len(d)} fields, {int(dated.sum())} with a usable eruption age '
          f'(0 to {a.tmax:.0f} Ma)')

    # Reconstruct the sites to their own eruption ages, in bins so that each
    # epoch's rotation is applied once rather than per field.
    bins = np.where(dated, np.round(age / a.dt) * a.dt, np.nan)
    rlat = np.full(len(d), np.nan); rlon = np.full(len(d), np.nan)
    for t in sorted(set(bins[np.isfinite(bins)])):
        k = np.where(bins == t)[0]
        if t == 0:
            rlat[k], rlon[k] = d.lat.values[k], d.lon_180.values[k]
            continue
        pts = Points(recon, d.lon_180.values[k], d.lat.values[k])
        lo, la = pts.reconstruct(float(t), return_array=True)
        rlon[k], rlat[k] = lo, la
    moved = np.full(len(d), np.nan)
    ok = np.isfinite(rlat)
    if ok.any():
        x0, x1 = to_xyz(d.lon_180.values[ok], d.lat.values[ok]), to_xyz(rlon[ok], rlat[ok])
        moved[ok] = R_EARTH * np.arccos(np.clip((x0 * x1).sum(1), -1, 1))
    print(f'  sites moved a median of {np.nanmedian(moved):.0f} km back to their '
          f'eruption ages (max {np.nanmax(moved):.0f} km)')

    lead = np.full(len(d), np.nan)          # t - T for the youngest hit
    tz_hit = np.zeros(len(d), bool)         # any epoch in the arrival window
    lo_w, hi_w = 410.0 / a.v_sink, 660.0 / a.v_sink
    epochs = np.arange(0.0, a.tmax + a.dt, a.dt)
    for t in epochs:
        cand = np.where(ok & (bins <= t - 1e-9) if t > 0 else np.zeros(len(d), bool))[0]
        if not len(cand):
            continue
        sz = recon.tessellate_subduction_zones(
            float(t), tessellation_threshold_radians=0.01, ignore_warnings=True)
        if sz is None or not len(sz):
            continue
        tlon, tlat = displace(sz[:, 0], sz[:, 1], sz[:, 7], a.offset)
        tree = cKDTree(to_xyz(tlon, tlat))
        chord = 2.0 * np.sin(0.5 * a.radius / R_EARTH)
        hit = tree.query_ball_point(to_xyz(rlon[cand], rlat[cand]), chord)
        for i, h in zip(cand, hit):
            if not h:
                continue
            dtt = t - bins[i]
            if not np.isfinite(lead[i]) or dtt < lead[i]:
                lead[i] = dtt
            if lo_w <= dtt <= hi_w:
                tz_hit[i] = True

    out = pd.DataFrame(dict(lat=d.lat, lon_180=d.lon_180, age_Ma=age,
                            recon_lat=rlat, recon_lon=rlon, moved_km=moved,
                            lead_Myr=lead, tz_timed=tz_hit))
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f'province_eruption_context{a.suffix}.csv')
    out.to_csv(f, index=False)
    n = int(np.isfinite(lead).sum())
    print(f'  {n} of {int(dated.sum())} dated fields had subduction within '
          f'{a.radius:.0f} km at some epoch before eruption')
    if n:
        print(f'  lead time before eruption: median {np.nanmedian(lead):.0f} Myr, '
              f'{int((lead <= 50).sum())} within 50 Myr')
    print(f'  {int(tz_hit.sum())} fields had delivery timed to put the parcel at '
          f'transition-zone depth at eruption ({lo_w:.0f} to {hi_w:.0f} Myr before)')
    print(f'wrote {f}')


if __name__ == '__main__':
    main()
