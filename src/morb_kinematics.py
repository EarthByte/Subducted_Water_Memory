"""What passed over each MORB site between its hydration and today.

The Dixon compilation dates the last delivery of slab water beneath every ridge
site in the mantle reference frame. The reviewer's objection to the Farallon
attribution was kinematic: for water delivered west of North America at 220 Ma
to be sampled by the Mid-Atlantic Ridge today, the whole continent, keel and
all, must have passed over it in between. Whether it did, when, for how long,
and whether the lithosphere that passed was a 200 km cratonic keel or a 100 km
margin is a question the reconstruction can answer exactly, and nothing in
either paper has asked it. This script does, for every site, and records the
second kinematic quantity the geochemistry needs: how long a spreading ridge
has sat over the site since the water arrived, which is the time the mantle
there has been processed by decompression melting.

For each site (fixed in the mantle frame, present-day coordinates) and each
reconstruction time from the model start to the present:

    cont[t]     a continental cell of the plate model lies over the site
    thick[t]    that cell's present-day lithosphere is at least THICK km
    keel[t]     at least KEEL km
    craton[t]   the cell lies inside one of the twenty present-day craton
                polygons of Shirmard et al. (2025), which are defined
                tomographically from REVEAL rather than from surface geology,
                so the keel is delimited by the same velocity model the rest of
                this paper reads. The polygons are rasterised at 0 Ma, given
                plate identities by the Zahirovic et al. (2022) static polygons
                and moved with the Zahirovic et al. (2022) rotations, like
                every other cell here.
    d_ridge[t]  distance to the nearest mid-ocean ridge, km
    d_trench[t] distance to the nearest subduction zone, km

Continental cells are the present-day continental mask rasterised at half a
degree, given plate identities by the model's static polygons and moved rigidly
with them. Their thickness is LithoRef18 (Afonso et al., 2019) at the present
day, carried back unchanged: a keel is assumed to have been a keel throughout,
which overstates thick lithosphere where a root has since been lost and is
stated as the assumption it is. The craton polygons are the alternative: an
outline of the seismically fast root rather than a thickness threshold, so a
craton whose root is thin in LithoRef18 today but lies inside the tomographic
outline counts under this definition and not under the other. Ridges and
trenches are the model's resolved plate boundaries at each time.

Writes out/morb_kinematics<suffix>.npz with the per-site, per-time arrays and
out/morb_kinematics<suffix>.csv with the summaries the tests use:

    t_cont, t_thick, t_keel     Myr the site spent under continent / >=THICK /
                                >=KEEL lithosphere since its hydration age
    t_craton                    Myr under a craton polygon since its hydration age
    first_keel, last_keel       when keel cover began and ended (Ma)
    t_ridge_R                   Myr with a ridge within R km since hydration
    ridge_onset_R               when a ridge first came within R km (Ma)
    d_ridge_0                   distance to the nearest ridge today, km

Alternative reference frames enter through --rotations, which replaces the
model's rotation file (the optAPM frames differ only in the 005-000 rotation);
the hydration ages under that frame are read from the Dixon frames table by
--frame {NNR,meanNR,maxNR}.
"""
import os, argparse, numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
import paths as P
import morb
import contmask

R_EARTH = 6371.0088
THICK, KEEL = 150.0, 200.0
RIDGE_RADII = (100.0, 200.0, 300.0)
OPTAPM_FILES = {'NNR': 'Zahirovic2022_NNR.rot',
                'meanNR': 'Zahirovic2022_optimised_mean_NR.rot',
                'maxNR': 'Zahirovic2022_optimised_max_NR.rot'}


def to_xyz(lon, lat):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord_to_km(d):
    return R_EARTH * 2.0 * np.arcsin(np.clip(d / 2.0, 0, 1))


def plate_ids(static_polygons, rotations, lon, lat):
    """Plate identity of each present-day point from the static polygons."""
    import pygplates
    part = pygplates.PlatePartitioner(static_polygons, pygplates.RotationModel(rotations))
    pid = np.zeros(len(lon), int)
    for k, (lo, la) in enumerate(zip(lon, lat)):
        f = part.partition_point(pygplates.PointOnSphere(float(la), float(lo)))
        pid[k] = f.get_feature().get_reconstruction_plate_id() if f is not None else -1
    return pid


def rotation_matrix(fr):
    """3x3 matrix of a pygplates FiniteRotation."""
    lat, lon, ang = fr.get_lat_lon_euler_pole_and_angle_degrees()
    ax = to_xyz(np.array([lon]), np.array([lat]))[0]
    th = np.radians(ang)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.cos(th) * np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * np.outer(ax, ax)


def reconstruct_xyz(rotation_model, xyz, pid, t):
    """Rigidly reconstruct unit vectors with their plate ids to time t.
    Points whose plate has no rotation at t, or no plate, come back not ok."""
    out = np.full_like(xyz, np.nan)
    ok = np.zeros(len(xyz), bool)
    for p in np.unique(pid):
        if p < 0:
            continue
        m = pid == p
        fr = rotation_model.get_rotation(t, int(p), anchor_plate_id=0)
        if fr is None:
            continue
        out[m] = xyz[m] @ rotation_matrix(fr).T
        ok[m] = True
    return out, ok


def craton_flags(lon, lat):
    """Whether each present-day point lies inside one of the Shirmard et al.
    (2025) craton polygons, read from data/ as a present-day outline only."""
    import pygplates
    fc = pygplates.FeatureCollection(
        P.need(P.CRATONS, 'PAPER_DATA', 'the Shirmard et al. (2025) craton polygons'))
    polys = []
    for f in fc:
        for g in f.get_geometries():
            if isinstance(g, pygplates.PolygonOnSphere):
                polys.append(g)
    flags = np.zeros(len(lon), bool)
    for k, (lo, la) in enumerate(zip(lon, lat)):
        p = pygplates.PointOnSphere(float(la), float(lo))
        for poly in polys:
            if poly.is_point_in_polygon(p):
                flags[k] = True
                break
    return flags


def continental_cells(res=0.5):
    """Present-day continental cell centres with LithoRef18 thickness in km."""
    lon = np.arange(-180 + res / 2, 180, res)
    lat = np.arange(-90 + res / 2, 90, res)
    LO, LA = np.meshgrid(lon, lat)
    cont = contmask.is_continental(LO.ravel(), LA.ravel())
    li = xr.open_dataset(P.need(P.LITHO, 'PAPER_DATA', 'LithoRef18 lithospheric thickness'))
    var = list(li.data_vars)[0]
    z = li[var].values / 1000.0
    glat, glon = li.lat.values, li.lon.values
    j = np.clip(np.rint((LA.ravel() - glat[0]) / (glat[1] - glat[0])).astype(int), 0, len(glat) - 1)
    i = np.clip(np.rint((LO.ravel() - glon[0]) / (glon[1] - glon[0])).astype(int), 0, len(glon) - 1)
    thick = z[j, i]
    return LO.ravel()[cont], LA.ravel()[cont], thick[cont]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='zahirovic2022')
    ap.add_argument('--rotations', default=None,
                    help='alternative rotation file, e.g. an optAPM frame')
    ap.add_argument('--frame', default='published',
                    choices=['published', 'NNR', 'meanNR', 'maxNR'],
                    help='which hydration age to summarise against')
    ap.add_argument('--step', type=float, default=5.0)
    ap.add_argument('--thick', type=float, default=THICK,
                    help='km; lithosphere counted as thick (default 150)')
    ap.add_argument('--keel', type=float, default=KEEL,
                    help='km; lithosphere counted as a keel (default 200)')
    ap.add_argument('--start', type=float, default=400.0)
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    thick_km, keel_km = a.thick, a.keel

    import pygplates
    from gplately import PlateReconstruction
    from plate_model_manager import PlateModelManager
    pmm = PlateModelManager().get_model(a.model, data_dir=P.MODELS)
    rot = a.rotations or pmm.get_rotation_model()
    if a.rotations is None and a.frame != 'published':
        # the optAPM frames live with the GPlately tutorials, as in the Dixon
        # workflow; DIXON_OPTAPM overrides the location
        optapm = os.environ.get('DIXON_OPTAPM', os.path.expanduser(
            '~/Documents/GPlates/GPlately-pyGMT_tutorials/data/optAPM_reference_frames'))
        rot = os.path.join(optapm, OPTAPM_FILES[a.frame])
        P.need(rot, 'DIXON_OPTAPM', f'the {a.frame} optAPM rotation file')
    recon = PlateReconstruction(rot, topology_features=pmm.get_topologies(),
                                static_polygons=pmm.get_static_polygons())

    d = morb.load()
    if a.frame != 'published':
        f = morb.load_frames(d)
        d['age'] = f[f'age_{a.frame}'].values
        d['hydrated'] = np.isfinite(d['age'].values)
        print(f'{a.frame} frame: {int(d.hydrated.sum())} hydrated samples')
    site_xyz = to_xyz(d.Longitude.values, d.Latitude.values)
    n = len(d)

    clon, clat, cthick = continental_cells()
    print(f'{len(clon)} continental cells; {int((cthick >= thick_km).sum())} at least '
          f'{thick_km:.0f} km thick, {int((cthick >= keel_km).sum())} at least {keel_km:.0f} km')
    cell_pid = plate_ids(pmm.get_static_polygons(), rot, clon, clat)
    ccraton = craton_flags(clon, clat)
    print(f'{int(ccraton.sum())} cells inside a craton polygon')
    cell_xyz = to_xyz(clon, clat)
    rotation_model = pygplates.RotationModel(rot)
    half_cell_km = 0.5 * 0.5 * 111.2 * 1.5     # a cell's half diagonal, generously

    times = np.arange(0.0, a.start + 1e-9, a.step)
    cont = np.zeros((len(times), n), bool)
    thick = np.zeros((len(times), n), bool)
    keel = np.zeros((len(times), n), bool)
    craton = np.zeros((len(times), n), bool)
    d_ridge = np.full((len(times), n), np.nan)
    d_trench = np.full((len(times), n), np.nan)
    print(f'\n{"t (Ma)":>7s} {"sites under continent":>22s} {"under keel":>11s} '
          f'{"ridge within 200 km":>20s}')
    for k, t in enumerate(times):
        rxyz, ok = reconstruct_xyz(rotation_model, cell_xyz, cell_pid, float(t))
        if ok.any():
            tree = cKDTree(rxyz[ok])
            dist, nb = tree.query(site_xyz)
            over = chord_to_km(dist) <= half_cell_km
            th = cthick[ok][nb]
            cont[k] = over
            thick[k] = over & (th >= thick_km)
            keel[k] = over & (th >= keel_km)
            craton[k] = over & ccraton[ok][nb]
        try:
            mor = recon.tessellate_mid_ocean_ridges(
                float(t), tessellation_threshold_radians=0.01, ignore_warnings=True)
            if mor is not None and len(mor):
                dd, _ = cKDTree(to_xyz(mor[:, 0], mor[:, 1])).query(site_xyz)
                d_ridge[k] = chord_to_km(dd)
        except Exception:
            pass
        try:
            sz = recon.tessellate_subduction_zones(
                float(t), tessellation_threshold_radians=0.01, ignore_warnings=True)
            if sz is not None and len(sz):
                dd, _ = cKDTree(to_xyz(sz[:, 0], sz[:, 1])).query(site_xyz)
                d_trench[k] = chord_to_km(dd)
        except Exception:
            pass
        if k % 4 == 0:
            print(f'{t:7.0f} {int(cont[k].sum()):22d} {int(keel[k].sum()):11d} '
                  f'{int(np.nansum(d_ridge[k] <= 200)):20d}')

    # summaries since the hydration age
    age = d.age.values
    rows = []
    for j in range(n):
        rec = dict(Sample=d.Sample[j], Latitude=d.Latitude[j], Longitude=d.Longitude[j],
                   H2O_Ce=d.H2O_Ce[j], age=age[j], hydrated=bool(np.isfinite(age[j])))
        if np.isfinite(age[j]):
            w = times <= age[j]
        else:
            w = np.ones(len(times), bool)     # the whole model span, for reference
        for name, arr in (('t_cont', cont), ('t_thick', thick), ('t_keel', keel),
                          ('t_craton', craton)):
            rec[name] = float(arr[w, j].sum() * a.step)
        kk = np.where(keel[:, j] & w)[0]
        rec['first_keel'] = float(times[kk].max()) if kk.size else np.nan
        rec['last_keel'] = float(times[kk].min()) if kk.size else np.nan
        cc = np.where(cont[:, j] & w)[0]
        rec['first_cont'] = float(times[cc].max()) if cc.size else np.nan
        rec['last_cont'] = float(times[cc].min()) if cc.size else np.nan
        for R in RIDGE_RADII:
            near = (d_ridge[:, j] <= R) & w
            rec[f't_ridge_{int(R)}'] = float(near.sum() * a.step)
            idx = np.where(near)[0]
            rec[f'ridge_onset_{int(R)}'] = float(times[idx].max()) if idx.size else np.nan
        rec['d_ridge_0'] = float(d_ridge[0, j])
        rec['min_d_trench_since'] = float(np.nanmin(d_trench[w, j]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    tag = a.suffix or ('' if a.frame == 'published' else f'_{a.frame}')
    out.to_csv(os.path.join(P.OUT, f'morb_kinematics{tag}.csv'), index=False)
    np.savez_compressed(os.path.join(P.OUT, f'morb_kinematics{tag}.npz'),
                        times=times, cont=cont, thick=thick, keel=keel, craton=craton,
                        d_ridge=d_ridge, d_trench=d_trench,
                        lon=d.Longitude.values, lat=d.Latitude.values,
                        sample=d.Sample.values.astype(str), age=age,
                        model=a.model, frame=a.frame, thick_km=thick_km, keel_km=keel_km)
    h = out[out.hydrated]
    print(f'\n{len(h)} hydrated samples. Since hydration:')
    print(f'  under continental lithosphere at some time: {int((h.t_cont > 0).sum())} '
          f'({100 * (h.t_cont > 0).mean():.0f}%), median {h.t_cont.median():.0f} Myr '
          f'among those')
    print(f'  under lithosphere >= {thick_km:.0f} km: {int((h.t_thick > 0).sum())} '
          f'({100 * (h.t_thick > 0).mean():.0f}%)')
    print(f'  under a keel >= {keel_km:.0f} km: {int((h.t_keel > 0).sum())} '
          f'({100 * (h.t_keel > 0).mean():.0f}%)')
    print(f'  under a craton polygon: {int((h.t_craton > 0).sum())} '
          f'({100 * (h.t_craton > 0).mean():.0f}%)')
    for R in RIDGE_RADII:
        c = f't_ridge_{int(R)}'
        print(f'  ridge within {R:.0f} km: median {h[c].median():.0f} Myr, '
              f'10-90% {h[c].quantile(.1):.0f}-{h[c].quantile(.9):.0f}')
    print(f'wrote out/morb_kinematics{tag}.csv and .npz')


if __name__ == '__main__':
    main()
