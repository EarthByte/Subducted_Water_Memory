"""Stage 21 — separating hydration from temperature with the Vp/Vs ratio.

Shear velocity alone cannot distinguish a cold slab from a hydrated one, which
is why every earlier stage in this project could only say that the transition
zone beneath continental intraplate volcanism is fast. Wang and Wang (2022) show
the two causes separate in the RATIO of the P and S anomalies rather than in
either alone: over the full temperature range their mineral-physics kernels give
about 4 per cent in Vp against 8 per cent in Vs, so a thermal anomaly has
dlnVs/dlnVp near 2, whereas water makes the P anomaly slightly LARGER than the S
anomaly, giving a ratio at or below 1. Harzburgite abundance contributes under
1 per cent and is neglected here.

This stage implements that discriminant on GLAD-M35 (Cui et al., 2024), which
carries P and S and, unusually, per-cell uncertainty estimates.

THREE THINGS THAT ARE APPROXIMATIONS, STATED RATHER THAN BURIED

The model is anisotropic and the kernels are isotropic, so Voigt averages are
formed first: Vs^2 = (2 Vsv^2 + Vsh^2)/3 and Vp^2 = (Vpv^2 + 4 Vph^2)/5. The
whole discriminant inherits whatever that conversion does, so the sensitivity of
the result to it is measured rather than assumed, using eta.

The ratio is unstable wherever the P anomaly approaches zero. The quantity
actually mapped is therefore the ANGLE of the anomaly pair in the (dlnVp, dlnVs)
plane, theta = atan2(dlnVs, dlnVp), which is well behaved everywhere. Thermal
corresponds to theta near 63 degrees, hydration to 45 degrees or less.

This is the first-order form of the Wang and Wang inversion, not the inversion
itself. They grid-search temperature, water and composition jointly including
anelastic terms; we report where the observed angle is inconsistent with a purely
thermal anomaly. That is enough to test the hydration prediction and not enough
to quote a water content, and we do not quote one.

Cells where either anomaly is smaller than the model's own reported uncertainty
are discarded before anything is computed.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# PATHS, not P: main() binds P to the P-wave anomaly at line 124, which
# shadows a module-level P and makes every default below unreachable.
import paths as PATHS
import plume_classifier as pc
from s16_plume_null import to_xyz
from s19_slab_provenance import label_seam, cell_area_km2, MIN_BODY_KM2

warnings.filterwarnings('ignore')

R_EARTH = 6371.0
MTZ = (410.0, 660.0)
THERMAL_DEG = np.degrees(np.arctan2(2.0, 1.0))    # dlnVs/dlnVp = 2
HYDROUS_DEG = 45.0                                # dlnVs/dlnVp = 1


def voigt(ds, exact=False):
    """Isotropic-equivalent Vp and Vs from the transversely isotropic model.

    The Love parameters are A = rho Vph^2, C = rho Vpv^2, N = rho Vsh^2,
    L = rho Vsv^2 and F = eta (A - 2L). The Voigt averages of the bulk and
    shear moduli are

        kappa = (C + 4A + 4F - 4N)/9 ,   mu = (C + A + 6L + 5N - 2F)/15 ,

    from which Vp^2 = kappa + 4 mu/3 and Vs^2 = mu (per unit density). Setting
    eta = 1 collapses Vp^2 to the familiar (Vpv^2 + 4 Vph^2)/5, and dropping the
    small P-anisotropy term (C - A)/15 from mu collapses Vs^2 to
    (2 Vsv^2 + Vsh^2)/3. Both approximations are in routine use; exact=True
    keeps every term so their effect can be measured rather than assumed.
    """
    L, N = ds['vsv'].values ** 2, ds['vsh'].values ** 2
    C, A = ds['vpv'].values ** 2, ds['vph'].values ** 2
    if exact:
        F = ds['eta'].values * (A - 2.0 * L)
        mu = (C + A + 6.0 * L + 5.0 * N - 2.0 * F) / 15.0
        kap = (C + 4.0 * A + 4.0 * F - 4.0 * N) / 9.0
        return np.sqrt(kap + 4.0 * mu / 3.0), np.sqrt(mu)
    return np.sqrt((C + 4.0 * A) / 5.0), np.sqrt((2.0 * L + N) / 3.0)


def shell_anomaly(v):
    m = np.nanmean(v, axis=(1, 2))[:, None, None]
    return (v - m) / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=os.path.join(PATHS.TOMO, 'GLADM35_mtz.nc'))
    ap.add_argument('--src', default=pc.DATA)
    ap.add_argument('--bodies',
                    default=os.path.join(PATHS.OUT, 's19', 'slab_bodies.csv'))
    ap.add_argument('--reveal', default=PATHS.REVEAL)
    ap.add_argument('--ipv', default=PATHS.IPV_V3)
    ap.add_argument('--roots', default=os.path.join(
                        PATHS.TOMO, 'Analysis_2026-08', 'tables',
                        'root_test_REVEAL_voigt_2880km.csv'))
    ap.add_argument('--out', default=PATHS.OUT)
    ap.add_argument('--snr', type=float, default=1.0,
                    help='require |anomaly| > snr x reported uncertainty')
    a = ap.parse_args()

    ds = xr.open_dataset(os.path.join(a.src, a.file))
    depth = ds['depth'].values.astype(float)
    lat = ds['latitude'].values.astype(float)
    lon = ds['longitude'].values.astype(float)
    vp, vs = voigt(ds)
    dvp, dvs = shell_anomaly(vp), shell_anomaly(vs)

    # fractional uncertainty from the model's own error fields
    fp = ds['std_vp'].values / np.nanmean(vp, axis=(1, 2))[:, None, None]
    fs = ds['std_vs'].values / np.nanmean(vs, axis=(1, 2))[:, None, None]

    k = (depth >= MTZ[0]) & (depth < MTZ[1])
    print(f'GLAD-M35 transition zone: {k.sum()} shells, '
          f'{depth[k].min():.0f}-{depth[k].max():.0f} km, grid {len(lat)}x{len(lon)}')

    P = np.nanmean(dvp[k], axis=0)
    S = np.nanmean(dvs[k], axis=0)
    UP = np.nanmean(fp[k], axis=0)
    US = np.nanmean(fs[k], axis=0)
    good = (np.abs(P) > a.snr * UP) & (np.abs(S) > a.snr * US) & np.isfinite(P) & np.isfinite(S)
    print(f'cells where both anomalies exceed the model uncertainty: '
          f'{100 * good.mean():.1f}%')

    theta = np.degrees(np.arctan2(S, P))
    theta = np.where(good, theta, np.nan)
    # fold the fast and slow half-planes together: only the ratio matters
    tf = np.where(theta < -90, theta + 180, np.where(theta > 90, theta - 180, theta))

    print(f'\nreference angles: purely thermal {THERMAL_DEG:.1f} deg '
          f'(dlnVs/dlnVp = 2), hydrous <= {HYDROUS_DEG:.0f} deg (ratio <= 1)')
    print(f'global median angle where resolved: {np.nanmedian(tf):.1f} deg')

    # sensitivity of the anisotropic-to-isotropic conversion
    vp2, vs2 = voigt(ds, exact=True)
    P2 = np.nanmean(shell_anomaly(vp2)[k], axis=0)
    S2 = np.nanmean(shell_anomaly(vs2)[k], axis=0)
    t2 = np.degrees(np.arctan2(S2, P2))
    t2 = np.where(good, np.where(t2 < -90, t2 + 180, np.where(t2 > 90, t2 - 180, t2)), np.nan)
    print(f'  keeping every term of the Voigt average (eta, P anisotropy in mu) '
          f'moves the median angle by {np.nanmedian(t2 - tf):+.2f} deg '
          f'(90th percentile of |shift| {np.nanpercentile(np.abs(t2 - tf), 90):.2f} deg)')

    LO, LA = np.meshgrid(lon, lat)

    # ---------------------------------------------------------------- bodies
    print('\n' + '=' * 78)
    print('ANGLE WITHIN THE ATTRIBUTED TRANSITION-ZONE SLAB BODIES')
    print('=' * 78)
    rv = pc.ModelSpec('REVEAL', a.reveal, 'voigt', 'dv/v')
    rdep, rlat, rlon, rarr = pc.load_anomaly(rv, os.path.join(a.src, a.reveal))
    rk = (rdep >= MTZ[0]) & (rdep < MTZ[1])
    rband = np.nanmean(rarr[rk], axis=0)
    fmask = np.isfinite(rband) & (rband >= np.percentile(rband[np.isfinite(rband)], 90))
    rl, _ = label_seam(fmask)
    bodies = pd.read_csv(a.bodies)
    att = set(bodies[bodies.attributed].body.astype(int))

    RLO, RLA = np.meshgrid(rlon, rlat)
    gi = np.clip(((RLO + 180) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    gj = np.clip(((RLA + 90) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    theta_on_reveal = tf[gj, gi]

    rows = []
    for _, b in bodies.iterrows():
        sel = rl == int(b.body)
        t = theta_on_reveal[sel]
        t = t[np.isfinite(t)]
        if t.size < 50:
            continue
        rows.append(dict(region=b.region, attributed=bool(b.attributed),
                         area_km2=b.area_km2, n=int(t.size),
                         median_angle=float(np.median(t)),
                         frac_below_45=float((t <= HYDROUS_DEG).mean())))
    tab = pd.DataFrame(rows).sort_values('area_km2', ascending=False)
    print(tab.head(12).to_string(index=False, float_format=lambda x: f'{x:.1f}'))
    for flag, name in ((True, 'attributed to subduction'), (False, 'unattributed')):
        g = tab[tab.attributed == flag]
        if len(g):
            print(f'\n  {name:26s} median angle {g.median_angle.median():.1f} deg, '
                  f'fraction below 45 deg {g.frac_below_45.median():.2f}')

    # ---------------------------------------------------------------- sites
    def sample(df, label):
        gi = np.clip(((df.lon_180.values + 180) / (lon[1] - lon[0])).astype(int),
                     0, len(lon) - 1)
        gj = np.clip(((df.lat.values + 90) / (lat[1] - lat[0])).astype(int),
                     0, len(lat) - 1)
        t = tf[gj, gi]
        ok = np.isfinite(t)
        print(f'  {label:34s} n={ok.sum():4d} resolved of {len(df):4d}, '
              f'median angle {np.nanmedian(t):5.1f} deg, '
              f'below 45 deg {100 * np.nanmean(t[ok] <= HYDROUS_DEG):4.0f}%')
        return t

    print('\n' + '=' * 78)
    print('ANGLE BENEATH VOLCANIC SITES')
    print('=' * 78)
    roots = pd.read_csv(a.roots) if os.path.exists(a.roots) else None
    if roots is None:
        print(f'no hotspot-root table at {a.roots}; hotspot samples skipped')
        roots = pd.DataFrame(columns=['root_verdict', 'lat', 'lon_180'])
    sample(roots[roots.root_verdict.str.lower().str.startswith('rootless')],
           'hotspots, no lower-mantle root')
    sample(roots[roots.root_verdict.str.startswith('deep-rooted')], 'deep-rooted hotspots')
    if a.ipv and os.path.exists(a.ipv):
        ipv = pd.read_csv(a.ipv).dropna(subset=['lat', 'lon_180'])
        sample(ipv, 'continental intraplate fields')
    print(f'  {"global background":34s} median angle '
          f'{np.nanmedian(tf):5.1f} deg, below 45 deg '
          f'{100 * np.nanmean(tf[np.isfinite(tf)] <= HYDROUS_DEG):4.0f}%')

    # ------------------------------------------------------ amplitude control
    # An angle is only as well determined as the vector whose direction it is.
    # Where both anomalies are a tenth of a per cent, atan2 returns a number but
    # that number carries no information about mechanism. Stratifying by the
    # length of the anomaly vector shows whether the low-angle population is a
    # physical population or simply the weak-anomaly population.
    print('\n' + '=' * 78)
    print('ANGLE AS A FUNCTION OF ANOMALY AMPLITUDE')
    print('=' * 78)
    amp = np.hypot(P, S)
    aa = amp[good]
    tt = tf[good]
    qs = np.percentile(aa, [0, 20, 40, 60, 80, 90, 95, 100])
    print(f'{"amplitude |(dlnVp,dlnVs)| %":30s} {"n":>7s} {"median":>8s} {"IQR":>8s} '
          f'{"<45 deg":>9s}')
    for i in range(len(qs) - 1):
        m = (aa >= qs[i]) & (aa <= qs[i + 1] if i == len(qs) - 2 else aa < qs[i + 1])
        if m.sum() < 20:
            continue
        q1, q3 = np.percentile(tt[m], [25, 75])
        print(f'  {100 * qs[i]:6.3f} - {100 * qs[i + 1]:6.3f}{"":12s} {m.sum():7d} '
              f'{np.median(tt[m]):8.1f} {q3 - q1:8.1f} {100 * (tt[m] <= HYDROUS_DEG).mean():8.1f}%')
    strong = aa >= np.percentile(aa, 90)
    print(f'\n  strongest decile only: median angle {np.median(tt[strong]):.1f} deg '
          f'(thermal reference {THERMAL_DEG:.1f}), '
          f'{100 * (tt[strong] <= HYDROUS_DEG).mean():.1f}% below 45 deg')

    # ------------------------------------------------- coherent low-angle patches
    # A hydrous reservoir of the kind GLAD-M35 could see has to be hundreds of
    # kilometres across, so isolated low-angle cells are of no interest. Only
    # connected patches larger than the model's own resolution length count.
    print('\n' + '=' * 78)
    print('COHERENT LOW-ANGLE REGIONS')
    print('=' * 78)
    # amplitude gate: a cell only enters the search if its anomaly vector is long
    # enough for its direction to mean anything (median length or greater)
    gate = np.nanmedian(aa)
    low = np.isfinite(tf) & (tf <= HYDROUS_DEG) & (amp >= gate)
    print(f'searching only cells with anomaly amplitude >= the global median '
          f'{100 * gate:.2f}%, where the angle is determined to better than '
          f'{np.percentile(tt[aa >= gate], 75) - np.percentile(tt[aa >= gate], 25):.0f} deg (IQR)')
    lab, nlab = label_seam(low)
    area = cell_area_km2(lat, lon)
    patches = []
    for i in range(1, nlab + 1):
        sel = lab == i
        km2 = float(area[sel].sum())
        if km2 < MIN_BODY_KM2:
            continue
        patches.append(dict(area_km2=km2, n=int(sel.sum()),
                            lat=float(LA[sel].mean()), lon=float(LO[sel].mean()),
                            median_angle=float(np.median(tf[sel])),
                            median_dlnVs=float(100 * np.median(S[sel])),
                            median_dlnVp=float(100 * np.median(P[sel]))))
    for p_ in patches:
        pass
    pat = pd.DataFrame(patches).sort_values('area_km2', ascending=False) \
        if patches else pd.DataFrame()
    if len(pat):
        pat['sense'] = np.where((pat.median_dlnVs < 0) & (pat.median_dlnVp < 0), 'slow',
                                np.where((pat.median_dlnVs > 0) & (pat.median_dlnVp > 0),
                                         'fast', 'mixed'))
    print(f'{len(pat)} connected patches with angle <= {HYDROUS_DEG:.0f} deg exceed '
          f'{MIN_BODY_KM2 / 1e5:.0f}e5 km2 '
          f'({100 * low[np.isfinite(tf)].mean() if np.isfinite(tf).any() else 0:.1f}% of '
          f'resolved cells are below the threshold in total)')
    if len(pat):
        print(pat.head(10).to_string(index=False, float_format=lambda x: f'{x:.1f}'))
        os.makedirs(a.out, exist_ok=True)
        pat.to_csv(os.path.join(a.out, 'vpvs_low_angle_patches.csv'), index=False)
    # Hydration lowers both velocities. A low ratio on a FAST anomaly cannot be
    # water whatever else it is, so the sign is the last filter.
    ns = int((pat.sense == 'slow').sum()) if len(pat) else 0
    print(f'\n  of these, {ns} are slow in both Vp and Vs, which is the only sense '
          f'hydration can produce')

    # ------------------------------------------------------- depth-by-depth
    # If the discriminant is measuring mineralogy rather than noise it should
    # behave sensibly across the 410 and 660 discontinuities, and any hydrous
    # reservoir should appear in the wadsleyite and ringwoodite fields alone.
    print('\n' + '=' * 78)
    print('ANGLE SHELL BY SHELL, STRONG ANOMALIES ONLY')
    print('=' * 78)
    prof = []
    for i, d in enumerate(depth):
        Pb, Sb = dvp[i], dvs[i]
        gb = (np.abs(Pb) > a.snr * fp[i]) & (np.abs(Sb) > a.snr * fs[i]) \
            & np.isfinite(Pb) & np.isfinite(Sb)
        ab = np.hypot(Pb, Sb)
        gb &= ab >= np.nanpercentile(ab[gb], 75) if gb.any() else gb
        tb = np.degrees(np.arctan2(Sb[gb], Pb[gb]))
        tb = np.where(tb < -90, tb + 180, np.where(tb > 90, tb - 180, tb))
        prof.append(dict(depth=float(d), n=int(gb.sum()),
                         median_angle=float(np.median(tb)),
                         frac_below_45=float((tb <= HYDROUS_DEG).mean())))
    pr = pd.DataFrame(prof)
    for _, r in pr.iterrows():
        if r.depth % 50 == 0 or r.depth in (410, 660):
            bar = '#' * int(round(r.median_angle / 2))
            print(f'  {r.depth:5.0f} km  {r.median_angle:5.1f} deg  '
                  f'{100 * r.frac_below_45:4.1f}% <45  {bar}')
    print(f'  thermal reference {THERMAL_DEG:.1f} deg would be '
          f'{"#" * int(round(THERMAL_DEG / 2))}')
    if os.path.isdir(a.out) or True:
        os.makedirs(a.out, exist_ok=True)
        pr.to_csv(os.path.join(a.out, 'vpvs_angle_profile.csv'), index=False)

    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(os.path.join(a.out, 'vpvs_angle_by_body.csv'), index=False)
    # UP and US are the model's own uncertainties on the two anomalies, in the
    # same units as them. They were computed here only to threshold `good` and
    # then discarded, which left the angle with a cut but no error bar; keeping
    # them lets vpvs_uncertainty.py propagate onto theta.
    np.savez_compressed(os.path.join(a.out, 'vpvs_angle_field.npz'),
                        lon=lon, lat=lat, theta=tf, dvp=P, dvs=S, good=good,
                        sig_p=UP, sig_s=US)
    print('\nwrote vpvs_angle_by_body.csv and vpvs_angle_field.npz')


if __name__ == '__main__':
    main()
