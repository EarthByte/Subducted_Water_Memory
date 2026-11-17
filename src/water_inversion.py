# -*- coding: utf-8 -*-
"""Water content and temperature at the base of the transition zone, and whether they track
reconstructed delivery.

Shear velocity alone cannot separate hydration from temperature: at transition-zone pressure
water lowers the velocity a little and the cold slab that carried it raises the velocity a lot,
so a map of fast structure is a map of the carrier. The depth of the 660 km discontinuity breaks
that degeneracy, because water and cold mantle both deepen it while they move velocity in
opposite senses. Two observables, two unknowns, solved per cell:

    dVs = a*dT + b*W            a = -1.4 %/230 K = -0.00609 %/K, b = -1.4 %/wt%
    dd660 = e*dT + f*W          e = -0.06 km/K, f = +6.0 km/wt%

The formulation is that of Liu et al. (2026), who map Asia this way. The coefficients come from
their sources: the velocity sensitivity to water in ringwoodite from Wang et al. (2021), the
depth sensitivity to water from Higo et al. (2001), and the Clapeyron slope of the post-spinel
transition from Bina & Helffrich (1994), Fei et al. (2004) and Yu et al. First-principles and
experimental calibrations of b and f differ, so --scale sweeps them.

What is new here is the comparison. Liu et al. score a cell by how many 1 Myr steps since 410 Ma
it lay within ten degrees of a reconstructed trench, which is proximity to subduction rather than
water. The forward model used here carries the water itself, in metres of equivalent sea level per
cell, with the age of each delivery, so the inferred water content can be compared with the water
the reconstruction says arrived, and separately for delivery of each age. That is the test the
paper needs: the carrier fades in a few tens of Myr, and the question is whether the water does.

Both fields are compared only after smoothing to the resolution of the discontinuity maps, which
is 600 km at best and coarser for three of the four products. The 250 km radius used for the
velocity comparisons is finer than anything these maps resolve.

Two things are not settled yet and are marked in the output rather than hidden: the velocity
band is 520-660 km until a 600-660 km band is extracted, which is the depth the calibration is
for; and the p-values printed with the correlations are those of the Spearman statistic over
164,143 smoothed cells, which are not independent, so the significance has to come from the
rotation test this paper uses elsewhere.

    python3 src/water_inversion.py --pilot
    python3 src/water_inversion.py --model REVEAL --d660 waszek
    python3 src/water_inversion.py                      # every model, every product

The inverted maps are written to the large-output folder `paths.big` resolves, PAPER_BIG or
out/large/, because the ensemble runs to fifteen files of tens of megabytes; the summary table
is small and stays in `out/water_inversion.csv` with the rest of the paper's numbers.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

import paths as P

# ---------------------------------------------------------------- the calibration
A_VS_T = -1.4 / 230.0      # % per K, from 1 wt% == -1.4 % Vs == +230 K (Wang et al., 2021)
B_VS_W = -1.4              # % per wt% of water in ringwoodite (Wang et al., 2021)
E_D_T = -0.06              # km per K, post-spinel Clapeyron slope (Bina & Helffrich, 1994)
F_D_W = 6.0                # km per wt% (Higo et al., 2001)

A_VP_T = -1.0 / 230.0      # % per K, from 1 wt% == -1.0 % Vp == +230 K (Wang et al., 2021)
B_VP_W = -1.0              # % per wt% of water in ringwoodite (Wang et al., 2021)

NOISE_VS, NOISE_D = 0.3, 3.0     # below these a cell carries no information (Liu et al., 2026)
# Measured uncertainties, used as weights in the three-observable solution: the shear and
# compressional figures are the means of GLAD-M35's own reported standard deviations over the
# transition zone (`out/vpvs_angle_field.npz`), and the discontinuity error is read per cell from
# the map itself, with this as the fallback where a map carries none.
SIG_VS, SIG_VP, SIG_D = 0.08, 0.12, 3.5
MODELS = ('REVEAL', 'RevealLO', 'GLADM35', 'SPiRaL', 'SEMUCBWM1')
BAND_WANTED, BAND_FALLBACK = '600_660', '520_660'
SMOOTH_KM = 600.0                 # the resolution the discontinuity maps claim at best
EARTH_KM = 6371.0



# Boxes for the independent check against the anomalies Liu et al. (2026) report, with two
# controls. Theirs are the Mongol-Okhotsk suture, the Central Asian Orogenic Belt, the Tethyan
# domain and, as their non-Asian case, northern South America around the Juina diamond locality.
REGIONS = {'Mongol-Okhotsk': (45, 57, 108, 132),
           'Central Asian Orogenic Belt': (40, 50, 72, 95),
           'Tethyan (Zagros-Iran)': (26, 38, 44, 62),
           'NW China and Tianshan': (38, 46, 78, 95),
           'northern South America': (-5, 10, -75, -55),
           'Pacific control': (-20, 20, -160, -120),
           'Atlantic control': (-20, 20, -35, -10)}


def regions(field, lat, lon, w):
    gm = float(np.nansum(field * w) / np.nansum(w * np.isfinite(field)))
    out = ['   {:<30}{:>9}{:>12}'.format('region', 'mean W', 'vs global')]
    for name, (la0, la1, lo0, lo1) in REGIONS.items():
        i, j = (lat >= la0) & (lat <= la1), (lon >= lo0) & (lon <= lo1)
        sub, ws = field[np.ix_(i, j)], w[np.ix_(i, j)]
        ok = np.isfinite(sub)
        m = float(np.sum(sub[ok] * ws[ok]) / np.sum(ws[ok]))
        out.append('   {:<30}{:>9.3f}{:>+11.0f}%'.format(name, m, 100 * (m / gm - 1)))
    return '\n'.join(out)


def area_weights(lat, lon):
    return np.repeat(np.cos(np.radians(lat))[:, None], lon.size, axis=1)


def read_d660(product):
    """Depth anomaly of the 660 in km, on the half-degree mesh, its own mean removed."""
    d = os.path.join(P.DATA_DIR, 'd660')
    if product == 'waszek':
        ds = xr.open_dataset(os.path.join(d, 'd660_waszek2021.nc'))
        lat, lon, dep = ds.lat.values, ds.lon.values, ds.d660.values
        err = ds.d660_err.values
    elif product == 'dai':
        t = np.loadtxt(os.path.join(d, 'd660_dai2026.csv'), delimiter=',')
        lat, lon, dep, err = _scatter(t[:, 1], t[:, 2], np.nanmean(t[:, 9:15], axis=1),
                                      np.nanstd(t[:, 9:15], axis=1), 600.0)
    elif product == 'huang':
        t = np.loadtxt(os.path.join(d, 'd660_huang2019.txt'))
        ok = np.isfinite(t[:, 3])
        lat, lon, dep, err = _scatter(t[ok, 2], t[ok, 1], t[ok, 3], t[ok, 4], 1000.0)
    else:
        raise SystemExit('unknown product: {}'.format(product))
    w = area_weights(lat, lon)
    good = np.isfinite(dep)
    mean = float(np.sum(dep[good] * w[good]) / np.sum(w[good]))
    return lat, lon, dep - mean, err, mean


def _scatter(plat, plon, val, err, radius_km):
    """Put scattered cap values on the half-degree mesh, nearest cap within its own radius."""
    lat = np.arange(-90.0, 90.0 + 0.5, 0.5)
    lon = np.arange(-180.0, 180.0 + 0.5, 0.5)
    glat, glon = np.meshgrid(lat, lon, indexing='ij')
    out = np.full(glat.shape, np.nan)
    oerr = np.full(glat.shape, np.nan)
    pxyz = _xyz(plat, plon)
    gxyz = _xyz(glat.ravel(), glon.ravel())
    for k in range(0, gxyz.shape[0], 20000):
        chunk = gxyz[k:k + 20000]
        d = np.arccos(np.clip(chunk @ pxyz.T, -1, 1)) * EARTH_KM
        j = np.argmin(d, axis=1)
        near = d[np.arange(d.shape[0]), j]
        take = near <= radius_km
        idx = np.arange(k, k + chunk.shape[0])[take]
        out.ravel()[idx] = val[j[take]]
        oerr.ravel()[idx] = err[j[take]]
    return lat, lon, out, oerr


def _xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def read_vs(model):
    """Shear-velocity anomaly in per cent over the deepest band of the transition zone."""
    for stem in (os.path.join(P.OUT, 'bands_{}.npz'.format(model)),
                 os.path.join(P.TOMO, 'REVEAL_mantle_tomography', 'bands_{}.npz'.format(model))):
        if os.path.exists(stem):
            z = np.load(stem)
            band = BAND_WANTED if BAND_WANTED in z else BAND_FALLBACK
            return z['lat'], z['lon'], z[band], band
    raise SystemExit('no band file for {}'.format(model))


def read_vp(model):
    """Compressional-wave anomaly over the same band, for a model that carries one."""
    for stem in (os.path.join(P.OUT, 'bands_{}_vp.npz'.format(model)),
                 os.path.join(P.TOMO, 'REVEAL_mantle_tomography',
                              'bands_{}_vp.npz'.format(model))):
        if os.path.exists(stem):
            z = np.load(stem)
            band = BAND_WANTED if BAND_WANTED in z else BAND_FALLBACK
            return z['lat'], z['lon'], z[band], band
    raise SystemExit(
        'no P-wave band file for {}: extract one with\n'
        '  python3 src/extract_bands.py --file <the volume> --tag {}_vp --pwave '
        '--out out/bands_{}_vp.npz'.format(model, model, model))


def invert(dvs, dd, scale_b=1.0, scale_f=1.0, scale_a=1.0, scale_e=1.0):
    """Solve the two equations per cell for temperature anomaly and water content.

    Which coefficient does what is worth stating, because it is not symmetric. Written out,
    W = (a*dd - e*dVs) / (a*f - b*e): the water sensitivities b and f appear only in the
    determinant, so scaling either one rescales the whole water map and leaves its pattern
    untouched, and with it every rank correlation. The pattern is set by a and e alone, the two
    temperature sensitivities, which fix how much of a deep discontinuity is read as cold rather
    than wet. On the published values the map is 0.0505*dd - 0.498*dVs, so velocity carries about
    twice the weight of the discontinuity at their respective amplitudes. The Clapeyron slope e is
    the least settled of the four, between about -0.04 and -0.09 km/K across experiments, so
    --scale-e is the sweep that tests whether a conclusion survives the calibration.
    """
    a, b = A_VS_T * scale_a, B_VS_W * scale_b
    e, f = E_D_T * scale_e, F_D_W * scale_f
    det = a * f - b * e
    dT = (dvs * f - b * dd) / det
    W = (a * dd - e * dvs) / det
    return dT, W



def invert3(dvs, dvp, dd, sig_d=None, scale_a=1.0, scale_e=1.0):
    """Temperature and water from three observables at once, by weighted least squares.

    Two observables determine two unknowns exactly, which means the two-observable inversion
    cannot be wrong: whatever it is given, it returns a temperature and a water content that
    reproduce both numbers. GLAD-M35 reports compressional as well as shear speed, so for that
    model there are three observables for two unknowns and the fit can fail, which is the only
    way to ask whether the calibration describes the mantle rather than assuming it does.

    Returned: temperature, water, and chi-squared on one degree of freedom. A cell whose
    chi-squared exceeds 3.84 cannot be explained by any single temperature and water content at
    95 per cent confidence, given the stated uncertainties.
    """
    a, e = A_VS_T * scale_a, E_D_T * scale_e
    rows = ((dvs, a, B_VS_W, SIG_VS),
            (dvp, A_VP_T * scale_a, B_VP_W, SIG_VP),
            (dd, e, F_D_W, sig_d if sig_d is not None else np.full_like(dd, SIG_D)))
    n11 = n12 = n22 = r1 = r2 = 0.0
    for obs, c_t, c_w, sig in rows:
        w = 1.0 / np.maximum(np.asarray(sig, float), 1e-6) ** 2
        n11 = n11 + w * c_t * c_t
        n12 = n12 + w * c_t * c_w
        n22 = n22 + w * c_w * c_w
        r1 = r1 + w * c_t * obs
        r2 = r2 + w * c_w * obs
    det = n11 * n22 - n12 * n12
    dT = (r1 * n22 - r2 * n12) / det
    W = (r2 * n11 - r1 * n12) / det
    chi2 = 0.0
    for obs, c_t, c_w, sig in rows:
        resid = obs - (c_t * dT + c_w * W)
        chi2 = chi2 + (resid / np.maximum(np.asarray(sig, float), 1e-6)) ** 2
    return dT, W, chi2


def smooth(field, lat, km=SMOOTH_KM):
    """Gaussian smoothing to a given full width, in grid cells of the half-degree mesh."""
    step_km = 0.5 * np.pi * EARTH_KM / 180.0
    sigma = (km / 2.355) / step_km
    m = np.isfinite(field).astype(float)
    filled = np.where(np.isfinite(field), field, 0.0)
    num = gaussian_filter(filled, sigma, mode=('nearest', 'wrap'))
    den = gaussian_filter(m, sigma, mode=('nearest', 'wrap'))
    out = np.where(den > 0.2, num / np.maximum(den, 1e-9), np.nan)
    return out


def run(model, product, args):
    """Invert one velocity model against one discontinuity map, and report what it gives.

    Prints the global statistics, the regional check against the anomalies Liu et al. report,
    and the correlation with delivery of each age, with the discontinuity and the velocity
    reported beside the water so that it is clear which observable carries an association.
    Writes the maps to the large-output folder unless asked not to.
    """
    glat, glon, dd, derr, dmean = read_d660(product)
    vlat, vlon, dvs, band = read_vs(model)
    assert np.allclose(glat, vlat) and np.allclose(glon, vlon), 'the two meshes differ'
    dT, W = invert(dvs, dd, args.scale_b, args.scale_f, args.scale_a, args.scale_e)
    if args.three:
        # Three observables for two unknowns, so the fit can fail: where it does, no single
        # temperature and water content explains the model's own P and S anomalies together with
        # the depth of the discontinuity, and that is the one test of the calibration that does
        # not assume the answer. Everything below then runs on this solution.
        plat, plon, dvp, pband = read_vp(model)
        assert np.allclose(plat, glat) and pband == band, \
            'the P-wave band file is on another mesh or another band'
        dT3, W3, chi2 = invert3(dvs, dvp, dd, derr, args.scale_a, args.scale_e)
        ok3 = np.isfinite(chi2)
        print('   three observables: {:.0%} of cells are explained by one temperature and one '
              'water content at 95 per cent, median chi-squared {:.1f} on one degree of '
              'freedom'.format(float(np.mean(chi2[ok3] <= 3.84)), float(np.median(chi2[ok3]))))
        both = np.isfinite(W3) & np.isfinite(W)
        print('      water from three observables correlates with the two-observable field at '
              '{:.3f}, and its area-weighted mean is {:.3f} against {:.3f} wt%'.format(
                  float(np.corrcoef(np.clip(W3[both], 0, None),
                                    np.clip(W[both], 0, None))[0, 1]),
                  float(np.nansum(np.clip(W3, 0, None) * area_weights(glat, glon)) /
                        np.nansum(area_weights(glat, glon))),
                  float(np.nansum(np.clip(W, 0, None) * area_weights(glat, glon)) /
                        np.nansum(area_weights(glat, glon)))))
        dT, W = dT3, W3

    informative = (np.abs(dvs) >= NOISE_VS) | (np.abs(dd) >= NOISE_D)
    solvable = np.isfinite(W) & (dT > -600) & (dT < 400)
    w = area_weights(glat, glon)
    keep = informative & solvable & np.isfinite(dd)
    # A negative solution is a dry cell, not a missing one: dropping those instead of reading them
    # as zero raises the global mean from 0.19 to 0.43 wt% and is the wrong statistic.
    Wc = np.where(solvable, np.clip(W, 0.0, None), np.nan)
    Wraw = np.where(solvable, W, np.nan)

    z = np.load(os.path.join(P.OUT, 'water_forward.npz'), allow_pickle=True)
    assert np.allclose(z['lat'], glat), 'the delivery field is on another mesh'
    delivery, age = z['water'], z['age']
    by_age, abins = z['by_age'], z['abins']

    Ws, Ds = smooth(Wc, glat), smooth(np.where(delivery > 0, delivery, 0.0), glat)
    both = np.isfinite(Ws) & np.isfinite(Ds)
    rho, p = spearmanr(Ws[both], Ds[both])

    # W = 0.0505*dd660 - 0.498*dVs on this calibration, so the two observables pull in opposite
    # directions and each one is reported against delivery as well, to show which carries it.
    dds, vss = smooth(dd, glat), smooth(dvs, glat)
    rows = []
    for k, (lo, hi) in enumerate(np.asarray(abins, float)):
        d_k = smooth(np.asarray(by_age)[k], glat)
        ok = both & np.isfinite(d_k)
        r_k, p_k = spearmanr(Ws[ok], d_k[ok])
        r_d = spearmanr(dds[ok], d_k[ok])[0]
        r_v = spearmanr(vss[ok], d_k[ok])[0]
        rows.append((lo, hi, r_k, p_k, int(ok.sum()), r_d, r_v))

    mean_W = float(np.sum(np.nan_to_num(Wc) * w * keep) / np.sum(w * keep))
    print('{} x {}: band {}, d660 mean {:.2f} km removed'.format(model, product, band, dmean))
    print('   cells informative {:.0%}, solvable {:.0%}, dry (negative solution) {:.0%}'.format(
        informative.mean(), (solvable & informative).mean(), float(np.nanmean(Wraw < 0))))
    print('   water content: area-weighted mean {:.3f} wt%, median {:.3f}, 90th pct {:.3f}, '
          'max {:.2f}; mean of the unclipped solution {:+.3f}'.format(
              mean_W, float(np.nanmedian(Wc)), float(np.nanpercentile(Wc, 90)),
              float(np.nanmax(Wc)), float(np.nansum(Wraw * w) / np.nansum(w))))
    print('   temperature: mean {:+.0f} K, 10th pct {:+.0f} K, 90th pct {:+.0f} K'.format(
        float(np.nanmean(np.where(solvable, dT, np.nan))),
        float(np.nanpercentile(np.where(solvable, dT, np.nan), 10)),
        float(np.nanpercentile(np.where(solvable, dT, np.nan), 90))))
    if args.regions:
        print(regions(Wc, glat, glon, w))
    print('   water against total delivery, smoothed to {:.0f} km: rho {:+.3f} (p {:.1e})'.format(
        SMOOTH_KM, rho, p))
    print('      {:<22}{:>9}{:>11}{:>11}'.format(
        'delivery interval', 'water', 'd660', 'Vs'))
    for lo, hi, r_k, p_k, n, r_d, r_v in rows:
        print('      {:>3.0f} to {:>3.0f} Ma{:>19.3f}{:>+11.3f}{:>+11.3f}'.format(
            lo, hi, r_k, r_d, r_v))

    if not args.no_write:
        ds = xr.Dataset({'water': (('lat', 'lon'), Wc), 'water_unclipped': (('lat', 'lon'), Wraw),
                         'temperature': (('lat', 'lon'), dT),
                         'd660_anomaly': (('lat', 'lon'), dd),
                         'vs_anomaly': (('lat', 'lon'), dvs),
                         'informative': (('lat', 'lon'), informative)},
                        coords={'lat': glat, 'lon': glon})
        ds.water.attrs = dict(units='wt%', long_name='water content at the base of the transition '
                                                     'zone from the joint inversion')
        ds.attrs = dict(model=model, d660=product, band=band, d660_mean_removed=dmean,
                        a_vs_T=A_VS_T * args.scale_a, b_vs_W=B_VS_W * args.scale_b,
                        e_d_T=E_D_T * args.scale_e, f_d_W=F_D_W * args.scale_f,
                        smoothing_km=SMOOTH_KM)
        # The maps are tens of megabytes each and there are fifteen of them across the
        # ensemble, so they go to the large-output folder rather than beside the tables.
        out = P.big('water_inversion_{}_{}.nc'.format(model, product),
                    'the inverted water and temperature maps')
        ds.to_netcdf(out)
        print('   wrote', out)
    return dict(model=model, d660=product, band=band, mean_W=mean_W,
                informative=float(informative.mean()),
                solvable=float((solvable & informative).mean()), rho_total=rho, p_total=p,
                **{'rho_{:.0f}_{:.0f}'.format(lo, hi): r for lo, hi, r, _, _, _, _ in rows})


def main():
    """Run the inversion over the models and maps asked for and write the summary table.

    The pilot is one model with one map and writes nothing, which is the run to make before
    the full ensemble.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS + ('all',), default='all')
    ap.add_argument('--d660', choices=('waszek', 'dai', 'huang', 'all'), default='all')
    ap.add_argument('--pilot', action='store_true',
                    help='REVEAL with the Waszek map only, and write nothing')
    ap.add_argument('--scale-b', type=float, default=1.0,
                    help='scale the velocity sensitivity to water')
    ap.add_argument('--scale-f', type=float, default=1.0,
                    help='scale the depth sensitivity to water; amplitude only')
    ap.add_argument('--scale-a', type=float, default=1.0,
                    help='scale the velocity sensitivity to temperature; changes the pattern')
    ap.add_argument('--scale-e', type=float, default=1.0,
                    help='scale the Clapeyron slope of the 660; changes the pattern')
    ap.add_argument('--three', action='store_true',
                    help='use the P-wave anomaly as a third observable, which makes the fit '
                         'falsifiable; needs a P-wave band file for the model')
    ap.add_argument('--regions', action='store_true',
                    help='the check against the regions Liu et al. report, with two controls')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    if args.pilot:
        args.model, args.d660, args.no_write = 'REVEAL', 'waszek', True
        args.regions = True
    models = MODELS if args.model == 'all' else (args.model,)
    products = ('waszek', 'dai', 'huang') if args.d660 == 'all' else (args.d660,)
    rows = [run(m, d, args) for m in models for d in products]
    if not args.no_write:
        out = os.path.join(P.OUT, 'water_inversion.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print('wrote', os.path.relpath(out, P.ROOT))


if __name__ == '__main__':
    main()
