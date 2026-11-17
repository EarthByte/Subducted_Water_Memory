# -*- coding: utf-8 -*-
"""The delay between water arriving at the base of the transition zone and intraplate volcanism
above it.

If a hydrous reservoir at the base of the transition zone feeds upwellings that melt on crossing
the 410 km discontinuity, then a dated volcanic field should sit above mantle that received water
some characteristic time before the field erupted. That time is what this measures. For every
dated continental intraplate field the reconstruction already gives the lead, the interval between
the last delivery of subducted water beneath the field and its eruption (`province_eruption_context`).
Against it is set the water content now inferred beneath the field from the joint inversion, and
the question is whether the two are related and at what lead the water is highest.

The field positions are their present ones, because the inferred water is a present-day field; the
lead is reconstructed. A field whose lead is near zero erupted while water was still arriving, and
one with a lead of 200 Myr erupted long after delivery stopped.

Significance comes from the spin null this paper uses elsewhere: the whole constellation is
rotated rigidly on the sphere, carrying its own geography with it, so only its alignment with the
water field is destroyed. Both nulls are reported, free rotations and rotations restricted to
continental lithosphere, and the second is the one to read, because continental mantle is drier in
this inversion by construction: cratons are fast, and the inversion reads fast as cold and dry.

    python3 src/water_plume_lag.py --pilot
    python3 src/water_plume_lag.py --model REVEAL --d660 waszek --rotations 500

Writes out/water_plume_lag.csv and out/water_plume_lag_bins.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import contmask
import paths as P
from sphere import spun, tail
from water_inversion import area_weights, invert, read_d660, read_vs, smooth

LEAD_BINS = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 130), (130, 200), (200, 400)]


def sample(field, lat, lon, flat, flon):
    """Nearest-cell value of a half-degree field at each field position."""
    i = np.clip(np.searchsorted(lat, flat), 0, lat.size - 1)
    j = np.clip(np.searchsorted(lon, flon), 0, lon.size - 1)
    return field[i, j]


def spin_mean(field, lat, lon, flat, flon, rng, rotations, continental, floor=None):
    """The mean of the field under the constellation, over admissible rigid rotations.

    Volcanic fields sit on continents, and continental mantle is drier in this inversion because
    cratons are fast and the inversion reads fast as cold and dry. A null that rotates the fields
    anywhere on the globe therefore compares them with mostly oceanic mantle and would make them
    look dry whatever the water does. With continental=True only the rotated positions that land
    on continental lithosphere are kept, and a rotation is used only if enough of the
    constellation survives that restriction: a quarter of it, with a floor of fifty points for
    the whole set of fields, which is the condition the field-scale tests in this paper apply,
    and a floor of three for a subset too small for fifty to be reachable. Attempts are capped,
    so a restriction that cannot be met returns fewer rotations rather than never returning.
    """
    need = floor if floor is not None else max(50, 0.25 * len(flon))
    out, attempts = [], 0
    while len(out) < rotations and attempts < 40 * rotations + 200:
        attempts += 1
        slon, slat = spun(flon, flat, rng)
        if continental:
            k = contmask.is_continental(slon, slat)
            if k.sum() < need:
                continue
            slon, slat = slon[k], slat[k]
        v = sample(field, lat, lon, slat, slon)
        v = v[np.isfinite(v)]
        if v.size:
            out.append(float(np.mean(v)))
    if len(out) < rotations:
        print('      (the continental restriction admitted {} of the {} rotations asked for)'
              .format(len(out), rotations))
    return np.asarray(out, float)

def run(model_name, product, args):
    """Set the inferred water beneath the dated fields against their delivery-to-eruption
    lead, for one model and one map.

    Returns the per-lead-interval rows and a summary. Both nulls are formed, free and
    restricted to continental lithosphere, and the second is the one to read.
    """
    glat, glon, dd, derr, dmean = read_d660(product)
    vlat, vlon, dvs, band = read_vs(model_name)
    dT, W = invert(dvs, dd)
    solvable = np.isfinite(W) & (dT > -600) & (dT < 400)
    Wc = smooth(np.where(solvable, np.clip(W, 0.0, None), np.nan), glat)

    # The lead comes from the reconstruction, the water from the present-day inversion, so a
    # field needs both to be usable: one without a delivery history has no lead to bin it by.
    f = pd.read_csv(os.path.join(P.OUT, 'province_eruption_context.csv'))
    f = f.dropna(subset=['lat', 'lon_180', 'lead_Myr']).reset_index(drop=True)
    w_field = sample(Wc, glat, glon, f.lat.values, f.lon_180.values)
    ok = np.isfinite(w_field)
    f, w_field = f[ok].reset_index(drop=True), w_field[ok]
    gm = float(np.nansum(Wc * area_weights(glat, glon)) /
               np.nansum(area_weights(glat, glon) * np.isfinite(Wc)))
    print('{} x {}: band {}, {} dated fields with a lead and a water value; '
          'global mean {:.3f} wt%'.format(model_name, product, band, len(f), gm))

    rng = np.random.default_rng(20261104)
    obs = float(np.mean(w_field))
    on_cont = float(contmask.is_continental(f.lon_180.values, f.lat.values).mean())
    free = spin_mean(Wc, glat, glon, f.lat.values, f.lon_180.values, rng, args.rotations, False) \
        if args.rotations else np.array([])
    cont = spin_mean(Wc, glat, glon, f.lat.values, f.lon_180.values, rng, args.rotations, True) \
        if args.rotations else np.array([])
    p_free, p_cont = tail(obs, free), tail(obs, cont)
    print('   {:.0%} of the fields are on continental lithosphere'.format(on_cont))
    print('   water beneath the fields {:.3f} wt% against {:.3f} global, {:+.0%}'.format(
        obs, gm, obs / gm - 1))
    print('      free rotations     median {:.3f} wt%, p {:.3f}'.format(
        float(np.median(free)) if free.size else float('nan'), p_free))
    print('      on continents only median {:.3f} wt%, p {:.3f}'.format(
        float(np.median(cont)) if cont.size else float('nan'), p_cont))
    p_all = p_cont

    # Rank correlation rather than linear: the leads are strongly skewed, with a third of the
    # fields erupting within 20 Myr of the last delivery and a tail out past 300 Myr.
    rho, p_rho = spearmanr(f.lead_Myr.values, w_field)
    print('   water against lead across fields: rho {:+.3f} (p {:.3f})'.format(rho, p_rho))

    rows = []
    print('      {:<16}{:>7}{:>10}{:>12}'.format('lead / Myr', 'fields', 'mean W', 'p(spin)'))
    for lo, hi in LEAD_BINS:
        m = (f.lead_Myr.values >= lo) & (f.lead_Myr.values < hi)
        if m.sum() < 3:
            continue
        o = float(np.mean(w_field[m]))
        nb = spin_mean(Wc, glat, glon, f.lat.values[m], f.lon_180.values[m], rng,
                       max(args.rotations // 2, 25), True,
                       floor=max(3, 0.25 * int(m.sum()))) if args.rotations else np.array([])
        p = tail(o, nb)
        rows.append(dict(model=model_name, d660=product, lead_lo=lo, lead_hi=hi,
                         n=int(m.sum()), mean_W=o, p_spin=p, global_mean=gm))
        print('      {:>3d} to {:>3d}{:>13d}{:>10.3f}{:>12.3f}'.format(lo, hi, int(m.sum()), o, p))
    best = max(rows, key=lambda r: r['mean_W']) if rows else None
    if best:
        print('   wettest lead interval: {} to {} Myr at {:.3f} wt%, {:+.0%} on the global mean'
              .format(best['lead_lo'], best['lead_hi'], best['mean_W'],
                      best['mean_W'] / gm - 1))
    return rows, dict(model=model_name, d660=product, band=band, n_fields=int(len(f)),
                      mean_W_fields=obs, global_mean=gm, p_spin_continental=p_cont,
                      p_spin_free=p_free, on_continent=on_cont, rho_lead=rho,
                      p_rho=p_rho, best_lead_lo=best['lead_lo'] if best else np.nan,
                      best_lead_hi=best['lead_hi'] if best else np.nan)


def main():
    """Run the lag test over the models and maps asked for and write the two tables.

    The pilot is one model with one map and writes nothing.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='all')
    ap.add_argument('--d660', default='all')
    ap.add_argument('--rotations', type=int, default=500)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    if args.pilot:
        args.model, args.d660, args.rotations, args.no_write = 'REVEAL', 'waszek', 100, True
    models = ('REVEAL', 'RevealLO', 'GLADM35', 'SPiRaL', 'SEMUCBWM1') if args.model == 'all' \
        else (args.model,)
    products = ('waszek', 'dai', 'huang') if args.d660 == 'all' else (args.d660,)
    bins, summary = [], []
    for m in models:
        for d in products:
            b, s = run(m, d, args)
            bins += b
            summary.append(s)
    if not args.no_write:
        pd.DataFrame(bins).to_csv(os.path.join(P.OUT, 'water_plume_lag_bins.csv'), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(P.OUT, 'water_plume_lag.csv'), index=False)
        print('wrote out/water_plume_lag.csv and out/water_plume_lag_bins.csv')


if __name__ == '__main__':
    main()
