# -*- coding: utf-8 -*-
"""How quickly agreement decays between reconstructed delivery and each observable at the base of
the transition zone: the inferred water content, the depth of the 660, and shear velocity.

The paper measures one decay already, of the excess of fast structure above reconstructed
delivery against the age of that delivery, and finds 37 Myr. Velocity, though, follows the cold
carrier. This script measures the same quantity for the water content inferred by the joint
inversion and for the discontinuity depth that goes into it, with the same enrichment statistic
and the same curve, so that the three numbers can be read side by side:

    E(t) = E_inf + (E_0 - E_inf) exp(-t / tau)

E(t) is the share of the cells that received delivery of age t which fall in the wettest tenth of
the field, divided by the share of all cells that are that wet, so one means delivery is no wetter
than the globe at large. Every field is smoothed to 600 km first, the resolution the discontinuity
maps claim at best, and the threshold is applied after smoothing, so the statistic is formed at the
scale at which the comparison is meaningful.

Significance is the spin null this paper uses elsewhere: the thresholded field is rotated rigidly
on the sphere, carrying its own autocorrelation with it, and the observed enrichment is read
against the distribution of the rotated ones.

    python3 src/water_decay.py --pilot
    python3 src/water_decay.py --model REVEAL --d660 waszek --rotations 200

Writes out/water_decay_<model>_<product>.csv and out/water_decay_fit.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import paths as P
from water_inversion import (area_weights, invert, read_d660, read_vs, smooth, NOISE_VS, NOISE_D,
                             SMOOTH_KM)

DECILE = 0.10


def model(t, e0, einf, tau):
    return einf + (e0 - einf) * np.exp(-t / tau)


def top_decile(field, w, frac=DECILE):
    """The wettest, deepest or fastest tenth by area, as a mask."""
    ok = np.isfinite(field)
    v, ww = field[ok].ravel(), w[ok].ravel()
    order = np.argsort(v)[::-1]
    cut = np.searchsorted(np.cumsum(ww[order]) / ww.sum(), frac)
    threshold = v[order][min(cut, v.size - 1)]
    return ok & (field >= threshold)


def enrichment(mask, delivery, w):
    """Share of delivered cells inside the mask, over the share of all cells inside it."""
    got = delivery > 0
    base = float(np.sum(w[mask]) / np.sum(w[np.isfinite(delivery)]))
    if not got.any() or base <= 0:
        return np.nan, 0, np.nan
    n = int(got.sum())
    frac = float(np.sum(w[got & mask]) / np.sum(w[got]))
    se = np.sqrt(max(frac * (1 - frac), 1e-12) / n)
    return frac / base, n, 1.96 * se / base


def spin(mask, w, delivery, rotations, rng):
    """The enrichment of the same mask rolled to other longitudes and flipped in latitude."""
    out = []
    for _ in range(rotations):
        shifted = np.roll(mask, rng.integers(mask.shape[1]), axis=1)
        if rng.random() < 0.5:
            shifted = shifted[::-1, :]
        e, _, _ = enrichment(shifted, delivery, w)
        out.append(e)
    return np.asarray(out, float)


def run(model_name, product, args):
    """Fit the enrichment curve for one model and one discontinuity map.

    The water, the discontinuity depth and the velocity are each thresholded at their top
    tenth after smoothing to 600 km, and the enrichment above delivery of each age is
    formed and fitted. Returns the per-interval rows and the fits.
    """
    glat, glon, dd, derr, dmean = read_d660(product)
    vlat, vlon, dvs, band = read_vs(model_name)
    dT, W = invert(dvs, dd)
    solvable = np.isfinite(W) & (dT > -600) & (dT < 400)
    Wc = np.where(solvable, np.clip(W, 0.0, None), np.nan)
    w = area_weights(glat, glon)

    z = np.load(os.path.join(P.OUT, 'water_forward.npz'), allow_pickle=True)
    by_age, abins = np.asarray(z['by_age']), np.asarray(z['abins'], float)

    fields = {'water': smooth(Wc, glat), 'd660': smooth(dd, glat), 'vs': smooth(dvs, glat)}
    rng = np.random.default_rng(20261104)
    rows, fits = [], []
    for what, field in fields.items():
        mask = top_decile(field, w)
        print('{} x {}: {} band {}, wettest tenth of {} covers {:.1%} of the area'.format(
            model_name, product, what, band, what, float(np.sum(w[mask]) / np.sum(w))))
        obs = []
        for k, (t0, t1) in enumerate(abins):
            d_k = smooth(by_age[k], glat)
            e, n, sigma = enrichment(mask, np.nan_to_num(d_k), w)
            null = spin(mask, w, np.nan_to_num(d_k), args.rotations, rng) if args.rotations else []
            p = (float(np.mean(np.asarray(null) >= e)) if len(null) else np.nan)
            rows.append(dict(model=model_name, d660=product, observable=what, t0=t0, t1=t1,
                             t_mid=0.5 * (t0 + t1), n=n, enrichment=e, sigma=sigma, p_spin=p))
            obs.append((0.5 * (t0 + t1), e, sigma))
            print('   delivery {:>3.0f} to {:>3.0f} Ma: enrichment {:5.2f} +/- {:4.2f}{}'.format(
                t0, t1, e, sigma, '' if np.isnan(p) else '   p(spin) {:.3f}'.format(p)))
        t_mid = np.array([r[0] for r in obs])
        e_obs = np.array([r[1] for r in obs])
        sig = np.array([max(r[2], 1e-3) for r in obs])
        # The curve is bounded, because an enrichment that falls almost linearly over the range
        # leaves the floor unidentifiable: unbounded, one map returned a floor of -1552 and a
        # decay time of 900,000 Myr with a respectable R squared. The floor is an enrichment and
        # cannot be negative, and a decay time outside 5 to 500 Myr is not a measurement of this
        # 400 Myr record. A fit that reaches a bound is reported as such.
        bounds = ([0.0, 0.0, 5.0], [10.0, 3.0, 500.0])
        try:
            popt, pcov = curve_fit(model, t_mid, e_obs,
                                   p0=[min(max(e_obs[0], 0.1), 9.9),
                                       min(max(e_obs[-1], 0.1), 2.9), 60.0],
                                   sigma=sig, absolute_sigma=True, maxfev=40000, bounds=bounds)
            err = np.sqrt(np.diag(pcov))
            at_bound = [n for n, v, lo, hi in zip(('E0', 'Einf', 'tau'), popt, *bounds)
                        if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6]
            if at_bound:
                print('   the fit reached a bound in {}, so it is a limit, not a measurement'
                      .format(', '.join(at_bound)))
            res = e_obs - model(t_mid, *popt)
            r2 = 1 - (res ** 2).sum() / ((e_obs - e_obs.mean()) ** 2).sum()
            print('   fit: E0 {:.2f}, Einf {:.2f}, tau {:.0f} +/- {:.0f} Myr, R2 {:.3f}'.format(
                popt[0], popt[1], popt[2], err[2], r2))
            fits.append(dict(model=model_name, d660=product, observable=what, E0=popt[0],
                             Einf=popt[1], tau=popt[2], tau_err=err[2], r2=r2))
        except Exception as exc:
            print('   fit failed: {}: {}'.format(type(exc).__name__, exc))
            fits.append(dict(model=model_name, d660=product, observable=what, E0=np.nan,
                             Einf=np.nan, tau=np.nan, tau_err=np.nan, r2=np.nan))
    return rows, fits


def main():
    """Run the decay fits over the models and maps asked for and write the two tables.

    The pilot is one model with one map and writes nothing, which is the run to make
    before committing to the full ensemble.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='all')
    ap.add_argument('--d660', default='all')
    ap.add_argument('--rotations', type=int, default=200)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    if args.pilot:
        args.model, args.d660, args.rotations, args.no_write = 'REVEAL', 'waszek', 50, True
    models = ('REVEAL', 'RevealLO', 'GLADM35', 'SPiRaL', 'SEMUCBWM1') if args.model == 'all' \
        else (args.model,)
    products = ('waszek', 'dai', 'huang') if args.d660 == 'all' else (args.d660,)
    rows, fits = [], []
    for m in models:
        for d in products:
            r, f = run(m, d, args)
            rows += r
            fits += f
    if not args.no_write:
        pd.DataFrame(rows).to_csv(os.path.join(P.OUT, 'water_decay.csv'), index=False)
        pd.DataFrame(fits).to_csv(os.path.join(P.OUT, 'water_decay_fit.csv'), index=False)
        print('wrote out/water_decay.csv and out/water_decay_fit.csv')
    print('\n{:<10}{:<9}{:>10}{:>12}'.format('model', 'd660', 'observable', 'tau / Myr'))
    for f in fits:
        print('{:<10}{:<9}{:>10}{:>12.0f}'.format(f['model'], f['d660'], f['observable'],
                                                  f['tau']))


if __name__ == '__main__':
    main()
