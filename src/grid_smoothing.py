# -*- coding: utf-8 -*-
"""How much smoothing is in the "300 km" subducted-water grids.

The water products of the subduction workflow are named for a smoothing length, and the
supplement compares that length with the distance convective stirring moves material. The
comparison is only meaningful once it is known whether the name is a standard deviation, a
full width or something else, and the grids do not say.

This measures it. The difference between two consecutive cumulative grids is the delivery of
one time step: a line source along the trenches convolved with whatever filter the workflow
applied. Across a locally straight trench the cross-profile of that convolution is the filter
itself, so profiles are taken perpendicular to the trench, normalised at their peak, stacked
along the margin and fitted with a Gaussian.

    python3 src/grid_smoothing.py
    python3 src/grid_smoothing.py --times 100 99 --product subducted

Writes out/grid_smoothing.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

import paths as P

# Margins active at 100 Ma, as latitude and longitude windows containing one trench each.
MARGINS = [('NW Pacific', 20.0, 55.0, 120.0, 165.0),
           ('SE Asia', -10.0, 25.0, 90.0, 135.0),
           ('eastern Pacific', -35.0, -5.0, -90.0, -55.0),
           ('Tethys', 30.0, 48.0, -20.0, 40.0)]
HALF_WIDTH_KM = 900.0


def gaussian(x, amplitude, sigma, offset):
    return amplitude * np.exp(-0.5 * (x / sigma) ** 2) + offset


def step_field(product, t_old, t_new):
    """Delivery during one time step, from the difference of two cumulative grids."""
    g = {}
    for t in (t_old, t_new):
        files = P.water(product, 'Z22', t)
        z = None
        for f in files:
            d = xr.open_dataset(P.need(f, 'GRIDS_DIR', 'the Z22 {} grid at {} Ma'.format(product, t)))
            v = np.nan_to_num(d['z'].values)
            z = v if z is None else z + v
        g[t] = z
        lat, lon = d.lat.values, d.lon.values
    return g[t_new] - g[t_old], lat, lon


def sigma_rows(field, lat, lon, window, half_km=HALF_WIDTH_KM, max_rms=0.02):
    """Gaussian width of the cross-trench profile, row by row, in one window.

    Each row that crosses a well-developed band is fitted on its own and kept only if the
    Gaussian describes it to better than `max_rms` of the peak. Fitting row by row rather
    than stacking avoids widening the profile where the margin curves, which is what spoils
    a stacked fit along an arc.
    """
    name, lat_lo, lat_hi, lon_lo, lon_hi = window
    peak = field.max()
    kept = []
    for j in np.where((lat >= lat_lo) & (lat <= lat_hi))[0]:
        m = (lon >= lon_lo) & (lon <= lon_hi)
        y = field[j, m]
        if y.max() <= 0.2 * peak:
            continue
        dx = 111.195 * np.cos(np.radians(lat[j])) * abs(lon[1] - lon[0])
        k, half = int(np.argmax(y)), int(half_km / dx)
        if k - half < 0 or k + half >= y.size:
            continue
        seg = y[k - half:k + half + 1] / y[k]
        x = np.arange(-half, half + 1) * dx
        try:
            popt, _ = curve_fit(gaussian, x, seg, p0=[1.0, 150.0, 0.0], maxfev=20000)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean((gaussian(x, *popt) - seg) ** 2)))
        if rms <= max_rms and 20.0 <= abs(popt[1]) <= 600.0:
            kept.append((abs(popt[1]), rms, float(lat[j])))
    if not kept:
        return dict(margin=name, rows=0, sigma_km=np.nan, fwhm_km=np.nan, rms=np.nan)
    s = np.array([k[0] for k in kept])
    return dict(margin=name, rows=len(kept), sigma_km=round(float(np.median(s))),
                fwhm_km=round(2.3548 * float(np.median(s))),
                rms=round(float(np.median([k[1] for k in kept])), 4),
                spread_km='{:.0f}-{:.0f}'.format(np.percentile(s, 25), np.percentile(s, 75)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--times', type=int, nargs=2, default=[100, 99], help='older then younger')
    ap.add_argument('--product', default='subducted', choices=['subducted', 'stored', 'outflux'])
    a = ap.parse_args()

    field, lat, lon = step_field(a.product, a.times[0], a.times[1])
    print('{} water, delivery between {} and {} Ma: {} cells carry water'
          .format(a.product, a.times[0], a.times[1], int((field > 0).sum())))
    rows = [sigma_rows(field, lat, lon, w) for w in MARGINS]
    print('{:<18} {:>5} {:>10} {:>10} {:>8} {:>12}'.format(
        'margin', 'rows', 'sigma', 'FWHM', 'rms', 'quartiles'))
    for r in rows:
        if r['rows'] == 0:
            print('{:<18} {:>5} {:>10} {:>10} {:>8}'.format(r['margin'], 0, '-', '-', '-'))
            continue
        print('{:<18} {:>5d} {:>7.0f} km {:>7.0f} km {:>8.4f} {:>12}'
              .format(r['margin'], r['rows'], r['sigma_km'], r['fwhm_km'], r['rms'],
                      r.get('spread_km', '')))
    good = [r for r in rows if r['rows'] >= 2]
    med = float(np.median([r['sigma_km'] for r in good]))
    print('\nmedian over the margins with a stacked profile: sigma = {:.0f} km, FWHM = {:.0f} km'
          .format(med, 2.3548 * med))
    print('the grids are named "300 km": that is 2 sigma, not sigma')
    rows.append(dict(margin='median of stacked margins', rows=sum(r['rows'] for r in good),
                     sigma_km=round(med), fwhm_km=round(2.3548 * med), rms=''))
    out = os.path.join(P.OUT, 'grid_smoothing.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
