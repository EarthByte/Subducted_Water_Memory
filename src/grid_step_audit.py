# -*- coding: utf-8 -*-
"""Does any single time step inject an implausible amount of water into the grids?

The cumulative water grids are built by adding one step's delivery to the previous grid, so a
fault in one step survives in every grid after it and, once smoothed, looks like a round patch
of delivered water. An audit of the Muller2025 carbon grids by the rift workflow found such
steps, which is reason enough to run the same test on the Z22 water grids this paper uses.

For every consecutive pair the script forms the step's delivery, takes the largest increment in
any one cell, and compares it with the median of that quantity over the preceding window. A
step whose maximum is more than `--factor` times that median is reported with the location of
the cell, so it can be looked at on the map.

    python3 src/grid_step_audit.py                        # subducted water, every step
    python3 src/grid_step_audit.py --product stored --factor 10 --window 50
    python3 src/grid_step_audit.py --times 120 80         # a range, oldest first

Writes out/grid_step_audit_<product>.csv: one row per step, with the flagged ones marked.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import paths as P


def cumulative(product, t):
    z, lat, lon = None, None, None
    for f in P.water(product, 'Z22', t):
        d = xr.open_dataset(P.need(f, 'GRIDS_DIR', 'the Z22 {} grid at {} Ma'.format(product, t)))
        v = np.nan_to_num(d['z'].values)
        z = v if z is None else z + v
        lat, lon = d.lat.values, d.lon.values
    return z, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--product', default='subducted', choices=['subducted', 'stored', 'outflux'])
    ap.add_argument('--times', type=int, nargs=2, default=None,
                    help='oldest and youngest reconstruction time to audit (default: all found)')
    ap.add_argument('--factor', type=float, default=10.0,
                    help='flag a step whose largest cell increment exceeds this many times the '
                         'median of the preceding window')
    ap.add_argument('--window', type=int, default=50, help='length of that window, in steps')
    a = ap.parse_args()

    if a.times:
        times = list(range(max(a.times), min(a.times) - 1, -1))
    else:
        times = []
        t = 0
        while os.path.exists(P.water(a.product, 'Z22', t)[0]):
            times.append(t)
            t += 1
        times = times[::-1]                      # oldest first
    assert len(times) > 2, 'need at least three grids to audit'
    print('{} water, {} grids from {} to {} Ma, flagging steps above {:.0f} x the running median'
          .format(a.product, len(times), times[0], times[-1], a.factor))

    rows, maxima = [], []
    prev, lat, lon = cumulative(a.product, times[0])
    for t_old, t_new in zip(times[:-1], times[1:]):
        cur, _, _ = cumulative(a.product, t_new)
        step = cur - prev
        prev = cur
        k = int(np.argmax(step))
        j, i = np.unravel_index(k, step.shape)
        peak = float(step[j, i])
        ref = float(np.median(maxima[-a.window:])) if len(maxima) >= 5 else np.nan
        ratio = peak / ref if ref and np.isfinite(ref) and ref > 0 else np.nan
        flag = bool(np.isfinite(ratio) and ratio > a.factor)
        rows.append(dict(t_old=t_old, t_new=t_new, max_increment=peak,
                         running_median=ref, ratio=ratio, flagged=flag,
                         lat=float(lat[j]), lon=float(lon[i]),
                         negative_cells=int((step < -1e-12).sum())))
        maxima.append(peak)
        if flag:
            print('  {:>4} -> {:<4} Ma  max increment {:.4g} is {:.0f} x the running median, '
                  'at {:.1f} N {:.1f} E'.format(t_old, t_new, peak, ratio, lat[j], lon[i]))

    df = pd.DataFrame(rows)
    out = os.path.join(P.OUT, 'grid_step_audit_{}.csv'.format(a.product))
    df.to_csv(out, index=False, float_format='%.6g')
    n_neg = int((df.negative_cells > 0).sum())
    print('\n{} steps audited, {} flagged, {} steps with a negative increment anywhere'
          .format(len(df), int(df.flagged.sum()), n_neg))
    print('wrote', out)


if __name__ == '__main__':
    main()
