"""Why the velocity models separate in probability at 410-520 km.

REVEAL and RevealLO share their short-wavelength structure - RevealLO is the same
inversion with long-period updates added - yet their adjusted probabilities in
Figure 3a differ by two orders of magnitude. This script measures why, and writes
the numbers the supplement quotes, so that none of them is asserted from
inspection.

For every pair of models it reports, at the depth band the test uses:

  correlation        of the two anomaly fields, area weighted
  long / short       the same correlation after splitting each field at a 10 deg
                     boxcar, which separates the long-period structure that was
                     added from the fine structure that was not
  decile overlap     the share of one model's fastest-10 per cent cells that the
                     other also calls fast, and the Jaccard index of the two sets
  largest region     the share of a model's fast cells lying in its largest
                     connected region, four-connected with the date line joined:
                     the coherence that sets how widely its rotated maps scatter

    python3 src/model_separation.py            # writes out/model_separation.csv
    python3 src/model_separation.py --band 410_660

Reads the cached band files only, so it runs without the velocity volumes.
"""
import argparse
import os

import numpy as np
import pandas as pd

import paths as P

BANDS = {'REVEAL': os.path.join(P.OUT, 'bands_REVEAL.npz')}
for tag in ('RevealLO', 'GLADM35', 'SPiRaL', 'SEMUCBWM1'):
    BANDS[tag] = os.path.join(P.TOMO, 'REVEAL_mantle_tomography', f'bands_{tag}.npz')
PCTL = 90.0
SMOOTH_DEG = 10.0


def field(path, band):
    z = np.load(path, allow_pickle=True)
    if band == '410_660':          # thickness weighted, as the maps of Figure 2
        a, b = z['410_520'].astype(float), z['520_660'].astype(float)
        return z['lat'], z['lon'], (110.0 * a + 140.0 * b) / 250.0
    return z['lat'], z['lon'], z[band].astype(float)


def boxcar(x, ok, n_lat, n_lon):
    """Mean over an n_lat x n_lon window, the date line joined, gaps ignored."""
    def run(arr, n, wrap):
        k = np.ones(n) / n
        if wrap:
            pad = np.concatenate([arr[:, -(n // 2):], arr, arr[:, :n // 2]], 1)
            return np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 1, pad)
        pad = np.concatenate([arr[:n // 2][::-1], arr, arr[-(n // 2):][::-1]], 0)
        return np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 0, pad)
    z = run(run(np.where(ok, x, 0.0), n_lon, True), n_lat, False)
    c = run(run(ok.astype(float), n_lon, True), n_lat, False)
    return np.where(c > 0, z / np.maximum(c, 1e-9), np.nan)


def wcorr(x, y, w, ok):
    ww = w * ok
    m = lambda v: (v * ww)[ok].sum() / ww[ok].sum()
    cx, cy = x - m(x), y - m(y)
    return float((cx * cy * ww)[ok].sum()
                 / np.sqrt((cx ** 2 * ww)[ok].sum() * (cy ** 2 * ww)[ok].sum()))


def wrms(x, w, ok):
    ww = w * ok
    m = (x * ww)[ok].sum() / ww[ok].sum()
    return float(np.sqrt(((x - m) ** 2 * ww)[ok].sum() / ww[ok].sum()))


def largest_region(mask):
    """Share of the fast cells in the largest four-connected region."""
    idx = np.argwhere(mask)
    if not len(idx):
        return float('nan')
    pos = {(int(i), int(j)): n for n, (i, j) in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nlon = mask.shape[1]
    for (i, j), n in pos.items():
        for di, dj in ((0, 1), (1, 0)):
            q = (i + di, (j + dj) % nlon)
            if q in pos:
                a, b = find(n), find(pos[q])
                if a != b:
                    parent[a] = b
    sizes = {}
    for n in range(len(idx)):
        r = find(n)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / len(idx)


def main():
    """Split two models into long and short wavelengths and compare them in each.

    REVEAL and RevealLO are the same inversion with long-period updates added, so the
    split shows what the updates carry and why a rank threshold on the sum moves so many
    cells across it. Writes the band-by-band correlations and amplitudes.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', default='410_520')
    a = ap.parse_args()

    have = {t: p for t, p in BANDS.items() if os.path.exists(p)}
    assert 'REVEAL' in have and 'RevealLO' in have, 'need at least REVEAL and RevealLO bands'

    lat = lon = None
    f, ok, mask, w = {}, {}, {}, None
    for tag, p in have.items():
        la, lo, x = field(p, a.band)
        if lat is None:
            lat, lon = la, lo
            w = np.cos(np.radians(lat))[:, None] * np.ones((1, lon.size))
        assert np.allclose(la, lat) and np.allclose(lo, lon), f'{tag} is on another grid'
        f[tag], ok[tag] = x, np.isfinite(x)
        mask[tag] = ok[tag] & (x >= np.percentile(x[ok[tag]], PCTL))

    n_lat = int(round(SMOOTH_DEG / abs(lat[1] - lat[0]))) | 1
    n_lon = int(round(SMOOTH_DEG / abs(lon[1] - lon[0]))) | 1

    rows = []
    for tag in have:
        o = ok[tag]
        lo_field = boxcar(f[tag], o, n_lat, n_lon)
        rows.append(dict(model=tag, band=a.band, other='',
                         rms=wrms(f[tag], w, o),
                         rms_long=wrms(lo_field, w, o & np.isfinite(lo_field)),
                         rms_short=wrms(f[tag] - lo_field, w, o & np.isfinite(lo_field)),
                         largest_region=largest_region(mask[tag]),
                         r=np.nan, r_long=np.nan, r_short=np.nan,
                         decile_shared=np.nan, jaccard=np.nan))

    pairs = [('REVEAL', 'RevealLO')]
    for x, y in pairs:
        o = ok[x] & ok[y]
        lx, ly = boxcar(f[x], ok[x], n_lat, n_lon), boxcar(f[y], ok[y], n_lat, n_lon)
        oo = o & np.isfinite(lx) & np.isfinite(ly)
        inter = (mask[x] & mask[y]).sum()
        rows.append(dict(model=x, band=a.band, other=y,
                         rms=np.nan, rms_long=np.nan, rms_short=np.nan,
                         largest_region=np.nan,
                         r=wcorr(f[x], f[y], w, o),
                         r_long=wcorr(lx, ly, w, oo),
                         r_short=wcorr(f[x] - lx, f[y] - ly, w, oo),
                         decile_shared=inter / mask[x].sum(),
                         jaccard=inter / (mask[x] | mask[y]).sum()))

    df = pd.DataFrame(rows)
    out = os.path.join(P.OUT, f'model_separation{"" if a.band == "410_520" else "_" + a.band}.csv')
    df.to_csv(out, index=False, float_format='%.6f')
    pd.set_option('display.width', 200)
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('wrote', out)


if __name__ == '__main__':
    main()
