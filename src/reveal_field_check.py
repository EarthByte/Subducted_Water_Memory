"""Which REVEAL field is in which cached product.

Two cached REVEAL products appeared to disagree. `out/bands_REVEAL.npz`, which
every depth statistic reads, and `out/tz_REVEAL.npz`, the field Figure 2 draws,
are both REVEAL over the transition zone, yet combining the two cached bands into
a 410-660 km mean by their thicknesses, 110 and 140 km, gives a field that
correlates with the cached 410-660 mean at only 0.954. Neither cache records the
file it came from, so the possibility that one of them held something other than
the field its name says had to be excluded from the model volumes themselves.

This settles it. For every REVEAL volume it can find the script forms the band
means with the project's own convention - the Voigt average of vsv and vsh, the
lateral mean removed shell by shell, in per cent - samples them onto the
half-degree grid the cached products use, and compares. It then combines the two
cached bands in both of the ways available, by depth thickness and by the number
of model shells each band contains, and compares those with the cached 410-660
mean as well.

Run on 15 October 2026 with the SSD mounted, the answer is that nothing is
mislabelled: `REVEAL_vs_full.nc` reproduces all three cached fields exactly,
r = 1.00000 with an RMS difference of 0.0000 in each case. The 0.954 is an
artefact of the thickness weighting. REVEAL is not sampled evenly in depth, and
carries ten shells between 410 and 520 km against three between 520 and 660 km,
so the mean over the whole transition zone weights the two bands 10 to 3 and not
110 to 140. Weighted by shell count the two cached bands reproduce the cached
410-660 mean exactly.

Needs the velocity volumes, so the SSD has to be mounted:

    PYTHONPATH=src python3 src/reveal_field_check.py
    PYTHONPATH=src python3 src/reveal_field_check.py --file /path/to/REVEAL.nc

It writes out/reveal_field_check.csv and changes nothing else.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import paths as P

BANDS = [(410.0, 520.0), (520.0, 660.0), (410.0, 660.0)]
CACHED_BANDS = os.path.join(P.OUT, 'bands_REVEAL.npz')
CACHED_TZ = os.path.join(P.OUT, 'tz_REVEAL.npz')


def candidates(extra):
    """Every REVEAL volume worth testing, in the order the project would find them."""
    seen, out = set(), []
    here = os.path.join(P.TOMO, 'REVEAL_mantle_tomography')
    for p in ([extra] if extra else []) + [
            P.REVEAL,
            os.path.join(here, 'REVEAL.nc'),
            os.path.join(here, 'REVEAL_0_1000km.nc'),
            os.path.join(P.TOMO, 'REVEAL_vs_full.nc')]:
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


def anomaly(path):
    """Lateral-mean-removed per cent anomaly, depth first, as the project takes it.

    The conventions are those of `plume_classifier.load_anomaly` and
    `extract_bands.py`: the Voigt average where both polarisations are present,
    the mean of each depth shell removed, and a depth axis that may arrive in
    kilometres, in metres or as a radius.
    """
    d = xr.open_dataset(path)
    names = {k.lower(): k for k in d.variables}
    dkey = names.get('depth') or names.get('radius') or names.get('z')
    assert dkey, 'no depth axis in {}'.format(os.path.basename(path))
    dep = np.asarray(d[dkey].values, float)
    if dep.max() > 1.0e5:
        dep = dep / 1.0e3
    if dep.min() > 3.0e3 and dep.max() <= 6.5e3:
        dep = 6371.0 - dep
    flip = dep[0] > dep[-1]
    if flip:
        dep = dep[::-1]
    lat = np.asarray(d[names.get('latitude', names.get('lat'))].values, float)
    lon = np.asarray(d[names.get('longitude', names.get('lon'))].values, float)

    vsv, vsh = names.get('vsv'), names.get('vsh')
    if vsv and vsh:
        v = np.sqrt((2 * d[vsv].values ** 2 + d[vsh].values ** 2) / 3.0)
        kind = 'Voigt'
    else:
        key = names.get('vs') or names.get('v_s')
        assert key, 'no shear velocity in {}'.format(os.path.basename(path))
        v = d[key].values
        kind = key
    v = np.asarray(v, float)
    if v.shape[0] != len(dep):
        v = np.moveaxis(v, int(np.argmax([s == len(dep) for s in v.shape])), 0)
    if flip:
        v = v[::-1]
    flat = v.reshape(len(dep), -1)
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.nanmean(flat, axis=1)
        a = ((v - ref[:, None, None]) * 100.0 / ref[:, None, None])
    return dep, lat, lon, np.clip(a, -100, 100), kind


def band_mean(dep, a, z0, z1):
    m = (dep >= z0) & (dep < z1)
    assert m.any(), 'no shell between {:.0f} and {:.0f} km'.format(z0, z1)
    with np.errstate(invalid='ignore'):
        return np.nanmean(a[m], axis=0), int(m.sum())


def to_grid(lat, lon, f, glat, glon):
    j = np.abs(lat[None, :] - glat[:, None]).argmin(1)
    i = np.abs(((lon[None, :] - glon[:, None] + 180) % 360) - 180).argmin(1)
    return f[np.ix_(j, i)]


def compare(x, y, glat, glon):
    w = np.cos(np.radians(glat))[:, None] * np.ones((1, glon.size))
    ok = np.isfinite(x) & np.isfinite(y)
    ww = w * ok
    m = lambda v: (v * ww)[ok].sum() / ww[ok].sum()
    cx, cy = x - m(x), y - m(y)
    r = (cx * cy * ww)[ok].sum() / np.sqrt((cx ** 2 * ww)[ok].sum() * (cy ** 2 * ww)[ok].sum())
    rms = np.sqrt(((x - y) ** 2 * ww)[ok].sum() / ww[ok].sum())
    return float(r), float(rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', help='an extra REVEAL volume to test')
    a = ap.parse_args()

    assert os.path.exists(CACHED_BANDS), 'no {}'.format(CACHED_BANDS)
    z = np.load(CACHED_BANDS, allow_pickle=True)
    glat, glon = z['lat'], z['lon']
    cached = {'bands 410-520': z['410_520'].astype(float),
              'bands 520-660': z['520_660'].astype(float)}
    if os.path.exists(CACHED_TZ):
        t = np.load(CACHED_TZ)
        assert np.allclose(t['lat'], glat) and np.allclose(t['lon'], glon), \
            'the two caches are on different grids'
        cached['tz_REVEAL 410-660'] = t['dvs'].astype(float)

    files = candidates(a.file)
    assert files, 'no REVEAL volume found; is the SSD mounted?'
    rows, shells_in = [], {}
    for path in files:
        dep, lat, lon, arr, kind = anomaly(path)
        print('{}: {} shells {:.0f}-{:.0f} km, {}x{}, {}'.format(
            os.path.basename(path), len(dep), dep.min(), dep.max(), len(lat), len(lon), kind))
        for z0, z1 in BANDS:
            f, n = band_mean(dep, arr, z0, z1)
            shells_in[(z0, z1)] = n
            g = to_grid(lat, lon, f, glat, glon)
            for name, c in cached.items():
                r, rms = compare(g, c, glat, glon)
                rows.append(dict(file=os.path.basename(path), velocity=kind,
                                 band='{:.0f}-{:.0f}'.format(z0, z1), shells=n,
                                 cached=name, r=r, rms_difference=rms))
                if r > 0.999:
                    print('   {:9s} reproduces "{}"  r = {:.5f}, RMS difference {:.4f}'
                          .format('{:.0f}-{:.0f}'.format(z0, z1), name, r, rms))

    # The two cached bands combined, each way, against the cached 410-660 mean: the
    # thickness weighting is the one that looks wrong, and the shell counts say why.
    if 'tz_REVEAL 410-660' in cached and shells_in:
        n1, n2 = shells_in[(410.0, 520.0)], shells_in[(520.0, 660.0)]
        print('REVEAL shells per band: {} between 410 and 520 km, {} between 520 and 660 km'
              .format(n1, n2))
        for label, w1, w2 in (('by depth thickness, 110:140', 110.0, 140.0),
                              ('by shell count, {}:{}'.format(n1, n2), float(n1), float(n2))):
            g = (w1 * cached['bands 410-520'] + w2 * cached['bands 520-660']) / (w1 + w2)
            r, rms = compare(g, cached['tz_REVEAL 410-660'], glat, glon)
            rows.append(dict(file='(the two cached bands combined)', velocity=label,
                             band='410-660', shells=n1 + n2, cached='tz_REVEAL 410-660',
                             r=r, rms_difference=rms))
            print('   combined {:28s} r = {:.5f}, RMS difference {:.4f}'.format(label, r, rms))

    df = pd.DataFrame(rows).sort_values('r', ascending=False)
    out = os.path.join(P.OUT, 'reveal_field_check.csv')
    df.to_csv(out, index=False, float_format='%.6f')
    print('\nbest match for each cached product')
    for name in cached:
        best = df[df.cached == name].iloc[0]
        verdict = 'reproduced' if best.r > 0.999 else 'NOT reproduced by any file tested'
        print('  {:22s} {:10s} {} {} ({} shells)  r = {:.5f}'.format(
            name, verdict, best.file, best.band, best.shells, best.r))
    print('wrote', out)


if __name__ == '__main__':
    main()
