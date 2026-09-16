"""The four axes that change the occupancy field, and what they do to tau.

tau_sensitivity.py varies what can be varied after the reconstruction has run:
the threshold, the age binning, the functional form. The four choices that go
into building the occupancy itself cannot be tested that way, because each needs
the trenches reconstructed again:

    search radius        how far from a displaced trench a cell is marked
    down-dip offset      how far the parcel is carried along the slab
    sinking rate         how the epoch maps onto the depth now sampled
    frame                the mantle reference frame: published, or the optAPM bounds

This drives deep_time_hydration.py once per setting, then refits the decay from
each occupancy with everything else held at the reference values, so the only
thing changing between rows is the axis named. Nulls are switched off in the
sweep runs with --n-null 0: the sweep needs the occupancy and the curve, not the
rotation test, and the null is most of the cost.

    python3 src/occupancy_sweep.py --quick     # 8 runs, about 40 minutes
    python3 src/occupancy_sweep.py             # 17 runs, about 90 minutes
    python3 src/occupancy_sweep.py --fit-only  # refit what is already on disk

A run whose npz is already in out/ is not repeated, so this can be interrupted
and restarted, and a single axis can be added later without redoing the rest.
"""
import argparse, os, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

try:
    import paths as P
    OUT, TOMO = P.OUT, getattr(P, 'TOMO', '..')
except Exception:
    OUT, TOMO = 'out', '..'

REF = dict(radius=250.0, offset=300.0, v_sink=50.0, model='zahirovic2022', frame='published')

# The reference-frame axis is the one that has to be chosen rather than swept.
# The attribution traces material sinking through the mantle, so a model is only
# admissible here if its absolute rotations are in a MANTLE reference frame, and
# only informative if it is independent of the reference model. Most of what
# plate_model_manager offers fails one test or the other:
#
#   muller2019    incorporated into zahirovic2022; not an independent check
#   merdith2021   palaeomagnetic reference frame: longitude is not constrained
#                 relative to the mantle, so a trench cannot be placed above the
#                 mantle column it fed
#   cao2024       the same palaeomagnetic frame, built on merdith2021 rotations;
#                 its occupancy differs from merdith2021 in 0.06 per cent of
#                 cells, so it is not a second test either
#   muller2025    an extension of muller2022, and its Palaeozoic reconstruction
#                 of the Americas against Gondwana and Laurasia is known to be
#                 wrong, which is inside the 400 Myr window used here
#
#   muller2022    a mantle reference frame built independently of
#                 zahirovic2022, but its Palaeozoic (Domeier & Torsvik, 2014)
#                 differs from the Young et al. (2018) Palaeozoic of
#                 zahirovic2022 inside the 400 Myr window, and this repository
#                 does not use it (CLAUDE.md). It was the alternative row of
#                 Table S2 until 16 September 2026.
#
# The reference frame is therefore varied within the same model: the optAPM
# no-net-rotation and maximum-net-rotation bounds of zahirovic2022 (Tetley et
# al., 2019), which are the frames the ridge-record kinematics use, so the
# whole paper is bounded by one pair of frames.
FRAMES = ['NNR', 'maxNR']
MODELS = []

FULL = {
    'radius':  [150.0, 200.0, 350.0, 500.0],
    'offset':  [150.0, 200.0, 400.0, 500.0],
    'v_sink':  [30.0, 40.0, 70.0, 90.0],
    'frame':   FRAMES,
}
QUICK = {'radius': [150.0, 500.0], 'offset': [150.0, 500.0],
         'v_sink': [30.0, 90.0], 'frame': FRAMES}


def tag_for(axis, value):
    v = value if isinstance(value, str) else f'{value:g}'
    return f'_{axis}{v}'.replace('.', 'p')


def run_one(axis, value, extra_null=0):
    """One reconstruction, skipped if its occupancy is already written."""
    tag = tag_for(axis, value)
    npz = os.path.join(OUT, f'hydration_age_map{tag}.npz')
    if os.path.exists(npz):
        return tag, npz, 0.0
    cfg = dict(REF); cfg[axis] = value
    cmd = [sys.executable, os.path.join('src', 'deep_time_hydration.py'),
           '--model', str(cfg['model']),
           '--radius', str(cfg['radius']),
           '--offset', str(cfg['offset']),
           '--v-sink', str(cfg['v_sink']),
           '--frame', str(cfg['frame']),
           '--n-null', str(extra_null),
           '--suffix', tag]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(npz):
        print(f'  {axis}={value}: FAILED\n    ' + '\n    '.join(
            (r.stderr or r.stdout).strip().splitlines()[-4:]))
        return tag, None, time.time() - t0
    return tag, npz, time.time() - t0


# ---- the fit, identical to tau_sensitivity.py so the rows are comparable ----
def load_fields():
    import xarray as xr
    out = {}
    ds = xr.open_dataset(os.path.join(TOMO, 'REVEAL_vs_full.nc'))
    v = np.sqrt((2.0 * ds['vsv'] ** 2 + ds['vsh'] ** 2) / 3.0)
    mu = v.mean(dim=['latitude', 'longitude'])
    a = ((v - mu) * 100.0 / mu).clip(min=-100, max=100)
    dep = np.asarray(ds['depth'].values, float)
    arr = a.transpose('depth', 'latitude', 'longitude').values
    if dep[0] > dep[-1]:
        o = np.argsort(dep); dep, arr = dep[o], arr[o]
    out['REVEAL'] = (np.asarray(ds['latitude'].values, float),
                     np.asarray(ds['longitude'].values, float),
                     np.nanmean(arr[(dep >= 410) & (dep < 660)], axis=0))
    for tag in ('RevealLO', 'GLADM35'):
        f = os.path.join(TOMO, 'REVEAL_mantle_tomography', f'bands_{tag}.npz')
        if not os.path.exists(f):
            continue
        b = np.load(f, allow_pickle=True)
        out[tag] = (b['lat'], b['lon'],
                    (110.0 * b['410_520'].astype(float)
                     + 140.0 * b['520_660'].astype(float)) / 250.0)
    return out


def tau_from(npz, fields):
    z = np.load(npz)
    occ, ABINS, olat, olon = z['occupancy'], z['bands'], z['lat'], z['lon']
    LO, LA = np.meshgrid(olon, olat)
    w = np.cos(np.radians(LA))
    masks = []
    for k in range(len(ABINS)):
        y = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
        m = occ[k] & ~y
        masks.append(m if m.sum() >= 300 else None)
    amid = 0.5 * (ABINS[:, 0] + ABINS[:, 1])

    def decay(t, e0, ei, tau):
        return ei + (e0 - ei) * np.exp(-t / tau)

    res = {}
    for tag, (blat, blon, band) in fields.items():
        j = np.abs(blat[None, :] - olat[:, None]).argmin(1)
        i = np.abs(((blon[None, :] - olon[:, None] + 180) % 360) - 180).argmin(1)
        g = band[np.ix_(j, i)]
        f = np.isfinite(g)
        fast = f & (g >= np.percentile(g[f], 90.0))
        base = w[fast].sum() / w.sum()
        x, y = [], []
        for k, m in enumerate(masks):
            if m is None:
                continue
            x.append(amid[k]); y.append((w[m & fast].sum() / w[m].sum()) / base)
        x, y = np.array(x, float), np.array(y, float)
        try:
            p, c = curve_fit(decay, x, y, p0=[6.0, 0.5, 40.0],
                             bounds=([0, 0, 3], [60, 3, 400]), maxfev=60000)
            t, dt = float(p[2]), float(np.sqrt(np.diag(c))[2])
            res[tag] = (t, dt) if np.isfinite(dt) and dt < 0.5 * t else (t, np.nan)
        except Exception:
            res[tag] = (np.nan, np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--fit-only', action='store_true',
                    help='refit occupancies already in out/, run nothing')
    ap.add_argument('--axes', nargs='*', default=list(FULL))
    a = ap.parse_args()
    grid = QUICK if a.quick else FULL

    jobs = [('reference', None)] + [(ax, v) for ax in a.axes for v in grid[ax]]
    fields = load_fields()
    print(f'fitting on {", ".join(fields)}\n')
    rows, t0 = [], time.time()
    for axis, value in jobs:
        if axis == 'reference':
            npz = os.path.join(OUT, 'hydration_age_map.npz')
            if not os.path.exists(npz):
                raise SystemExit(f'{npz} missing; run deep_time_hydration.py first')
            tag, dt = '', 0.0
        elif a.fit_only:
            tag = tag_for(axis, value)
            npz = os.path.join(OUT, f'hydration_age_map{tag}.npz')
            dt = 0.0
            if not os.path.exists(npz):
                continue
        else:
            tag, npz, dt = run_one(axis, value)
            if npz is None:
                continue
        r = tau_from(npz, fields)
        rows.append(dict(axis=axis, value=REF.get(axis, '') if value is None else value,
                         seconds=round(dt),
                         **{k: round(v[0], 1) for k, v in r.items()}))
        print(f'  {axis:9} {str(rows[-1]["value"]):14} '
              + '  '.join(f'{k} {v[0]:5.1f}' for k, v in r.items())
              + (f'   [{dt / 60:.0f} min]' if dt else ''))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'occupancy_sweep.csv'), index=False)
    cols = [c for c in ('REVEAL', 'RevealLO', 'GLADM35') if c in d.columns]
    allv = d[cols].values.astype(float)
    allv = allv[np.isfinite(allv)]
    ref = d[d.axis == 'reference'][cols].values.astype(float).ravel()
    print(f'\n{len(d)} settings, {len(allv)} fits')
    print(f'  reference: {", ".join(f"{c} {v:.1f}" for c, v in zip(cols, ref))} Myr')
    print(f'  across every setting and volume: {allv.min():.1f} to {allv.max():.1f} Myr, '
          f'median {np.median(allv):.1f}')
    for ax in d.axis.unique():
        if ax == 'reference':
            continue
        v = d[d.axis == ax][cols].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v):
            print(f'  varying {ax:8}: {v.min():5.1f} to {v.max():5.1f} Myr')
    print(f'\nwrote {OUT}/occupancy_sweep.csv   [{(time.time() - t0) / 60:.0f} min total]')


if __name__ == '__main__':
    main()
