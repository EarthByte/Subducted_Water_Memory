"""How much of the 37 Myr is a choice?

The persistence constant is fitted to an enrichment curve built from four
decisions: where the fast transition zone is thresholded, how the subduction
history is cut into age bands, where those bands start, and what function is
fitted to the result. None of them is forced by the data. This varies each one
and reports what tau does.

Three axes are here because they need only the occupancy field and a band map:
threshold, binning, and functional form. The four that change the occupancy
itself -- search radius, down-dip displacement, sinking rate and reference frame
-- have to be run where the plate model is, and reuse this same fit.
"""
import itertools
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

import argparse, os
try:
    import paths as _P
    _DEF = getattr(_P, 'TOMO', '..')
except Exception:
    _P, _DEF = None, '..'
_ap = argparse.ArgumentParser()
_ap.add_argument('--data', default=_DEF,
                 help='folder holding the models, the band files and the catalogue')
_a, _ = _ap.parse_known_args()
U = _a.data
OUT = getattr(_P, 'OUT', 'out') if _P is not None else 'out'
OCC = os.path.join(OUT, 'hydration_age_map.npz')
R = 6371.0088

z = np.load(OCC)
occ, ABINS, olat, olon = z['occupancy'], z['bands'], z['lat'], z['lon']
LO, LA = np.meshgrid(olon, olat)
w = np.cos(np.radians(LA))
print(f'occupancy {occ.shape} on {len(olat)}x{len(olon)}, bands {ABINS[0]} .. {ABINS[-1]}')

# The three models on the occupancy grid, each thresholded the same way.
import xarray as xr
def reveal_band():
    ds = xr.open_dataset(f'{U}/REVEAL_vs_full.nc')
    v = np.sqrt((2.0*ds['vsv']**2 + ds['vsh']**2)/3.0)
    m = v.mean(dim=['latitude','longitude'])
    a = ((v-m)*100.0/m).clip(min=-100, max=100)
    dep = np.asarray(ds['depth'].values, float)
    arr = a.transpose('depth','latitude','longitude').values
    if dep[0] > dep[-1]:
        o = np.argsort(dep); dep, arr = dep[o], arr[o]
    return (np.asarray(ds['latitude'].values,float), np.asarray(ds['longitude'].values,float),
            np.nanmean(arr[(dep>=410)&(dep<660)], axis=0))

def npz_band(tag):
    b = np.load(f'{U}/REVEAL_mantle_tomography/bands_{tag}.npz', allow_pickle=True)
    a, c = b['410_520'].astype(float), b['520_660'].astype(float)
    return b['lat'], b['lon'], (110.0*a + 140.0*c)/250.0

FIELDS = {'REVEAL': reveal_band(), 'RevealLO': npz_band('RevealLO'),
          'GLADM35': npz_band('GLADM35')}

def on_occ_grid(blat, blon, band):
    j = np.abs(blat[None,:]-olat[:,None]).argmin(1)
    i = np.abs(((blon[None,:]-olon[:,None]+180)%360)-180).argmin(1)
    return band[np.ix_(j, i)]

GRIDDED = {k: on_occ_grid(*v) for k, v in FIELDS.items()}


def curve(fast, width, offset):
    """Enrichment against age, conditioned on no younger subduction.

    The occupancy is stored in 2 Myr epochs collapsed into fixed bands, so a
    different binning is built by merging those bands rather than by re-running
    the reconstruction. width is in units of the stored band; offset shifts
    where the merged bands start.
    """
    n = len(ABINS)
    edges = list(range(offset % width, n + 1, width))
    if edges[0] > 0:
        edges = [0] + edges
    if edges[-1] < n:
        edges.append(n)
    base = w[fast].sum() / w.sum()
    ages, es = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        if hi_ <= lo_:
            continue
        younger = occ[:lo_].any(axis=0) if lo_ else np.zeros_like(occ[0], bool)
        m = occ[lo_:hi_].any(axis=0) & ~younger
        if m.sum() < 300:
            continue
        ages.append(0.5*(ABINS[lo_][0] + ABINS[hi_-1][1]))
        es.append((w[m & fast].sum()/w[m].sum())/base)
    return np.array(ages, float), np.array(es, float)


def fit(x, y, form):
    if form == 'exp+floor':
        f = lambda t, e0, ei, tau: ei + (e0-ei)*np.exp(-t/tau)
        p0, bd = [6.0, 0.5, 40.0], ([0,0,3],[60,3,400])
        k = 2
    elif form == 'exp':
        f = lambda t, e0, tau: e0*np.exp(-t/tau)
        p0, bd = [6.0, 40.0], ([0,3],[60,400])
        k = 1
    elif form == 'power':          # tau read as the half-value age, for comparison
        f = lambda t, e0, ei, tau: ei + (e0-ei)/(1.0 + t/tau)
        p0, bd = [6.0, 0.5, 40.0], ([0,0,3],[60,3,400])
        k = 2
    elif form == 'stretched':
        f = lambda t, e0, ei, tau, b: ei + (e0-ei)*np.exp(-(t/tau)**b)
        p0, bd = [6.0, 0.5, 40.0, 1.0], ([0,0,3,0.3],[60,3,400,3.0])
        k = 2
    try:
        p, c = curve_fit(f, x, y, p0=p0, bounds=bd, maxfev=60000)
        return float(p[k]), float(np.sqrt(np.diag(c))[k])
    except Exception:
        return np.nan, np.nan


# Everything printed below is also collected, so the supplement can be generated
# from a file rather than from a terminal transcript. build_si3.py reads it.
ROWS = []
def keep(axis, value, model, tau, err, nb):
    ROWS.append(dict(axis=axis, value=value, model=model,
                     tau=tau, tau_err=err, n_bands=nb))


print('\n=== threshold on the fast transition zone (REVEAL, native binning, exp+floor)')
for p in (80, 85, 88, 90, 92, 95):
    g = GRIDDED['REVEAL']; f = np.isfinite(g)
    fast = f & (g >= np.percentile(g[f], p))
    x, y = curve(fast, 1, 0)
    t, dt = fit(x, y, 'exp+floor')
    keep('threshold', f'top {100-p} per cent', 'REVEAL', t, dt, len(x))
    print(f'  top {100-p:2d} per cent   base {100*w[fast].sum()/w.sum():4.1f} %   '
          f'tau = {t:5.1f} +/- {dt:4.1f} Myr   ({len(x)} bands)')

print('\n=== age binning: width in stored bands, and where the bands start (REVEAL, 90th)')
g = GRIDDED['REVEAL']; f = np.isfinite(g)
fast = f & (g >= np.percentile(g[f], 90.0))
for width, offset in itertools.product((1, 2, 3), (0, 1, 2)):
    if offset >= width:
        continue
    x, y = curve(fast, width, offset)
    t, dt = fit(x, y, 'exp+floor')
    keep('binning', f'width {width}, start {offset}', 'REVEAL', t, dt, len(x))
    print(f'  width {width}, offset {offset}   {len(x):2d} bands   tau = {t:5.1f} +/- {dt:4.1f} Myr')

print('\n=== functional form (REVEAL, 90th, native binning)')
x, y = curve(fast, 1, 0)
for form in ('exp+floor', 'exp', 'power', 'stretched'):
    t, dt = fit(x, y, form)
    keep('decay function', form, 'REVEAL', t, dt, len(x))
    print(f'  {form:11} tau = {t:5.1f} +/- {dt:4.1f} Myr')

print('\n=== the same three axes, all three models, exp+floor at the 90th')
for tag in FIELDS:
    g = GRIDDED[tag]; f = np.isfinite(g)
    fast = f & (g >= np.percentile(g[f], 90.0))
    ts = []
    for width, offset in ((1,0),(2,0),(2,1),(3,0),(3,1),(3,2)):
        x, y = curve(fast, width, offset)
        t, _ = fit(x, y, 'exp+floor')
        if np.isfinite(t):
            ts.append(t)
    x, y = curve(fast, 1, 0)
    t0, d0 = fit(x, y, 'exp+floor')
    keep('native binning', 'width 1, start 0', tag, t0, d0, len(x))
    keep('binning range', f'{min(ts):.1f} to {max(ts):.1f} Myr', tag,
         np.nan, np.nan, len(ts))
    print(f'  {tag:9} native {t0:5.1f} +/- {d0:4.1f} Myr   '
          f'across binnings {min(ts):5.1f} to {max(ts):5.1f} Myr')

_f = os.path.join(OUT, 'tau_sensitivity.csv')
os.makedirs(os.path.dirname(_f) or '.', exist_ok=True)
pd.DataFrame(ROWS).to_csv(_f, index=False)
print(f'\nwrote {_f}')
