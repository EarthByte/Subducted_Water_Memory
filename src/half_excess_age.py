"""A summary of the decay that does not depend on fitting a function to it.

The fitted e-folding moves by a factor of two across defensible choices of age
binning and functional form, so it is not the statistic to lead with. The age at
which the enrichment has fallen half way from its youngest value to its floor is
read off the curve by interpolation instead, and needs no model of the decay at
all. This reports it across every axis the fit was tested on.
"""
import itertools, numpy as np
exec(open('tau_sensitivity.py').read().split("def fit(")[0].split("print('\\n=== threshold")[0])

def half_age(x, y, floor=None):
    """Age at which enrichment falls half way from its first value to the floor.

    The floor is the mean of the bands older than 150 Ma, which the paper already
    treats as a property of the reference class rather than of ancient slabs; it
    is taken from the data rather than fitted so that nothing here is a fit.
    """
    old = x > 150.0
    ei = float(np.mean(y[old])) if old.sum() >= 2 else float(y[-1])
    e0 = float(y[0])
    if not np.isfinite(e0) or e0 <= ei:
        return np.nan
    target = ei + 0.5 * (e0 - ei)
    for k in range(1, len(x)):
        if y[k] <= target:
            if y[k-1] == y[k]:
                return float(x[k])
            f = (y[k-1] - target) / (y[k-1] - y[k])
            return float(x[k-1] + f * (x[k] - x[k-1]))
    return np.nan

print(f'{"model":10} {"top %":>6} {"binning":>10} {"bands":>6} {"half-excess age":>16}')
rows = []
for tag in FIELDS:
    g = GRIDDED[tag]; f = np.isfinite(g)
    for p in (85, 90, 95):
        fast = f & (g >= np.percentile(g[f], p))
        for width, offset in ((1,0),(2,0),(2,1),(3,1),(3,2)):
            x, y = curve(fast, width, offset)
            h = half_age(x, y)
            rows.append((tag, 100-p, f'{width}/{offset}', len(x), h))
            print(f'{tag:10} {100-p:6d} {width}/{offset:>8} {len(x):6d} {h:16.1f}')
h = np.array([r[4] for r in rows], float)
h = h[np.isfinite(h)]
print(f'\nhalf-excess age over {len(h)} combinations of model, threshold and binning:')
print(f'  median {np.median(h):.1f} Myr, range {h.min():.1f} to {h.max():.1f}, '
      f'interquartile {np.percentile(h,25):.1f} to {np.percentile(h,75):.1f}')

# the non-parametric statement that needs no curve at all
z = np.load(OCC)
occ2 = z['occupancy']; A = z['bands']
first = np.full(occ2.shape[1:], np.nan)
for k in range(len(A)-1, -1, -1):
    first[occ2[k]] = 0.5*(A[k][0]+A[k][1])
for tag in FIELDS:
    g = GRIDDED[tag]; f = np.isfinite(g)
    fast = f & (g >= np.percentile(g[f], 90.0))
    v = first[fast & np.isfinite(first)]
    allv = first[np.isfinite(first)]
    print(f'{tag:10} median youngest explaining age: fast set {np.median(v):5.1f} Ma, '
          f'everywhere with an explanation {np.median(allv):5.1f} Ma')
