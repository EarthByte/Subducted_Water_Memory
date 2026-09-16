"""Do the fast provinces have a different subduction history from the rest?

Reads out/province_eruption_context.csv when it exists, which reconstructs every
field to its own eruption age before asking what had been delivered beneath it.
Falls back to the present-day occupancy map when it does not, which is wrong for
anything old and is kept only so the script still runs.

Step 5 of the recipe. Province-level, one observation each, compared with a
permutation test on the class labels: no weighting by fields or cells, and no
per-province p-values.

The subduction quantity is taken from the occupancy field the earlier analysis
already produced: for each province, the median over its fields of the age of
the youngest reconstructed subduction that can account for material now at
transition-zone depth beneath them, and the fraction of its fields that have any
such explanation within 400 Myr.
"""
import argparse, os, re
import numpy as np, pandas as pd

_ap = argparse.ArgumentParser()
_ap.add_argument('--max-age', type=float, default=None,
                 help='keep only provinces whose median eruption age is below '
                      'this, in Ma. The seismic class is present-day and the '
                      'eruption context is not, so the two are only about the '
                      'same mantle where the eruption is young: the transition '
                      'zone remembers subduction for a few tens of Myr. Without '
                      'this the old provinces are included and dilute the test.')
_a, _ = _ap.parse_known_args()
import paths as P
U = os.path.dirname(P.IPV_V3)
z = np.load(f'{U}/Paper_hydration/out/hydration_age_map.npz')
tmin, olat, olon = z['t_min'], z['lat'], z['lon']
d = pd.read_csv(f'{U}/ipv_catalogue_georoc_v3.csv').dropna(subset=['lat','lon_180']).reset_index(drop=True)
cls = pd.read_csv('out/province_classes_final.csv')

import importlib.util, sys
spec = importlib.util.spec_from_file_location('pp','province_pivot.py')
pp = importlib.util.module_from_spec(spec); sys.argv=['x']; spec.loader.exec_module(pp)
pname = pp.provinces(d)

j = np.abs(olat[None,:]-d.lat.values[:,None]).argmin(1)
i = np.abs(((olon[None,:]-d.lon_180.values[:,None]+180)%360)-180).argmin(1)
age = tmin[j,i]

rows=[]
for p in sorted(set(pname)):
    m = pname==p
    a = age[m]
    rows.append(dict(province=p, n=int(m.sum()),
                     frac_explained=float(np.isfinite(a).mean()),
                     median_age=float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan))
s = pd.DataFrame(rows).merge(cls[['province','cls','n_fields']], on='province')
s = s.copy()
s['group'] = np.where(s.cls=='fast','fast','other')
print(f'{len(s)} adequately sampled provinces: '
      f'{int((s.group=="fast").sum())} fast, {int((s.group=="other").sum())} other\n')
print(f'{"province":38}{"n":>3} {"class":>13} {"explained":>10} {"median age":>11}')
for _, r in s.sort_values(['group','median_age']).iterrows():
    ma = f'{r.median_age:.0f}' if np.isfinite(r.median_age) else '-'
    print(f'{r.province[:37]:38}{r.n:3d} {r.cls:>13} {100*r.frac_explained:9.0f}% {ma:>11}')

rng = np.random.default_rng(3)
def perm(col, higher_is):
    x = s[col].values.astype(float); g = (s.group=='fast').values
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    if g.sum()==0 or (~g).sum()==0: return np.nan, np.nan, np.nan
    obs = np.median(x[g]) - np.median(x[~g])
    null = np.array([np.median(x[p_]) - np.median(x[~p_])
                     for p_ in (rng.permutation(g) for _ in range(20000))])
    p = ((null <= obs).sum()+1)/(len(null)+1) if higher_is=='lower' else ((null >= obs).sum()+1)/(len(null)+1)
    return float(np.median(x[g])), float(np.median(x[~g])), float(p)

print()
for col, direction, label in (('frac_explained','higher','fraction with any explaining subduction'),
                              ('median_age','lower','age of the youngest explaining subduction (Ma)')):
    a,b,p = perm(col, direction)
    print(f'  {label:48} fast {a:7.2f}   other {b:7.2f}   permutation p = {p:.4f}')
