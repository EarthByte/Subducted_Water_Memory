"""Test the Wang et al. (2025) prediction against tomography.

Wang et al. use no seismic data: their wet-MTZ field is a GPlates reconstruction
of water delivery compared directly to eruption locations. Tomography has never
been used to check it, and doing so is not a rebuttal of their method — it is the
independent test the model has not had.

Their condition, rebuilt on this project's reconstruction: water enters the MTZ
410/v after subduction, stays for a residence time R, and the eruption happens
while it is there, so a field is predicted wet when its lead time from
`province_eruption_context.py` falls in [410/v, 410/v + R]. Their stated ranges
are v = 1-9 cm/yr and R = 30-100 Myr, and both are swept, because the conclusion
should not rest on one choice inside them.

The discriminating question is not how many fields come out wet — our proxy is
tighter than their field and yields fewer — but whether the provinces the model
calls wetter are seismically SLOWER, which is what hydrous upwelling requires.

    python3 src/wang_test.py
    python3 src/wang_test.py --band 520-660

Caveats worth carrying into any text: this uses trench proximity rather than a
slab water budget from seafloor age, a fixed cap rather than a spread hydration
field, and a different plate model, so it is a noisier proxy for their
prediction than their own grid would be; and a null cannot separate "no water"
from "water that velocity cannot see".
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

try:
    import paths as P
    OUT = getattr(P, 'OUT', 'out')
except Exception:
    OUT = 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', default='410-520')
    ap.add_argument('--null', default='continental')
    ap.add_argument('--sink', type=float, nargs='+', default=[1, 3, 5, 9],
                    help='cm/yr; Wang et al. examine 1-9')
    ap.add_argument('--residence', type=float, nargs='+', default=[30, 60, 100],
                    help='Myr the water stays in the MTZ; Wang et al. give 30-100')
    ap.add_argument('--sample', default='cenozoic',
                    choices=('cenozoic', 'same-column', 'both', 'all'),
                    help='which provinces the analysis may be applied to. The '
                         'seismic class is present-day and the eruption context '
                         'is not, so only young, little-moved provinces let the '
                         'two be about the same mantle. See province_sample.py.')
    a = ap.parse_args()
    try:
        from province_sample import sample as _sample
        _keep = _sample(OUT, a.sample)
        _keep = set(_keep[_keep].index)
    except Exception as _e:
        print(f'  (sample restriction unavailable: {_e})'); _keep = None

    ec = os.path.join(OUT, 'province_eruption_context.csv')
    if not os.path.exists(ec):
        raise SystemExit(f'\n{ec} missing; run province_eruption_context.py first\n')
    e = pd.read_csv(ec)
    if 'province' not in e:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from province_pivot import provinces, CAT
        d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
        e['province'] = provinces(d)
    t = pd.read_csv(os.path.join(OUT, f'province_scores_{a.null}.csv'))
    S = t[t.band == a.band].groupby('province').S.mean()

    print(f'\nWang et al. (2025) predicted-wet fraction against the seismic anomaly '
          f'at {a.band} km\n')
    print(f'{"sink":>7}{"residence":>11}{"lead window":>15}{"fields wet":>12}'
          f'{"rho":>9}{"p":>8}')
    rows = []
    for v_cm in a.sink:
        for R in a.residence:
            v = v_cm * 10.0
            lo, hi = 410.0 / v, 410.0 / v + R
            e['wet'] = ((e.lead_Myr >= lo) & (e.lead_Myr <= hi)).astype(float)
            w = e.groupby('province').wet.mean()
            j = pd.concat([w.rename('wet'), S.rename('S')], axis=1).dropna()
            r, p = spearmanr(j.wet, j.S)
            rows.append(dict(sink_cm_yr=v_cm, residence_Myr=R, lo=lo, hi=hi,
                             wet_fraction=float(e.wet.mean()), rho=r, p=p))
            print(f'{v_cm:6.0f}c{R:11.0f}{f"{lo:.0f}-{hi:.0f}":>15}'
                  f'{100 * e.wet.mean():11.0f}%{r:9.3f}{p:8.3f}')
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, f'wang_test_{a.band.replace("-", "_")}.csv'), index=False)
    sig = int((d.p <= 0.05).sum())
    print(f'\n  median rho {d.rho.median():+.3f}; {int((d.rho > 0).sum())} of '
          f'{len(d)} parameter choices positive; {sig} of {len(d)} reach p <= 0.05')
    print('  positive rho means the provinces the model calls wetter are FASTER.')
    if sig == 0:
        print('\n  No parameter choice inside their stated ranges gives a significant\n'
              '  relation in either direction. Shear velocity carries no information\n'
              '  about this prediction, which is a measurement of what tomography can\n'
              '  contribute here rather than an argument about whether water is there.')


if __name__ == '__main__':
    main()
