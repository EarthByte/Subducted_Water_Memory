"""Which provinces the analysis may be applied to, and why.

The seismic class is present-day; the eruption context is not. The two are only
about the same mantle where the province is young enough, and near enough to
where it was, for the present transition zone to be the one that was beneath it
when it erupted. This project's persistence work puts the transition zone's
memory of subduction at a few tens of Myr, so the reconstruction's 400 Myr reach
is not the limit — the memory is.

Two criteria, both computed from what is already on disk:

    Cenozoic      median eruption age <= 66 Ma. The default sample.
    same column   median reconstruction displacement <= 550 km, one sampling
                  footprint. This is the criterion the age cut is a proxy for:
                  a province on a slowly moving plate could be older and still
                  sit above the mantle it erupted over.

In this catalogue the two nearly coincide. No province is pre-Cenozoic and has
moved less than a footprint, so the slow-plate extension buys nothing here; five
Cenozoic provinces have moved further than a footprint and are dropped by the
stricter rule.

Import `sample()` rather than reimplementing the cut.
"""
import os
import numpy as np, pandas as pd

FOOTPRINT_KM = 550.0        # the 5 degree sampling cap
CENOZOIC_MA = 66.0


def sample(out_dir, rule='cenozoic'):
    """province -> bool, for rule in {cenozoic, same-column, both, all}."""
    e = pd.read_csv(os.path.join(out_dir, 'province_eruption_context.csv'))
    if 'province' not in e:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from province_pivot import provinces, CAT
        d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
        e['province'] = provinces(d)
    g = e.groupby('province').agg(age=('age_Ma', 'median'),
                                  moved=('moved_km', 'median'))
    cz = g.age <= CENOZOIC_MA
    sc = g.moved <= FOOTPRINT_KM
    return {'cenozoic': cz, 'same-column': sc, 'both': cz & sc,
            'all': pd.Series(True, index=g.index)}[rule]
