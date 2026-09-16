"""Where does the hydrated-unhydrated H2O/Ce contrast live?

The keel and ridge-residence tests of this paper are reported globally and
along the Atlantic-Arctic corridor, and the underlying contrast between mantle
the reconstruction dates as hydrated and mantle it does not is taken from the
companion study as a global result. Neither asks the prior question this script
asks: is that contrast present outside the corridor at all?

It matters because of where the compilation's hydrated sites are. Almost all of
them lie in the Atlantic and the Arctic, while most of the unhydrated clusters
are Pacific. A global comparison of the two therefore sets one ocean against
another, and a difference of provenance, spreading rate or melting regime
between the Atlantic and the Pacific would produce the same number as a
difference between hydrated and unhydrated mantle. The within-corridor
comparison, hydrated against unhydrated clusters of the same ocean, is the one
that separates them, and the outside-corridor comparison is the independent
test of the same claim.

Three things are measured for each domain, always on 300 km clusters, the
Dixon unit of observation, and always with the geochemical plume exclusion:

    contrast    median of hydrated cluster medians against unhydrated, with
                the cluster-label permutation tail and a cluster bootstrap
    composition how many hydrated clusters the domain holds, how old their
                hydration is and how long a ridge has been above them, which
                says whether the domain can test the claim at all
    relations   H2O/Ce against ridge residence and against the cumulative
                shallow water delivered, on hydrated clusters

    PYTHONPATH=src python3 src/morb_domain_test.py

Writes out/morb_domain_test.csv. Reads out/morb_kinematics.csv and
out/water_partition_sites.csv, so run morb_kinematics.py and
water_partition.py first.
"""
import os
import numpy as np
import pandas as pd
import paths as P
import morb

KEY = ['Sample', 'Latitude', 'Longitude']
OLD = 100.0          # Ma; hydration old enough for the water to have been
                     # advected into a melting column, not still arriving
DOSE = 300.0         # t/m2 of cumulative shallow delivery; the step in H2O/Ce
                     # sits between 200 and 300 and is flat above it
# The Red Sea, the Gulf of Aden and the easternmost Mediterranean: young
# spreading centres opening through continental lithosphere, where elevated
# H2O/Ce has a contamination explanation that has nothing to do with subducted
# water, and where the reconstruction assigns large delivery because Tethyan
# subduction ran alongside. Every cluster there holds one or two samples. They
# are not excluded from the main tests; the sensitivity below removes them.
RIFT_BOX = dict(lon=(25.0, 62.0), lat=(3.0, 40.0))


def table():
    d = morb.load()
    k = pd.read_csv(P.need(os.path.join(P.OUT, 'morb_kinematics.csv'), 'OUT',
                           'the per-site kinematic summaries'))
    d = d.merge(k[KEY + ['t_craton', 't_keel', 't_ridge_200']], on=KEY,
                how='left', validate='one_to_one')
    w = os.path.join(P.OUT, 'water_partition_sites.csv')
    if os.path.exists(w):
        d = d.merge(pd.read_csv(w)[KEY + ['cumulative_outflux']], on=KEY,
                    how='left', validate='one_to_one')
    else:
        d['cumulative_outflux'] = np.nan
    return d


def main():
    d = table()
    keep = ~morb.plume_mask(d)
    d = d[keep].reset_index(drop=True)
    cl = morb.clusters(d.Longitude.values, d.Latitude.values)
    corr, pac = morb.corridor(d), morb.pacific(d)
    hyd = d.hydrated.values
    old = np.isfinite(d.age.values) & (d.age.values >= OLD)

    domains = [
        ('global', np.ones(len(d), bool)),
        ('Atlantic-Arctic corridor', corr),
        ('outside the corridor', ~corr),
        ('outside the corridor, hydration >= %g Ma' % OLD, (~corr) & (old | ~hyd)),
        ('Pacific box', pac),
    ]

    rows = []
    print(f'{len(d)} samples after the plume exclusion, {len(set(cl))} clusters\n')
    print(f'{"domain":42s} {"hyd":>4s} {"unhyd":>6s} {"med in":>7s} {"med out":>8s} '
          f'{"diff":>6s} {"95% CI":>16s} {"p":>7s}')
    for name, m in domains:
        c = morb.contrast(d.H2O_Ce.values[m], hyd[m], cl[m])
        print(f'{name:42s} {c["n_in"]:4d} {c["n_out"]:6d} {c["med_in"]:7.0f} '
              f'{c["med_out"]:8.0f} {c["diff"]:+6.0f} '
              f'[{c["lo"]:+6.0f},{c["hi"]:+6.0f}] {c["p"]:7.4f}')
        rows.append(dict(domain=name, test='hydrated vs unhydrated', **c))

    print(f'\n{"domain":42s} {"hyd cl":>7s} {"age med":>8s} {"age>=%g" % OLD:>8s} '
          f'{"t_ridge med":>12s}')
    for name, m in domains[:3] + domains[4:]:
        g = d[m & hyd]
        if not len(g):
            continue
        cm = morb.cluster_medians(g.age.values, cl[m & hyd],
                                  extra=dict(tr=g.t_ridge_200.values))
        n_old = int((cm['median'].values >= OLD).sum())
        print(f'{name:42s} {len(cm):7d} {np.nanmedian(cm["median"]):8.0f} '
              f'{n_old:8d} {np.nanmedian(cm["tr"]):12.0f}')
        rows.append(dict(domain=name, test='hydrated composition',
                         n_in=len(cm), med_in=float(np.nanmedian(cm['median'])),
                         n_out=n_old, med_out=float(np.nanmedian(cm['tr']))))

    print(f'\n{"domain":42s} {"against":26s} {"rho":>6s} {"n":>4s} '
          f'{"95% CI":>16s} {"p":>7s}')
    for name, m in domains[:3]:
        g = m & hyd
        for v, label in (('t_ridge_200', 'ridge residence'),
                         ('cumulative_outflux', 'cumulative shallow water'),
                         ('age', 'hydration age'),
                         ('t_craton', 'time beneath a craton')):
            s = morb.spearman_clusters(d[v].values[g], d.H2O_Ce.values[g], cl[g])
            print(f'{name:42s} {label:26s} {s["rho"]:+6.2f} {s["n"]:4d} '
                  f'[{s["lo"]:+6.2f},{s["hi"]:+6.2f}] {s["p"]:7.4f}')
            rows.append(dict(domain=name, test=f'rho H2O/Ce vs {v}', **s))

    # ------------------------------------------------------------ the dose
    # The hydration age is a threshold on a rate in a 20 Myr window, so it
    # divides the compilation in two and discards how much water arrived. The
    # cumulative shallow delivery keeps it, and it is defined for every
    # cluster, hydrated or not, which makes the comparison independent of the
    # convention.
    w = d.cumulative_outflux.values * 1e6                 # Mt/m2 -> t/m2
    rift = (d.Longitude.between(*RIFT_BOX['lon'])
            & d.Latitude.between(*RIFT_BOX['lat'])).values
    print(f'\n{"domain":42s} {"against cumulative shallow water":32s} {"rho":>6s} '
          f'{"n":>4s} {"95% CI":>16s} {"p":>7s}')
    for name, m in (('all clusters', np.ones(len(d), bool)),
                    ('Atlantic-Arctic corridor', corr),
                    ('outside the corridor', ~corr),
                    ('outside the corridor, no rift box', (~corr) & ~rift)):
        s = morb.spearman_clusters(w[m], d.H2O_Ce.values[m], cl[m])
        print(f'{name:42s} {"":32s} {s["rho"]:+6.2f} {s["n"]:4d} '
              f'[{s["lo"]:+6.2f},{s["hi"]:+6.2f}] {s["p"]:7.4f}')
        rows.append(dict(domain=name, test='rho H2O/Ce vs cumulative shallow water', **s))

    print(f'\n{"domain":42s} {"above %g t/m2" % DOSE:>14s} {"below":>9s} '
          f'{"diff":>6s} {"95% CI":>16s} {"p":>7s}')
    for name, m in (('all clusters', np.ones(len(d), bool)),
                    ('Atlantic-Arctic corridor', corr),
                    ('outside the corridor', ~corr),
                    ('outside the corridor, no rift box', (~corr) & ~rift)):
        c = morb.contrast(d.H2O_Ce.values[m], w[m] >= DOSE, cl[m])
        print(f'{name:42s} {c["n_in"]:4d} at {c["med_in"]:6.0f} {c["n_out"]:4d} at '
              f'{c["med_out"]:4.0f} {c["diff"]:+6.0f} [{c["lo"]:+6.0f},{c["hi"]:+6.0f}] '
              f'{c["p"]:7.4f}')
        rows.append(dict(domain=name, test=f'H2O/Ce above vs below {DOSE:g} t/m2', **c))

    out = os.path.join(P.OUT, 'morb_domain_test.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
