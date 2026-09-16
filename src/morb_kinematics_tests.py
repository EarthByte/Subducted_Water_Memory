"""The two kinematic questions the chemistry can answer, tested on clusters.

Reads what morb_kinematics.py wrote and asks, for the sites the reconstruction
hydrated:

1. Keel passage. Does H2O/Ce differ between hydrated sites that a cratonic keel
   (>= 200 km lithosphere, or a Shirmard et al. 2025 craton polygon) later passed over and hydrated sites that only thin
   continental lithosphere, or none, passed over? The reviewer's objection to
   the Farallon attribution predicts that keel passage erases the signal, so
   overrun sites should be no wetter than unhydrated ones. Reported as the
   contrast and as a rank correlation with the time spent under a keel.

2. Ridge processing. Does H2O/Ce fall with the time a spreading ridge has spent
   over the site since the water arrived? Decompression melting extracts water
   from the column it processes, so a signal that survives long ridge residence
   must reside below or beside the melting column, and a signal that does not
   was within it. Reported as a rank correlation with the ridge dwell time at
   three radii, with the hydration age partialled out, on clusters.

Both are run globally, in the Atlantic-Arctic corridor, and with the
plume-influenced segments removed. Every statistic is computed on one median
per 300 km cluster and its null is a permutation across clusters, so the
numbers are comparable with the Dixon contrasts and with each other.

Writes out/morb_kinematics_tests<suffix>.csv.
"""
import os, argparse, numpy as np, pandas as pd
from scipy import stats
import paths as P
import morb


def partial_spearman_clusters(x, y, z, cl, n_boot=2000, seed=0):
    """Spearman of x and y with z regressed out of both ranks, on cluster medians."""
    rng = np.random.default_rng(seed)
    cx = morb.cluster_medians(x, cl); cy = morb.cluster_medians(y, cl)
    cz = morb.cluster_medians(z, cl)
    m = cx.merge(cy, on='cluster', suffixes=('_x', '_y')).merge(
        cz.rename(columns={'median': 'median_z'})[['cluster', 'median_z']], on='cluster')
    X, Y, Z = (m[c].values for c in ('median_x', 'median_y', 'median_z'))
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[ok], Y[ok], Z[ok]
    if len(X) < 8:
        return dict(rho=np.nan, n=int(len(X)), lo=np.nan, hi=np.nan, p=np.nan)

    def pr(X, Y, Z):
        rx, ry, rz = (stats.rankdata(v) for v in (X, Y, Z))
        def res(v):
            A = np.column_stack([np.ones(len(rz)), rz])
            return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        return stats.pearsonr(res(rx), res(ry)).statistic
    r = pr(X, Y, Z)
    b = np.array([pr(X[i], Y[i], Z[i]) for i in
                  (rng.integers(0, len(X), len(X)) for _ in range(n_boot))])
    null = np.array([pr(X, rng.permutation(Y), Z) for _ in range(n_boot)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (1 + len(null))
    return dict(rho=float(r), n=int(len(X)), lo=float(np.nanpercentile(b, 2.5)),
                hi=float(np.nanpercentile(b, 97.5)), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    ap.add_argument('--n-perm', type=int, default=10000)
    a = ap.parse_args()
    k = pd.read_csv(os.path.join(P.OUT, f'morb_kinematics{a.suffix}.csv'))
    d = morb.load()
    assert (k.Sample.values == d.Sample.values).all(), 'kinematics are for another table'
    plume = morb.plume_mask(d)
    hyd = k.hydrated.values
    rows = []
    domains = {'global': np.ones(len(k), bool), 'corridor': morb.corridor(k),
               'no plume segments': ~plume, 'corridor, no plumes': morb.corridor(k) & ~plume}

    print('\n1. Keel passage among hydrated sites\n')
    print(f'{"domain":20s} {"comparison":34s} {"in":>5s} {"out":>5s} {"n":>7s} {"diff":>6s} {"p":>6s}')
    for dom, dm in domains.items():
        m = dm & hyd
        cl = morb.clusters(k.Longitude.values[m], k.Latitude.values[m])
        y = k.H2O_Ce.values[m]
        for label, grp in (('keel >=200 km passed over', k.t_keel.values[m] > 0),
                           ('craton polygon passed over', k.t_craton.values[m] > 0),
                           ('lithosphere >=150 km passed over', k.t_thick.values[m] > 0),
                           ('any continent passed over', k.t_cont.values[m] > 0)):
            c = morb.contrast(y, grp, cl, n_perm=a.n_perm, n_boot=500)
            rows.append(dict(section='keel', domain=dom, test=label, **c))
            print(f'{dom:20s} {label:34s} {c["med_in"]:5.0f} {c["med_out"]:5.0f} '
                  f'{c["n_in"]:3d}/{c["n_out"]:<3d} {c["diff"]:+6.0f} {c["p"]:6.3f}')
        for col in ('t_keel', 't_craton', 't_thick', 't_cont'):
            r = morb.spearman_clusters(k[col].values[m], y, cl)
            rows.append(dict(section='keel', domain=dom, test=f'rho H2O/Ce vs {col}', **r))
            print(f'{dom:20s} {"rho vs " + col:34s} {"":5s} {"":5s} {r["n"]:7d} '
                  f'{r["rho"]:+6.2f} {r["p"]:6.3f}')
        print()

    # and the same contrast against the unhydrated population, which is what
    # the objection actually predicts: overrun hydrated sites should look
    # unhydrated
    print('2. Overrun hydrated sites against unhydrated sites\n')
    for dom, dm in domains.items():
        for label, grp in (('keel-overrun hydrated', (k.t_keel.values > 0) & hyd),
                           ('craton-overrun hydrated', (k.t_craton.values > 0) & hyd),
                           ('never-overrun hydrated', (k.t_cont.values == 0) & hyd)):
            m = dm & (grp | ~hyd)
            cl = morb.clusters(k.Longitude.values[m], k.Latitude.values[m])
            c = morb.contrast(k.H2O_Ce.values[m], grp[m], cl, n_perm=a.n_perm, n_boot=500)
            rows.append(dict(section='vs unhydrated', domain=dom, test=label, **c))
            print(f'{dom:20s} {label + " vs unhydrated":34s} {c["med_in"]:5.0f} '
                  f'{c["med_out"]:5.0f} {c["n_in"]:3d}/{c["n_out"]:<3d} {c["diff"]:+6.0f} '
                  f'{c["p"]:6.3f}')
    print()

    print('3. Ridge processing among hydrated sites (rho on clusters; partial | age)\n')
    print(f'{"domain":20s} {"predictor":16s} {"rho":>6s} {"[95%]":>15s} {"p":>6s} '
          f'{"partial|age":>12s} {"p":>6s} {"n":>4s}')
    for dom, dm in domains.items():
        m = dm & hyd
        cl = morb.clusters(k.Longitude.values[m], k.Latitude.values[m])
        y = k.H2O_Ce.values[m]; age = k.age.values[m]
        for col in ('t_ridge_100', 't_ridge_200', 't_ridge_300', 'ridge_onset_200'):
            x = k[col].values[m]
            r = morb.spearman_clusters(x, y, cl)
            pr = partial_spearman_clusters(x, y, age, cl)
            rows.append(dict(section='ridge', domain=dom, test=f'rho H2O/Ce vs {col}',
                             **r, partial_rho=pr['rho'], partial_p=pr['p']))
            print(f'{dom:20s} {col:16s} {r["rho"]:+6.2f} [{r["lo"]:+5.2f},{r["hi"]:+5.2f}] '
                  f'{r["p"]:6.3f} {pr["rho"]:+12.2f} {pr["p"]:6.3f} {r["n"]:4d}')
        print()

    # the same for every site, hydrated or not, over the whole model span: does
    # ridge residence alone organise H2O/Ce?
    print('4. Ridge residence over the full 400 Myr, all sites\n')
    for dom, dm in domains.items():
        cl = morb.clusters(k.Longitude.values[dm], k.Latitude.values[dm])
        r = morb.spearman_clusters(k.t_ridge_200.values[dm], k.H2O_Ce.values[dm], cl)
        rows.append(dict(section='ridge, all sites', domain=dom, test='rho vs t_ridge_200', **r))
        print(f'{dom:20s} rho(H2O/Ce, t_ridge_200) = {r["rho"]:+.2f} '
              f'[{r["lo"]:+.2f},{r["hi"]:+.2f}] p={r["p"]:.3f} n={r["n"]}')
    print('\n5. Joint rank model on hydrated clusters: H2O/Ce ~ t_ridge_200 + t_keel + age\n')
    for dom, dm in domains.items():
        m = dm & hyd
        cl = morb.clusters(k.Longitude.values[m], k.Latitude.values[m])
        cols = {c: morb.cluster_medians(k[c].values[m], cl).set_index('cluster')['median']
                for c in ('H2O_Ce', 't_ridge_200', 't_keel', 'age')}
        X = pd.DataFrame(cols).dropna()
        if len(X) < 10:
            continue
        R = X.rank()
        A = np.column_stack([np.ones(len(R)), R['t_ridge_200'], R['t_keel'], R['age']])
        beta = np.linalg.lstsq(A, R['H2O_Ce'].values, rcond=None)[0]
        rng = np.random.default_rng(0)
        null = np.array([np.linalg.lstsq(A, rng.permutation(R['H2O_Ce'].values), rcond=None)[0]
                         for _ in range(5000)])
        p = ((1 + (np.abs(null) >= np.abs(beta)).sum(0)) / (1 + len(null)))
        print(f'{dom:20s} n={len(X):3d}  ridge {beta[1]:+.2f} (p={p[1]:.3f})  '
              f'keel {beta[2]:+.2f} (p={p[2]:.3f})  age {beta[3]:+.2f} (p={p[3]:.3f})')
        rows.append(dict(section='joint', domain=dom, test='rank regression', n=len(X),
                         beta_ridge=beta[1], p_ridge=p[1], beta_keel=beta[2], p_keel=p[2],
                         beta_age=beta[3], p_age=p[3]))

    print('\n6. How fast a ridge erases the excess: corridor clusters, excess over the '
          'Pacific median\n')
    pac = morb.pacific(k)
    clp = morb.clusters(k.Longitude.values[pac], k.Latitude.values[pac])
    base = np.median(morb.cluster_medians(k.H2O_Ce.values[pac], clp)['median'])
    m = morb.corridor(k) & ~plume
    cl = morb.clusters(k.Longitude.values[m], k.Latitude.values[m])
    cm = morb.cluster_medians(k.H2O_Ce.values[m], cl,
                              extra=dict(dwell=k.t_ridge_200.values[m]))
    ex = cm['median'].values - base; dw = cm['dwell'].values
    from scipy.optimize import curve_fit
    try:
        pp, _ = curve_fit(lambda t, e0, tau: e0 * np.exp(-t / tau), dw, ex,
                          p0=[80, 60], bounds=([0, 5], [500, 1000]))
        boot = []
        rng = np.random.default_rng(1)
        for _ in range(1000):
            i = rng.integers(0, len(dw), len(dw))
            try:
                q, _ = curve_fit(lambda t, e0, tau: e0 * np.exp(-t / tau), dw[i], ex[i],
                                 p0=pp, bounds=([0, 5], [500, 1000]))
                boot.append(q[1])
            except Exception:
                pass
        boot = np.array(boot)
        print(f'Pacific baseline {base:.0f}; excess at zero dwell {pp[0]:.0f}; '
              f'e-folding {pp[1]:.0f} Myr of ridge residence '
              f'(cluster bootstrap 68%: {np.percentile(boot, 16):.0f}-{np.percentile(boot, 84):.0f}); '
              f'n={len(dw)} clusters')
        rows.append(dict(section='erasure', domain='corridor, no plumes', test='exp fit',
                         n=len(dw), baseline=base, excess0=pp[0], tau=pp[1],
                         lo=np.percentile(boot, 16), hi=np.percentile(boot, 84)))
    except Exception as e:
        print('fit failed:', e)

    pd.DataFrame(rows).to_csv(os.path.join(P.OUT, f'morb_kinematics_tests{a.suffix}.csv'),
                              index=False)
    print(f'\nwrote out/morb_kinematics_tests{a.suffix}.csv')


if __name__ == '__main__':
    main()
