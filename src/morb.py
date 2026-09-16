"""The Dixon et al. MORB compilation, read from that project and never written.

One place for everything the MORB scripts in this repository share, so that the
sample table, the hydration age, the 300 km clusters and the plume exclusion are
the same objects in every analysis here as they are in the paper that owns them.

    load()          the 1,344-sample table with the columns this repository uses
    clusters()      complete-linkage 300 km clusters, as in Dixon et al.
    plume_mask()    samples on ridge segments under demonstrable plume influence
    corridor()      the Atlantic-Arctic corridor, 50W-5E
    contrast()      cluster-median contrast between two groups with a permutation
                    tail and a cluster bootstrap

Conventions inherited from the Dixon workflow and not to be changed here:

- the hydration age is `age_min_total_mantle_rate0p5`: the youngest 20 Myr window
  in which the total subducted-water delivery rate at the site exceeded 0.5 t/m2/Myr
  in the Zahirovic et al. (2022) mantle reference frame;
- a site with no such window has no hydration age, and that absence is a result:
  no subduction water came within reach of the site in 410 Myr;
- groups are compared as one median per 300 km cluster, never per sample.
"""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import paths as P

R_EARTH = 6371.0088
AGE = 'age_min_total_mantle_rate0p5'
DUR = 'duration_total_mantle_rate0p5'
EPW = 'episode_water_total_mantle_rate0p5'
BLOCK_KM = 300.0

# Ridge segments under demonstrable plume influence, geochemically defined
# (the geochemically defined rule the Dixon paper adopts). Five are
# given as the start and end of a stretch of ridge axis and a sample counts as on
# the segment when it lies within PLUME_KM of that line. The Sierra Leone stretch
# crosses the equatorial transforms, where the axis is a staircase rather than a
# line, so it is a latitude band across the ridge between the two longitudes the
# note gives. The exclusion removes 77 samples in the Dixon workflow; the count
# is printed by any script that applies it so that a drift would be noticed.
PLUME_SEGMENTS = {
    'Jan Mayen':         ((-4.79, 71.43), (-8.63, 71.48)),
    'Reykjanes >61N':    ((-28.11, 61.00), (-21.80, 63.70)),
    'Azores Platform':   ((-32.91, 37.00), (-29.67, 40.00)),
    '35N OH-1':          ((-34.96, 35.10), (-34.28, 35.80)),
    '14N Researcher':    ((-45.00, 13.70), (-45.00, 14.20)),
}
PLUME_BOX = {'Sierra Leone 1.7N': dict(lat=(-0.30, 2.18), lon=(-30.68, -16.24))}
PLUME_KM = 40.0


def to_xyz(lon, lat):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def gc_km(lon1, lat1, lon2, lat2):
    a = to_xyz(lon1, lat1); b = to_xyz(lon2, lat2)
    return R_EARTH * 2.0 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2.0, 0, 1))


def load():
    d = pd.read_csv(P.need(P.MORB_CSV, 'DIXON_DIR', 'the Dixon MORB sample table'))
    d['hydrated'] = np.isfinite(d[AGE].astype(float))
    d['age'] = d[AGE].astype(float)
    d['duration'] = d[DUR].astype(float)
    d['episode_water'] = d[EPW].astype(float)
    return d


def load_frames(d):
    """Hydration age under the four reference frames, aligned row for row with
    the sample table d. Sample names are not unique in the compilation (36 are
    reused at different positions), so the join is on name and position
    together, and every row of d must find exactly one match."""
    f = pd.read_csv(P.need(P.MORB_FRAMES, 'DIXON_DIR', 'the reference-frame ages'))
    key = ['Sample', 'Latitude', 'Longitude']
    m = d[key].merge(f, on=key, how='left', validate='one_to_one')
    if len(m) != len(d):
        raise SystemExit('reference-frame ages do not align with the sample table')
    ok = np.isfinite(m['age_published'].values) == np.isfinite(d['age'].values)
    if not ok.all():
        raise SystemExit(f'{(~ok).sum()} samples whose published-frame age in the '
                         'frames table disagrees with the sample table')
    return m


def clusters(lon, lat, km=BLOCK_KM):
    """Complete-linkage clusters at km, the Dixon definition of one observation."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if len(lon) < 2:
        return np.zeros(len(lon), int)
    gc = 2 * R_EARTH * np.arcsin(np.clip(pdist(to_xyz(lon, lat)) / 2, 0, 1))
    return fcluster(linkage(gc, method='complete'), t=km, criterion='distance')


def _dist_to_segment_km(lon, lat, a, b, n=200):
    """Great-circle distance from each point to the nearest of n points along
    the great-circle arc from a to b, which at these lengths is the segment."""
    pa, pb = to_xyz(*a)[0], to_xyz(*b)[0]
    w = np.linspace(0, 1, n)[:, None]
    line = (1 - w) * pa + w * pb
    line /= np.linalg.norm(line, axis=1)[:, None]
    pts = to_xyz(lon, lat)
    dots = np.clip(pts @ line.T, -1, 1)
    return R_EARTH * np.arccos(dots).min(axis=1)


def plume_mask(d, km=PLUME_KM, report=True):
    m = np.zeros(len(d), bool)
    counts = {}
    for name, (a, b) in PLUME_SEGMENTS.items():
        s = _dist_to_segment_km(d.Longitude.values, d.Latitude.values, a, b) <= km
        counts[name] = int(s.sum()); m |= s
    for name, box in PLUME_BOX.items():
        s = (d.Latitude.between(*box['lat']) & d.Longitude.between(*box['lon'])).values
        counts[name] = int(s.sum()); m |= s
    if report:
        print('plume exclusion: ' + ', '.join(f'{k} {v}' for k, v in counts.items())
              + f'; {int(m.sum())} samples in all (Dixon workflow: 77)')
    return m


def corridor(d):
    """The Atlantic-Arctic corridor, 50W to 5E, all latitudes."""
    return (d.Longitude.between(-50, 5)).values | (d.Latitude > 70).values


def pacific(d):
    return (((d.Longitude + 360) % 360).between(175, 280) & d.Latitude.between(-60, 50)).values


def cluster_medians(values, cl, extra=None):
    """One median per cluster; returns a frame with the cluster id, n and median."""
    v = np.asarray(values, float)
    rows = []
    for c in np.unique(cl):
        m = (cl == c) & np.isfinite(v)
        if m.sum() == 0:
            continue
        rec = dict(cluster=int(c), n=int(m.sum()), median=float(np.median(v[m])))
        if extra is not None:
            for k, arr in extra.items():
                arr = np.asarray(arr, float)
                rec[k] = float(np.nanmedian(arr[cl == c]))
        rows.append(rec)
    return pd.DataFrame(rows)


def contrast(values, group, cl, n_perm=20000, n_boot=2000, seed=0):
    """Median of cluster medians in group against not-group.

    A cluster is assigned to a group by the majority of its samples. The
    permutation tail shuffles the group label across clusters, which is the
    exchangeable unit; the bootstrap resamples clusters with replacement.
    Two-sided on the difference of medians, as in the Dixon analysis.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float); g = np.asarray(group, bool)
    cm = cluster_medians(v, cl, extra=dict(g=g.astype(float)))
    if len(cm) < 4:
        return dict(n_in=0, n_out=0, med_in=np.nan, med_out=np.nan, diff=np.nan,
                    p=np.nan, lo=np.nan, hi=np.nan)
    med = cm['median'].values; gg = cm['g'].values >= 0.5
    if gg.sum() < 2 or (~gg).sum() < 2:
        return dict(n_in=int(gg.sum()), n_out=int((~gg).sum()), med_in=np.nan,
                    med_out=np.nan, diff=np.nan, p=np.nan, lo=np.nan, hi=np.nan)

    def stat(gmask):
        return np.median(med[gmask]) - np.median(med[~gmask])
    obs = stat(gg)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = stat(rng.permutation(gg))
    p = (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + n_perm)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(med), len(med))
        gb = gg[i]
        boot[k] = (np.median(med[i][gb]) - np.median(med[i][~gb])
                   if gb.sum() > 1 and (~gb).sum() > 1 else np.nan)
    boot = boot[np.isfinite(boot)]
    return dict(n_in=int(gg.sum()), n_out=int((~gg).sum()),
                med_in=float(np.median(med[gg])), med_out=float(np.median(med[~gg])),
                diff=float(obs), p=float(p),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def spearman_clusters(x, y, cl, n_boot=2000, seed=0):
    """Spearman rho between two per-sample quantities on cluster medians, with a
    cluster bootstrap interval. Returns rho, n_clusters, lo, hi, p (permutation)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    cx = cluster_medians(np.asarray(x, float), cl)
    cy = cluster_medians(np.asarray(y, float), cl)
    m = cx.merge(cy, on='cluster', suffixes=('_x', '_y'))
    xs, ys = m['median_x'].values, m['median_y'].values
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 8:
        return dict(rho=np.nan, n=int(len(xs)), lo=np.nan, hi=np.nan, p=np.nan)
    r = stats.spearmanr(xs, ys).statistic
    b = np.array([stats.spearmanr(xs[i], ys[i]).statistic
                  for i in (rng.integers(0, len(xs), len(xs)) for _ in range(n_boot))])
    b = b[np.isfinite(b)]
    null = np.array([stats.spearmanr(xs, rng.permutation(ys)).statistic
                     for _ in range(n_boot)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (1 + len(null))
    return dict(rho=float(r), n=int(len(xs)), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)), p=float(p))
