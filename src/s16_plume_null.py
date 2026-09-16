"""Stage 16 — is the hotspot/structure agreement real, or is it what any 49 points give?

Two problems have to be dealt with before the stage-C numbers mean anything.

CIRCULARITY. Courtillot's `count` is the number of his five criteria a hotspot
satisfies, and one of the five IS a deep tomographic low-velocity anomaly. Testing
"deep structure in REVEAL" against `count` therefore correlates a quantity with
itself at strength 1/5. The non-circular target is

    count_indep = count - (tomo_anomaly == 'slow')

which counts only age progression, LIP link, high 3He/4He and buoyancy flux —
four criteria with no seismic input at all. If deep REVEAL structure predicts
THAT, the classifier is telling us something tomography was not already told.

EFFECTIVE SAMPLE SIZE. Both the hotspot distribution and the velocity field are
strongly spatially autocorrelated, and this project has already shown (s14) that
naive p-values on such a pair are wrong by twenty orders of magnitude. So the
null is a spin test: the hotspot constellation is rotated rigidly on the sphere,
carrying its labels with it, and the whole analysis is recomputed at the rotated
positions. Hotspot-to-hotspot geometry, the label distribution and the velocity
field's own autocorrelation are all preserved exactly; only the alignment between
them is destroyed.

Rotations that put a hotspot back near its true position would leak signal, so
each spin is required to displace the constellation by a minimum arc.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import plume_classifier as pc

warnings.filterwarnings('ignore')

OUT = os.environ.get('PLUME_OUT', '.')
N_SPIN = int(os.environ.get("NSPIN", 300))
MIN_DISPLACE_DEG = 20.0
SEED = 20260803


def to_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def to_lonlat(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return (np.degrees(np.arctan2(y, x)),
            np.degrees(np.arcsin(np.clip(z / np.linalg.norm(xyz, axis=-1), -1, 1))))


def random_rotation(rng):
    """A rotation drawn uniformly from SO(3).

    The QR factorisation of a Gaussian matrix gives the Haar measure only
    after the signs of the diagonal of R are folded into Q (Mezzadri, 2007)
    and the determinant is then forced positive. Without the sign step the
    draw is not uniform either: it starves the small rotations and gives a
    mean rotation angle of 144 degrees against the correct 126.5. The obvious alternative, a uniform axis with
    an angle uniform on [0, 2*pi), is NOT uniform on the rotation group: the Haar
    angle density goes as 1 - cos(theta), so a uniform angle over-weights small
    rotations. About an eighth of such draws move the transition-zone fast set by
    less than 20 degrees, which leaves it close to where it was observed and
    carries the observed association into the null. On the 410-520 km target that
    raised the 95th percentile of the null from 2.35 to 2.76.

    That is what this function used to do, and it is kept below as
    axis_angle_rotation for reproducing numbers published from it.
    """
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def axis_angle_rotation(rng):
    """The former random_rotation: uniform axis, uniform angle. Not uniform on
    SO(3); see random_rotation. Present only to reproduce earlier output."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def descriptors(names, lons, lats, depth, lat, lon, arr, labels, scan):
    """The two descriptors the null is run on, for one constellation."""
    um, deep = [], []
    for nm, o, a in zip(names, lons, lats):
        c = pc.column_descriptors(nm, float(a), float(o), depth, lat, lon, arr)
        r5 = [r for r in c if r['radius_deg'] == 5.0][0]
        um.append(r5.get('um_200_410_mean', np.nan))

        allrows = pc.persistence(nm, float(a), float(o), depth, lat, lon, labels, scan)
        rows = [r for r in allrows if r['below_pc'] and r['seeded']]
        m = [r['pct'] for r in allrows if r.get('merged') is True]
        mp = min(m) if m else np.inf
        d = [r['max_depth'] for r in rows
             if r['pct'] < mp and np.isfinite(r['max_depth'] or np.nan)]
        deep.append(max(d) if d else np.nan)
    return np.array(um, float), np.array(deep, float)


def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    return stats.spearmanr(a[m], b[m]).statistic


def main():
    global OUT, N_SPIN
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='REVEAL', choices=list(pc.MODELS))
    ap.add_argument('--file', default=None)
    ap.add_argument('--var', default=None)
    ap.add_argument('--src', default=pc.DATA,
                    help='folder holding the tomography file (default: cwd)')
    ap.add_argument('--hotspots', default=pc.HOTSPOTS, help='Courtillot table CSV')
    ap.add_argument('--out', default=OUT, help='where to write the CSV')
    ap.add_argument('--spins', type=int, default=N_SPIN)
    a = ap.parse_args()

    OUT, N_SPIN = a.out, a.spins
    pc.HOTSPOTS = a.hotspots
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(pc.HOTSPOTS):
        raise SystemExit(f'hotspot table not found: {pc.HOTSPOTS}\n'
                         '  pass --hotspots /path/to/courtillot_2003_table1.csv')
    spec = pc.MODELS[a.model]
    if a.var:
        spec = pc.ModelSpec(spec.name, spec.file, a.var, spec.anomaly_label)
    path = a.file if a.file and os.path.isabs(a.file) else os.path.join(
        a.src, a.file or spec.file)
    depth, lat, lon, arr = pc.load_anomaly(spec, path)
    tag = f'{spec.name}_{spec.velocity_var}_{int(round(depth.max()))}km'
    scan, labels, p_c = pc.percolation_scan(arr, depth)
    print(f'{tag}: {len(depth)} shells to {depth.max():.0f} km, '
          f'percolation threshold p{p_c:g}', flush=True)

    hs = pd.read_csv(pc.HOTSPOTS).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    hs['count_indep'] = hs['count'] - (hs.tomo_anomaly == 'slow').astype(int)
    print(f'{len(hs)} hotspots; count 0-{hs["count"].max()}, '
          f'count_indep 0-{hs.count_indep.max()} '
          f'(mean {hs["count"].mean():.2f} -> {hs.count_indep.mean():.2f})\n', flush=True)

    names = hs.hotspot.values
    lon0, lat0 = hs.lon_180.values.astype(float), hs.lat.values.astype(float)
    um0, deep0 = descriptors(names, lon0, lat0, depth, lat, lon, arr, labels, scan)

    targets = {'count (circular, 1 of 5 criteria is tomographic)': hs['count'].values,
               'count_indep (no seismic input)': hs.count_indep.values,
               'he_ratio high (3He/4He only)':
                   np.where(hs.he_ratio.isna(), np.nan,
                            (hs.he_ratio == 'high').astype(float))}

    obs = {}
    for tname, tv in targets.items():
        obs[tname] = (rho(np.asarray(tv, float), um0),
                      rho(np.asarray(tv, float), deep0))

    # ------------------------------------------------------------------ spins
    rng = np.random.default_rng(SEED)
    xyz0 = to_xyz(lon0, lat0)
    null_um, null_deep = [], []
    done = 0
    while done < N_SPIN:
        R = random_rotation(rng)
        xyz = xyz0 @ R.T
        disp = np.degrees(np.arccos(np.clip((xyz * xyz0).sum(1), -1, 1)))
        if np.median(disp) < MIN_DISPLACE_DEG:
            continue
        slon, slat = to_lonlat(xyz)
        u, d = descriptors(names, slon, slat, depth, lat, lon, arr, labels, scan)
        null_um.append(u)
        null_deep.append(d)
        done += 1
        if done % 25 == 0:
            print(f'  spin {done}/{N_SPIN}', flush=True)
    null_um = np.array(null_um)
    null_deep = np.array(null_deep)

    print('\n' + '=' * 92)
    print('SPIN TEST — hotspot constellation rotated rigidly, labels carried along')
    print(f'{N_SPIN} rotations, median displacement >= {MIN_DISPLACE_DEG:.0f} deg')
    print('=' * 92)
    print(f'{"target":48s} {"descriptor":18s} {"rho":>7s} {"null 2.5-97.5%":>20s} {"p":>7s}')
    res = []
    for tname, tv in targets.items():
        tvf = np.asarray(tv, float)
        for dname, o, nl in (('upper mantle dv', obs[tname][0], null_um),
                             ('pre-merge depth', obs[tname][1], null_deep)):
            nr = np.array([rho(tvf, row) for row in nl])
            nr = nr[np.isfinite(nr)]
            if not np.isfinite(o) or len(nr) < 30:
                continue
            p = ((np.abs(nr) >= abs(o)).sum() + 1) / (len(nr) + 1)
            print(f'{tname:48s} {dname:18s} {o:+7.3f} '
                  f'[{np.percentile(nr, 2.5):+6.3f},{np.percentile(nr, 97.5):+6.3f}] '
                  f'{p:7.3f}{"  SURVIVES" if p < 0.05 else ""}')
            res.append(dict(target=tname, descriptor=dname, rho=o,
                            null_lo=np.percentile(nr, 2.5),
                            null_hi=np.percentile(nr, 97.5), p=p, n_null=len(nr)))
    pd.DataFrame(res).to_csv(f'{OUT}/plume_spin_test_{tag}.csv', index=False)
    print(f'\nwrote plume_spin_test_{tag}.csv')


if __name__ == '__main__':
    main()
