"""The volcanic-field tests, in every model that can carry them.

Two of the paper's three refutations were measured in REVEAL alone: the sign of
the transition-zone anomaly beneath the fields, and the 350-410 km interval where
dehydration melting is predicted. The depth-localisation result is checked in
five volumes; these were not, which is the wrong way round, because these are the
claims and that one is the control.

This runs both in every model given, on that model's own grid, and reports how
many depth levels each interval actually contains. Compatibility is per test, not
per model: SPiRaL resolves 410-520 against 520-660 better than any other volume
but carries a single level between 350 and 410 km, so it can bear the first test
and barely the second.

Three quantities per model:

  transition zone      mean 410-660 anomaly beneath the fields; hydration
                       predicts slow, and the sign is what refutes it
  350-410 alone        the interval where a dehydration layer is predicted,
                       against the same interval at rotated locations
  350-410 minus 250-350  the band difference, which is a trap: the reference
                       lies in cratonic roots and manufactures a slow layer

Fields are grouped once, by geography, so every model is scored on the same 27
provinces.

    python3 src/field_clusters.py
    python3 src/field_clusters.py --continental
    python3 src/field_clusters.py --models REVEAL SPiRaL --nspin 1000
"""
import argparse, os, sys
import numpy as np, pandas as pd, xarray as xr

try:
    import paths as P
    CAT, TOMO, OUT = P.IPV_V3, getattr(P, 'TOMO', '..'), getattr(P, 'OUT', 'out')
    REVEAL_NC, REVEALLO_NC = P.REVEAL, P.REVEALLO
except Exception:
    _D = os.environ.get('HYD_DATA', '..')
    CAT, TOMO, OUT = os.path.join(_D, 'ipv_catalogue_georoc_v3.csv'), _D, 'out'
    REVEAL_NC = os.path.join(_D, 'REVEAL_vs_full.nc')
    REVEALLO_NC = os.path.join(_D, 'REVEAL_mantle_tomography', 'RevealLO.nc')

R = 6371.0088
TZ, MELT, REF = (410.0, 660.0), (350.0, 410.0), (250.0, 350.0)
RADIUS_DEG, LINK = 5.0, 500.0
MODEL_FILE = {
    'REVEAL':    REVEAL_NC,
    'RevealLO':  REVEALLO_NC,
    'GLADM35':   os.path.join(TOMO, 'REVEAL_mantle_tomography', 'GLADM35.nc'),
    'SPiRaL':    os.path.join(TOMO, 'REVEAL_mantle_tomography', 'SPiRaL.nc'),
    'SEMUCBWM1': os.path.join(TOMO, 'REVEAL_mantle_tomography', 'SEMUCB-WM1.nc'),
}
NICE = {'GLADM35': 'GLAD-M35', 'SEMUCBWM1': 'SEMUCB-WM1'}


def to_xyz(lon, lat):
    a, b = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])


def haar(rng):
    """Uniform on SO(3): the QR of a Gaussian matrix with the signs of diag(R)
    folded in (Mezzadri 2007). Without that step it is not uniform."""
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def groups(la, lo, link_km):
    """Single linkage on great-circle distance, by union-find. Chaining is the
    conservative direction: fewer, larger groups means a smaller effective
    sample and a wider interval."""
    x = to_xyz(lo, la)
    d = R * 2.0 * np.arcsin(np.clip(
        np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2) / 2.0, 0, 1))
    parent = np.arange(len(la))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(d <= link_km, 1))):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    _, inv = np.unique(np.array([find(i) for i in range(len(la))]),
                       return_inverse=True)
    return inv


class Model:
    """One volume, reduced to the three intervals on its own grid.

    The normalisation follows extract_bands.py, because the models do not agree
    on any of it: RevealLO uses uppercase variable names and gives the vertical
    axis as a radius in metres, SEMUCB-WM1 carries an isotropic vs with no vsv
    and vsh, and axes run in either direction. Guessing any of this wrongly puts
    every shell outside every interval and returns a silent nothing.

    Only the shells between 200 and 700 km are read; the whole of RevealLO is
    several gigabytes and none of it below 700 km is wanted here.
    """
    def __init__(self, path):
        ds = xr.open_dataset(path)
        names = {k.lower(): k for k in ds.variables}
        latn = names.get('latitude') or names.get('lat')
        lonn = names.get('longitude') or names.get('lon')
        dkey = names.get('depth') or names.get('radius') or names.get('z')
        if not (latn and lonn and dkey):
            raise SystemExit(f'{os.path.basename(path)}: cannot find lat, lon and a '
                             f'vertical axis among {sorted(ds.variables)}')
        raw = np.asarray(ds[dkey].values, float)
        dep = raw / 1.0e3 if raw.max() > 1.0e5 else raw.copy()   # metres of some kind
        if dep.min() > 3.0e3 and dep.max() <= 6.5e3:             # a radius in km
            dep = 6371.0 - dep
        if dep.max() < 400:
            raise SystemExit(f'{os.path.basename(path)}: vertical axis tops out at '
                             f'{dep.max():.0f} km after conversion (raw '
                             f'{raw.min():.4g} to {raw.max():.4g}); check the units')
        dn = [d for d in ds[names.get('vsv') or names.get('vs') or dkey].dims
              if d not in (latn, lonn)]
        dn = dn[0] if dn else dkey
        keep = np.where((dep >= 200.0) & (dep <= 700.0))[0]
        if len(keep) < 3:
            raise SystemExit(f'{os.path.basename(path)}: only {len(keep)} shells '
                             f'between 200 and 700 km')
        sub = ds.isel({dn: keep})
        get = lambda *c: next((sub[names[k]] for k in c if k in names), None)
        vsv, vsh, vs = get('vsv'), get('vsh'), get('vs', 'v_s')
        if vsv is not None and vsh is not None:
            v = np.sqrt((2.0 * vsv ** 2 + vsh ** 2) / 3.0)
        elif vs is not None:
            v = vs
        else:
            raise SystemExit(f'{os.path.basename(path)}: no vsv/vsh and no vs among '
                             f'{sorted(ds.data_vars)}')
        mu = v.mean(dim=[latn, lonn])
        a = ((v - mu) * 100.0 / mu).clip(min=-100, max=100)
        arr = a.transpose(dn, latn, lonn).values
        self.dep = dep[keep]
        o = np.argsort(self.dep)                      # axes run in either direction
        self.dep, arr = self.dep[o], arr[o]
        self.lat = np.asarray(ds[latn].values, float)
        self.lon = np.asarray(ds[lonn].values, float)
        self.NLON = len(self.lon)
        self.DLAT = abs(self.lat[1] - self.lat[0])
        self.DLON = abs(self.lon[1] - self.lon[0])
        self.maps, self.nlev = {}, {}
        for nm, (b0, b1) in (('tz', TZ), ('melt', MELT), ('ref', REF)):
            k = (self.dep >= b0) & (self.dep < b1)
            self.nlev[nm] = int(k.sum())
            self.maps[nm] = np.nanmean(arr[k], axis=0) if k.any() else None
        ds.close()
        self.ii = {k: self._integral(m) for k, m in self.maps.items()
                   if m is not None}

    def _integral(self, m):
        p = np.nan_to_num(m, nan=0.0)
        g = np.zeros((m.shape[0] + 1, 2 * self.NLON + 1))
        g[1:, 1:] = np.cumsum(np.cumsum(np.concatenate([p, p], axis=1), 0), 1)
        return g

    def _boxes(self, la, lo, r=RADIUS_DEG):
        lo = ((np.asarray(lo, float) + 180.0) % 360.0) - 180.0
        near = lambda ax, v: np.abs(ax[None, :] - np.asarray(v, float)[:, None]).argmin(1)
        jl = near(self.lat, la)
        dj = max(1, int(round(r / self.DLAT)))
        j0 = np.clip(jl - dj, 0, len(self.lat)); j1 = np.clip(jl + dj + 1, 0, len(self.lat))
        cl = np.maximum(np.cos(np.radians(np.asarray(la, float))), 0.05)
        di = np.maximum(1, np.round(r / self.DLON / cl).astype(int))
        i0 = (near(self.lon, lo) - di) % self.NLON
        return j0, j1, i0, i0 + 2 * di + 1

    def sample(self, la, lo):
        j0, j1, i0, i1 = self._boxes(la, lo)
        return {k: ((g[j1, i1] - g[j0, i1] - g[j1, i0] + g[j0, i0])
                    / ((j1 - j0) * (i1 - i0))) for k, g in self.ii.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODEL_FILE))
    ap.add_argument('--continental', action='store_true',
                    help='restrict the rotation null to continental lithosphere')
    ap.add_argument('--nspin', type=int, default=3000)
    ap.add_argument('--nboot', type=int, default=4000)
    a = ap.parse_args()
    if a.continental:
        import contmask

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180'])
    la0 = d.lat.values.astype(float)
    lo0 = d.lon_180.values.astype(float)
    xyz0 = to_xyz(lo0, la0)
    g = groups(la0, lo0, LINK)
    ng = g.max() + 1
    print(f'{len(d)} fields in {ng} provinces at {LINK:.0f} km linkage; null is '
          + ('continent-restricted' if a.continental else 'free') + f', {a.nspin} rotations\n')

    def prov(x, gg=None):
        gg = g if gg is None else gg
        ks = [k for k in range(ng) if (gg == k).any()]
        return float(np.nanmean([np.nanmean(x[gg == k]) for k in ks]))

    rows = []
    for tag in a.models:
        f = MODEL_FILE.get(tag)
        if not f or not os.path.exists(f):
            print(f'  {NICE.get(tag, tag):11} no model file at {f}'); continue
        try:
            M = Model(f)
        except SystemExit as e:
            print(f'  {NICE.get(tag, tag):11} skipped: {e}'); continue
        s = M.sample(la0, lo0)
        if 'melt' not in s or 'ref' not in s or 'tz' not in s:
            print(f'  {NICE.get(tag, tag):11} does not sample all three intervals'); continue
        obs = dict(tz=prov(s['tz']), melt=prov(s['melt']),
                   con=prov(s['melt'] - s['ref']))
        rng = np.random.default_rng(11)
        nul = {k: [] for k in obs}
        tries = 0
        while len(nul['tz']) < a.nspin and tries < a.nspin * 60:
            tries += 1
            x = xyz0 @ haar(rng).T
            if np.degrees(np.arccos(np.clip((x * xyz0).sum(1), -1, 1))).mean() < 20.0:
                continue
            sla = np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))
            slo = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
            gg = g
            if a.continental:
                keep = contmask.is_continental(slo, sla)
                if keep.sum() < max(50, 0.25 * len(slo)):
                    continue
                sla, slo, gg = sla[keep], slo[keep], g[keep]
            q = M.sample(sla, slo)
            nul['tz'].append(prov(q['tz'], gg))
            nul['melt'].append(prov(q['melt'], gg))
            nul['con'].append(prov(q['melt'] - q['ref'], gg))
        tail = lambda o, n: (((n >= o).sum() + 1) / (len(n) + 1) if o >= np.median(n)
                             else ((n <= o).sum() + 1) / (len(n) + 1))
        # the interval is a bootstrap over provinces and needs no rotations
        gm = np.array([np.nanmean(s['tz'][g == k]) for k in range(ng)])
        bt = rng.integers(0, ng, size=(a.nboot, ng))
        q = np.percentile(np.nanmean(gm[bt], axis=1), [2.5, 97.5])
        r = dict(model=NICE.get(tag, tag),
                 n_tz=M.nlev['tz'], n_melt=M.nlev['melt'], n_ref=M.nlev['ref'],
                 tz=obs['tz'], tz_lo=q[0], tz_hi=q[1],
                 tz_p=tail(obs['tz'], np.array(nul['tz'])),
                 melt_band=obs['melt'], melt_band_p=tail(obs['melt'], np.array(nul['melt'])),
                 contrast=obs['con'], contrast_null=float(np.median(nul['con'])),
                 contrast_p=tail(obs['con'], np.array(nul['con'])))
        rows.append(r)
        print(f'  {r["model"]:11} levels {M.nlev["ref"]}/{M.nlev["melt"]}/{M.nlev["tz"]}'
              f'   TZ {r["tz"]:+.3f} [{q[0]:+.3f},{q[1]:+.3f}] p={r["tz_p"]:.3f}'
              f'   350-410 {r["melt_band"]:+.3f} p={r["melt_band_p"]:.3f}'
              f'   minus ref {r["contrast"]:+.3f} p={r["contrast_p"]:.3f}')

    t = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, 'field_clusters'
                       + ('_continental' if a.continental else '') + '.csv')
    t.to_csv(out, index=False)
    if len(t):
        print(f'\n  transition zone positive in {int((t.tz > 0).sum())} of {len(t)} volumes'
              f'; 350-410 km positive in {int((t.melt_band > 0).sum())} of {len(t)}')
        print(f'  levels are 250-350 / 350-410 / 410-660; a volume with one level in an'
              f'\n  interval is not really measuring it')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
