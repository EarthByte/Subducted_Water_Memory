"""Go/no-go diagnostic for the province-classification pivot.

Step 1 of province_classification_pivot_recipe.md: group the fields into
geological provinces, score each province against a matched rotated null in
three depth intervals in every tomographic volume, and see whether reproducible
classes exist before any manuscript is rewritten.

Provinces are defined without reference to the tomography, as the recipe
requires. The catalogue's `source` column carries GEOROC's own regional
compilations, which are geological groupings made by someone else for another
purpose; name variants (part1/part2, Cenozoic/Mesozoic of one belt) are merged,
and each merged group is then split into geographically disjoint pieces by
single linkage at 600 km, because a belt like the Central Asian Foldbelt spans
5,000 km and is not one mantle setting. Nothing here looks at a velocity.

A province's value in an interval is the MEDIAN over its fields of each field's
footprint mean, rather than the median over every cell in the union footprint.
The two differ when fields are unevenly spaced: the per-cell version lets one
large or high-latitude footprint dominate a province, and the recipe asks for
one observation per province.

The score is
    S = (province value - median matched null) / (1.4826 * MAD of matched null)
with the null built by rigidly rotating the whole constellation, so a province's
null is its own footprint placed elsewhere, and the continent-restricted variant
keeps only rotations that land it on continental lithosphere.

    python3 src/province_pivot.py                  # continent-matched null
    python3 src/province_pivot.py --null free      # for comparison
    python3 src/province_pivot.py --models REVEAL SPiRaL --nspin 400
"""
import argparse, os, re, sys
import numpy as np, pandas as pd, xarray as xr

try:
    import paths as P
    CAT, TOMO, OUT = P.IPV_V3, getattr(P, 'TOMO', '..'), getattr(P, 'OUT', 'out')
    REVEAL_NC, REVEALLO_NC = P.REVEAL, P.REVEALLO
except Exception:
    _D = os.environ.get('HYD_DATA', '..')
    CAT = os.path.join(_D, 'ipv_catalogue_georoc_v3.csv')
    TOMO, OUT, REVEAL_NC = _D, 'out', os.path.join(_D, 'REVEAL_vs_full.nc')
    REVEALLO_NC = os.path.join(_D, 'REVEAL_mantle_tomography', 'RevealLO.nc')

R = 6371.0088
BAND_LIMITS = [('350-410', 350.0, 410.0), ('410-520', 410.0, 520.0),
               ('520-660', 520.0, 660.0)]
BANDS = [b for b, _, _ in BAND_LIMITS]   # names only; the limits are above
RADIUS_DEG, SPLIT_KM = 5.0, 600.0
MIN_FIELDS = 3                  # below this a province is flagged unresolved
MODEL_FILE = {
    'REVEAL':    REVEAL_NC,
    'RevealLO':  REVEALLO_NC,
    'GLADM35':   os.path.join(TOMO, 'REVEAL_mantle_tomography', 'GLADM35.nc'),
    'SPiRaL':    os.path.join(TOMO, 'REVEAL_mantle_tomography', 'SPiRaL.nc'),
    'SEMUCBWM1': os.path.join(TOMO, 'REVEAL_mantle_tomography', 'SEMUCB-WM1.nc'),
}
NICE = {'GLADM35': 'GLAD-M35', 'SEMUCBWM1': 'SEMUCB-WM1'}
FAMILY = {'REVEAL': 'REVEAL', 'RevealLO': 'REVEAL', 'GLADM35': 'GLAD',
          'SPiRaL': 'SPiRaL', 'SEMUCBWM1': 'SEMUCB'}


def to_xyz(lon, lat):
    a, b = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])


def haar(rng):
    z = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r) / np.abs(np.diagonal(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def geo_group(src):
    """GEOROC's own regional compilation name, with the variants merged."""
    s = re.sub(r'^GEOROC:reduced_[0-9-]+\s*', '', src)
    s = re.sub(r'^RZZ9VM_?', '', s).replace('.csv', '')
    s = re.sub(r'_(part\d)$', '', s, flags=re.I)
    s = re.sub(r'[_ ](CENOZOIC|MESOZOIC|PROTEROZOIC|PALEOZOIC).*$', '', s, flags=re.I)
    return s.strip('_ -').replace('_', ' ').title()


def linkage(lat, lon, km):
    x = to_xyz(lon, lat)
    d = R * 2.0 * np.arcsin(np.clip(
        np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2) / 2.0, 0, 1))
    par = np.arange(len(lat))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in zip(*np.where(np.triu(d <= km, 1))):
        a, b = find(int(i)), find(int(j))
        if a != b:
            par[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(lat))])


def provinces(d):
    """Geological group, then geographically disjoint pieces within it."""
    la = d.lat.values.astype(float); lo = d.lon_180.values.astype(float)
    grp = np.array([geo_group(s) for s in d.source.values])
    name = np.empty(len(d), object)
    for g in sorted(set(grp)):
        idx = np.where(grp == g)[0]
        sub = linkage(la[idx], lo[idx], SPLIT_KM)
        multi = len(np.unique(sub)) > 1
        for k, s in enumerate(np.unique(sub)):
            m = idx[sub == s]
            # a split piece is named for where it is, not by an index
            tag = f'{g} ({np.mean(la[m]):+.0f}, {np.mean(lo[m]):+.0f})' if multi else g
            name[m] = tag
    return name


class Model:
    """One volume on its own grid, reduced to the three intervals."""
    def __init__(self, path):
        ds = xr.open_dataset(path)
        names = {k.lower(): k for k in ds.variables}
        latn = names.get('latitude') or names.get('lat')
        lonn = names.get('longitude') or names.get('lon')
        dkey = names.get('depth') or names.get('radius') or names.get('z')
        raw = np.asarray(ds[dkey].values, float)
        dep = raw / 1.0e3 if raw.max() > 1.0e5 else raw.copy()
        if dep.min() > 3.0e3 and dep.max() <= 6.5e3:
            dep = 6371.0 - dep
        if dep.max() < 400:
            raise SystemExit(f'{os.path.basename(path)}: vertical axis wrong after '
                             f'conversion ({dep.min():.0f}-{dep.max():.0f} km)')
        vv = names.get('vsv') or names.get('vs') or dkey
        dn = next((x for x in ds[vv].dims if x not in (latn, lonn)), dkey)
        keep = np.where((dep >= 300.0) & (dep <= 700.0))[0]
        sub = ds.isel({dn: keep})
        get = lambda *c: next((sub[names[k]] for k in c if k in names), None)
        vsv, vsh, vs = get('vsv'), get('vsh'), get('vs', 'v_s')
        v = (np.sqrt((2.0 * vsv ** 2 + vsh ** 2) / 3.0) if (vsv is not None and vsh is not None)
             else vs)
        if v is None:
            raise SystemExit(f'{os.path.basename(path)}: no vsv/vsh and no vs')
        mu = v.mean(dim=[latn, lonn])
        arr = ((v - mu) * 100.0 / mu).clip(min=-100, max=100).transpose(dn, latn, lonn).values
        dd = dep[keep]; o = np.argsort(dd); dd, arr = dd[o], arr[o]
        self.lat = np.asarray(ds[latn].values, float)
        self.lon = np.asarray(ds[lonn].values, float)
        self.NLON = len(self.lon)
        self.DLAT = abs(self.lat[1] - self.lat[0]); self.DLON = abs(self.lon[1] - self.lon[0])
        self.nlev, self.ii = {}, {}
        for nm, b0, b1 in BAND_LIMITS:
            k = (dd >= b0) & (dd < b1)
            self.nlev[nm] = int(k.sum())
            if k.any():
                self.ii[nm] = self._integral(np.nanmean(arr[k], axis=0))
        ds.close()

    def _integral(self, m):
        p = np.nan_to_num(m, nan=0.0)
        g = np.zeros((m.shape[0] + 1, 2 * self.NLON + 1))
        g[1:, 1:] = np.cumsum(np.cumsum(np.concatenate([p, p], axis=1), 0), 1)
        return g

    def field_means(self, la, lo):
        lo = ((np.asarray(lo, float) + 180.0) % 360.0) - 180.0
        near = lambda ax, v: np.abs(ax[None, :] - np.asarray(v, float)[:, None]).argmin(1)
        jl = near(self.lat, la)
        dj = max(1, int(round(RADIUS_DEG / self.DLAT)))
        j0 = np.clip(jl - dj, 0, len(self.lat)); j1 = np.clip(jl + dj + 1, 0, len(self.lat))
        cl = np.maximum(np.cos(np.radians(np.asarray(la, float))), 0.05)
        di = np.maximum(1, np.round(RADIUS_DEG / self.DLON / cl).astype(int))
        i0 = (near(self.lon, lo) - di) % self.NLON; i1 = i0 + 2 * di + 1
        n = (j1 - j0) * (i1 - i0)
        return {k: (g[j1, i1] - g[j0, i1] - g[j1, i0] + g[j0, i0]) / n
                for k, g in self.ii.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODEL_FILE))
    ap.add_argument('--null', choices=('continental', 'free'), default='continental')
    ap.add_argument('--nspin', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--lat-tol', type=float, default=None,
                    help='also build a latitude-matched null: for each province, '
                         'use only the rotations that land it within this many '
                         'degrees of its own latitude. The continent-restricted '
                         'null matches the DOMAIN but not the latitude, and most '
                         'continental area is at middle latitudes, so an Arctic '
                         'province is otherwise compared against a mid-latitude '
                         'null.')
    a = ap.parse_args()
    cm = None
    if a.null == 'continental':
        try:
            import contmask as cm
        except ImportError:
            raise SystemExit('\n--null continental needs contmask; use --null free '
                             'to run without it\n')

    d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180']).reset_index(drop=True)
    la0 = d.lat.values.astype(float); lo0 = d.lon_180.values.astype(float)
    pname = provinces(d)
    order = sorted(set(pname))
    pid = np.array([order.index(p) for p in pname])
    npv = len(order)
    counts = np.bincount(pid, minlength=npv)
    print(f'{len(d)} fields in {npv} provinces '
          f'({int((counts >= MIN_FIELDS).sum())} with {MIN_FIELDS} or more fields)\n')
    xyz0 = to_xyz(lo0, la0)

    def by_province(vals, ids):
        out = np.full(npv, np.nan)
        for k in range(npv):
            m = ids == k
            if m.any():
                out[k] = np.nanmedian(vals[m])
        return out

    rows = []
    for tag in a.models:
        f = MODEL_FILE.get(tag)
        if not f or not os.path.exists(f):
            print(f'  {NICE.get(tag, tag):11} no model file'); continue
        try:
            M = Model(f)
        except SystemExit as e:
            print(f'  {NICE.get(tag, tag):11} skipped: {e}'); continue
        obs = {k: by_province(v, pid) for k, v in M.field_means(la0, lo0).items()}
        nul = {k: [] for k in obs}
        rot_lat = []                       # each province's latitude per rotation
        obs_lat = np.array([np.mean(la0[pid == k]) for k in range(npv)])
        rng = np.random.default_rng(a.seed)
        tries = 0
        while len(nul[list(obs)[0]]) < a.nspin and tries < a.nspin * 80:
            tries += 1
            x = xyz0 @ haar(rng).T
            if np.degrees(np.arccos(np.clip((x * xyz0).sum(1), -1, 1))).mean() < 20.0:
                continue
            sla = np.degrees(np.arcsin(np.clip(x[:, 2], -1, 1)))
            slo = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
            ids = pid
            if cm is not None:
                keep = cm.is_continental(slo, sla)
                if keep.sum() < max(50, 0.25 * len(slo)):
                    continue
                sla, slo, ids = sla[keep], slo[keep], pid[keep]
            fm = M.field_means(sla, slo)
            for k in obs:
                nul[k].append(by_province(fm[k], ids))
            rot_lat.append(by_province(sla, ids))
        RL = np.array(rot_lat)
        for k in obs:
            N = np.array(nul[k])                     # rotations x provinces
            med = np.nanmedian(N, axis=0)
            mad = 1.4826 * np.nanmedian(np.abs(N - med), axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                S = (obs[k] - med) / mad
            Sm = np.full(npv, np.nan); nmatch = np.zeros(npv, int)
            if a.lat_tol:
                for p in range(npv):
                    sel = np.abs(RL[:, p] - obs_lat[p]) <= a.lat_tol
                    nmatch[p] = int(sel.sum())
                    if sel.sum() >= 50:
                        v = N[sel, p]
                        m_ = np.nanmedian(v)
                        s_ = 1.4826 * np.nanmedian(np.abs(v - m_))
                        if s_ > 0:
                            Sm[p] = (obs[k][p] - m_) / s_
            for p in range(npv):
                rows.append(dict(province=order[p], n_fields=int(counts[p]),
                                 model=NICE.get(tag, tag), family=FAMILY[tag],
                                 band=k, levels=M.nlev[k], value=obs[k][p],
                                 null_median=med[p], null_mad=mad[p], S=S[p],
                                 S_latmatched=Sm[p], n_latmatched=int(nmatch[p]),
                                 resolved=bool(counts[p] >= MIN_FIELDS)))
        print(f'  {NICE.get(tag, tag):11} levels ' +
              ' '.join(f'{b}:{M.nlev[b]}' for b in BANDS) +
              f'   {len(nul[list(obs)[0]])} rotations accepted')

    t = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tag_ = os.environ.get('PIVOT_TAG', '')
    out = os.path.join(OUT, f'province_scores_{a.null}{tag_}.csv')
    t.to_csv(out, index=False)
    print(f'\nwrote {out}')
    return t


if __name__ == '__main__':
    main()
