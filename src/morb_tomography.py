"""What the tomography shows beneath the MORB sites the Dixon compilation dates.

The persistence measurement in this paper says that the fast transition zone
stays attributable to the subduction that made it for an e-folding of about
37 Myr. Every hydration age in the Dixon compilation is older than 100 Myr.
So the cold-carrier hypothesis makes a prediction that can be checked directly:
beneath ridge sites whose mantle the reconstruction hydrated 120 to 380 Myr ago
there should be no excess fast transition-zone structure at all, whatever the
chemistry says. This script measures that, at every depth band, in every
velocity model, comparing hydrated ridge sites with unhydrated ones.

Two things it is not. It is not a test of whether the water is in the transition
zone, because hydration is nearly invisible to shear velocity there (Schulze et
al., 2018) and the slab that carried it has equilibrated. And it is not a global
correlation of chemistry with velocity, which is the comparison the
paper_combination outline warns against.

The comparison population is other ridge sites, never a rotated map: a ridge
site rotated at random lands on continents and old ocean basins whose velocity
structure has nothing to do with the question. So the null is a permutation of
the hydrated label across 300 km clusters, exactly as the Dixon workflow tests
H2O/Ce, and the two tests are therefore comparable.

Writes out/morb_tomography.csv and prints the table.
"""
import os, argparse, dataclasses, numpy as np, pandas as pd, xarray as xr
import paths as P
import morb

BANDS = [(0, 100), (100, 200), (200, 300), (300, 410), (410, 520), (520, 660),
         (660, 720), (720, 780), (780, 880), (880, 1010), (1000, 1300),
         (1300, 1600), (1600, 2000), (2000, 2500)]
BAND_FILES = {'RevealLO': 'bands_RevealLO.npz', 'GLADM35': 'bands_GLADM35.npz',
              'SPiRaL': 'bands_SPiRaL.npz', 'SEMUCBWM1': 'bands_SEMUCBWM1.npz'}


def reveal_bands():
    """Depth-band means of REVEAL on the half-degree grid, cached in out/."""
    cache = os.path.join(P.OUT, 'bands_REVEAL.npz')
    if os.path.exists(cache):
        return np.load(cache)
    import plume_classifier as pc
    spec = dataclasses.replace(pc.MODELS['REVEAL'], velocity_var='voigt')
    depth, lat, lon, arr = pc.load_anomaly(spec, P.need(P.REVEAL, 'TOMO_DIR', 'REVEAL'))
    out = dict(lat=lat, lon=lon, bands=np.array(BANDS, float), tag='REVEAL')
    for b0, b1 in BANDS:
        m = (depth >= b0) & (depth < b1)
        out[f'{b0}_{b1}'] = (np.nanmean(arr[m], axis=0).astype(np.float32)
                             if m.any() else np.full(arr.shape[1:], np.nan, np.float32))
    np.savez_compressed(cache, **out)
    return np.load(cache)


def load_models():
    models = {'REVEAL': reveal_bands()}
    bdir = os.path.join(P.TOMO, 'REVEAL_mantle_tomography')
    for tag, f in BAND_FILES.items():
        p = os.path.join(bdir, f)
        if os.path.exists(p):
            models[tag] = np.load(p)
        else:
            print(f'  {tag}: {p} not found, skipped')
    return models


def sample(model, key, lon, lat):
    """Nearest-cell value of one band beneath each site."""
    glat, glon = model['lat'], model['lon']
    arr = model[key]
    j = np.clip(np.rint((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, len(glat) - 1)
    i = np.rint((lon - glon[0]) / (glon[1] - glon[0])).astype(int) % len(glon)
    return arr[j, i].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=5000)
    ap.add_argument('--no-plumes', action='store_true',
                    help='drop samples on plume-influenced segments')
    a = ap.parse_args()

    d = morb.load()
    keep = np.ones(len(d), bool)
    if a.no_plumes:
        keep = ~morb.plume_mask(d)
    d = d[keep].reset_index(drop=True)
    lon, lat = d.Longitude.values, d.Latitude.values
    models = load_models()
    print(f'{len(d)} samples, {int(d.hydrated.sum())} with a hydration age; '
          f'models: {", ".join(models)}\n')

    domains = {
        'global': np.ones(len(d), bool),
        'Atlantic-Arctic corridor': morb.corridor(d),
        'Pacific': morb.pacific(d),
    }
    rows = []
    for dom, dm in domains.items():
        cl = morb.clusters(lon[dm], lat[dm])
        hyd = d.hydrated.values[dm]
        print(f'--- {dom}: {int(dm.sum())} samples, {len(np.unique(cl))} clusters, '
              f'{int(hyd.sum())} hydrated')
        print(f'{"model":10s} {"band (km)":>10s} {"hydrated":>9s} {"not":>8s} '
              f'{"diff":>7s} {"p":>7s}   {"fast-decile share, hydr/not":>28s}')
        for tag, mdl in models.items():
            for b0, b1 in BANDS:
                key = f'{b0}_{b1}'
                if key not in mdl.files:
                    continue
                v = sample(mdl, key, lon[dm], lat[dm])
                if np.isfinite(v).sum() < 20:
                    continue
                r = morb.contrast(v, hyd, cl, n_perm=a.n_perm, n_boot=300)
                # share of sites over the fastest decile of this band, globally
                full = mdl[key]; f = np.isfinite(full)
                thr = np.percentile(full[f], 90)
                fh = np.nanmean(v[hyd] >= thr); fn = np.nanmean(v[~hyd] >= thr)
                rows.append(dict(domain=dom, model=tag, band=key, **r,
                                 fast_share_hydrated=fh, fast_share_not=fn))
                flag = '*' if r['p'] < 0.05 else ' '
                print(f'{tag:10s} {key:>10s} {r["med_in"]:9.3f} {r["med_out"]:8.3f} '
                      f'{r["diff"]:7.3f} {r["p"]:7.3f}{flag}  {fh:8.2f} {fn:8.2f}')
        print()
    out = pd.DataFrame(rows)
    suffix = '_noplumes' if a.no_plumes else ''
    out.to_csv(os.path.join(P.OUT, f'morb_tomography{suffix}.csv'), index=False)
    print(f'wrote out/morb_tomography{suffix}.csv')

    # The transition zone specifically, by hydration age class, in the corridor:
    # this is the row the combined paper quotes.
    print('\n410-520 km beneath corridor sites by hydration-age class (cluster medians)')
    dm = domains['Atlantic-Arctic corridor']
    cl = morb.clusters(lon[dm], lat[dm])
    age = d.age.values[dm]
    classes = [('none', ~np.isfinite(age)), ('100-200 Ma', (age >= 100) & (age < 200)),
               ('200-300 Ma', (age >= 200) & (age < 300)), ('300-400 Ma', age >= 300)]
    print(f'{"model":10s} ' + ' '.join(f'{c:>12s}' for c, _ in classes))
    for tag, mdl in models.items():
        v = sample(mdl, '410_520', lon[dm], lat[dm])
        cm = morb.cluster_medians(v, cl, extra=dict(age=age))
        line = []
        for c, m in classes:
            a_ = cm['age'].values
            sel = ~np.isfinite(a_) if c == 'none' else (
                (a_ >= float(c.split('-')[0])) & (a_ < float(c.split('-')[1].split()[0])))
            line.append(f'{np.median(cm["median"].values[sel]):8.3f} ({sel.sum():2d})'
                        if sel.sum() else f'{"-":>12s}')
        print(f'{tag:10s} ' + ' '.join(f'{s:>12s}' for s in line))


if __name__ == '__main__':
    main()
