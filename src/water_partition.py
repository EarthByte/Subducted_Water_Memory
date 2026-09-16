"""Where the subducted water leaves the slab: above or below 125 km.

The subduction workflow behind the Dixon et al. water grids (Mather et al.,
2026) carries three cumulative fields on the same 0.2 degree grid in the
mantle reference frame: the total water subducted, the part released from the
slab between the trench and 125 km slab depth (slab outflux, the mantle-wedge
component), and the part still chemically bound below 125 km (stored). The
two components sum to the total. This script measures their partition,
globally and beneath the hydrated ridge sites, which is the quantity the
depth argument of the discussion rests on: the fraction of subducted water
that is released within reach of the ridge melting column.

    python3 src/water_partition.py                 # 0, 100, 200, 300 Ma
    python3 src/water_partition.py --times 0 50 100 150 200 250 300 350 400

Writes out/water_partition.csv (global totals and shares by time) and
out/water_partition_sites.csv (the cumulative shallow share at every ridge
site at 0 Ma), and prints the segment medians. The per-episode shallow share,
computed from the per-site histories over each site's own hydration window,
is morb_release_depth.py's and is reported beside these for comparison.
"""
import argparse, os, numpy as np, pandas as pd, xarray as xr
import paths as P
import morb

R = 6371.0088e3


def grid(product, t):
    files = P.water(product, 'Z22', t)
    z = None
    for f in files:
        d = xr.open_dataset(P.need(f, 'GRIDS_DIR', f'the Z22 {product} water grid at {t} Ma'))
        v = d['z'].values
        z = v if z is None else z + v
        lat, lon = d.lat.values, d.lon.values
    return z, lat, lon


def area_weights(lat, lon, shape):
    dlat = np.radians(abs(lat[1] - lat[0])); dlon = np.radians(abs(lon[1] - lon[0]))
    w = (R ** 2 * np.cos(np.radians(lat)) * dlat * dlon)[:, None] * np.ones(shape)
    if abs((lon[-1] - lon[0]) - 360.0) < 1e-6:
        w[:, -1] = 0.0                       # the duplicated meridian
    return w


def total(z, w):
    return float(np.nansum(np.where(np.isfinite(z), z, 0.0) * w))


def sample(z, lat, lon, la, lo):
    i = np.clip(np.rint((lo - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    j = np.clip(np.rint((la - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    return z[j, i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--times', type=float, nargs='+', default=[0, 100, 200, 300])
    a = ap.parse_args()

    rows = []
    for t in a.times:
        t = int(t)
        zt, lat, lon = grid('subducted', t)
        zo, _, _ = grid('outflux', t)
        zs, _, _ = grid('stored', t)
        w = area_weights(lat, lon, zt.shape)
        T, O, S = total(zt, w), total(zo, w), total(zs, w)
        m = np.isfinite(zt) & (zt > 0.01 * np.nanmax(zt))
        share = zo[m] / (zo[m] + zs[m])
        ww = w[m]
        rows.append(dict(time_Ma=t, total=T, outflux=O, stored=S,
                         outflux_share=O / T, stored_share=S / T,
                         closure=(O + S) / T,
                         cell_share_mean=float(np.average(share, weights=ww)),
                         cell_share_p10=float(np.percentile(share, 10)),
                         cell_share_p50=float(np.percentile(share, 50)),
                         cell_share_p90=float(np.percentile(share, 90))))
        print(f'{t:4d} Ma  released above 125 km {100 * O / T:5.1f} %   bound below {100 * S / T:5.1f} %'
              f'   (closure {(O + S) / T:.4f}; cellwise share p10/p50/p90 '
              f'{np.percentile(share, 10):.2f}/{np.percentile(share, 50):.2f}/{np.percentile(share, 90):.2f})')
    pd.DataFrame(rows).to_csv(os.path.join(P.OUT, 'water_partition.csv'), index=False)

    # the ridge sites, cumulative at 0 Ma
    zt, lat, lon = grid('subducted', 0); zo, _, _ = grid('outflux', 0); zs, _, _ = grid('stored', 0)
    d = morb.load()
    k = pd.read_csv(os.path.join(P.OUT, 'morb_kinematics.csv'))
    assert (k.Sample.values == d.Sample.values).all()
    so = sample(zo, lat, lon, k.Latitude.values, k.Longitude.values)
    ss = sample(zs, lat, lon, k.Latitude.values, k.Longitude.values)
    st = sample(zt, lat, lon, k.Latitude.values, k.Longitude.values)
    with np.errstate(invalid='ignore', divide='ignore'):
        share = so / (so + ss)
    out = k[['Sample', 'Latitude', 'Longitude', 'H2O_Ce', 'age', 'hydrated']].copy()
    out['cumulative_total'] = st; out['cumulative_outflux'] = so; out['cumulative_stored'] = ss
    out['shallow_share_cumulative'] = share
    ep = os.path.join(P.OUT, 'morb_release_depth_sites.csv')
    if os.path.exists(ep):
        e = pd.read_csv(ep)
        if (e.Sample.values == k.Sample.values).all():
            out['shallow_share_episode'] = e.shallow_share.values
    out.to_csv(os.path.join(P.OUT, 'water_partition_sites.csv'), index=False)

    h = k.hydrated.values
    print(f'\nhydrated sites ({int(h.sum())}): cumulative shallow share median '
          f'{np.nanmedian(share[h]):.3f}, quartiles {np.nanpercentile(share[h], 25):.3f}–'
          f'{np.nanpercentile(share[h], 75):.3f}, minimum {np.nanmin(share[h]):.3f}')
    if 'shallow_share_episode' in out:
        ee = out.shallow_share_episode.values
        print(f'  episode shallow share (morb_release_depth) median {np.nanmedian(ee[h]):.3f}, '
              f'quartiles {np.nanpercentile(ee[h], 25):.3f}–{np.nanpercentile(ee[h], 75):.3f}')
    cor = morb.corridor(k)
    segs = [('Arctic ridges, >70N', k.Latitude.values > 70),
            ('Northern MAR, 50-70N', (k.Latitude.values >= 50) & (k.Latitude.values <= 70)),
            ('Farallon band, 20-35N', (k.Latitude.values >= 20) & (k.Latitude.values < 35)),
            ('MAR 10-20N', (k.Latitude.values >= 10) & (k.Latitude.values < 20)),
            ('South Atlantic', k.Latitude.values < 0)]
    print('  by corridor segment (hydrated sites):')
    for name, m in segs:
        s = m & cor & h
        if s.sum():
            line = f'    {name:24s} n={int(s.sum()):3d}  cumulative {np.nanmedian(share[s]):.3f}'
            if 'shallow_share_episode' in out:
                line += f'  episode {np.nanmedian(out.shallow_share_episode.values[s]):.3f}'
            print(line)


if __name__ == '__main__':
    main()
