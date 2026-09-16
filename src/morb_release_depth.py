"""Does the chemistry care whether the water left the slab shallow or deep?

The subduction workflow splits the water each slab carries into what it releases
between the trench and 125 km slab depth and what stays chemically bound below
that. The two components share the geography of subduction almost exactly, which
is why this paper's carrier control found them collinear at rho 0.97 to 0.99 and
why the Dixon paper reports the same hydration age from either. Geography cannot
tell them apart. Their proportion can: the shallow share at a site varies from
two thirds to all of the delivery, set by the thermal state of the slab that
delivered it, and that variation is independent of where the site is.

If the MORB signature is wedge water, sites whose delivery was more shallow
should be wetter at a given total. If it is water that survived past 125 km and
came back, the opposite. If neither, the release depth is not what the chemistry
records. This script measures the shallow share of the water delivered during
each site's hydration episode, from the per-site histories the Dixon workflow
saved, and tests H2O/Ce against it on 300 km clusters.

Writes out/morb_release_depth.csv.
"""
import os, numpy as np, pandas as pd
import paths as P
import morb


def episode_water(times, W, age_min, age_end):
    """Water delivered between age_end and age_min at each site, from cumulative
    histories W (times x sites); cumulative grows into the past, so delivery in
    a window is the difference across it."""
    t = np.asarray(times, float)
    out = np.full(W.shape[1], np.nan)
    for j in range(W.shape[1]):
        if not np.isfinite(age_min[j]):
            continue
        k0 = np.argmin(np.abs(t - age_min[j])); k1 = np.argmin(np.abs(t - age_end[j]))
        out[j] = max(W[k0, j] - W[k1, j], 0.0)
    return out


def main():
    d = morb.load()
    h = np.load(P.need(P.MORB_HISTORIES, 'DIXON_DIR', 'the per-site water histories'))
    assert np.allclose(h['lat'], d.Latitude.values), 'histories are for another sample table'
    times = h['times']
    age_min = d['age'].values
    age_end = d['age_end_total_mantle_rate0p5'].astype(float).values
    shallow = episode_water(times, h['slab_outflux_mantle'], age_min, age_end)
    deep = episode_water(times, h['stored_water_mantle'], age_min, age_end)
    tot = shallow + deep
    share = np.where(tot > 0, shallow / np.maximum(tot, 1e-12), np.nan)
    d['shallow_episode'] = shallow; d['deep_episode'] = deep; d['shallow_share'] = share

    hyd = d.hydrated.values & np.isfinite(share)
    print(f'{int(hyd.sum())} hydrated samples with a computable split')
    print(f'shallow share of episode water: median {np.nanmedian(share[hyd]):.2f}, '
          f'IQR {np.nanpercentile(share[hyd], 25):.2f}-{np.nanpercentile(share[hyd], 75):.2f}'
          f'  (Dixon SI quotes 0.75, 0.64-0.99 for all water at all sites)')

    rows = []
    for dom, m in (('global', np.ones(len(d), bool)), ('corridor', morb.corridor(d)),
                   ('no plume segments', ~morb.plume_mask(d))):
        mm = m & hyd
        cl = morb.clusters(d.Longitude.values[mm], d.Latitude.values[mm])
        y = d.H2O_Ce.values[mm]
        for name, x in (('shallow_share', share[mm]), ('deep_episode', deep[mm]),
                        ('shallow_episode', shallow[mm]), ('total_episode', tot[mm])):
            r = morb.spearman_clusters(x, y, cl)
            rows.append(dict(domain=dom, predictor=name, **r))
            print(f'{dom:18s} rho(H2O/Ce, {name:16s}) = {r["rho"]:+.3f} '
                  f'[{r["lo"]:+.2f}, {r["hi"]:+.2f}] n={r["n"]:3d} p={r["p"]:.3f}')
        # above and below the median share, as a contrast
        med = np.nanmedian(share[mm])
        c = morb.contrast(y, share[mm] >= med, cl, n_perm=5000, n_boot=500)
        rows.append(dict(domain=dom, predictor=f'share>={med:.2f}', **c))
        print(f'{dom:18s} H2O/Ce, shallow share above/below median {med:.2f}: '
              f'{c["med_in"]:.0f} vs {c["med_out"]:.0f} (p={c["p"]:.3f}, '
              f'{c["n_in"]}/{c["n_out"]} clusters)')
    pd.DataFrame(rows).to_csv(os.path.join(P.OUT, 'morb_release_depth.csv'), index=False)
    d[['Sample', 'Latitude', 'Longitude', 'H2O_Ce', 'age', 'shallow_episode',
       'deep_episode', 'shallow_share']].to_csv(
        os.path.join(P.OUT, 'morb_release_depth_sites.csv'), index=False)
    print('wrote out/morb_release_depth.csv and morb_release_depth_sites.csv')


if __name__ == '__main__':
    main()
