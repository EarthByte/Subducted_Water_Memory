"""Reduce a large tomographic model to the depth-band means this paper needs.

RevealLO is five gigabytes, which is more than can be moved around, and all any
analysis here wants from a tomographic model is the mean anomaly in a handful of
depth bands on a half-degree grid. That is about ten megabytes.

Run it wherever the model lives:

    python3 extract_bands.py --file /path/to/RevealLO.nc --tag RevealLO

It writes bands_<tag>.npz next to itself. The anomaly is the lateral-mean-removed
fractional perturbation within each depth shell, in per cent, formed from the
Voigt average of the vertically and horizontally polarised shear velocities,
which is what everything else in this repository uses.
"""
import argparse, contextlib, os, warnings, numpy as np, xarray as xr


@contextlib.contextmanager
def quiet():
    """Silence the arithmetic of empty shells, never a result.

    Shells above the topography and below the core-mantle boundary are all-NaN,
    and their means are meant to come out NaN. numpy says so twice, once as a
    floating-point state and once through the warnings module, and neither is
    news. A band that ends up with nothing in it is reported on its own line.
    """
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        yield

BANDS = [(0, 100), (100, 200), (200, 300), (300, 410), (410, 520), (520, 660),
         # the base of the transition zone on its own: the joint inversion of velocity with the
         # depth of the 660 is calibrated for about 23.5 GPa, and the 520-660 mean dilutes the
         # velocity anomaly there over 140 km of depth
         (600, 660),
         (660, 720), (720, 780), (780, 880), (880, 1010), (1000, 1300),
         (1300, 1600), (1600, 2000), (2000, 2500)]


def main():
    """Reduce one velocity volume to the depth-band means and write them as one file.

    The depth axis is checked rather than assumed, since it arrives in kilometres, in
    metres or as a radius, and guessing wrongly puts every shell outside every band. One
    file is written per run, which also keeps bulk extraction clear of the endpoint
    protection that reads many small writes as ransomware.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--var', default=None,
                    help='the velocity variable to reduce, for a model that carries more than '
                         'one; the default picks the shear-wave variable as before')
    ap.add_argument('--pwave', action='store_true',
                    help='reduce the compressional speed instead of the shear speed, as the '
                         'isotropic equivalent of the transversely isotropic model')
    ap.add_argument('--res', type=float, default=0.5,
                    help='degrees; the grid the rest of the workflow uses')
    a = ap.parse_args()

    d = xr.open_dataset(a.file)
    names = {k.lower(): k for k in list(d.variables)}
    dkey = names.get('depth') or names.get('radius') or names.get('z') or 'depth'
    dep = np.asarray(d[dkey].values, float)
    raw = (float(dep.min()), float(dep.max()))
    # Depth axes arrive in kilometres, in metres, or as a radius in either unit.
    # Guessing wrongly puts every shell outside every band and yields a file with
    # one band in it, which is how this was found.
    if dep.max() > 1.0e5:                       # metres of some kind
        dep = dep / 1.0e3
    if dep.min() > 3.0e3 and dep.max() <= 6.5e3:   # radius in km
        dep = 6371.0 - dep
    if dep[0] > dep[-1]:
        dep = dep[::-1]
        _flip = True
    else:
        _flip = False
    print(f'  depth axis {dkey}: raw {raw[0]:.4g} to {raw[1]:.4g} -> '
          f'{dep.min():.0f} to {dep.max():.0f} km, {len(dep)} shells')
    if dep.max() < 400:
        raise SystemExit(f'\ndepth axis tops out at {dep.max():.0f} km after '
                         'conversion, which cannot be right for a mantle model.\n'
                         f'  raw range was {raw[0]:.4g} to {raw[1]:.4g}\n'
                         '  check the units before trusting anything downstream\n')
    la = d[names.get('latitude', names.get('lat', 'latitude'))].values.astype(float)
    lo = d[names.get('longitude', names.get('lon', 'longitude'))].values.astype(float)
    print(f'{a.tag}: {len(dep)} shells {dep.min():.0f}-{dep.max():.0f} km, '
          f'{len(la)}x{len(lo)}')

    def get(*cands):
        for c in cands:
            if c in names:
                return d[names[c]]
        return None
    vsv, vsh = get('vsv'), get('vsh')
    vs = get('vs', 'v_s')
    vpv, vph, eta = get('vpv'), get('vph'), get('eta')
    vp = get('vp', 'v_p')
    if a.pwave:
        # The isotropic equivalent of a transversely isotropic model, by the same Voigt average
        # the P-S test of this paper uses; the derivation is in the docstring of `voigt` in
        # s21_vpvs_water.py. With the Love parameters A = Vph^2, C = Vpv^2, N = Vsh^2, L = Vsv^2
        # and F = eta (A - 2L), kappa = (C + 4A + 4F - 4N)/9 and mu = (C + A + 6L + 5N - 2F)/15
        # per unit density, and Vp^2 = kappa + 4 mu/3. Setting eta = 1 collapses that to the
        # familiar (Vpv^2 + 4 Vph^2)/5, which is used where the model carries no eta.
        if vpv is not None and vph is not None:
            A, C = vph.values ** 2, vpv.values ** 2
            if vsv is not None and vsh is not None:
                N, L = vsh.values ** 2, vsv.values ** 2
                e = eta.values if eta is not None else 1.0
                FF = e * (A - 2 * L)
                kappa = (C + 4 * A + 4 * FF - 4 * N) / 9.0
                mu = (C + A + 6 * L + 5 * N - 2 * FF) / 15.0
                v = np.sqrt(np.clip(kappa + 4 * mu / 3.0, 0, None))
                print('  Voigt average of vpv and vph through the Love parameters'
                      + (' with eta' if eta is not None else ' with eta = 1'))
            else:
                v = np.sqrt((C + 4 * A) / 5.0)
                print('  Voigt average of vpv and vph, (vpv^2 + 4 vph^2)/5')
        elif vp is not None:
            v = vp.values
            print('  single compressional velocity')
        else:
            raise SystemExit('\n{} carries no compressional velocity; it has {}\n'.format(
                os.path.basename(a.file), ', '.join(sorted(names))))
        vsv = vsh = vs = None
    elif a.var:
        # A named variable overrides the shear-wave default, which is how the P-wave speed of a
        # model that carries both is reduced. Voigt averaging is for the shear case only, so a
        # named variable is taken as it stands.
        if a.var.lower() not in names:
            raise SystemExit('\n{} has no variable {!r}; it carries {}\n'.format(
                os.path.basename(a.file), a.var, ', '.join(sorted(names))))
        v = d[names[a.var.lower()]].values
        print(f'  variable {a.var} as named')
        vsv = vsh = vs = None
    elif vsv is not None and vsh is not None:
        v = np.sqrt((2 * vsv.values ** 2 + vsh.values ** 2) / 3.0)
        print('  Voigt average of vsv and vsh')
    elif vs is not None:
        v = vs.values
        print('  single shear velocity')
    else:
        raise SystemExit(f'no shear velocity found; variables are {list(d.variables)}')
    v = np.asarray(v, float)
    if v.shape[0] != len(dep):              # put depth first
        ax = int(np.argmax([s == len(dep) for s in v.shape]))
        v = np.moveaxis(v, ax, 0)
    if _flip:
        v = v[::-1]

    # A model file holds either an absolute velocity or an anomaly someone has
    # already taken the mean out of. Dividing the second kind by its own lateral
    # mean divides by very nearly zero, which produced the warning and the band
    # means of order 1e-13 that this was found by.
    flat = v.reshape(len(dep), -1)
    with quiet():
        ref = np.nanmean(flat, axis=1)
    live = np.isfinite(ref)
    if not live.any():
        raise SystemExit('\nevery depth shell is empty; the velocity variable '
                         'read as all-NaN\n')
    scale = float(np.nanmedian(np.abs(ref[live])))
    if scale < 0.5:                         # absolute Vs is 2-8 km/s, or 1e3x that
        with quiet():
            rms = float(np.sqrt(np.nanmean(flat[live] ** 2)))
        pct = rms < 0.5                     # a fraction, not already per cent
        anom = v * (100.0 if pct else 1.0)
        print(f'  field is already an anomaly: lateral mean {scale:.3g}, '
              f'rms {rms:.3g}, ' + ('scaled to per cent' if pct else 'read as per cent'))
    else:
        live &= np.abs(ref) > 0.01 * scale  # a shell whose mean is ~0 is not usable
        with quiet():
            anom = (v - ref[:, None, None]) / ref[:, None, None] * 100.0
        anom[~live] = np.nan
        print(f'  fractional anomaly about a lateral mean of '
              f'{ref[live].min():.4g} to {ref[live].max():.4g} per shell')
    if (~live).any():
        print(f'  {int((~live).sum())} of {len(dep)} shells carry no usable data')

    nlat = int(round(180.0 / a.res)) + 1
    nlon = int(round(360.0 / a.res)) + 1
    LAT = np.linspace(-90.0, 90.0, nlat)
    LON = np.linspace(-180.0, 180.0, nlon)
    lo180 = ((lo + 180.0) % 360.0) - 180.0
    order = np.argsort(lo180)
    j = np.abs(la[None, :] - LAT[:, None]).argmin(1)
    i = np.abs(lo180[order][None, :] - LON[:, None]).argmin(1)

    out, kept, missed = {}, [], []
    for b in BANDS:
        k = (dep >= b[0]) & (dep < b[1])
        if not k.any():
            missed.append(b)
            continue
        with quiet():
            m = np.nanmean(anom[k], axis=0)[:, order][np.ix_(j, i)]
        cover = float(np.isfinite(m).mean())
        if cover < 0.01:                    # shells present but nothing in them
            missed.append(b)
            continue
        out[f'{b[0]}_{b[1]}'] = m.astype(np.float32)
        kept.append(b)
        with quiet():
            mu = float(np.nanmean(m))
        print(f'  {b[0]:5d}-{b[1]:<5d} {int(k.sum()):3d} shells, '
              f'mean {mu:+.3f} per cent, {100 * cover:.0f} per cent covered')
    if missed:
        print('  empty: ' + ', '.join(f'{b[0]}-{b[1]}' for b in missed))
    if len(kept) < 4:
        raise SystemExit(f'\nonly {len(kept)} bands have shells in them, which '
                         'means the depth axis is still wrong.\n')
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.file)),
                                 f'bands_{a.tag}.npz')
    np.savez_compressed(path, lat=LAT, lon=LON,
                        bands=np.array(kept, float), tag=a.tag, **out)
    print(f'\nwrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
