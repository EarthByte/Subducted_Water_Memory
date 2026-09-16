"""How well is the P-to-S angle actually determined?

Section 4.3 reports a median angle of 60.4 degrees beneath the volcanic fields
against a thermal reference of 63.4 and a hydrous one at or below 45, and then
says that a definitive assessment should propagate the uncertainty rather than
only threshold on it. This does that.

theta = atan2(S, P), so for independent errors on the two anomalies

    sigma_theta = sqrt(P^2 sigma_S^2 + S^2 sigma_P^2) / (P^2 + S^2)

in radians. GLAD-M35 distributes a standard deviation for each of Vp and Vs but
not their cross-covariance, so independence is assumed. That is the conservative
direction: errors from one inversion are more likely positively correlated than
not, and correlated errors partly cancel in a ratio, so the interval below is an
upper bound on the true one rather than an estimate of it.

The number worth reading is not the median angle but the last block: the share
of cells whose angle is determined well enough to tell 63.4 from 45 at all.

    python3 src/vpvs_uncertainty.py

Needs sig_p and sig_s in out/vpvs_angle_field.npz, which s21_vpvs_water.py
writes only after the change that added them; re-run it if they are missing.
"""
import os
import numpy as np
import pandas as pd

try:
    import paths as P
    OUT = P.OUT
    CAT = P.IPV_V3
except Exception:
    OUT, CAT = 'out', None

THERMAL, HYDROUS = 63.4, 45.0
SEP = 0.5 * (THERMAL - HYDROUS)      # 9.2 deg: closer than this cannot separate them


def load():
    f = os.path.join(OUT, 'vpvs_angle_field.npz')
    z = np.load(f)
    missing = [k for k in ('sig_p', 'sig_s') if k not in z.files]
    if missing:
        raise SystemExit(
            f'\n{f} has no {" or ".join(missing)}.\n'
            's21_vpvs_water.py computes them as UP and US and now saves them;\n'
            'if this file predates that change, re-run:\n\n'
            '    python3 src/s21_vpvs_water.py\n')
    return z


def sigma_theta(p, s, sp, ss):
    """First-order propagation onto atan2(s, p), in degrees."""
    d = p * p + s * s
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.degrees(np.sqrt(p * p * ss * ss + s * s * sp * sp) / d)


def summarise(name, th, sg, w=None):
    k = np.isfinite(th) & np.isfinite(sg)
    if not k.any():
        print(f'  {name:34} no cells')
        return
    t, g = th[k], sg[k]
    ww = np.ones_like(t) if w is None else w[k]
    def wmedian(x):
        """Area-weighted median. The weights have to be reordered with the
        values: perturbing the angles changes their order, and interpolating
        the new values against the old cumulative weights silently pairs each
        angle with another cell's area."""
        o = np.argsort(x)
        c = np.cumsum(ww[o]) / ww.sum()
        return float(np.interp(0.5, c, x[o]))

    med = wmedian(t)
    # the median's own interval, drawing each cell's angle within its own error
    rng = np.random.default_rng(3)
    draws = [wmedian(t + rng.normal(0, g)) for _ in range(400)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    frac = float(np.average(t <= HYDROUS, weights=ww))
    res = float(np.average(g < SEP, weights=ww))
    print(f'  {name:34} {med:5.1f} deg  [{lo:5.1f}, {hi:5.1f}]   '
          f'below 45: {100 * frac:4.1f} %   determined to better '
          f'than {SEP:.1f} deg: {100 * res:4.1f} %')


def main():
    z = load()
    th, p, s = z['theta'], z['dvp'], z['dvs']
    sp, ss, good = z['sig_p'], z['sig_s'], z['good']
    lat, lon = z['lat'], z['lon']
    sg = np.where(good, sigma_theta(p, s, sp, ss), np.nan)
    thg = np.where(good, th, np.nan)
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))

    print(f'\nangle, its 95 per cent interval, and how much of the map can '
          f'separate {THERMAL} from {HYDROUS}\n')
    summarise('global, both anomalies resolved', thg, sg, w)

    if CAT and os.path.exists(CAT):
        d = pd.read_csv(CAT).dropna(subset=['lat', 'lon_180'])
        j = np.abs(lat[None, :] - d.lat.values[:, None]).argmin(1)
        i = np.abs(((lon[None, :] - d.lon_180.values[:, None] + 180) % 360) - 180).argmin(1)
        m = np.zeros(thg.shape, bool)
        m[j, i] = True
        summarise('beneath the volcanic fields', np.where(m, thg, np.nan), sg, w)

    strong = np.hypot(p, s) >= np.nanmedian(np.hypot(p, s)[good])
    summarise('strongest half by amplitude', np.where(strong, thg, np.nan), sg, w)

    q = np.nanpercentile(np.hypot(p, s)[good], 90)
    top = np.hypot(p, s) >= q
    summarise('strongest decile', np.where(top, thg, np.nan), sg, w)

    print(f'\n  A cell whose angle carries an uncertainty above {SEP:.1f} degrees '
          f'cannot be\n  assigned to either hypothesis, whichever side of 45 it '
          f'happens to fall.\n  Read the last column before the third.')


if __name__ == '__main__':
    main()
