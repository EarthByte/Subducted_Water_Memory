"""How fast does the transition zone lose the signature of subduction?

The enrichment curve computed alongside the hydration ages has a leak. A cell that
saw subduction at 300 Ma and again at 20 Ma is counted in the 300 Ma band, and it
is fast because of the 20 Ma event, so every old band is inflated by cells with a
younger explanation. The same leak inverted the ridge-preservation test once it
was closed, so it is closed here too: for each age band, only cells with no
subduction in any younger band are counted. What is left is the quantity the decay
law needs, which is how often the transition zone is fast given subduction at age
t and nothing since.

An exponential decaying to a floor is fitted to that, giving a time constant for
the loss and an asymptote for whatever does not decay.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import paths as P

z = np.load(os.path.join(P.OUT, 'hydration_age_map.npz'))
occ = z['occupancy']            # bands x nlat x nlon
bands = z['bands']
obs = z['observed']
lat = z['lat']
w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(z['lon'])))
base = w[obs].sum() / w.sum()
print(f'base rate of a fast transition zone: {100 * base:.1f} per cent of the surface\n')

print('given subduction at age t and NO subduction since:')
print(f'  {"band (Ma)":>14s} {"cells":>9s} {"fast TZ":>9s} {"enrichment":>11s} {"95 % band":>16s}')
rows = []
for k, (t0, t1) in enumerate(bands):
    younger = occ[:k].any(axis=0) if k else np.zeros_like(occ[0], bool)
    m = occ[k] & ~younger
    n = int(m.sum())
    if n < 300:
        print(f'  {t0:6.0f}-{t1:<7.0f} {n:9d}   too few cells with nothing since')
        continue
    wm = w[m]
    frac = wm[obs[m]].sum() / wm.sum()
    se = np.sqrt(max(frac * (1 - frac), 1e-12) / n)
    lo, hi = (frac - 1.96 * se) / base, (frac + 1.96 * se) / base
    print(f'  {t0:6.0f}-{t1:<7.0f} {n:9d} {100 * frac:8.1f}% {frac / base:11.2f} '
          f'{lo:7.2f} {hi:7.2f}')
    rows.append(dict(t0=t0, t1=t1, t_mid=0.5 * (t0 + t1), n=n, frac=frac,
                     enrichment=frac / base, sigma=1.96 * se / base))
d = pd.DataFrame(rows)

def model(t, e0, einf, tau):
    return einf + (e0 - einf) * np.exp(-t / tau)

p0 = [d.enrichment.iloc[0], d.enrichment.iloc[-1], 60.0]
popt, pcov = curve_fit(model, d.t_mid, d.enrichment, p0=p0,
                       sigma=d.sigma, absolute_sigma=True, maxfev=20000)
err = np.sqrt(np.diag(pcov))
e0, einf, tau = popt
print(f'\nfit  E(t) = E_inf + (E_0 - E_inf) exp(-t / tau)')
print(f'  E_0    {e0:6.2f} +/- {err[0]:.2f}   enrichment at zero age')
print(f'  E_inf  {einf:6.2f} +/- {err[1]:.2f}   floor that does not decay')
print(f'  tau    {tau:6.1f} +/- {err[2]:.1f} Myr   e-folding time')
res = d.enrichment - model(d.t_mid, *popt)
ss = 1 - (res ** 2).sum() / ((d.enrichment - d.enrichment.mean()) ** 2).sum()
print(f'  R^2    {ss:6.3f}')
half = tau * np.log((e0 - einf) / (0.5 * (e0 - einf))) if e0 > einf else np.nan
print(f'\n  half of the decaying excess is gone by {tau * np.log(2):.0f} Myr')
for frac_left in (0.1, 0.05):
    print(f'  {100 * frac_left:.0f} per cent of it remains at '
          f'{tau * np.log(1 / frac_left):.0f} Myr')
print(f'\n  the floor is {einf:.2f}, which is '
      f'{"above" if einf - 1.96 * err[1] > 1 else "not distinguishable from"} '
      f'unity at 95 per cent')
d['fit'] = model(d.t_mid, *popt)
d.to_csv(os.path.join(P.OUT, 'persistence_decay.csv'), index=False)
pd.DataFrame([dict(E0=e0, E0_err=err[0], Einf=einf, Einf_err=err[1],
                   tau=tau, tau_err=err[2], r2=ss)]).to_csv(
    os.path.join(P.OUT, 'persistence_decay_fit.csv'), index=False)
print(f'\nwrote persistence_decay.csv and persistence_decay_fit.csv')
