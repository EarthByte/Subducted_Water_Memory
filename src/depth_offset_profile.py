"""How the depth localisation depends on the down-dip offset.

`depth_fwer.py` corrects for the ten depths searched in one occupancy field. It
does not correct for the offset used to build that field, and the reference
offset was chosen because the observed fast transition zone preferred it. This
reads every `depth_fwer*.csv` and reports, for each offset, which volumes still
place the association at 410-520 km after the depth correction.

    python3 src/depth_fwer.py --occupancy hydration_age_map_offset150.npz --suffix _off150
    python3 src/depth_fwer.py --occupancy hydration_age_map_offset400.npz --suffix _off400
    python3 src/depth_fwer.py --occupancy hydration_age_map_offset500.npz --suffix _off500
    python3 src/depth_offset_profile.py

A claim that holds at one offset and not at its neighbours is a property of the
offset. One that holds across the range is a property of the Earth. The point of
this table is to say which of the two we have.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import paths as P
    OUT = getattr(P, 'OUT', 'out')
except Exception:
    OUT = 'out'

NICE = {'GLADM35': 'GLAD-M35', 'SEMUCBWM1': 'SEMUCB-WM1'}
ORDER = ['REVEAL', 'RevealLO', 'GLAD-M35', 'SPiRaL', 'SEMUCB-WM1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', default='410-520')
    ap.add_argument('--alpha', type=float, default=0.05)
    a = ap.parse_args()
    z0, z1 = (float(x) for x in a.band.split('-'))

    files = {}
    for f in sorted(glob.glob(os.path.join(OUT, 'depth_fwer*.csv'))):
        m = re.search(r'depth_fwer_off(\d+)\.csv$', f)
        files[float(m.group(1)) if m else 300.0] = f
    if not files:
        raise SystemExit(f'\nno depth_fwer*.csv in {OUT}\n')

    tab, allmods = {}, []
    for off, f in sorted(files.items()):
        d = pd.read_csv(f)
        d['model'] = d.model.map(lambda m: NICE.get(m, m))
        s = d[(d.z0 == z0) & (d.z1 == z1)].set_index('model')
        tab[off] = s.p_adj
        allmods += list(s.index)
    mods = [m for m in ORDER if m in set(allmods)] + \
           [m for m in dict.fromkeys(allmods) if m not in ORDER]

    w = max(len(m) for m in mods) + 2
    print(f'\nadjusted probability at {a.band} km, by down-dip offset\n')
    print(' ' * w + ''.join(f'{o:>10.0f}' for o in sorted(tab)))
    for m in mods:
        cells = []
        for o in sorted(tab):
            v = tab[o].get(m, np.nan)
            cells.append('         -' if not np.isfinite(v)
                         else f'{v:>9.3f}' + ('*' if v <= a.alpha else ' '))
        print(f'{m:<{w}}' + ''.join(cells))
    print(' ' * w + ''.join(
        f'{int(np.sum([np.isfinite(tab[o].get(m, np.nan)) and tab[o].get(m, 1) <= a.alpha for m in mods])):>10d}'
        for o in sorted(tab)))
    print(' ' * (w - 9) + 'survive:')
    print(f'\n  * p <= {a.alpha}. The reference offset is 300 km and was chosen because '
          'the\n    observed fast transition zone preferred it, so its column is the one '
          'that\n    cannot be read as an independent test.')

    rows = [dict(offset=o, model=m, p_adj=tab[o].get(m, np.nan),
                 survives=bool(np.isfinite(tab[o].get(m, np.nan))
                               and tab[o].get(m, 1) <= a.alpha))
            for o in sorted(tab) for m in mods]
    f = os.path.join(OUT, 'depth_offset_profile.csv')
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f'\nwrote {f}')


if __name__ == '__main__':
    main()
