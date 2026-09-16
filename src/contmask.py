"""Present-day continental-lithosphere mask, rasterised once from the plate model.

Needed because a spin test that rotates a continental constellation onto ocean
basins is testing the wrong null. "Continental intraplate volcanoes sit close to
transition-zone slabs" is only interesting if it is still true when the null is
also constrained to continental lithosphere - otherwise it can be restating
"continents are near subduction zones".
"""
from __future__ import annotations

import os
import numpy as np

# Which definition of continental lithosphere the null domain uses.
#   cob      every polygon valid at 0 Ma in COBfile_1800_0.gpml (the default)
#   agegrid  cells the plate model gives no seafloor age
#   polygons rasterised from the model's ContinentalPolygons layer
# The polygon layer is a reconstruction product and has enclosed holes over
# Hudson Bay, the region east of the Caspian and part of central Europe, none of
# which is oceanic; mask_audit.py measures them. Both are kept so that every
# statistic depending on the mask can be run under either.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MASKS = {'cob': os.path.join(_HERE, 'continental_mask_0Ma_cob.npz'),
          'agegrid': os.path.join(_HERE, 'continental_mask_0Ma_agegrid.npz'),
          'polygons': os.path.join(_HERE, 'continental_mask_0Ma.npz')}
DEFINITION = os.environ.get('CONTMASK', 'cob')
if DEFINITION not in _MASKS:
    raise SystemExit(f'CONTMASK must be one of {sorted(_MASKS)}, not {DEFINITION!r}')
_CACHE = _MASKS[DEFINITION]


def build(res_deg=0.25):
    """Rebuild the cache by running build_mask.py.

    The rasterisation lives there because the earlier in-line version imported a
    plate-model loader from another project, which in turn imported a helper
    module that is not part of this repository, so calling this function raised
    ImportError rather than rebuilding anything. build_mask.py reads the
    continental polygons directly and reproduces the cached mask exactly.
    """
    import subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    if abs(res_deg - 0.25) > 1e-9:
        raise ValueError('build_mask.py rasterises at 0.25 degrees; edit its RES '
                         'to change the resolution')
    subprocess.run([sys.executable, os.path.join(here, 'build_mask.py')],
                   check=True, cwd=here)
    z = np.load(_CACHE)
    return z['lon'], z['lat'], z['mask']


def load():
    if not os.path.exists(_CACHE):
        return build()
    z = np.load(_CACHE)
    return z['lon'], z['lat'], z['mask']


def is_continental(lons, lats):
    lon, lat, mask = load()
    res = lon[1] - lon[0]
    i = np.clip(((np.asarray(lons, float) + 180.0) / res).astype(int),
                0, len(lon) - 1)
    j = np.clip(((np.asarray(lats, float) + 90.0) / res).astype(int),
                0, len(lat) - 1)
    return mask[j, i]


if __name__ == '__main__':
    lon, lat, mask = build()
    frac = mask.mean()
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))
    print(f'grid {mask.shape}, {100 * frac:.1f}% of cells, '
          f'{100 * (mask * w).sum() / w.sum():.1f}% of surface area continental')
