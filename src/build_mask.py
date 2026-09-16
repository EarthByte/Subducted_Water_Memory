"""Build contmask's continental-lithosphere cache without the s01 dependency.

contmask.build() imports load_model from s01_reconstruct, which pulls in
geopandas, gplately and a config module carrying paths that no longer exist. The
only thing it needs from all that is the ContinentalPolygons layer, so this
fetches that directly and reproduces the rest of build() verbatim.
"""
import os, numpy as np, pygplates
from plate_model_manager import PlateModelManager
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep

MODEL = 'muller2025'          # as config.PLATE_MODEL; the mask is present-day
RES = 0.25
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'continental_mask_0Ma.npz')

import paths as P
m = PlateModelManager().get_model(MODEL, data_dir=P.MODELS)
conts = pygplates.FeatureCollection()
for f in m.get_layer('ContinentalPolygons'):
    conts.add(pygplates.FeatureCollection(f))
print(f'{MODEL}: {len(conts)} continental features')

polys = []
for f in conts:
    g = f.get_geometry()
    if g is None:
        continue
    try:
        ll = g.to_lat_lon_list()
    except Exception:
        continue
    if len(ll) < 4:
        continue
    try:
        p = Polygon([(lo, la) for la, lo in ll])
        if not p.is_valid:
            p = p.buffer(0)
        if not p.is_empty:
            polys.append(p)
    except Exception:
        continue
print(f'{len(polys)} valid polygons')
pre = prep(unary_union(polys))

lon = np.arange(-180.0, 180.0, RES) + RES / 2
lat = np.arange(-90.0, 90.0, RES) + RES / 2
mask = np.zeros((len(lat), len(lon)), bool)
for j, la in enumerate(lat):
    row = mask[j]
    for i, lo in enumerate(lon):
        if pre.contains(Point(lo, la)):
            row[i] = True
np.savez_compressed(CACHE, lon=lon, lat=lat, mask=mask, res=RES)

# area-weighted fraction of the sphere, which is the number the paper quotes
w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))
frac = float((mask * w).sum() / w.sum())
print(f'wrote {CACHE}')
print(f'continental fraction of the surface: {100 * frac:.1f} per cent '
      f'(the manuscript quotes 40.2)')
