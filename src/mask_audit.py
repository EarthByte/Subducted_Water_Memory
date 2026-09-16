"""Audit the continental mask, and build an independent one from the age grid.

The restricted null in this paper occupies whatever the continental mask says is
continental lithosphere, so the mask is not a presentational choice. The one in
use is rasterised from muller2025's ContinentalPolygons, and that layer has holes:
enclosed regions covered by no polygon at all. Hudson Bay is the largest at about
360 square degrees, then a region east of the Caspian, then several marginal
basins, then part of central Europe. The Norwegian margin is not a hole but simply
falls outside the outline. Together the enclosed holes are 2.5 per cent of the
filled outline.

Those holes are in the model rather than in the rasterisation: on a half-degree
grid, "inside any one of the 868 polygons" and "inside their union" agree on every
cell of the globe.

ContinentalPolygons is a reconstruction product. It defines what rifts and what
gets reconstructed, and it is not a map of where the crust is continental, so
excluding Hudson Bay from a null domain is a defensible choice for its own purpose
and a poor one for ours. The alternative built here uses the same plate model's
seafloor age grid and calls a cell continental where the model gives it no
seafloor age. That needs no judgement about individual basins, and it keeps the
marginal basins that do have ages out.

Both masks are written, and every statistic that uses one should be run under
both.
"""
import os, numpy as np, xarray as xr, pygplates
from plate_model_manager import PlateModelManager
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import paths as P

RES = 0.25
OUT_AGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'continental_mask_0Ma_agegrid.npz')
lon = np.arange(-180.0, 180.0, RES) + RES / 2
lat = np.arange(-90.0, 90.0, RES) + RES / 2
w = np.cos(np.radians(lat))[:, None] * np.ones((1, len(lon)))
def frac(m):
    return 100 * (m * w).sum() / w.sum()

m = PlateModelManager().get_model('muller2025', data_dir=P.MODELS)

# ------------------------------------------------------------------ polygons
conts = pygplates.FeatureCollection()
for f in m.get_layer('ContinentalPolygons'):
    conts.add(pygplates.FeatureCollection(f))
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
    p = Polygon([(lo, la) for la, lo in ll])
    if not p.is_valid:
        p = p.buffer(0)
    if not p.is_empty:
        polys.append(p)
uni = unary_union(polys)
holes = sorted((Polygon(r).area for g in (uni.geoms if hasattr(uni, 'geoms') else [uni])
                for r in g.interiors), reverse=True)
print(f'{len(polys)} polygons, {len(holes)} enclosed holes, '
      f'{sum(holes):.0f} square degrees, '
      f'{100 * sum(holes) / (uni.area + sum(holes)):.1f} per cent of the filled outline')
print(f'  largest holes, square degrees: '
      + ', '.join(f'{h:.0f}' for h in holes[:6]))

pre = prep(uni)
poly_mask = np.zeros((len(lat), len(lon)), bool)
for j, la in enumerate(lat):
    row = poly_mask[j]
    for i, lo in enumerate(lon):
        if pre.contains(Point(lo, la)):
            row[i] = True

# ------------------------------------------------------------------ age grid
age_path = m.get_raster('AgeGrids', 0)
d = xr.open_dataset(age_path)
v = [k for k in d.data_vars if d[k].ndim == 2][0]
alat = d[d[v].dims[0]].values.astype(float)
alon = d[d[v].dims[1]].values.astype(float)
age = d[v].values.astype(float)
print(f'\nage grid {os.path.basename(age_path)}: {age.shape}, '
      f'{100 * np.isfinite(age).mean():.1f} per cent of cells carry a seafloor age')
j = np.abs(alat[None, :] - lat[:, None]).argmin(1)
i = np.abs(((alon[None, :] - lon[:, None] + 180) % 360) - 180).argmin(1)
age_mask = ~np.isfinite(age[np.ix_(j, i)])

print(f'\ncontinental fraction of the surface')
print(f'  from ContinentalPolygons : {frac(poly_mask):.2f} per cent')
print(f'  from the age grid        : {frac(age_mask):.2f} per cent')
print(f'  in the age mask only     : {frac(age_mask & ~poly_mask):.2f} per cent')
print(f'  in the polygon mask only : {frac(poly_mask & ~age_mask):.2f} per cent')

print(f'\nregions, per cent of cells called continental')
print(f'  {"region":22s} {"polygons":>9s} {"age grid":>9s}')
for nm, lo0, lo1, la0, la1 in [
        ('Hudson Bay', -95, -70, 52, 65), ('central Europe', 5, 25, 45, 55),
        ('W Scandinavia', 5, 15, 58, 70), ('Caspian and east', 48, 62, 45, 58),
        ('South China Sea', 111, 120, 12, 20), ('W Mediterranean', 2, 9, 37, 42),
        ('Siberia', 80, 110, 55, 70), ('central Australia', 125, 140, -28, -20),
        ('central Pacific', -150, -130, -20, 0)]:
    a = (lon >= lo0) & (lon <= lo1); b = (lat >= la0) & (lat <= la1)
    print(f'  {nm:22s} {100 * poly_mask[np.ix_(b, a)].mean():8.1f}% '
          f'{100 * age_mask[np.ix_(b, a)].mean():8.1f}%')

np.savez_compressed(OUT_AGE, lon=lon, lat=lat, mask=age_mask, res=RES)
print(f'\nwrote {OUT_AGE}')
