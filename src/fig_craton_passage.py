"""Cratons passing over the hydrated ridge sites: the kinematic history on maps.

Six reconstruction times in the mantle reference frame of Zahirovic et al.
(2022). The ridge sites are fixed in that frame, so they sit at their
present-day positions in every panel; the continents move over them. Each
panel shows the continental lithosphere of the model at that time (grey),
the cells whose present-day lithosphere is at least 200 km thick (dark grey)
and the Shirmard et al. (2025) craton outlines carried with their plates
(red), the resolved mid-ocean ridges (blue) and subduction zones (black
barbed), and the hydrated sites coloured by H2O/Ce, sites not yet hydrated at
that time and unhydrated sites in grey. The default times follow the Farallon
band of the Mid-Atlantic Ridge: hydration at 220 Ma, beneath the Wyoming
craton at 160 Ma, beneath the Appalachian margin at 100 Ma, North America
clearing the band at 85 Ma, the ridge arriving at 30 Ma, and today.

Reuses continental_cells, plate_ids, craton_flags and reconstruct_xyz from
morb_kinematics.py, so what is drawn is what Table S13 counted.

    python3 src/fig_craton_passage.py
    python3 src/fig_craton_passage.py --times 300 220 160 100 30 0 --region atlantic
"""
import argparse, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygplates
from gplately import PlateReconstruction
import paths as P
import figstyle as F
import morb
from morb_kinematics import (continental_cells, plate_ids, craton_flags, reconstruct_xyz,
                             to_xyz, KEEL)

F.apply(10.0)
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='zahirovic2022')
ap.add_argument('--times', type=float, nargs='+', default=[220, 160, 100, 85, 30, 0])
ap.add_argument('--region', default='atlantic', choices=('atlantic', 'global'),
                help='atlantic: 120W-60E, 60S-85N; global: whole Earth')
ap.add_argument('--suffix', default='')
a = ap.parse_args()

from plate_model_manager import PlateModelManager
pmm = PlateModelManager().get_model(a.model, data_dir=P.MODELS)
rot = pmm.get_rotation_model()
recon = PlateReconstruction(rot, topology_features=pmm.get_topologies(),
                            static_polygons=pmm.get_static_polygons())
rotation_model = pygplates.RotationModel(rot)

d = morb.load()
k = pd.read_csv(os.path.join(P.OUT, 'morb_kinematics.csv'))
assert (k.Sample.values == d.Sample.values).all()

clon, clat, cthick = continental_cells()
cell_pid = plate_ids(pmm.get_static_polygons(), rot, clon, clat)
ccraton = craton_flags(clon, clat)
cell_xyz = to_xyz(clon, clat)
print(f'{len(clon)} continental cells, {int(ccraton.sum())} inside a craton outline, '
      f'{int((cthick >= KEEL).sum())} at least {KEEL:.0f} km thick')


def lonlat(xyz):
    return np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])), np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))


if a.region == 'atlantic':
    extent, proj = (-120, 60, -60, 85), ccrs.Robinson(central_longitude=-30)
    ncol, size = 2, (16.0 * F.CM, 19.5 * F.CM)
else:
    extent, proj = None, ccrs.Robinson(central_longitude=-30)
    ncol, size = 2, (16.0 * F.CM, 17.0 * F.CM)
nrow = int(np.ceil(len(a.times) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=size, subplot_kw=dict(projection=proj))
axes = np.atleast_1d(axes).ravel()
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.07, wspace=0.03, hspace=0.05)
geo = ccrs.PlateCarree()
norm = matplotlib.colors.Normalize(100, 400)
from cmcrameri import cm as ccm

for ax, t in zip(axes, a.times):
    ax.set_global() if extent is None else ax.set_extent(extent, crs=geo)
    ax.spines['geo'].set_linewidth(0.8)
    rxyz, ok = reconstruct_xyz(rotation_model, cell_xyz, cell_pid, float(t))
    lo, la = lonlat(rxyz[ok])
    th, cr = cthick[ok], ccraton[ok]
    ax.scatter(lo, la, s=1.2, c=F.LAND, marker='s', lw=0, transform=geo, zorder=1)
    ax.scatter(lo[th >= KEEL], la[th >= KEEL], s=1.2, c='#b9b4ad', marker='s', lw=0,
               transform=geo, zorder=2)
    ax.scatter(lo[cr], la[cr], s=1.2, c='#d9a79a', marker='s', lw=0, transform=geo, zorder=3)
    # plate boundaries of the model at t
    try:
        mor = recon.tessellate_mid_ocean_ridges(float(t), tessellation_threshold_radians=0.01,
                                                ignore_warnings=True)
        if mor is not None and len(mor):
            ax.scatter(mor[:, 0], mor[:, 1], s=0.5, c=F.BLU, lw=0, transform=geo, zorder=4)
        sz = recon.tessellate_subduction_zones(float(t), tessellation_threshold_radians=0.01,
                                               ignore_warnings=True)
        if sz is not None and len(sz):
            ax.scatter(sz[:, 0], sz[:, 1], s=0.5, c=F.INK, lw=0, transform=geo, zorder=4)
    except Exception as e:
        print(f'  no boundaries at {t:.0f} Ma: {e}')
    # the sites: hydrated by time t and coloured by H2O/Ce; the rest grey
    lit = k.hydrated.values & (k.age.values >= t)
    ax.scatter(k.Longitude[~lit], k.Latitude[~lit], s=4, c='#c8c8c8', lw=0, transform=geo,
               zorder=5)
    sc = ax.scatter(k.Longitude[lit], k.Latitude[lit], c=k.H2O_Ce[lit], s=6, cmap=ccm.lajolla,
                    norm=norm, lw=0.2, edgecolor=F.INK, transform=geo, zorder=6)
    ax.text(0.02, 0.97, f'{t:.0f} Ma', transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold', zorder=7)
    print(f'{t:6.0f} Ma  {int(lit.sum())} sites hydrated by then, '
          f'{int(ok.sum())} continental cells reconstructed')
for ax in axes[len(a.times):]:
    ax.set_visible(False)

cb = fig.colorbar(sc, ax=list(axes[:len(a.times)]), orientation='horizontal', fraction=0.018,
                  pad=0.02, aspect=45, shrink=0.6, ticks=[100, 200, 300, 400])
cb.set_label('H$_2$O/Ce of sites hydrated by that time')
F.check(fig)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(P.FIG, f'fig_craton_passage{a.suffix}.{ext}'), bbox_inches='tight',
                dpi=400)
print(f'wrote figures/fig_craton_passage{a.suffix}.pdf and .png')
