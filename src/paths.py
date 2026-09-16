"""Where everything lives.

Every script resolves its inputs and outputs here. Defaults are relative to
the repository; each can be overridden by an environment variable.

    TOMO_DIR     the tomography models (REVEAL_vs_full.nc, GLADM35_mtz.nc and
                 REVEAL_mantle_tomography/{RevealLO,GLADM35,SPiRaL,SEMUCB-WM1}.nc)
                 default data/tomography; the depth-band reductions
                 bands_<model>.npz that most scripts read are shipped there
    REVEALLO_NC  the RevealLO volume itself, when not in TOMO_DIR
    GRIDS_DIR    the cumulative subducted-water grids of Dixon et al.
                 (Z22_water_grids_300km and the A24 and M25 products)
                 default data/water_grids
    H2O_DIR      the Muller et al. (2025) carrier grids; only the M25 rows of
                 the water tables need them; default data/H2O
    MODEL_DIR    plate_model_manager cache; models download into it if absent
                 default models/
    DIXON_DIR    a checkout of the Dixon et al. repository; by default the
                 three files this repository needs are read from data/morb
    PAPER_OUT    run outputs        default out/
    PAPER_FIG    finished figures   default figures/
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')

OUT = os.environ.get('PAPER_OUT', os.path.join(ROOT, 'out'))
FIG = os.environ.get('PAPER_FIG', os.path.join(ROOT, 'figures'))
TOMO = os.environ.get('TOMO_DIR', os.path.join(DATA_DIR, 'tomography'))
GRIDS = os.environ.get('GRIDS_DIR', os.path.join(DATA_DIR, 'water_grids'))
H2O = os.environ.get('H2O_DIR', os.path.join(DATA_DIR, 'H2O'))
MODELS = os.environ.get('MODEL_DIR', os.path.join(ROOT, 'models'))
HOTSPOTS = os.environ.get('HOTSPOTS_CSV',
                          os.path.join(DATA_DIR, 'hotspots_courtillot2003.csv'))

REVEAL = os.path.join(TOMO, 'REVEAL_vs_full.nc')
# RevealLO: the September 2026 revision Reveal_33s_with_LO.nc is read when
# present, the file it supersedes otherwise. Same grid. REVEALLO_NC overrides.
_LO = os.path.join(TOMO, 'REVEAL_mantle_tomography')
REVEALLO = os.environ.get('REVEALLO_NC') or next(
    (p for p in (os.path.join(_LO, 'Reveal_33s_with_LO.nc'), os.path.join(_LO, 'RevealLO.nc'))
     if os.path.exists(p)), os.path.join(_LO, 'Reveal_33s_with_LO.nc'))
IPV_V3 = os.path.join(DATA_DIR, 'catalogues', 'ipv_catalogue_georoc_v3.csv')
IPV_GVP = os.path.join(DATA_DIR, 'catalogues', 'ipv_catalogue_gvp.csv')

for d in (OUT, FIG):
    os.makedirs(d, exist_ok=True)


def need(path, var, what):
    """Fail with the name of the variable to set, rather than a traceback."""
    if not os.path.exists(path):
        raise SystemExit(
            f'\ncannot find {what}\n  looked in: {path}\n'
            f'  set {var} to the directory that holds it, for example\n'
            f'      export {var}=/path/to/that/directory\n')
    return path


def water(product, model, t):
    """One water grid. product is stored, outflux or subducted; model is A24,
    Z22 or M25; t is the reconstruction time in Ma.

    M25 is distributed by carrier rather than as a mantle total, so it is
    returned as the list of carrier files that sum to it. The other two are
    single files. Callers sum whatever they are given."""
    if model == 'M25' and product in ('stored', 'outflux'):
        stem = 'stored_water' if product == 'stored' else 'slab_outflux'
        return [os.path.join(H2O, f'Muller2025_mean_{stem}_300km_{c}',
                             f'cumulative_{stem}_{c}_{t}.nc')
                for c in ('crust_bound', 'lithosphere_bottom', 'sediment_bound')]
    if model == 'M25':
        return [os.path.join(GRIDS, 'M25_water_grids_300km_600-0Ma',
                             f'cumulative_subducted_water_mantle_{t}.nc')]
    tag = {'stored': 'cumulative_stored_water_mantle',
           'outflux': 'cumulative_slab_outflux_water_mantle',
           'subducted': 'cumulative_subducted_water_mantle'}[product]
    # the outflux files themselves drop "water" from the name in A24 only
    fname = {'stored': 'cumulative_stored_water_mantle',
             'outflux': ('cumulative_slab_outflux_mantle' if model == 'A24'
                         else 'cumulative_slab_outflux_water_mantle'),
             'subducted': 'cumulative_subducted_water_mantle'}[product]
    return [os.path.join(GRIDS, f'{model}_water_grids_300km',
                         f'{tag}_{model}', f'{fname}_{t}.nc')]


# The Dixon et al. mid-ocean ridge compilation, its per-site hydration ages in
# the three mantle reference frames, and its per-site subduction histories.
# They are read and never written.
_DIXON = os.environ.get('DIXON_DIR')
_MORB = os.path.join(_DIXON, 'scripts', 'data') if _DIXON else os.path.join(DATA_DIR, 'morb')
MORB_CSV = os.path.join(_MORB, 'MORB_H2O_Ce_with_subduction_episodes.csv')
MORB_FRAMES = os.path.join(_MORB, 'hydration_age_reference_frames.csv')
MORB_HISTORIES = os.path.join(_MORB, 'subduction_histories_variants.npz')
# Present-day lithospheric thickness, LithoRef18 (Afonso et al., 2019, GJI),
# as distributed through the GPlates portal; metres on a 0.2 degree grid.
LITHO = os.path.join(DATA_DIR, 'LithoRef18_lithospheric_thickness.nc')
# Present-day craton polygons, defined tomographically from REVEAL by Shirmard
# et al. (2025, Geoscience Frontiers; github.com/EarthByte/Craton_Boundaries).
CRATONS = os.path.join(DATA_DIR, 'Craton_Boundaries_Shirmard2025',
                       'Craton_Boundaries_Inferred.shp')

MODEL_LENGTH = {'A24': 170.0, 'Z22': 410.0, 'M25': 600.0}
MODEL_NAME = {'A24': 'Alfonso2024', 'Z22': 'Zahirovic2022', 'M25': 'Muller2025'}
