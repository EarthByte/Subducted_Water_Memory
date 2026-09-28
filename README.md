# Subducted_Water_Memory

Code, figures and derived data for

> Müller, R. D., Fichtner, A., Schiller, C. J., Mather, B., Dutkiewicz, A., & Dixon, J. E. (2026). Where subducted water resides: seismic carriers, shallow storage and removal at mid-ocean ridges. In preparation for *Earth and Planetary Science Letters*.

The paper investigates what becomes of the water that subducting plates carry into the mantle, by reading two independent records against one plate reconstruction: global seismic tomography of the transition zone, and the H₂O/Ce of mid-ocean ridge basalts erupted above mantle that the reconstruction dates as hydrated. Reconstructed slab delivery overlaps fast 410–520 km structure in five velocity models, and the overlap decays with delivery age with an e-folding time of about 37 Myr; the joint P- and S-wave pattern beneath volcanic provinces is thermal. Ridge basalts above mantle hydrated 120–380 Myr ago retain elevated H₂O/Ce, keep it beneath sites that cratonic keels later passed over, and lose it with the duration of spreading-ridge residence, with an e-folding time of about 30 Myr.

This repository holds everything needed to rebuild every figure and every supplementary table of the paper. The derived products of each analysis step are archived on Zenodo (see *Citation*), so the figures and tables rebuild from them without the tomography models or the plate model; the analysis scripts that produce those products are here too, with the inputs they need named below.

## Layout

```
src/        analysis and figure scripts (Python 3)
data/       small inputs carried with the repository (see Inputs)
figures/    the ten main-text and six supplementary figures, PDF and PNG
out/        derived products of every step (not in git; in the Zenodo archive)
checkfigs.sh   checks every PDF figure for text below 10 pt and for label collisions
```

Every script resolves its inputs and outputs through `src/paths.py`. Defaults are relative to the repository; each can be overridden by an environment variable named in that file.

## Inputs

Carried in `data/`:

- `LithoRef18_lithospheric_thickness.nc`: present-day lithospheric thickness of Afonso et al. (2019), as distributed through the GPlates portal.
- `Craton_Boundaries_Shirmard2025/`: the twenty craton outlines that Shirmard et al. (2025) delimited tomographically from REVEAL (github.com/EarthByte/Craton_Boundaries).
- `hotspots_courtillot2003.csv`: the hotspot list of Courtillot et al. (2003).
- `catalogues/`: the continental intraplate volcanic-field catalogues built from GEOROC (Lehnert et al., 2000) and the Global Volcanism Program by `src/build_ipv_catalogue.py`; `ipv_catalogue_georoc_v3.csv` is the 385-field catalogue the paper uses.
- `morb/`: the mid-ocean ridge H₂O/Ce compilation of Dixon et al. (in review) with each sample's hydration age in the published, no-net-rotation and maximum-net-rotation reference frames and the per-site subduction histories. These files are reproduced here with the permission of that study's authors and are to be cited as Dixon et al.
- `tomography/REVEAL_mantle_tomography/bands_*.npz`: depth-band means of the four published velocity models that `src/extract_bands.py` reduces from the full volumes (RevealLO, GLAD-M35, SPiRaL, SEMUCB-WM1); `out/bands_REVEAL.npz` in the archive is the same reduction of REVEAL.

Obtained from their sources when an analysis step is rerun from scratch:

- REVEAL (Thrastarson et al., 2024), as `REVEAL_vs_full.nc` in `TOMO_DIR`; GLAD-M35 (Cui et al., 2024) as `GLADM35_mtz.nc`; SPiRaL (Simmons et al., 2021) and SEMUCB-WM1 (French & Romanowicz, 2014) as NetCDF volumes in `TOMO_DIR/REVEAL_mantle_tomography/`. RevealLO is an unpublished development of REVEAL from the Seismology and Wave Physics group at ETH Zürich (the volume `Reveal_33s_with_LO.nc`, or `REVEALLO_NC`); its depth-band reduction is carried here.
- The Zahirovic et al. (2022) plate reconstruction, fetched by `plate_model_manager` into `MODEL_DIR` on first use, and the optAPM reference-frame rotation files of Tetley et al. (2019) for the two bounding frames.
- The cumulative subducted-water grids of Dixon et al. (in review) in `GRIDS_DIR`: the total, the part released above 125 km and the part bound below it, for the water-delivery field (`src/water_forward.py`) and the partition (`src/water_partition.py`).

## Rebuilding the figures from the archive

With `out/` in place from the Zenodo archive, every figure script reads what an earlier step wrote and runs in seconds:

```
pip install -r requirements.txt
python3 src/fig_water_history.py        # Figure 1
python3 src/fig_tomography.py           # Figure 2
python3 src/fig_depth_offset.py         # Figure 3
python3 src/fig_visibility_budget.py    # Figure 4
python3 src/fig_geological_test.py      # Figure S7
python3 src/fig_vpvs.py                 # Figure 5
python3 src/fig_morb_map.py             # Figure 6 (needs the plate model)
python3 src/fig_craton_passage.py       # Figure 7 (needs the plate model)
python3 src/fig_morb_corridor.py        # Figure 8
python3 src/fig_three_clocks.py         # Figure 9
python3 src/fig_schematic.py            # Figure 10 and the graphical abstract
python3 src/fig_age_map.py              # Figure S2
python3 src/fig_map.py                  # Figure S3
python3 src/fig_depth_memory.py         # Figure S4
python3 src/fig_depth_profile.py        # Figure S5
python3 src/fig_province_heatmap.py     # Figure S6
./checkfigs.sh
```

Figure S1 (`figures/fig_workflow.svg`) is a drawn diagram. `fig_craton_passage.py` reconstructs the continents and craton outlines with the plate model, which `plate_model_manager` fetches on first use. `fig_tomography.py` and `fig_map.py` also read `REVEAL_vs_full.nc` for the REVEAL panels; `fig_tomography.py` reads its cached reduction `out/tz_REVEAL.npz` unless `--refresh` is passed.

## The analysis steps

In the order the paper uses them. Each writes into `out/`; the archived products are what these commands produced.

Slab delivery and its overlap with tomography (Sections 2.2, 2.3, 3.1, 3.2; Tables S1–S4, S10, S11):

```
python3 src/mask_audit.py                       # continental mask from the age grid
python3 src/deep_time_hydration.py              # occupancy field: youngest delivery age per cell
python3 src/persistence_decay.py                # enrichment curve and its fit (Table S4)
python3 src/persistence_bootstrap.py            # spatial block bootstrap of the e-folding
python3 src/tau_sensitivity.py                  # threshold, binning, functional form (Table S3)
python3 src/occupancy_sweep.py                  # radius, offset, sinking rate, frame (Table S2)
python3 src/depth_fwer.py                       # ten depths, five models, corrected for the depth search (Table S1)
python3 src/depth_offset_profile.py             # the same across the offset range (Table S11)
python3 src/depth_decay.py                      # the enrichment curve by depth band (Figure S4)
python3 src/offset_confirm.py --offset 200      # headline results at the conservative geometry (Table S10)
```

The water-delivery field and its age budget (Sections 2.2, 3.2; Figure 4; Table S6):

```
python3 src/water_forward.py
python3 src/water_offset_sweep.py
python3 src/water_bootstrap.py
```

Volcanic provinces (Section 3.3; Figures S6 and S7; Tables S5, S7–S9):

```
python3 src/province_pivot.py                   # province scores against continent-restricted nulls
python3 src/province_classify.py                # the descriptive classes and the leave-one-out test
python3 src/province_eruption_context.py        # reconstructed eruption positions and delivery timing
python3 src/age_limit_scan.py
python3 src/depth_profile.py                    # the profile beneath the fields (Figure S5, Table S8)
python3 src/table1.py                           # volcanic fields against rotated locations, anywhere and on continents (Table S7)
python3 src/wang_test.py                        # the comparison with Wang et al. (2025)
```

P- and S-wave test (Section 3.4; Figure 5):

```
python3 src/s19_slab_provenance.py              # fast bodies and their attribution (Figure S3)
python3 src/s21_vpvs_water.py
python3 src/vpvs_uncertainty.py
```

The ridge record (Sections 2.4, 3.5; Figures 6–9; Tables S12–S15):

```
python3 src/morb_tomography.py                  # tomography beneath hydrated and unhydrated sites
python3 src/morb_tomography.py --no-plumes
python3 src/morb_kinematics.py                  # what lay above each site since its hydration
python3 src/morb_kinematics_tests.py
python3 src/morb_kinematics.py --frame NNR   && python3 src/morb_kinematics_tests.py --suffix _NNR
python3 src/morb_kinematics.py --frame maxNR && python3 src/morb_kinematics_tests.py --suffix _maxNR
python3 src/morb_release_depth.py
python3 src/water_partition.py                  # water released above and below 125 km (Table S15)
```

The rotation null is tested by `src/rotation_check.py`; `src/extract_bands.py` reduces a full velocity volume to the depth-band means the other scripts read. Each script's docstring states what it computes, what it reads and what it writes.

## Conventions

Every reconstruction is Zahirovic et al. (2022) in its mantle reference frame, with the optAPM no-net-rotation and maximum-net-rotation frames as bounds. Craton outlines are the tomographically defined set of Shirmard et al. (2025), used as present-day outlines carried back with the plates. REVEAL is read as the Voigt average of its radial and transverse components. Rotation nulls are drawn uniformly on SO(3); continental populations are tested against continent-restricted nulls; ridge-site populations are tested by permutation across 300 km clusters, following the conventions of Dixon et al. Figures carry no titles and no text below 10 pt at the printed width; `checkfigs.sh` enforces this on the PDFs.

## Licences

The code in `src/` is released under the MIT licence (`LICENSE`). The figures and the derived products in `out/`, and the catalogues in `data/catalogues/`, are released under CC BY 4.0. Third-party data in `data/` retain the licences of their sources: LithoRef18 (Afonso et al., 2019), the craton outlines (Shirmard et al., 2025), the hotspot list (Courtillot et al., 2003), the Dixon et al. compilation and the depth-band reductions of the published tomography models, which are derived from the cited models.

## Citation

Cite the paper, and the archive for the code and data:

> Müller, R. D., Fichtner, A., Schiller, C. J., Mather, B., Dutkiewicz, A., & Dixon, J. E. (2026). Subducted_Water_Memory: code, figures and derived data for "Where subducted water resides: seismic carriers, shallow storage and removal at mid-ocean ridges" (v1.0.0). Zenodo. DOI to be added on deposit.

`CITATION.cff` carries the same in machine-readable form.

## References for the inputs

Afonso, J. C., et al. (2019). *Geophysical Journal International*, 217, 1602–1628. https://doi.org/10.1093/gji/ggz094
Courtillot, V., Davaille, A., Besse, J., & Stock, J. (2003). Three distinct types of hotspots in the Earth's mantle. *Earth and Planetary Science Letters*, 205, 295–308. https://doi.org/10.1016/S0012-821X(02)01048-8
Cui, C., et al. (2024). GLAD-M35. *Geophysical Journal International*, 239, 478–502. https://doi.org/10.1093/gji/ggae270
Dixon, J. E., Müller, R. D., Dutkiewicz, A., & Mather, B. R. (in review). A 400-million-year memory of subduction in the mid-ocean ridge source. *Geology*.
French, S. W., & Romanowicz, B. A. (2014). SEMUCB-WM1. *Geophysical Journal International*, 199, 1303–1327. https://doi.org/10.1093/gji/ggu334
Lehnert, K., et al. (2000). GEOROC. *Geochemistry, Geophysics, Geosystems*, 1, 1999GC000026. https://doi.org/10.1029/1999GC000026
Shirmard, H., et al. (2025). Craton boundaries from REVEAL. github.com/EarthByte/Craton_Boundaries
Simmons, N. A., et al. (2021). SPiRaL. *Geophysical Journal International*, 227, 1366–1391. https://doi.org/10.1093/gji/ggab277
Tetley, M. G., Williams, S. E., Gurnis, M., Flament, N., & Müller, R. D. (2019). Constraining absolute plate motions since the Triassic. *Journal of Geophysical Research: Solid Earth*, 124, 7231–7258. https://doi.org/10.1029/2019JB017442
Thrastarson, S., et al. (2024). REVEAL. *Bulletin of the Seismological Society of America*, 114, 1392–1406. https://doi.org/10.1785/0120230273
Zahirovic, S., Eleish, A., Doss, S., Pall, J., Cannon, J., Pistone, M., et al. (2022). Subduction and carbonate platform interactions. *Geoscience Data Journal*, 9, 371–383. https://doi.org/10.1002/gdj3.146
