# Boreal Ledger

A reproducible Earth Engine/Python experiment testing whether AlphaEarth satellite embeddings retain information about disturbance history in a managed boreal landscape near Lebel-sur-Quevillon, Quebec.

## Research Question

Can satellite embeddings distinguish wildfire from probable harvest at matched time since canopy loss, and do they retain greater disturbance-type separability than conventional Sentinel-2 spectral information?

## Study Design

The experiment uses a 30,277.84 ha study landscape near Lebel-sur-Quevillon, Quebec. Disturbance timing comes from Hansen dated canopy loss. Wildfire attribution is defined as Hansen loss spatially and temporally matched to NBAC fire polygons. Probable harvest is defined as the non-fire residual Hansen canopy-loss class; it is not directly observed.

Matched disturbance years are observed in a common 2024 feature space:

- Sentinel-2 NBR;
- Sentinel-2 spectral baseline using B2, B3, B4, B8, B11, B12, and NBR;
- AlphaEarth 64-dimensional annual satellite embeddings.

The validation uses one-to-one geographic matching, spatial-block cross-validation, and fire-interior sensitivity tests that exclude fire-perimeter edge pixels before sampling.

## Matched Ages

| Disturbance year | Years since disturbance in 2024 |
| ---: | ---: |
| 2023 | 1 |
| 2012 | 12 |
| 2005 | 19 |

## Main Result

The most defensible result is the 19-year matched comparison. Fire and probable-harvest samples remain geographically close after explicit fire-interior filtering, while AlphaEarth retains higher disturbance-type separability than the conventional spectral baselines.

| Fire-interior buffer | Matched pairs | Median pair distance | NBR AUC | Sentinel-2 AUC | AlphaEarth AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 90 m | 222 | 0.160 km | 0.380 | 0.473 | 0.736 |
| 150 m | 222 | 0.133 km | 0.366 | 0.472 | 0.666 |

The stricter 150 m buffer reduces AlphaEarth performance, but the result does not collapse to the Sentinel-2 spectral baseline.

## Why The Validation Matters

The initial AlphaEarth comparison showed spatial confounding and sensitivity to fire-perimeter edge proximity. The final design therefore uses:

- one-to-one geographic matching between wildfire and probable-harvest samples;
- spatial-block cross-validation instead of random pixel splits;
- fire-interior masks before sampling;
- a stronger Sentinel-2 multiband baseline in addition to NBR.

This sequence is part of the result: the claim is not that a single high AUC proves disturbance history, but that separability persists under stricter spatial controls for the 19-year comparison.

## Interpretation

AlphaEarth embeddings retained greater disturbance-type separability than conventional spectral baselines for the 19-year matched comparison in this study landscape.

This does not mean that AlphaEarth measures field ecological variables, live biomass, or causal disturbance effects. It also does not mean that probable harvest is directly observed or that AlphaEarth universally outperforms Sentinel-2.

The 12-year comparison is interesting but secondary: it has fewer matched pairs and weaker geographic matching, and Sentinel-2 becomes comparable to or stronger than AlphaEarth under the stricter fire-interior buffers. The 1-year comparison is modest and is not the central result.

## Workflow

AOI selection -> disturbance attribution -> matched samples -> 2024 observations -> geographic matching -> spatial-block cross-validation -> fire-interior sensitivity.

## Key Scripts

| Path | Purpose |
| --- | --- |
| `analysis/recovery_by_disturbance.py` | Builds the 2024 Sentinel-2 NBR baseline by disturbance type and matched disturbance year. |
| `analysis/alphaearth_disturbance_comparison.py` | Extracts AlphaEarth embeddings and compares the first balanced sample against the NBR baseline. |
| `analysis/validate_alphaearth_disturbance.py` | Adds spatial diagnostics, geographic matching, Sentinel-2 spectral baseline, spatial-block CV, and permutation checks. |
| `analysis/final_2005_spatial_audit.py` | Audits the 2005 / 19-year matched sample for edge, clustering, and mask-overlap artifacts. |
| `analysis/fire_interior_sensitivity.py` | Repeats the matched classification using fire-interior masks at 60 m, 90 m, and 150 m. |

## Key Outputs

| Path | Purpose |
| --- | --- |
| `outputs/recovery_by_disturbance.csv` | NBR summaries by disturbance type and year. |
| `outputs/alphaearth_centroid_distances.csv` | Fire-harvest centroid distances in embedding and PCA space. |
| `outputs/alphaearth_group_dispersion.csv` | Within-group dispersion in embedding space. |
| `outputs/alphaearth_classification.csv` | Initial AlphaEarth classifier results. |
| `outputs/nbr_classification_baseline.csv` | Initial NBR-only classifier baseline. |
| `outputs/alphaearth_vs_nbr.csv` | Initial AlphaEarth vs NBR comparison. |
| `outputs/spatial_separation_diagnostics.csv` | Geographic separation diagnostics for the first balanced sample. |
| `outputs/alphaearth_matching_summary.csv` | Geographic matching summary by disturbance age. |
| `outputs/alphaearth_spatially_matched_classification.csv` | AlphaEarth results after geographic matching and spatial-block CV. |
| `outputs/nbr_spatially_matched_classification.csv` | NBR baseline after geographic matching and spatial-block CV. |
| `outputs/sentinel2_spectral_baseline.csv` | Sentinel-2 multiband baseline after geographic matching and spatial-block CV. |
| `outputs/alphaearth_permutation_test.csv` | Label-permutation sanity check under spatial CV. |
| `outputs/final_model_comparison.csv` | NBR, Sentinel-2, and AlphaEarth comparison for the matched sample. |
| `outputs/final_2005_spatial_audit.csv` | Final 2005 spatial audit metrics. |
| `outputs/fire_interior_sensitivity.csv` | Fire-interior sample availability and geographic matching metrics. |
| `outputs/fire_interior_model_comparison.csv` | Final fire-interior model comparison. |

## Figures

| Path | Purpose |
| --- | --- |
| `figures/recovery_by_disturbance.png` | NBR by disturbance type and years since disturbance. |
| `figures/alphaearth_pca.png` | PCA view of AlphaEarth embedding samples. |
| `figures/alphaearth_sample_locations.png` | Sample geography by matched disturbance age. |
| `figures/final_alphaearth_validation.png` | Strict matched comparison of NBR, Sentinel-2, and AlphaEarth. |
| `figures/final_2005_matched_pairs_map.png` | Diagnostic map of 2005 matched fire-harvest pairs. |
| `figures/fire_interior_sensitivity.png` | AUC sensitivity to fire-interior buffer distance. |

## Reproducibility

Create an environment with the packages in `requirements.txt`, authenticate Earth Engine, then run:

```bash
python analysis/recovery_by_disturbance.py
python analysis/alphaearth_disturbance_comparison.py
python analysis/validate_alphaearth_disturbance.py
python analysis/final_2005_spatial_audit.py
python analysis/fire_interior_sensitivity.py
```

## Limitations

- Probable harvest is inferred from non-fire residual Hansen canopy loss; it is not directly observed.
- This is an observational spatial comparison, not a randomized treatment design.
- Only three matched fire years are available.
- Residual site-condition confounding may remain after geographic matching.
- AlphaEarth embeddings are representation features, not direct ecological variables.
- Classification success indicates distinguishability, not mechanism.

## References

- Hansen, M. C., Potapov, P. V., Moore, R., et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342, 850-853.
- Natural Resources Canada, Canadian Forest Service. National Burned Area Composite, 1972-2023.
- Copernicus Sentinel-2 MSI Surface Reflectance Harmonized collection.
- Google DeepMind. AlphaEarth satellite embedding annual V1 collection.
