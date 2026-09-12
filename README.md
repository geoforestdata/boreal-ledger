# Boreal Ledger

Boreal Ledger is a reproducible Google Earth Engine and Python study of boreal forest recovery after wildfire and timber harvest near Lebel-sur-Quevillon, Quebec. The project asks how disturbed forests converge toward matched reference forest conditions over time, and whether that convergence looks the same in Landsat NBR, Sentinel-2 multiband spectral space, and AlphaEarth satellite representation space.

## Main finding

Landsat NBR shows nonlinear reference-relative convergence after disturbance. Wildfire produces a substantially larger early post-disturbance NBR departure from matched reference forest than harvest, and most NBR convergence occurs during roughly the first decade. Sentinel-2 multiband distance shows a related but non-identical pattern, while AlphaEarth representation-space distance can remain elevated at ages where NBR has substantially converged.

A forest can become spectrally similar to its reference without becoming indistinguishable in every satellite representation.

## Study design

The final design uses official disturbance histories and matched reference forests:

- 76 wildfire events;
- 76 harvest events;
- 76 exact-year fire-harvest event pairs;
- event-level inference, with sites treated as subsamples;
- nearby screened reference forest for each disturbed site;
- observation years spanning Landsat 2002-2025 and Sentinel-2 / AlphaEarth 2017-2025.

The core Landsat metric is:

```text
NBR gap = reference NBR - disturbed NBR
```

A shrinking spectral gap indicates increasing similarity to matched reference forest, not complete ecological recovery.

## Public story

The public-facing scientific web publication is served from GitHub Pages:

https://geoforestdata.github.io/boreal-ledger/

The web page is the main narrative product. This repository is the reproducibility backend.

## Reproducibility

The final workflow is implemented as scripts under `analysis/`. The principal public-facing outputs are:

| Path | Purpose |
| --- | --- |
| `analysis/09_model_landsat_recovery.py` | Landsat NBR reference-relative recovery model. |
| `analysis/10_extract_multivariate_recovery_timeseries.py` | Sentinel-2 and AlphaEarth event-year extraction and feasibility audit. |
| `analysis/11_model_multivariate_recovery.py` | Multivariate recovery models for Sentinel-2 and AlphaEarth. |
| `outputs/landsat_recovery_model/` | Landsat model tables and diagnostics. |
| `outputs/multivariate_recovery/` | Sentinel-2 and AlphaEarth event-year extraction outputs. |
| `outputs/multivariate_recovery_model/` | Sentinel-2 and AlphaEarth model outputs. |
| `figures/landsat_recovery_model/` | Landsat recovery figures. |
| `figures/multivariate_recovery_model/` | Multivariate recovery figures. |
| `docs/` | GitHub Pages scientific publication. |

The earlier scripts `analysis/00_*` through `analysis/08_*` document the study extent, disturbance histories, sampling frame, reference forest validation, and time-series extraction that support the final models.

## Data sources

- Official Quebec forest disturbance and forestry intervention records used in the retained workflow.
- Landsat surface reflectance time series.
- Copernicus Sentinel-2 Surface Reflectance Harmonized.
- Google AlphaEarth annual satellite embeddings.
- Hansen Global Forest Change for dated canopy-loss context where used in earlier screening.

## Repository structure

```text
analysis/   Reproducible Python analysis scripts.
outputs/    Derived tables, model outputs, and audit summaries.
figures/    Analysis and publication figures.
docs/       Static GitHub Pages publication.
config/     Project configuration used by the workflow.
```

## Limitations

This is an observational spatial analysis, not a randomized experiment. AlphaEarth distance is a satellite representation-space measure; it is not a direct measure of biomass, biodiversity, forest structure, ecosystem function, or complete ecological recovery. Sentinel-2 and AlphaEarth observations span 2017-2025, so older apparent recovery ages combine within-event temporal change with cross-cohort comparisons. The results should be read as evidence about satellite-observed recovery patterns in this study landscape.
