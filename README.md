# Applied Remote Sensing for Boreal Silviculture

Four Google Earth Engine analyses of a managed boreal forest zone in Abitibi-Temiscamingue, Quebec, addressing questions relevant to boreal forest management: directly dated canopy loss, disturbance attribution, probable post-harvest spectral recovery, and carbon stocks.

## Study zone

A ~10,000 ha (10 km x 10 km) area in Abitibi-Temiscamingue, chosen because it was verified (not assumed) to be genuine managed forest: Hansen tree cover >50% over ~90% of the box, at 10 m resolution. An earlier attempt using the full administrative region boundary (25 km, drawn from the region's official shapefile) turned out to be only ~15% real forest -- Abitibi is lake country, and an unverified polygon silently averaged in enormous amounts of water. See `src/zones.py`, `get_abitibi_analysis_zone()`.

## The four analyses

### 1. Stand age structure (`analysis/01_stand_age.py`)

~10% of the zone has directly dated Hansen canopy loss since 2001 (`lossyear`), with a mean of 14.6 years since loss within that dated segment. The other ~90% is undated by this method -- not necessarily old, just outside Hansen's detection window. ETH Global Canopy Height 2020 is used separately as a structural proxy: mean canopy height on forest pixels is 17.7 m (n=1.3M pixels), consistent with a well-developed canopy over much of the undated area.

**Limitation:** canopy height is not converted into stand age or silvicultural maturity. A calibrated age estimate would require species/site-specific growth relationships and a field-estimated site quality index (IQS), which this remote-sensing-only pipeline does not have.

### 2. Disturbance attribution: harvest vs fire (`analysis/02_disturbance_attribution.py`)

Splits Hansen-detected loss into fire-caused (matches an NBAC fire perimeter for its own loss year) vs non-fire residual loss. Result: 1,015.5 ha non-fire loss, 0 ha fire over 2001-2023. Verified as a real absence, not a data-matching bug, by an independent check: zero NBAC fire perimeters touch this specific zone at all across the full 1972-2023 record. Non-fire loss is interpreted as probable harvest in this managed forest context, but it is not a direct harvest observation.

### 3. Probable post-harvest spectral recovery chronosequence (`analysis/03_recovery_curve.py`)

Chronosequence method (space-for-time substitution): a single Sentinel-2 NBR composite from 2024 grouped by years since Hansen canopy loss, using pixels disturbed in different years that coexist in the landscape today rather than tracking the same stands over decades. Covers ages 0-24 (Hansen's detection window). NBR represents spectral vegetation recovery, not biomass recovery, structural forest recovery, or maturity. Because the separate NBAC diagnostic finds no mapped fires in this AOI, the curve is described as probable post-harvest recovery, but the script itself uses all Hansen loss pixels in the age window.

**Rigor applied:** an unfiltered version of this curve showed spurious zigzagging; age classes with fewer than 100 pixels (some had as few as 5) were dropped before drawing conclusions. This matters -- with 24 age classes and only ~10% of the landscape dated, several classes are severely undersampled even in a 10,000 ha zone.

### 4. Carbon (`analysis/04_carbon.py`)

Aboveground biomass is product-dependent in this AOI, so the repo does not present ESA CCI as a single definitive AGB estimate.

| Source | Scope | Biomass | Carbon |
| --- | --- | ---: | ---: |
| ESA CCI Method C | Forested AOI, Hansen forest-fraction weighted | 171.9 Mg biomass/ha | 80.8 Mg C/ha |
| SCANFI v1.2 | Forested AOI | 68.4 Mg biomass/ha | 32.2 Mg C/ha |
| MRNF Quebec | Productive stands covered by MRNF (~2,489.6 ha, ~25% of AOI) | 103.0 Mg biomass/ha | 51.1 Mg C/ha |
| SoilGrids | Whole AOI, 0-30 cm | NA | 59.8 Mg C/ha |

ESA CCI is the high mapped AGB estimate; SCANFI is the low mapped AGB estimate. MRNF Quebec lies between them on the productive-forest subset and is the most locally relevant provincial inventory-derived reference for productive stands, but it is not ground truth for the full AOI.

The common-footprint diagnostic (`analysis/10_agb_mrnf_common_footprint.py`) shows that the divergence is not mainly caused by comparing different surfaces. On the identical MRNF productive-stand footprint, ESA remains positively biased relative to MRNF, while SCANFI is negatively biased; SCANFI has lower absolute bias and RMSE relative to MRNF.

SoilGrids is reported separately as soil organic carbon stock, 0-30 cm. AGB and SOC are separate mapped pools in this repo. The workflow does not present AGB + SOC as total ecosystem carbon or total forest carbon.

## Data sources

- **Hansen Global Forest Change v1.13** (`UMD/hansen/global_forest_change_2025_v1_13`): tree cover and directly dated canopy loss since 2001 -- the backbone for the forest mask, dated loss, and recovery chronosequence.
- **Canadian National Burned Area Composite (NBAC)** (`projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530`): fire perimeters, 1972-2023, for fire overlap vs residual non-fire loss.
- **Sentinel-2 SR Harmonized**: growing-season 2024 NBR composite for spectral recovery.
- **ESA CCI Biomass v6.0** (Santoro & Cartus, 2025): aboveground biomass dry matter, converted to AGB carbon with a 0.47 carbon fraction and weighted by explicit Hansen forest fraction.
- **SoilGrids 250m v2.0** (ISRIC): organic carbon stock 0-30 cm (`ocs_mean`), reported as Mg C/ha.
- **ETH Global Canopy Height** (Lang et al., 2022): canopy height, descriptive only (see stand-age limitation above).
- **SCANFI v1.2** (Natural Resources Canada / Canadian Forest Service): live aboveground dry tree biomass, used as an independent mapped-product cross-check in `analysis/06_agb_crosscheck_scanfi.py`.

## How to run

```
pip install -r requirements.txt
earthengine authenticate
```

Edit `PROJECT` at the top of each script in `analysis/` to your own GEE project, then run each independently:

```
cd analysis
python 01_stand_age.py
python 02_disturbance_attribution.py
python 03_recovery_curve.py
python 04_carbon.py
python 05_agb_sensitivity.py
python 06_agb_crosscheck_scanfi.py
python 07_agb_product_diagnostics.py
python 08_agb_crosscheck_mrnf.py
python 10_agb_mrnf_common_footprint.py
```

Each script is self-contained (no shared notebook state, no import caching issues) and prints its result directly; `03_recovery_curve.py` also saves a figure to `figures/`. `05_agb_sensitivity.py` documents the ESA CCI masking/resolution sensitivity; `06`, `07`, `08`, and `10` document the AGB cross-product diagnostics behind the consolidated carbon table. `09_agb_crosscheck_ntems.py` is exploratory and is not required for the main workflow.

## References and context

- ESA CCI Biomass v6.0: Santoro & Cartus (2025), aboveground biomass product; Earth Engine catalog: https://developers.google.com/earth-engine/datasets/catalog/ESA_CCI_Above_Ground_Biomass_V6_0
- SoilGrids 2.0: Poggio et al. (2021), global soil information with quantified uncertainty; ISRIC documentation: https://docs.isric.org/globaldata/soilgrids/
- SCANFI: Spatialized Canadian National Forest Inventory, Natural Resources Canada / Canadian Forest Service; Earth Engine catalog: https://developers.google.com/earth-engine/datasets/catalog/projects_gcpm041u-lemur_assets_scanfi_v12_SCANFI_v1_2
- Boreal carbon context: use regional and plot-based literature only as ecological context. Similarity to published ranges is not validation; methodological validation here comes from dimensional checks, explicit spatial weighting, and cross-product comparison.

## Why this project changed shape along the way

This repo started as a cross-biome carbon comparison (Quebec, Brazil, Chile) before being refocused entirely on Abitibi for a boreal silviculture application. Several methodological corrections happened during development and are worth stating plainly rather than hiding:

- An unsupervised AlphaEarth land-cover clustering approach, used earlier to build forest masks, was found to over-segment small, homogeneous areas into artificial sub-clusters, undercounting real forest. It was replaced with a direct Hansen tree-cover threshold for this repo's zone.
- A GEDI L4B cross-check was dropped after diagnostics showed its `MU` band blends forest and non-forest by design, and that real GEDI sampling density varied sharply by location -- unreliable at this AOI scale.
- A soil-carbon scale factor was initially mis-applied (dividing by 10 when the source was already in the correct units), producing values roughly 10x too low; corrected after cross-referencing literature ranges.
- An early version of the "carbon at risk" total multiplied a masked density by the full bounding-box area instead of the true covered area, inflating results by over 20x for a peatland zone with low real peat coverage; corrected by summing density x pixel-area directly over valid pixels only.

## Next step

This repo is the applied-methods companion to the author's PhD thesis MILP model for forest road restoration -- a separate, dedicated repo -- and is not a substitute for it.
