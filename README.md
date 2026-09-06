# Applied Remote Sensing for Boreal Silviculture

Four Google Earth Engine analyses of a managed boreal forest zone in Abitibi-Temiscamingue, Quebec, addressing questions relevant to boreal forest management: stand age structure, disturbance regime, post-harvest recovery, and carbon dynamics.

## Study zone

A ~10,000 ha (10 km x 10 km) area in Abitibi-Temiscamingue, chosen because it was verified (not assumed) to be genuine managed forest: Hansen tree cover >50% over ~90% of the box, at 10 m resolution. An earlier attempt using the full administrative region boundary (25 km, drawn from the region's official shapefile) turned out to be only ~15% real forest -- Abitibi is lake country, and an unverified polygon silently averaged in enormous amounts of water. See `src/zones.py`, `get_abitibi_analysis_zone()`.

## The four analyses

### 1. Stand age structure (`analysis/01_stand_age.py`)

~10% of the zone has a dated harvest since 2001 (Hansen `lossyear`), with a mean age of 14.6 years within that dated segment. The other ~90% is undated by this method -- not necessarily old, just outside Hansen's detection window. Mean canopy height there (17.7 m, ETH 2020, n=1.3M pixels) suggests that undated majority is substantially mature forest, consistent with a landscape structure of "mostly resting, a smaller fraction in active rotation" typical of sustainably managed public boreal forest.

**Limitation:** converting canopy height to a calibrated stand age would require Pothier & Savard (1998) -- the standard Quebec MRNF growth and yield equations -- which need a field-estimated site quality index (IQS) as input. This remote-sensing-only pipeline does not have that input, so no age number is derived from height; height is reported as a descriptive statistic only.

### 2. Disturbance attribution: harvest vs fire (`analysis/02_disturbance_attribution.py`)

Splits Hansen-detected loss into fire-caused (matches an NBAC fire perimeter for its own loss year) vs non-fire (residual, mostly harvest). Result: 1,015.5 ha harvest, 0 ha fire over 2001-2023. Verified as a real absence, not a data-matching bug, by an independent check: zero NBAC fire perimeters touch this specific zone at all across the full 1972-2023 record.

### 3. Post-harvest spectral recovery (`analysis/03_recovery_curve.py`)

Chronosequence method (space-for-time substitution): a single recent NBR composite (2024) grouped by years-since-harvest, using stands of different ages that coexist in the landscape today rather than tracking one stand over decades. Covers ages 0-24 (Hansen's detection window). Shows rapid establishment (years 2-9) and a spectral saturation plateau (years 17+), consistent with the literature. Real cohort-to-cohort variability appears in years 4-13 (not noise -- those classes have 500-2,800 pixels each), suggesting site/treatment differences between harvest cohorts, not just age, drive early recovery.

**Rigor applied:** an unfiltered version of this curve showed spurious zigzagging; age classes with fewer than 100 pixels (some had as few as 5) were dropped before drawing conclusions. This matters -- with 24 age classes and only ~10% of the landscape dated, several classes are severely undersampled even in a 10,000 ha zone.

### 4. Carbon (`analysis/04_carbon.py`)

AGB (ESA CCI Biomass, forest-masked): 80.6 Mg C/ha. SOC 0-30 cm (SoilGrids): 59.8 Mg C/ha. Cross-checked against an independent 500 m point sample at the same location, computed with an entirely different method (individual point + replicate offsets rather than a 10,000 ha aggregate): 82.8 and 58.2 Mg C/ha respectively -- close agreement across very different sampling scales, which is real evidence of measurement stability, not something engineered to match.

## Data sources

- **Hansen Global Forest Change v1.13** (`UMD/hansen/global_forest_change_2025_v1_13`): tree cover, loss year -- the backbone for the forest mask, stand age, and disturbance/recovery chronosequence.
- **Canadian National Burned Area Composite (NBAC)** (`projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530`): fire perimeters, 1972-2023, for disturbance attribution.
- **Sentinel-2 SR Harmonized**: growing-season NBR composites for the recovery curve.
- **ESA CCI Biomass v6.0** (Santoro & Cartus, 2025): aboveground carbon, converted from AGB (IPCC fraction 0.47).
- **SoilGrids 250m v2.0** (ISRIC): soil organic carbon, 0-30 cm.
- **ETH Global Canopy Height** (Lang et al., 2022): canopy height, descriptive only (see stand-age limitation above).

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
```

Each script is self-contained (no shared notebook state, no import caching issues) and prints its result directly; `03_recovery_curve.py` also saves a figure to `figures/`.

## Why this project changed shape along the way

This repo started as a cross-biome carbon comparison (Quebec, Brazil, Chile) before being refocused entirely on Abitibi for a boreal silviculture application. Several methodological corrections happened during development and are worth stating plainly rather than hiding:

- An unsupervised AlphaEarth land-cover clustering approach, used earlier to build forest masks, was found to over-segment small, homogeneous areas into artificial sub-clusters, undercounting real forest. It was replaced with a direct Hansen tree-cover threshold for this repo's zone.
- A GEDI L4B cross-check was dropped after diagnostics showed its `MU` band blends forest and non-forest by design, and that real GEDI sampling density varied sharply by location -- unreliable at this AOI scale.
- A soil-carbon scale factor was initially mis-applied (dividing by 10 when the source was already in the correct units), producing values roughly 10x too low; corrected after cross-referencing literature ranges.
- An early version of the "carbon at risk" total multiplied a masked density by the full bounding-box area instead of the true covered area, inflating results by over 20x for a peatland zone with low real peat coverage; corrected by summing density x pixel-area directly over valid pixels only.

## Next step

This repo is the applied-methods companion to the author's PhD thesis MILP model for forest road restoration -- a separate, dedicated repo -- and is not a substitute for it.

