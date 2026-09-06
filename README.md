# Aboveground Carbon Across Biomes

Compares aboveground carbon density (Mg C/ha) across four contrasting forest types using ESA CCI Biomass, plus a complementary structural comparison of the same zones using Google's AlphaEarth satellite embeddings.

## Research question

Where is carbon actually stored — in forest biomass, in forest soil, or in wetland soil? Which forest type stores the most? And, in concrete terms, how much carbon is at risk if a wetland is lost?

## Zones

| Zone | Type | Location |
|---|---|---|
| Abitibi | Boreal managed forest | Quebec, Canada |
| Tapajos | Intact tropical rainforest | Para, Brazil |
| Alerce Costero | Native temperate forest | Los Rios, Chile |
| Radiata Biobio | Even-aged Pinus radiata plantation | Biobio, Chile |

Bounding boxes (~20-25 km) are illustrative placeholders centered on well-known sites for each forest type — see `src/zones.py` to adjust with more precise boundaries.

## Data

- **ESA CCI Biomass v6.0** (Santoro & Cartus, 2025): continuous, gap-free annual AGB maps (2007, 2010, 2015–2022), converted to carbon (IPCC default fraction 0.47). Community-curated GEE asset: `projects/sat-io/open-datasets/ESA/ESA_CCI_AGB`.
- **AlphaEarth Satellite Embedding** (Google DeepMind): 64-band, 10 m, annual since 2017. Used for unsupervised forest-mask identification and a complementary structural comparison between zones — not for carbon estimation. GEE asset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- **Hansen Global Forest Change v1.13**: used to label which AlphaEarth cluster is "forest" (highest mean 2000 tree cover) and, via the `lossyear` band, to estimate years since last harvest for the radiata plantation. GEE asset: `UMD/hansen/global_forest_change_2025_v1_13`.
- **ETH Global Canopy Height** (Lang et al., 2022): 10 m canopy top height, 2020 snapshot, cross-referenced against an illustrative Pinus radiata height-age curve to estimate stand age. GEE asset: `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1`.
- **SoilGrids 250m v2.0** (ISRIC): soil organic carbon stock, 0-30 cm depth. Official GEE asset: `projects/soilgrids-isric/ocs_mean`.
- **PEATGRIDS** (Widyastuti et al., 2025): global peat thickness and carbon-stock model, used to locate a real peatland within Abitibi-Temiscamingue via a grid search rather than a guessed coordinate. GEE asset: `projects/sat-io/open-datasets/PEATGRIDS/C_STOCK_MGC_PER_M2` (band `C_STOCK_MgC_per_m2_MEAN`, converted from Mg C/m2 to Mg C/ha).
- **IPCC default factors**: root:shoot ratios (belowground biomass) and deadwood/litter fractions by biome type, applied to AGB — see `src/carbon_sources.py` for the exact values and source table.

## Why GEDI L4B was dropped

An earlier version cross-checked ESA CCI against GEDI L4B (1 km gridded biomass, `LARSE/GEDI/GEDI04_B_002`). Two diagnostics ruled it out for this use case:

1. **Definitional mismatch.** GEDI L4B's `MU` band is explicitly defined as the mean biomass "including forest and non-forest" within each 1 km cell — it is not a forest-only value, so a coarse cell that is partly water, road, or clearing can't be corrected after the fact by masking; the dilution is already baked into the source pixel.
2. **Uneven real coverage.** Checking the `QF` (quality flag) and `NC`/`NS` (ground-track/footprint count) bands showed real sampling density varies sharply by zone — Tapajos averaged under 1 GEDI ground track per 1 km cell, meaning most of its "data" there was a statistical model fill-in (per GEDI's own documentation) rather than a direct measurement, while Abitibi and Alerce Costero had much denser real coverage (~5 tracks/cell). This is exactly the zone where GEDI diverged most from ESA CCI (37 vs 107 Mg C/ha) — not a coincidence.

GEDI L4A (25 m footprint-level biomass) would avoid both problems and remains a possible future addition, but was out of scope here.

## Methodology

1. Define four small AOIs, one per forest type — `src/zones.py`.
2. Forest mask per zone: unsupervised k-means clustering on AlphaEarth embeddings, cross-referenced with Hansen tree cover (2000) to identify which cluster is forest — `src/embedding_utils.py`.
3. Forest-weighted carbon: combine ESA CCI with the fine (10 m) forest mask in a single `reduceRegion`, so each 10 m forest pixel contributes the value of the (coarser, 100 m) cell it falls in, weighted proportionally rather than by a hard threshold — `forest_weighted_mean_carbon`.
4. Comparison chart across the four zones.
5. Forest-only AlphaEarth embedding signature per zone (same mask as carbon) and cosine similarity between zones — computed over forest pixels only, since the raw bounding-box mean is dominated by regional climate/geography signal rather than forest structure (an earlier version using the unmasked mean showed two structurally different Chilean zones as 96% similar).
6. Similarity heatmap.
7. Total ecosystem carbon per forest zone: AGB + belowground biomass (IPCC root:shoot ratio) + deadwood/litter (IPCC fraction) + soil organic carbon (SoilGrids, 0-30 cm) — `src/carbon_sources.py`, `src/soil_carbon.py`.
8. Stacked bar chart of carbon by pool, per forest zone.
9. Three wetland zones (Rocuant-Andalien, a data-located Abitibi peatland, Rio Cruces) compared on soil/peat carbon alongside the forest zones. The Abitibi peatland was found via a PEATGRIDS grid search (`src/peatland_finder.py`) rather than a guessed coordinate, scanning 9 candidate boxes across the region and picking the one with the highest real peat-carbon signal. Where PEATGRIDS detects real peat (full-depth carbon stock), that value is used as the total rather than SoilGrids' 0-30 cm figure — for the Abitibi peatland this was 2,309 Mg C/ha (full depth) vs 63 Mg C/ha (0-30 cm only), i.e. the shallow layer alone misses roughly 97% of the actual stock. For the two Chilean wetlands (marsh/estuarine, not peat-forming), PEATGRIDS shows no peat signal, so the SoilGrids 0-30 cm value is used as a floor estimate instead — see `get_wetland_total_carbon` for the exact logic.
10. Dominant-pool comparison: what share of each ecosystem's carbon is aboveground vs belowground — the core forest-vs-wetland contrast.
11. Carbon at risk: total tonnes (not density) per zone, converted to CO2-equivalent — the "what's actually at stake if this is lost" framing.
12. Stand age of the radiata plantation: years since last Hansen-detected harvest, cross-checked against canopy height (ETH 2020) via an illustrative height-age curve — `src/stand_age_utils.py`.

## Wetland zones

| Zone | Location | Coordinate confidence |
|---|---|---|
| Rocuant-Andalien | Biobio, Chile (coastal wetland) | Verified against published site description |
| Abitibi peatland | Quebec, Canada (boreal peatland) | Located via PEATGRIDS grid search: highest of 9 candidates, 2,309 Mg C/ha full-depth peat carbon |
| Rio Cruces | Los Rios, Chile (freshwater wetland) | Verified against published site description |

Bounding boxes for Rocuant-Andalien and Rio Cruces were corrected after an initial version (unverified placeholders) produced implausibly low SOC results (6-14 Mg C/ha — far below typical wetland soils), which turned out to be because the boxes missed the actual wetland extent by several km. The Abitibi peatland zone was located with a data-driven grid search over PEATGRIDS (see `src/peatland_finder.py`) rather than a guessed named site, since no single documented peatland coordinate could be verified with confidence. See `src/zones.py` for exact coordinates and sourcing notes.

## Structure

```
src/                  zone definitions, carbon sources, embedding utilities
notebooks/            analysis pipeline and results
figures/              final charts
data/                 (unused in this version — no local downloads required)
```

## How to run

1. `pip install -r requirements.txt`
2. `earthengine authenticate`
3. Replace `PROJECT` in the notebook with your own GEE project.
4. This repo is mid-pivot toward a boreal-silviculture focus (stand age, disturbance attribution, recovery, carbon dynamics for Abitibi) — the notebooks referenced above from the biome-comparison phase are no longer part of the plan. A full README rewrite is pending once the four new modules are complete.

## Methodological note

Zone boundaries are illustrative bounding boxes, not validated forest-type or wetland-boundary polygons. The AlphaEarth + Hansen forest mask removes the most obvious non-forest contamination but is an unsupervised approximation, not a validated land-cover classification. ESA CCI Biomass is a single-year (2022) snapshot; no independent second source was available at this AOI scale (see above), so treat absolute carbon values as estimates from one methodology, not as cross-validated measurements. BGB and deadwood/litter use IPCC default factors by biome type, not site-calibrated values. SoilGrids' 0-30 cm depth substantially underestimates true carbon in peat-forming wetlands, where organic layers extend far deeper — wetland SOC numbers here are a floor, not a full accounting; a peatland-specific product (e.g. PEATGRIDS) would be needed for that. The "carbon at risk" framing (total CO2e per zone) represents the carbon stock, not a claim that all of it would be emitted instantly upon disturbance — actual emission depends on the disturbance type and timescale. The height-age curve for stand-age estimation is illustrative (Chile-typical, moderate site index), not calibrated to this specific stand; a pixel with no recorded Hansen loss is age-censored, not confirmed old.

