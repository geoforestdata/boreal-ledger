# Aboveground Carbon Across Biomes

Compares aboveground carbon density (Mg C/ha) across four contrasting forest types using two independent satellite-derived sources, plus a complementary structural comparison using Google's AlphaEarth satellite embeddings.

## Research question

How does aboveground carbon density differ across a boreal managed forest, an intact tropical rainforest, a native temperate forest, and an even-aged industrial plantation — and do two independent remote-sensing sources agree?

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
- **GEDI L4B** (1 km gridded AGBD): spaceborne lidar, aggregated across the mission period — a cross-check, not a like-for-like year match with ESA CCI. GEE asset: `LARSE/GEDI/GEDI04_B_002` (a single Image, not an ImageCollection).
- **AlphaEarth Satellite Embedding** (Google DeepMind): 64-band, 10 m, annual since 2017. Used for a complementary structural comparison (embedding similarity between zones), not for carbon estimation. GEE asset: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.

## Methodology

1. Define four small AOIs, one per forest type — `src/zones.py`.
2. Forest mask per zone: unsupervised k-means clustering on AlphaEarth embeddings, cross-referenced with Hansen tree cover (2000) to identify which cluster is forest — `src/embedding_utils.py`. This removes roads, water, clearings, and secondary cover from the bounding box before averaging carbon.
3. Coarse-grid correction: for each carbon source, the fine (10 m) forest mask is aggregated to that source's native grid as a true forest FRACTION (not nearest-neighbor resampling), and only cells above a threshold (default 70%) are kept — `get_coarse_forest_mask`. This matters most for GEDI's 1 km grid, where a naive mask combination can include or exclude mixed cells (forest edges, lakes, coastline) almost arbitrarily.
4. Aboveground carbon density per zone from ESA CCI Biomass (2022), forest-fraction masked — `src/carbon_sources.py`.
5. Aboveground carbon density per zone from GEDI L4B, forest-fraction masked — `src/carbon_sources.py`.
6. Cross-source comparison chart (grouped bars, 4 zones x 2 sources).
7. Mean AlphaEarth embedding per zone, and cosine similarity between zones.
8. Similarity heatmap: how structurally distinct each zone is from the others.

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
4. Run `notebooks/01_carbon_comparison.ipynb` in order.

## Methodological note

Zone boundaries are illustrative bounding boxes, not validated forest-type polygons. The AlphaEarth + Hansen forest mask removes the most obvious non-forest contamination but is an unsupervised approximation, not a validated land-cover classification. The forest-fraction threshold (default 70%) at each source's native grid corrects the worst coarse-grid mixing (notably GEDI's 1 km cells near lakes, coastlines, or forest edges) but a lower or higher threshold will shift results — treat absolute values as approximate, and cross-source agreement as the more meaningful signal. ESA CCI (2022) and GEDI L4B (multi-year aggregate) are not from the same time window either.

