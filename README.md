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
2. Aboveground carbon density per zone from ESA CCI Biomass (2022) — `src/carbon_sources.py`.
3. Aboveground carbon density per zone from GEDI L4B — `src/carbon_sources.py`.
4. Cross-source comparison chart (grouped bars, 4 zones x 2 sources).
5. Mean AlphaEarth embedding per zone, and cosine similarity between zones — `src/embedding_utils.py`.
6. Similarity heatmap: how structurally distinct each zone is from the others.

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

Zone boundaries are illustrative bounding boxes, not validated forest-type polygons — a single 20-25 km box may include some non-target land cover (roads, water, mixed stands). ESA CCI (2022) and GEDI L4B (multi-year aggregate) are not from the same time window, so treat cross-source agreement as directional, not exact. The AlphaEarth similarity analysis is descriptive, not a validated classification.

