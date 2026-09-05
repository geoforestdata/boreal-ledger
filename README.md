# Deforestation and Recovery Balance — Managed Forest of Quebec

Analysis of the net balance between forest cover loss and recovery in the publicly managed forest of Quebec, using Sentinel-2 time series in Google Earth Engine, combined with GEDI-derived aboveground carbon density.

## Research question

What is the annual net balance between lost forest cover (harvesting, fire, pest outbreaks) and recovering cover within the managed forest, and how much aboveground carbon is stored — and potentially at risk — in those zones?

## What this repo shows

1. **Change dynamics (Sentinel-2, 2017–2025):** annual net balance of cover loss vs recovery per spatial unit.
2. **Aboveground carbon flux (ESA CCI Biomass v6.0):** carbon density change (Mg C/ha) between two years per spatial unit — positive values are carbon sinks (net gain), negative values are carbon sources (net loss). Continuous, gap-free annual maps (2007, 2010, 2015–2022), unlike GEDI's sparse orbital-track sampling.

## Data

- **Imagery:** Sentinel-2 SR Harmonized (2017–2025), annual growing-season composites (June–September).
- **Carbon:** ESA CCI Biomass v6.0 (Santoro & Cartus, 2025), annual aboveground biomass maps (2007, 2010, 2015–2022), converted to carbon (IPCC default fraction 0.47). Community-curated GEE asset: `projects/sat-io/open-datasets/ESA/ESA_CCI_AGB`. Citation required in any published output — see module docstring in `src/biomass_utils.py`.
- **Study area:** Abitibi-Témiscamingue administrative region, full extent. Boundaries from the "Découpages administratifs" layer (Données Québec / MRNF).
- **Spatial aggregation unit:** [to be defined — hexagon grid or MRC subdivision].

## Methodology

1. Annual composite extraction with cloud masking (SCL) — `src/gee_utils.py`.
2. NDVI and NBR computation per composite.
3. Year-over-year change detection (dNBR) and classification into loss / recovery / stable — `src/change_detection.py`.
4. Spatial aggregation: loss and recovery area (ha) per unit, net balance = recovery − loss.
5. Aboveground carbon flux from ESA CCI Biomass (Mg C/ha, 2017 vs 2022) per unit — `src/biomass_utils.py`.
6. Classification of carbon flux into source / sink / stable, cross-referenced with loss/recovery zones.
7. Visualization: annual change maps, carbon map, and cumulative net balance chart.

## Structure

```
src/                  reusable functions (GEE, change detection, GEDI carbon)
notebooks/            analysis pipeline and results
figures/              final maps and charts
data/                 public layers or download instructions
```

## How to run

1. `pip install -r requirements.txt`
2. `earthengine authenticate`
3. Download the "Découpages administratifs" layer (Données Québec) and place it under `data/`.
4. Replace `PROJECT` in the notebook with your own GEE project.
5. Define the spatial unit layer for aggregation.
6. Run `notebooks/01_extraccion_y_balance.ipynb` in order.

## Methodological note

The dNBR thresholds used to classify loss/recovery are a starting point and should be calibrated against known disturbance events (e.g. fire perimeters mapped by SOPFEU) before treating results as definitive.

## Next step

This repo is the first of two planned publications. The second part uses the net balance per unit as input to an optimization model (MILP) prioritizing restoration under a budget constraint.

