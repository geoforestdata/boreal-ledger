"""
peatland_finder.py

Instead of guessing a named wetland site from memory, this scans a grid of
candidate sub-boxes across the Abitibi-Temiscamingue region using PEATGRIDS
(a global peat thickness/carbon-stock model) and reports which candidate
has the highest mean peat carbon stock — a data-driven way to locate a
real peatland within a "lake country" region instead of picking blind
coordinates.

GEE assets (community-hosted, verify exact band names with
list_peatgrids_bands() before trusting them):
  projects/sat-io/open-datasets/PEATGRIDS/CSTOCK_MGC  (peat carbon stock, Mg C/ha)
  projects/sat-io/open-datasets/PEATGRIDS/BD_MEAN      (bulk density)
  projects/sat-io/open-datasets/PEATGRIDS/CC_MEAN      (carbon content)

Source: global peat thickness/carbon stock model (see repo README for full
citation) — a modeled product, not field-verified for any specific pixel.
"""

import ee

PEAT_CARBON_STOCK_ASSET = "projects/sat-io/open-datasets/PEATGRIDS/C_STOCK_MGC_PER_M2"
PEAT_CARBON_BAND = "C_STOCK_MgC_per_m2_MEAN"

# Search grid spanning the Abitibi-Temiscamingue administrative region
# (roughly 47.0-49.5N, -80.0 to -77.0W). Each candidate is a ~20 km box.
ABITIBI_SEARCH_GRID = {
    f"candidate_{i}_{j}": [lon, lat, lon + 0.25, lat + 0.20]
    for i, lon in enumerate([-79.5, -78.75, -78.0])
    for j, lat in enumerate([47.3, 48.0, 48.7])
}


def list_peatgrids_bands() -> list:
    """Diagnostic: band names in the PEATGRIDS carbon stock asset."""
    return ee.Image(PEAT_CARBON_STOCK_ASSET).bandNames().getInfo()


def get_peat_carbon_stock(aoi: ee.Geometry) -> ee.Image:
    """
    Peat carbon stock (Mg C/ha) from PEATGRIDS, clipped to aoi. The source
    band is Mg C/m2 (full peat depth), converted to Mg C/ha (x 10,000);
    negative values are no-data and are masked out per the dataset's own
    usage notes.
    """
    band = ee.Image(PEAT_CARBON_STOCK_ASSET).select(PEAT_CARBON_BAND)
    band = band.updateMask(band.gte(0))
    return band.multiply(10000).rename("peat_C_Mg_ha").clip(aoi)


def find_best_peatland_candidate(search_grid: dict = ABITIBI_SEARCH_GRID, scale: int = 1000) -> dict:
    """
    Computes mean peat carbon stock for every candidate box in search_grid
    and returns a dict of {candidate_key: mean_peat_C_Mg_ha}, sorted
    descending. The top entry is the best data-driven pick for a real
    peatland within the search area.
    """
    results = {}
    for key, bbox in search_grid.items():
        geom = ee.Geometry.Rectangle(bbox)
        mean_c = get_peat_carbon_stock(geom).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=scale,
            maxPixels=1e13, bestEffort=True,
        ).get("peat_C_Mg_ha").getInfo()
        results[key] = mean_c or 0

    return dict(sorted(results.items(), key=lambda kv: kv[1], reverse=True))

