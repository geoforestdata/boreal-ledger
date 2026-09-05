"""
carbon_sources.py

Aboveground carbon density (Mg C/ha) from two independent global sources,
so each zone gets a cross-checked estimate rather than a single-source
number:

  - ESA CCI Biomass v6.0 (Santoro & Cartus, 2025): continuous, gap-free
    annual AGB maps (2007, 2010, 2015-2022). GEE asset:
    projects/sat-io/open-datasets/ESA/ESA_CCI_AGB
  - GEDI L4B gridded AGBD (1 km): spaceborne lidar-derived biomass density,
    aggregated across the mission period (not a single calendar year).
    GEE asset: LARSE/GEDI/GEDI04_B_002 (a single Image, not an
    ImageCollection — ee.ImageCollection.load on this asset fails).

Both are converted to carbon using the IPCC default fraction (0.47).
"""

import ee

ESA_CCI_AGB_COLLECTION = "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
GEDI_L4B_ASSET = "LARSE/GEDI/GEDI04_B_002"
DEFAULT_CARBON_FRACTION = 0.47


def get_esa_cci_carbon(
    year: int,
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """Aboveground carbon density (Mg C/ha) from ESA CCI Biomass for the given year."""
    collection = ee.ImageCollection(ESA_CCI_AGB_COLLECTION).filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    agb = collection.select(0).mosaic().rename("AGB")
    return agb.multiply(carbon_fraction).rename("carbon_Mg_ha").clip(aoi)


def get_gedi_carbon(
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """Aboveground carbon density (Mg C/ha) from GEDI L4B, clipped to aoi."""
    agbd = ee.Image(GEDI_L4B_ASSET).select("MU")
    return agbd.multiply(carbon_fraction).rename("carbon_Mg_ha").clip(aoi)


def mean_carbon_over_zone(
    carbon_image: ee.Image,
    aoi: ee.Geometry,
    scale: int,
) -> float:
    """Mean carbon density (Mg C/ha) over the zone, as a plain Python float."""
    stats = carbon_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return stats.get("carbon_Mg_ha", 0) or 0

