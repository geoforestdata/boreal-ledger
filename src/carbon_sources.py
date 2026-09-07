"""
ESA CCI aboveground biomass helpers for the Abitibi carbon workflow.

ESA CCI Biomass v6.0 provides annual aboveground biomass density. The
workflow converts biomass to carbon with the common 0.47 carbon fraction
for comparison with other mapped products.
"""

import ee


ESA_CCI_AGB_COLLECTION = "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
ESA_CCI_AGB_BAND = "AGB"
DEFAULT_CARBON_FRACTION = 0.47


def get_esa_cci_carbon(
    year: int,
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """Aboveground biomass carbon density (Mg C/ha) from ESA CCI Biomass."""
    collection = ee.ImageCollection(ESA_CCI_AGB_COLLECTION).filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    agb = collection.select(ESA_CCI_AGB_BAND).mosaic()
    return agb.multiply(carbon_fraction).rename("carbon_Mg_ha").clip(aoi)


def mean_carbon_over_zone(
    carbon_image: ee.Image,
    aoi: ee.Geometry,
    scale: int,
) -> float:
    """Mean carbon density (Mg C/ha) over the AOI."""
    stats = carbon_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return stats.get("carbon_Mg_ha", 0) or 0
