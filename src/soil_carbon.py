"""
soil_carbon.py

Soil organic carbon stock (SOC stock), 0-30 cm depth, from SoilGrids
250m v2.0 (ISRIC) - the standard global gridded soil carbon product,
official GEE hosting under the soilgrids-isric project.

GEE asset: projects/soilgrids-isric/ocs_mean (band 'ocs_0-30cm_mean')
Units: the SoilGrids 'ocs' product maps organic carbon stock in t C/ha.
Because 1 tonne equals 1 Mg, no scale conversion is needed to express the
0-30 cm stock as Mg C/ha. The conversion factor of 10 in ISRIC's layer
table converts t/ha to kg/m2; it should not be applied when reporting
Mg C/ha.

Note: standard SoilGrids depths (0-30 cm) substantially
UNDERESTIMATE total carbon in peatlands and other wetlands, where organic
soil layers can extend several meters deep. This module is not a
peatland-specific carbon estimator - for that, a dedicated product (e.g.
the "PEATGRIDS" dataset) would be needed. Treat wetland SOC numbers from
this module as a floor, not a full accounting.

Citation: Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021). SoilGrids
2.0: producing soil information for the globe with quantified spatial
uncertainty. SOIL, 7, 217-240.
"""

import ee

SOILGRIDS_OCS_ASSET = "projects/soilgrids-isric/ocs_mean"
SOILGRIDS_OCS_BAND_0_30CM = "ocs_0-30cm_mean"
OCS_SCALE_FACTOR = 1  # t C/ha is numerically identical to Mg C/ha.


def list_ocs_bands() -> list:
    """Diagnostic: band names available in the SoilGrids OCS asset."""
    return ee.Image(SOILGRIDS_OCS_ASSET).bandNames().getInfo()


def get_soil_carbon_0_30cm(aoi: ee.Geometry) -> ee.Image:
    """Soil organic carbon stock, 0-30 cm depth (Mg C/ha), clipped to aoi."""
    ocs_raw = ee.Image(SOILGRIDS_OCS_ASSET).select(SOILGRIDS_OCS_BAND_0_30CM)
    return ocs_raw.divide(OCS_SCALE_FACTOR).rename("soc_stock_MgC_ha").clip(aoi)


def mean_soil_carbon(aoi: ee.Geometry, scale: int = 250) -> float:
    """Mean soil organic carbon (Mg C/ha, 0-30 cm) over the zone."""
    stats = get_soil_carbon_0_30cm(aoi).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return stats.get("soc_stock_MgC_ha", 0) or 0
