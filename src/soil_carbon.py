"""
soil_carbon.py

Soil organic carbon (SOC) stock, 0-30 cm depth, from SoilGrids 250m v2.0
(ISRIC) — the standard global gridded soil carbon product, official GEE
hosting under the soilgrids-isric project.

GEE asset: projects/soilgrids-isric/ocs_mean (band 'ocs_0-30cm_mean')
Units: raw pixel values are stored as t/ha x 10 (integer scaling); divide
by 10 to get true Mg C/ha. Verify the exact band name with
list_ocs_bands() before trusting the default in get_soil_carbon() — this
is ISRIC's official asset, but band names have changed across versions of
similar community-hosted datasets elsewhere in this repo.

IMPORTANT LIMITATION: standard SoilGrids depths (0-30 cm) substantially
UNDERESTIMATE total carbon in peatlands and other wetlands, where organic
soil layers can extend several meters deep. This module is not a
peatland-specific carbon estimator — for that, a dedicated product (e.g.
the "PEATGRIDS" dataset) would be needed. Treat wetland SOC numbers from
this module as a floor, not a full accounting.

Citation: Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021). SoilGrids
2.0: producing soil information for the globe with quantified spatial
uncertainty. SOIL, 7, 217-240.
"""

import ee

SOILGRIDS_OCS_ASSET = "projects/soilgrids-isric/ocs_mean"
# NOTE: earlier versions of this module divided the raw value by 10,
# based on a generic SoilGrids scaling table that applies to concentration
# properties (e.g. 'soc' in dg/kg). ISRIC's own OCS product description
# states the stored value is ALREADY in t/ha for the 0-30cm layer with no
# further scaling needed — the /10 division was likely wrong and produced
# SOC estimates roughly 10x too low across all four forest zones compared
# to published ranges (boreal ~60-120, tropical ~40-100, southern Chile
# andisols ~100-300+ Mg C/ha). Verify against a location with known SOC
# before trusting either version.
OCS_SCALE_FACTOR = 1  # raw stored value IS true Mg C/ha per ISRIC's own docs


def list_ocs_bands() -> list:
    """Diagnostic: band names available in the SoilGrids OCS asset."""
    return ee.Image(SOILGRIDS_OCS_ASSET).bandNames().getInfo()


def get_soil_carbon_0_30cm(aoi: ee.Geometry) -> ee.Image:
    """Soil organic carbon stock, 0-30 cm depth (Mg C/ha), clipped to aoi."""
    ocs_raw = ee.Image(SOILGRIDS_OCS_ASSET).select(0)
    return ocs_raw.divide(OCS_SCALE_FACTOR).rename("soc_Mg_ha").clip(aoi)


def mean_soil_carbon(aoi: ee.Geometry, scale: int = 250) -> float:
    """Mean soil organic carbon (Mg C/ha, 0-30 cm) over the zone."""
    stats = get_soil_carbon_0_30cm(aoi).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return stats.get("soc_Mg_ha", 0) or 0

