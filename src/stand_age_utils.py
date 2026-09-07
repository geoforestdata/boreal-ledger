"""
Hansen dated-loss and ETH canopy-height helpers for the Abitibi workflow.

Hansen lossyear provides directly dated canopy loss after 2000. Pixels with
lossyear == 0 are age-censored by this method; they cannot be assigned a
stand age from Hansen alone. ETH Global Canopy Height is used as
independent structure information only, not as stand age.
"""

import ee


HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
CANOPY_HEIGHT_ASSET = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"


def get_years_since_loss(aoi: ee.Geometry, current_year: int = 2025) -> ee.Image:
    """
    Years since the last Hansen-detected canopy loss.
    Masked where lossyear == 0: undated by Hansen, not assigned an age.
    """
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear")
    calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0))
    years_since = ee.Image.constant(current_year).subtract(calendar_year)
    return years_since.rename("years_since_loss").clip(aoi)


def get_canopy_height(aoi: ee.Geometry) -> ee.Image:
    """Canopy top height (m), 2020 snapshot, clipped to the AOI."""
    return ee.Image(CANOPY_HEIGHT_ASSET).rename("canopy_height_m").clip(aoi)
