"""
stand_age_utils.py

Estimates stand age for a plantation zone from two independent signals:

  1. Years since the last Hansen-detected stand-replacement disturbance
     (lossyear band). A pixel with no recorded loss since 2000 is
     age-censored: it could be an older stand (pre-dating or exceeding the
     Hansen record) or non-plantation cover (e.g. a native remnant) — it
     is NOT necessarily "very old", just "undated by this method".
  2. Canopy height (ETH Global Canopy Height, 2020 snapshot), cross
     -referenced against an illustrative Pinus radiata height-age growth
     curve for Chile.

GEE assets:
  UMD/hansen/global_forest_change_2025_v1_13 (band 'lossyear')
  users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1
"""

import ee

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
CANOPY_HEIGHT_ASSET = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"

# Illustrative Pinus radiata height-age curve for Chile (moderate site
# index, INFOR/CONAF-type yield tables), height in meters at stand age in
# years. Piecewise-linear breakpoints, NOT a calibrated site-specific model
# — real growth depends on site index, thinning regime, and genetics.
RADIATA_HEIGHT_AGE_CURVE = [
    (0, 0), (5, 6), (10, 15), (15, 22), (20, 28), (25, 32), (30, 34),
]


def get_years_since_loss(aoi: ee.Geometry, current_year: int = 2025) -> ee.Image:
    """
    Years since the last Hansen-detected stand-replacement disturbance.
    Masked (no value) where lossyear == 0 — age-censored, not necessarily old.
    """
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear")
    calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0))
    years_since = ee.Image.constant(current_year).subtract(calendar_year)
    return years_since.rename("years_since_loss").clip(aoi)


def get_canopy_height(aoi: ee.Geometry) -> ee.Image:
    """Canopy top height (m), 2020 snapshot, clipped to aoi."""
    return ee.Image(CANOPY_HEIGHT_ASSET).rename("canopy_height_m").clip(aoi)


def height_to_age(height_m: float, curve: list = RADIATA_HEIGHT_AGE_CURVE) -> float:
    """
    Approximate stand age (years) from canopy height via piecewise-linear
    interpolation on the given height-age curve. Illustrative only.
    """
    if height_m <= curve[0][1]:
        return curve[0][0]
    if height_m >= curve[-1][1]:
        return curve[-1][0]
    for (age0, h0), (age1, h1) in zip(curve[:-1], curve[1:]):
        if h0 <= height_m <= h1:
            frac = (height_m - h0) / (h1 - h0) if h1 != h0 else 0
            return age0 + frac * (age1 - age0)
    return None

