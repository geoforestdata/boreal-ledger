"""
recovery_utils.py

Builds a post-disturbance recovery curve for a managed boreal forest using
a chronosequence (space-for-time substitution): instead of tracking one
stand over decades, it looks at MANY stands disturbed in different years
-- all coexisting in the landscape today -- and relates their CURRENT NBR
value to how long ago each was disturbed. A single recent-year composite
is enough; no historical time series is needed, because the "time axis"
is provided by the landscape itself (different stands, different ages).

Coverage: years-since-disturbance from 0 to ~24 (Hansen's lossyear band
only covers disturbances from 2001 onward). This captures SPECTRAL
recovery (canopy closure, understory establishment), which typically
saturates within 10-20 years for NBR -- NOT full structural/biomass
recovery to a mature boreal stand, which takes 60-100+ years and cannot
be observed this way. State that distinction explicitly wherever this
curve is shown.

GEE assets:
  UMD/hansen/global_forest_change_2025_v1_13 (band 'lossyear')
  COPERNICUS/S2_SR_HARMONIZED (for the reference-year NBR composite)
"""

import ee

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"


def get_sentinel2_nbr_composite(year: int, aoi: ee.Geometry) -> ee.Image:
    """Cloud-masked median NBR composite (growing season) for the given year."""

    def mask_clouds(image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
            .And(scl.neq(10)).And(scl.neq(11))
        )
        return image.updateMask(mask).divide(10000)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )
    composite = collection.median().clip(aoi)
    return composite.normalizedDifference(["B8", "B12"]).rename("NBR")


def get_years_since_disturbance(
    aoi: ee.Geometry,
    reference_year: int = 2024,
    max_age: int = 24,
) -> ee.Image:
    """
    Years since the last Hansen-detected stand-replacement disturbance, as
    of reference_year. Masked outside [0, max_age] -- pixels with no
    recorded loss, or a loss more recent than reference_year (shouldn't
    happen), are excluded.
    """
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear")
    disturbance_year = lossyear.add(2000).updateMask(lossyear.gt(0))
    years_since = ee.Image.constant(reference_year).subtract(disturbance_year)
    years_since = years_since.updateMask(
        years_since.gte(0).And(years_since.lte(max_age))
    )
    return years_since.rename("years_since_disturbance").clip(aoi)


def build_age_nbr_image(
    aoi: ee.Geometry,
    reference_year: int = 2024,
    max_age: int = 24,
) -> ee.Image:
    """
    Returns a 2-band image: years_since_disturbance and NBR (both from
    reference_year), ready to be reduced by age class to build the
    recovery curve.
    """
    age = get_years_since_disturbance(aoi, reference_year, max_age)
    nbr = get_sentinel2_nbr_composite(reference_year, aoi)
    return age.addBands(nbr)

