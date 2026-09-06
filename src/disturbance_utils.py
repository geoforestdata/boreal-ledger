"""
disturbance_utils.py

Splits Hansen-detected forest loss into fire-caused vs non-fire (harvest
or other) disturbance, using the Canadian National Burned Area Composite
(NBAC) -- a public GEE asset from Natural Resources Canada with annual
fire perimeters since 1972.

Method: a loss pixel for year Y is FIRE if it falls inside an NBAC fire
perimeter for that same year Y; otherwise it is classified NON-FIRE
(harvest, or another cause NBAC doesn't map, such as unmapped windthrow
or pest outbreaks -- this is a residual category by exclusion, not a
positive identification of harvesting).

GEE asset: projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530
Coverage: 1972-2023.
"""

import ee

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NBAC_COLLECTION = "projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530"


def list_nbac_fields() -> list:
    """Diagnostic: property names of the first NBAC feature (confirm the year field)."""
    first = ee.FeatureCollection(NBAC_COLLECTION).first()
    return first.propertyNames().getInfo()


def get_fire_perimeters(year: int, aoi: ee.Geometry, year_field: str = "YEAR") -> ee.FeatureCollection:
    """Returns NBAC fire perimeters for the given year, filtered to aoi."""
    fires = ee.FeatureCollection(NBAC_COLLECTION).filter(ee.Filter.eq(year_field, year))
    return fires.filterBounds(aoi)


def get_disturbance_cause_map(
    aoi: ee.Geometry,
    year_field: str = "YEAR",
) -> ee.Image:
    """
    Returns a single image with two bands:
      'fire_loss'      -- Hansen loss pixels that fall inside an NBAC fire
                           perimeter for their own loss year
      'non_fire_loss'  -- Hansen loss pixels that don't (harvest or other)

    Implementation note: this uses ONE reduceToImage() call to rasterize
    every NBAC fire perimeter's year in a single pass, instead of looping
    over each year separately and painting+comparing 23 times -- the
    per-year loop scales very poorly to large AOIs (the same kind of
    timeout risk seen with big bounding boxes elsewhere in this repo).
    """
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)
    loss_calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0))

    fires = ee.FeatureCollection(NBAC_COLLECTION).filterBounds(aoi)
    fire_year_img = fires.reduceToImage(
        properties=[year_field], reducer=ee.Reducer.first()
    ).rename("fire_year")

    year_matches = loss_calendar_year.eq(fire_year_img).unmask(0)

    fire_loss = year_matches.And(lossyear.gt(0)).selfMask().rename("fire_loss")
    non_fire_loss = year_matches.Not().And(lossyear.gt(0)).selfMask().rename("non_fire_loss")

    return fire_loss.addBands(non_fire_loss)

