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
    min_year: int = 2001,
    max_year: int = 2023,
) -> ee.Image:
    """
    Returns a single image over the full year range with two bands:
      'fire_loss'      -- Hansen loss pixels that fall inside an NBAC fire
                           perimeter for their own loss year
      'non_fire_loss'  -- Hansen loss pixels that don't (harvest or other)

    Both bands are binary masks (1 = that class, masked elsewhere) so they
    can be summed with pixelArea() for hectare totals, or combined with a
    'lossyear' band for a per-year breakdown.
    """
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)

    fire_mask = ee.Image(0)
    for year in range(min_year, max_year + 1):
        year_offset = year - 2000
        year_loss = lossyear.eq(year_offset)
        fire_perimeters = get_fire_perimeters(year, aoi, year_field=year_field)
        year_fire_mask = ee.Image(0).paint(fire_perimeters, 1)
        fire_mask = fire_mask.where(year_loss.And(year_fire_mask.eq(1)), 1)

    all_loss = lossyear.gt(0)
    fire_loss = all_loss.And(fire_mask.eq(1)).selfMask().rename("fire_loss")
    non_fire_loss = all_loss.And(fire_mask.eq(0)).selfMask().rename("non_fire_loss")

    return fire_loss.addBands(non_fire_loss)

