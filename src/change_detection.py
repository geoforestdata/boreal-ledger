"""
change_detection.py

Year-over-year change detection from annual NDVI/NBR composites. Classifies
each pixel as loss, recovery, or stable, and aggregates net balance per
spatial unit (e.g. hexagon grid or UAF subdivision).
"""

import ee


def compute_delta(img_t0: ee.Image, img_t1: ee.Image, band: str = "NBR") -> ee.Image:
    """
    Computes the year-over-year delta for a given band (default NBR).
    Positive delta = greening/recovery; negative delta = loss/disturbance.
    """
    return img_t1.select(band).subtract(img_t0.select(band)).rename(f"d{band}")


def classify_change(
    delta: ee.Image,
    loss_threshold: float = -0.10,
    recovery_threshold: float = 0.10,
) -> ee.Image:
    """
    Classifies a delta image into three classes:
      1 = loss        (delta < loss_threshold)
      2 = recovery     (delta > recovery_threshold)
      0 = stable       (otherwise)

    Thresholds are on dNBR by default; calibrate against known disturbance
    events (e.g. a mapped fire perimeter) before finalizing.
    """
    loss = delta.lt(loss_threshold)
    recovery = delta.gt(recovery_threshold)
    classified = (
        ee.Image(0)
        .where(loss, 1)
        .where(recovery, 2)
        .rename("change_class")
    )
    return classified.updateMask(delta.mask())


def build_change_series(annual_stack: ee.ImageCollection, band: str = "NBR"):
    """
    Given an ImageCollection of annual composites (sorted by 'year'), returns
    a list of (year_t1, classified_change_image) pairs for each consecutive
    year transition.
    """
    years = annual_stack.aggregate_array("year").getInfo()
    years = sorted(years)
    image_list = annual_stack.sort("year").toList(annual_stack.size())

    pairs = []
    for i in range(len(years) - 1):
        img_t0 = ee.Image(image_list.get(i))
        img_t1 = ee.Image(image_list.get(i + 1))
        delta = compute_delta(img_t0, img_t1, band=band)
        classified = classify_change(delta)
        pairs.append((years[i + 1], classified))
    return pairs


def aggregate_by_units(
    classified: ee.Image,
    units_fc: ee.FeatureCollection,
    scale: int = 10,
) -> ee.FeatureCollection:
    """
    Aggregates a classified change image (loss=1, recovery=2, stable=0) over
    a spatial unit layer (e.g. hexagon grid), computing the loss and
    recovery area (ha) per unit.

    units_fc must have a unique 'unit_id' property.
    """
    pixel_area_ha = ee.Image.pixelArea().divide(10000)

    loss_area = pixel_area_ha.updateMask(classified.eq(1)).rename("loss_ha")
    recovery_area = pixel_area_ha.updateMask(classified.eq(2)).rename("recovery_ha")
    combined = loss_area.addBands(recovery_area)

    stats = combined.reduceRegions(
        collection=units_fc,
        reducer=ee.Reducer.sum(),
        scale=scale,
    )

    def add_net_balance(feature: ee.Feature) -> ee.Feature:
        loss = ee.Number(feature.get("loss_ha"))
        recovery = ee.Number(feature.get("recovery_ha"))
        return feature.set("net_balance_ha", recovery.subtract(loss))

    return stats.map(add_net_balance)

