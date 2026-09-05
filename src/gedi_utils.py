"""
gedi_utils.py

Loads GEDI L4B gridded aboveground biomass density (AGBD, 1 km resolution),
converts it to aboveground carbon density, and combines it with the
Sentinel-2 change classification to estimate carbon exposed in loss vs
recovery zones.

Coverage note: GEDI collects data between 51.6 N and 51.6 S. Data gap from
2023-03-17 to 2024-04-24 (instrument stored on the ISS). L4B is a static
gridded product (not a dense annual time series) — it represents biomass
density as sampled across the mission period, not a specific year.
"""

import ee

GEDI_L4B_COLLECTION = "LARSE/GEDI/GEDI04_B_002"
DEFAULT_CARBON_FRACTION = 0.47  # IPCC default: biomass -> carbon


def get_carbon_density(aoi: ee.Geometry,
                        carbon_fraction: float = DEFAULT_CARBON_FRACTION) -> ee.Image:
    """
    Returns aboveground carbon density (Mg C/ha) clipped to the AOI, derived
    from the GEDI L4B gridded AGBD mean band ('MU').
    """
    agbd = ee.ImageCollection(GEDI_L4B_COLLECTION).mosaic().select("MU")
    carbon_density = agbd.multiply(carbon_fraction).rename("carbon_Mg_ha")
    return carbon_density.clip(aoi)


def aggregate_carbon_by_units(
    carbon_density: ee.Image,
    units_fc: ee.FeatureCollection,
    scale: int = 1000,
) -> ee.FeatureCollection:
    """
    Aggregates mean carbon density and estimated total carbon stock (Mg) per
    spatial unit. units_fc must carry a unique 'unit_id' property.
    """
    pixel_area_ha = ee.Image.pixelArea().divide(10000)
    stock_per_pixel = carbon_density.multiply(pixel_area_ha).rename("carbon_stock_Mg")
    combined = carbon_density.addBands(stock_per_pixel)

    stats = combined.reduceRegions(
        collection=units_fc,
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.sum(), sharedInputs=False
        ),
        scale=scale,
    )
    return stats


def carbon_in_change_zones(
    carbon_density: ee.Image,
    classified_change: ee.Image,
) -> dict:
    """
    Masks carbon density by change class (loss=1, recovery=2) and returns
    the two masked images, ready for regional reduction (e.g. ee.Reducer.sum
    over pixelArea-weighted carbon to get Mg C in loss vs recovery zones).
    """
    carbon_loss_zone = carbon_density.updateMask(classified_change.eq(1)).rename(
        "carbon_in_loss_Mg_ha"
    )
    carbon_recovery_zone = carbon_density.updateMask(classified_change.eq(2)).rename(
        "carbon_in_recovery_Mg_ha"
    )
    return {"loss": carbon_loss_zone, "recovery": carbon_recovery_zone}

