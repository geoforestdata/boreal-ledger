"""
biomass_utils.py

Loads ESA CCI Biomass (v6.0) annual aboveground biomass (AGB) maps —
continuous global coverage, no orbital-track gaps like GEDI footprints —
converts AGB to aboveground carbon density, and computes carbon flux
(source vs sink) between two available years.

Source: ESA Climate Change Initiative (CCI) Biomass, Santoro & Cartus (2025).
Community-curated GEE asset (Samapriya Roy):
  projects/sat-io/open-datasets/ESA/ESA_CCI_AGB
Available years: 2007, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
(annual only from 2015 onward). Band order: band 0 = AGB (Mg/ha),
band 1 = per-pixel standard deviation (Mg/ha) — selected by index since
band naming is not guaranteed across catalog versions.

Citation (required if used in any published output):
Santoro, M.; Cartus, O. (2025): ESA Biomass Climate Change Initiative
(Biomass_cci): Global datasets of forest above-ground biomass for the years
2007, 2010, 2015-2022, v6.0. NERC EDS CEDA.
doi:10.5285/95913ffb6467447ca72c4e9d8cf30501
"""

import ee

AGB_COLLECTION = "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
DEFAULT_CARBON_FRACTION = 0.47  # IPCC default: biomass -> carbon
AVAILABLE_YEARS = [2007, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def list_collection_dates() -> ee.List:
    """
    Diagnostic helper: returns the system:time_start dates of every image in
    the collection. Run this once to confirm how years are indexed before
    trusting calendarRange filtering.
    """
    return (
        ee.ImageCollection(AGB_COLLECTION)
        .aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
    )


def get_agb(year: int, aoi: ee.Geometry) -> ee.Image:
    """Returns the AGB (Mg/ha) image for the given year, clipped to aoi."""
    if year not in AVAILABLE_YEARS:
        raise ValueError(f"Year {year} not available. Choose from {AVAILABLE_YEARS}")
    collection = ee.ImageCollection(AGB_COLLECTION).filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    agb = collection.select(0).mosaic().rename("AGB")
    return agb.clip(aoi)


def get_carbon_density(
    year: int,
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """Returns aboveground carbon density (Mg C/ha) for the given year."""
    agb = get_agb(year, aoi)
    return agb.multiply(carbon_fraction).rename("carbon_Mg_ha")


def carbon_flux(
    year_t0: int,
    year_t1: int,
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """
    Computes carbon density change between two years (t1 - t0).
    Positive = carbon sink (net gain); negative = carbon source (net loss).
    """
    c0 = get_carbon_density(year_t0, aoi, carbon_fraction)
    c1 = get_carbon_density(year_t1, aoi, carbon_fraction)
    return c1.subtract(c0).rename("carbon_flux_Mg_ha")


def classify_flux(
    flux: ee.Image,
    source_threshold: float = -5,
    sink_threshold: float = 5,
) -> ee.Image:
    """
    Classifies carbon flux into:
      1 = source (net loss, flux < source_threshold)
      2 = sink   (net gain, flux > sink_threshold)
      0 = stable (otherwise)
    Thresholds are in Mg C/ha over the period compared; calibrate against
    known disturbance events before treating results as definitive.
    """
    source = flux.lt(source_threshold)
    sink = flux.gt(sink_threshold)
    classified = ee.Image(0).where(source, 1).where(sink, 2).rename("flux_class")
    return classified.updateMask(flux.mask())


def aggregate_flux_by_units(
    flux: ee.Image,
    units_fc: ee.FeatureCollection,
    scale: int = 100,
) -> ee.FeatureCollection:
    """
    Aggregates mean carbon flux (Mg C/ha) and total carbon stock change (Mg)
    per spatial unit. units_fc must carry a unique 'unit_id' property.
    """
    pixel_area_ha = ee.Image.pixelArea().divide(10000)
    stock_change = flux.multiply(pixel_area_ha).rename("carbon_stock_change_Mg")
    combined = flux.addBands(stock_change)

    stats = combined.reduceRegions(
        collection=units_fc,
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.sum(), sharedInputs=False
        ),
        scale=scale,
    )
    return stats

