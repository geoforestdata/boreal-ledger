"""
carbon_sources.py

Aboveground carbon density (Mg C/ha) from ESA CCI Biomass v6.0 (Santoro &
Cartus, 2025): continuous, gap-free annual AGB maps (2007, 2010, 2015-2022).
GEE asset: projects/sat-io/open-datasets/ESA/ESA_CCI_AGB

Converted to carbon using the IPCC default fraction (0.47).

Note on GEDI: an earlier version of this module also queried GEDI L4B
(1 km gridded AGBD, asset LARSE/GEDI/GEDI04_B_002). It was dropped after
diagnostics showed it unreliable for AOIs this small: the L4B `MU` band is
defined as the mean biomass "including forest and non-forest" per 1 km
cell (not separable after the fact by masking), and real GEDI ground-track
density varied sharply across zones - one zone averaged under 1 track per
cell, meaning most of its value was a statistical fill-in rather than a
direct measurement. See the repo README for the diagnostic numbers.
"""

import ee

ESA_CCI_AGB_COLLECTION = "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
ESA_CCI_AGB_BAND = "AGB"
DEFAULT_CARBON_FRACTION = 0.47


def get_esa_cci_carbon(
    year: int,
    aoi: ee.Geometry,
    carbon_fraction: float = DEFAULT_CARBON_FRACTION,
) -> ee.Image:
    """Aboveground carbon density (Mg C/ha) from ESA CCI Biomass for the given year."""
    collection = ee.ImageCollection(ESA_CCI_AGB_COLLECTION).filter(
        ee.Filter.calendarRange(year, year, "year")
    )
    agb = collection.select(ESA_CCI_AGB_BAND).mosaic()
    return agb.multiply(carbon_fraction).rename("carbon_Mg_ha").clip(aoi)


def mean_carbon_over_zone(
    carbon_image: ee.Image,
    aoi: ee.Geometry,
    scale: int,
) -> float:
    """Mean carbon density (Mg C/ha) over the zone, as a plain Python float."""
    stats = carbon_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    return stats.get("carbon_Mg_ha", 0) or 0


def forest_weighted_mean_carbon(
    carbon_image: ee.Image,
    forest_mask_10m: ee.Image,
    aoi: ee.Geometry,
    scale: int = 10,
) -> float:
    """
    Forest-area-weighted mean carbon density (Mg C/ha): each 10 m forest
    pixel contributes the carbon value of whichever (possibly coarser)
    source cell it falls within; dividing summed weighted carbon by summed
    weight yields a mean restricted to forest pixels, naturally weighted by
    how much of each source cell is actually forest.
    """
    forest_binary = forest_mask_10m.unmask(0).rename("weight")
    weighted_carbon = carbon_image.multiply(forest_binary).rename("weighted_carbon")

    stats = weighted_carbon.addBands(forest_binary).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()

    numerator = stats.get("weighted_carbon", 0) or 0
    denominator = stats.get("weight", 0) or 0
    return numerator / denominator if denominator else 0


# IPCC default root:shoot ratios and deadwood/litter fractions (applied to
# aboveground carbon) by biome type. Source: IPCC 2006 Guidelines for
# National Greenhouse Gas Inventories, Vol. 4, Ch. 4, Table 4.4 (root:shoot)
# and generalized deadwood/litter fractions used in national GHG inventories.
# These are illustrative defaults, not site-calibrated values.
ROOT_SHOOT_RATIO = {
    "boreal": 0.29,
    "temperate": 0.24,
    "tropical": 0.24,
    "plantation": 0.25,
}

DEADWOOD_LITTER_FRACTION = {
    "boreal": 0.10,
    "temperate": 0.10,
    "tropical": 0.06,
    "plantation": 0.05,
}


def combined_mapped_forest_carbon_pools(agb_carbon_mg_ha: float, biome_type: str, soc_mg_ha: float) -> dict:
    """
    Combines aboveground carbon with belowground biomass (via IPCC
    root:shoot ratio), deadwood/litter (via IPCC fraction), and soil
    organic carbon (0-30 cm) into a mapped-pool summary.

    Returns a breakdown dict, not just the sum - the pools are
    illustrative-default-based (BGB, deadwood/litter) or single-depth
    (SOC), not independently validated for each site. Do not treat the
    summed value as total ecosystem carbon or total forest carbon.
    """
    if biome_type not in ROOT_SHOOT_RATIO:
        raise ValueError(f"Unknown biome_type '{biome_type}'. Choose from {list(ROOT_SHOOT_RATIO)}")

    bgb = agb_carbon_mg_ha * ROOT_SHOOT_RATIO[biome_type]
    deadwood_litter = agb_carbon_mg_ha * DEADWOOD_LITTER_FRACTION[biome_type]
    total = agb_carbon_mg_ha + bgb + deadwood_litter + soc_mg_ha

    return {
        "AGB": agb_carbon_mg_ha,
        "BGB": bgb,
        "deadwood_litter": deadwood_litter,
        "SOC_0_30cm": soc_mg_ha,
        "combined_mapped_pools": total,
    }
