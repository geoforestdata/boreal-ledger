"""
06_agb_crosscheck_scanfi.py

Independent AGB cross-check using SCANFI (Spatialized Canadian National
Forest Inventory), Natural Resources Canada / Canadian Forest Service.

The Earth Engine Data Catalog exposes SCANFI v1.2 as:
projects/gcpm041u-lemur/assets/scanfi_v12/SCANFI_v1_2

Band used:
  biomass -- live aboveground dry tree biomass, tonnes/ha.

SCANFI and ESA CCI are not field inventory data for this AOI, and their
mapped biomass pools are not guaranteed to be identical. The comparison
uses Hansen treecover2000 > 50 as a consistent forest definition and
converts biomass to carbon with 0.47 only for a common reporting unit.

Run: python 06_agb_crosscheck_scanfi.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee

from carbon_sources import ESA_CCI_AGB_BAND, ESA_CCI_AGB_COLLECTION, DEFAULT_CARBON_FRACTION, get_esa_cci_carbon
from zones import get_abitibi_analysis_zone

PROJECT = "ibfra2026"  # replace with your own GEE project
YEAR = 2022
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
SCANFI_ASSET = "projects/gcpm041u-lemur/assets/scanfi_v12/SCANFI_v1_2"
SCANFI_BIOMASS_BAND = "biomass"


def pct_diff(value: float, baseline: float) -> float:
    return 100 * (value - baseline) / baseline if baseline else 0


def esa_cci_method_c(aoi: ee.Geometry, forest_mask: ee.Image) -> float:
    pixel_ha = ee.Image.pixelArea().divide(10000)
    carbon_density = get_esa_cci_carbon(YEAR, aoi)
    esa_agb = (
        ee.ImageCollection(ESA_CCI_AGB_COLLECTION)
        .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
        .select(ESA_CCI_AGB_BAND)
        .mosaic()
        .clip(aoi)
    )
    esa_projection = esa_agb.projection()
    forest_fraction = (
        forest_mask.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=esa_projection, scale=100)
        .rename("forest_fraction")
    )
    weighted_carbon = carbon_density.multiply(forest_fraction).multiply(pixel_ha).rename("weighted_carbon_Mg")
    weighted_area = forest_fraction.multiply(pixel_ha).rename("forest_fraction_area_ha")
    stats = weighted_carbon.addBands(weighted_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    total = stats.get("weighted_carbon_Mg", 0) or 0
    area = stats.get("forest_fraction_area_ha", 0) or 0
    return total / area if area else 0


def main():
    ee.Initialize(project=PROJECT)

    aoi = get_abitibi_analysis_zone()
    forest_mask = ee.Image(HANSEN_ASSET).select("treecover2000").gt(50)

    scanfi = ee.Image(SCANFI_ASSET)
    scanfi_biomass = scanfi.select(SCANFI_BIOMASS_BAND).rename("scanfi_biomass_Mg_ha")
    scanfi_biomass_mg_ha = scanfi_biomass.updateMask(forest_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e13,
        bestEffort=True,
    ).get("scanfi_biomass_Mg_ha").getInfo()
    scanfi_carbon_mg_ha = scanfi_biomass_mg_ha * DEFAULT_CARBON_FRACTION if scanfi_biomass_mg_ha else 0

    esa_carbon_mg_ha = esa_cci_method_c(aoi, forest_mask)
    esa_biomass_mg_ha = esa_carbon_mg_ha / DEFAULT_CARBON_FRACTION if DEFAULT_CARBON_FRACTION else 0

    print("=== AGB cross-check: ESA CCI Method C vs SCANFI, Abitibi analysis zone ===\n")
    print(f"SCANFI asset: {SCANFI_ASSET}")
    print(f"SCANFI bands: {scanfi.bandNames().getInfo()}")
    print(f"SCANFI selected band: {SCANFI_BIOMASS_BAND}")
    print(f"SCANFI projection: {scanfi_biomass.projection().getInfo()}")
    print(f"Forest definition: Hansen treecover2000 > 50")
    print(f"Carbon fraction for comparison: {DEFAULT_CARBON_FRACTION}\n")
    print("Source | Mg biomass/ha | Mg C/ha | Difference vs ESA CCI | Difference %")

    rows = [
        ("ESA CCI Method C", esa_biomass_mg_ha, esa_carbon_mg_ha),
        ("SCANFI v1.2", scanfi_biomass_mg_ha, scanfi_carbon_mg_ha),
    ]
    for source, biomass, carbon in rows:
        diff = carbon - esa_carbon_mg_ha
        diff_pct = pct_diff(carbon, esa_carbon_mg_ha)
        print(f"{source} | {biomass:.2f} | {carbon:.2f} | {diff:+.2f} | {diff_pct:+.2f}%")

    print("\nInterpretation note: SCANFI is live aboveground dry tree biomass, "
          "while ESA CCI is an independent mapped aboveground biomass product. "
          "Differences should be read as cross-product uncertainty, not as a "
          "field validation error.")


if __name__ == "__main__":
    main()
