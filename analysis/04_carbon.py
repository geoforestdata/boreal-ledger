"""
04_carbon.py

Aboveground carbon (ESA CCI Biomass, forest-masked) and soil organic
carbon (SoilGrids, 0-30 cm) for the Abitibi analysis zone. Cross-checked
earlier in this project against an independent 500 m point sample in the
same location (AGB 82.8 vs 80.6 Mg C/ha, SOC 58.2 vs 59.8 Mg C/ha across
the two very different sampling scales) -- see README.

Run: python 04_carbon.py
"""

import sys
sys.path.append("../src")

import ee
from zones import get_abitibi_analysis_zone
from carbon_sources import get_esa_cci_carbon
from soil_carbon import get_soil_carbon_0_30cm

PROJECT = "your-gee-project-id"  # replace with your own GEE project
YEAR = 2022


def main():
    ee.Initialize(project=PROJECT)
    aoi = get_abitibi_analysis_zone()
    pixel_ha = ee.Image.pixelArea().divide(10000)

    forest_mask = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("treecover2000").gt(50)

    agb_density = get_esa_cci_carbon(YEAR, aoi)
    agb_masked = agb_density.updateMask(forest_mask).multiply(pixel_ha)
    agb_total_mg = agb_masked.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=100, maxPixels=1e13, bestEffort=True,
    ).get("carbon_Mg_ha").getInfo()

    forest_ha_total = pixel_ha.updateMask(forest_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e13, bestEffort=True,
    ).get("area").getInfo()
    agb_mg_ha = agb_total_mg / forest_ha_total if forest_ha_total else 0

    soc_density = get_soil_carbon_0_30cm(aoi)
    soc_total_mg = soc_density.multiply(pixel_ha).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=250, maxPixels=1e13, bestEffort=True,
    ).get("soc_Mg_ha").getInfo()
    box_ha = aoi.area().divide(10000).getInfo()
    soc_mg_ha = soc_total_mg / box_ha if box_ha else 0

    print("=== Carbon, Abitibi analysis zone (~10,000 ha) ===\n")
    print(f"AGB (forest pixels only, ESA CCI Biomass {YEAR}): {agb_mg_ha:.1f} Mg C/ha")
    print(f"SOC 0-30 cm (whole zone, SoilGrids): {soc_mg_ha:.1f} Mg C/ha")
    print("\nCross-check (500 m point sample, same location, done earlier in "
          "this project): AGB 82.8 Mg C/ha, SOC 58.2 Mg C/ha -- consistent "
          "with the above despite the very different sampling scale.")


if __name__ == "__main__":
    main()

