"""
05_agb_sensitivity.py

Sensitivity check for the Abitibi AGB carbon estimate. This script
documents the method comparison that led analysis/04_carbon.py to use
Method C as the official AGB calculation.

Methods:
  A. Original method used before the Method C update:
     ESA carbon density x pixel area, masked by Hansen forest, summed at
     100 m, divided by Hansen forest area computed at 30 m.
  B. Direct mean:
     Mean ESA carbon density over the Hansen forest mask, evaluated at
     ESA scale.
  C. Explicit forest fraction:
     Hansen forest mask aggregated to the ESA support as forest fraction,
     then ESA carbon density is area-weighted by that fraction.

Run: python 05_agb_sensitivity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee

from carbon_sources import ESA_CCI_AGB_BAND, ESA_CCI_AGB_COLLECTION, get_esa_cci_carbon
from zones import get_abitibi_analysis_zone

PROJECT = "ibfra2026"  # replace with your own GEE project
YEAR = 2022
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"


def pct_diff(value: float, baseline: float) -> float:
    return 100 * (value - baseline) / baseline if baseline else 0


def main():
    ee.Initialize(project=PROJECT)

    aoi = get_abitibi_analysis_zone()
    pixel_ha = ee.Image.pixelArea().divide(10000)

    hansen = ee.Image(HANSEN_ASSET)
    forest_mask = hansen.select("treecover2000").gt(50)
    carbon_density = get_esa_cci_carbon(YEAR, aoi)

    esa_collection = ee.ImageCollection(ESA_CCI_AGB_COLLECTION).filter(
        ee.Filter.calendarRange(YEAR, YEAR, "year")
    )
    esa_agb = esa_collection.select(ESA_CCI_AGB_BAND).mosaic().clip(aoi)
    esa_projection = esa_agb.projection()

    aoi_ha = aoi.area().divide(10000).getInfo()
    forest_ha_30m = pixel_ha.updateMask(forest_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=30,
        maxPixels=1e13,
        bestEffort=True,
    ).get("area").getInfo()

    # Method A: original method used before the Method C update.
    method_a_total = carbon_density.updateMask(forest_mask).multiply(pixel_ha).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    ).get("carbon_Mg_ha").getInfo()
    method_a = method_a_total / forest_ha_30m if forest_ha_30m else 0

    # Method B: direct masked mean at the ESA nominal scale.
    method_b = carbon_density.updateMask(forest_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    ).get("carbon_Mg_ha").getInfo()

    # Method C: aggregate Hansen forest mask into the ESA grid as a fractional
    # weight before applying it to ESA carbon density.
    forest_fraction = (
        forest_mask.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=esa_projection, scale=100)
        .rename("forest_fraction")
    )
    weighted_carbon = carbon_density.multiply(forest_fraction).multiply(pixel_ha).rename("weighted_carbon_Mg")
    weighted_area = forest_fraction.multiply(pixel_ha).rename("forest_fraction_area_ha")
    method_c_stats = weighted_carbon.addBands(weighted_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    method_c_area = method_c_stats.get("forest_fraction_area_ha", 0) or 0
    method_c_total = method_c_stats.get("weighted_carbon_Mg", 0) or 0
    method_c = method_c_total / method_c_area if method_c_area else 0

    forest_fraction_mean = forest_fraction.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
    ).get("forest_fraction").getInfo()

    print("=== AGB carbon sensitivity, Abitibi analysis zone ===\n")
    print("Note: Method C is now the official method in analysis/04_carbon.py.")
    print(f"ESA CCI collection: {ESA_CCI_AGB_COLLECTION}")
    print(f"ESA CCI selected band: {ESA_CCI_AGB_BAND}")
    print(f"ESA CCI bands: {esa_collection.first().bandNames().getInfo()}")
    print(f"ESA CCI projection: {esa_projection.getInfo()}")
    print(f"Hansen forest mask: treecover2000 > 50")
    print(f"AOI area: {aoi_ha:,.1f} ha")
    print(f"Hansen forest area at 30 m: {forest_ha_30m:,.1f} ha")
    print(f"Mean forest fraction at ESA support: {forest_fraction_mean:.3f}")
    print(f"Forest-fraction area at ESA support: {method_c_area:,.1f} ha\n")

    rows = [
        ("A - original mask/sum", method_a),
        ("B - direct mean", method_b),
        ("C - explicit forest fraction", method_c),
    ]

    print("Method | Mg C/ha | Delta vs A | Delta %")
    for name, value in rows:
        delta = value - method_a
        delta_pct = pct_diff(value, method_a)
        print(f"{name} | {value:.2f} | {delta:+.2f} | {delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
