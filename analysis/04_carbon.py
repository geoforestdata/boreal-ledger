"""
04_carbon.py

Consolidated carbon-product summary for the Abitibi analysis zone.

AGB estimates are strongly product-dependent in this AOI. This script no
longer presents ESA CCI as a single definitive AGB estimate. It reports two
wall-to-wall mapped AGB products (ESA CCI Method C and SCANFI v1.2), plus
the MRNF Quebec provincial productive-stand reference derived in
08_agb_crosscheck_mrnf.py and 10_agb_mrnf_common_footprint.py.

This is not direct field validation. MRNF is the most locally relevant
provincial inventory-derived reference for productive stands, but it is not
ground truth for the whole AOI. On the identical MRNF footprint, ESA shows a
strong positive bias relative to MRNF; SCANFI shows a negative bias relative
to MRNF, with lower absolute bias and RMSE than ESA.

SOC is reported only as SoilGrids soil organic carbon stock, 0-30 cm. AGB
and SOC are kept as separate mapped pools; no total ecosystem carbon or total
forest carbon is computed.

Run: python 04_carbon.py
"""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


import ee
from zones import get_abitibi_analysis_zone
from carbon_sources import DEFAULT_CARBON_FRACTION, ESA_CCI_AGB_BAND, ESA_CCI_AGB_COLLECTION, get_esa_cci_carbon
from soil_carbon import get_soil_carbon_0_30cm

PROJECT = "ibfra2026"  # replace with your own GEE project
YEAR = 2022
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
SCANFI_ASSET = "projects/gcpm041u-lemur/assets/scanfi_v12/SCANFI_v1_2"
SCANFI_BIOMASS_BAND = "biomass"

# Provincial productive-stand reference from:
#   analysis/08_agb_crosscheck_mrnf.py
#   analysis/10_agb_mrnf_common_footprint.py
# These are area-weighted over MRNF productive stands that intersect the AOI.
MRNF_COVERAGE_HA = 2489.6
MRNF_BIOMASS_MG_HA = 102.97
MRNF_CARBON_MGC_HA = 51.06


def write_product_comparison(rows: list[dict]) -> Path:
    output_path = ROOT / "outputs" / "carbon_product_comparison.csv"
    output_path.parent.mkdir(exist_ok=True)
    fieldnames = [
        "source",
        "spatial_scope",
        "coverage_ha",
        "coverage_pct_aoi",
        "pool",
        "biomass_Mg_ha",
        "carbon_MgC_ha",
        "reference_type",
        "notes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    ee.Initialize(project=PROJECT)
    aoi = get_abitibi_analysis_zone()
    pixel_ha = ee.Image.pixelArea().divide(10000)

    forest_mask = ee.Image(HANSEN_ASSET).select("treecover2000").gt(50)

    agb_density = get_esa_cci_carbon(YEAR, aoi)
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
    agb_weighted = agb_density.multiply(forest_fraction).multiply(pixel_ha).rename("agb_carbon_Mg")
    forest_fraction_area = forest_fraction.multiply(pixel_ha).rename("forest_fraction_area_ha")
    agb_stats = agb_weighted.addBands(forest_fraction_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=100, maxPixels=1e13, bestEffort=True,
    ).getInfo()
    agb_total_mg = agb_stats.get("agb_carbon_Mg", 0) or 0
    forest_fraction_ha = agb_stats.get("forest_fraction_area_ha", 0) or 0
    agb_mg_ha = agb_total_mg / forest_fraction_ha if forest_fraction_ha else 0
    esa_biomass_mg_ha = agb_mg_ha / DEFAULT_CARBON_FRACTION if DEFAULT_CARBON_FRACTION else 0

    scanfi_biomass = ee.Image(SCANFI_ASSET).select(SCANFI_BIOMASS_BAND).rename("scanfi_biomass_Mg_ha")
    scanfi_masked = scanfi_biomass.updateMask(forest_mask)
    scanfi_stats = scanfi_masked.addBands(
        pixel_ha.updateMask(forest_mask).updateMask(scanfi_masked.mask()).rename("area_ha")
    ).reduceRegion(
        reducer=ee.Reducer.mean().setOutputs(["scanfi_biomass_Mg_ha"]).combine(
            ee.Reducer.sum().setOutputs(["scanfi_valid_area_ha"]),
            sharedInputs=False,
        ),
        geometry=aoi,
        scale=30,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    scanfi_biomass_mg_ha = scanfi_stats.get("scanfi_biomass_Mg_ha", 0) or 0
    scanfi_carbon_mg_ha = scanfi_biomass_mg_ha * DEFAULT_CARBON_FRACTION
    scanfi_valid_area_ha = scanfi_stats.get("scanfi_valid_area_ha", 0) or 0

    soc_density = get_soil_carbon_0_30cm(aoi)
    soc_total_mg = soc_density.multiply(pixel_ha).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=250, maxPixels=1e13, bestEffort=True,
    ).get("soc_stock_MgC_ha").getInfo()
    box_ha = aoi.area().divide(10000).getInfo()
    soc_mg_ha = soc_total_mg / box_ha if box_ha else 0
    mrnf_coverage_pct = 100 * MRNF_COVERAGE_HA / box_ha if box_ha else 0

    rows = [
        {
            "source": "ESA CCI Biomass",
            "spatial_scope": "forested AOI / Method C",
            "coverage_ha": f"{forest_fraction_ha:.1f}",
            "coverage_pct_aoi": f"{100 * forest_fraction_ha / box_ha:.1f}" if box_ha else "",
            "pool": "aboveground biomass",
            "biomass_Mg_ha": f"{esa_biomass_mg_ha:.2f}",
            "carbon_MgC_ha": f"{agb_mg_ha:.2f}",
            "reference_type": "global raster product",
            "notes": "High AGB estimate; Hansen treecover2000 > 50 aggregated to ESA support as explicit forest fraction.",
        },
        {
            "source": "SCANFI v1.2",
            "spatial_scope": "forested AOI",
            "coverage_ha": f"{scanfi_valid_area_ha:.1f}",
            "coverage_pct_aoi": f"{100 * scanfi_valid_area_ha / box_ha:.1f}" if box_ha else "",
            "pool": "live aboveground tree biomass",
            "biomass_Mg_ha": f"{scanfi_biomass_mg_ha:.2f}",
            "carbon_MgC_ha": f"{scanfi_carbon_mg_ha:.2f}",
            "reference_type": "Canadian raster product",
            "notes": "Low AGB estimate; carbon reported only by applying the common 0.47 comparison factor.",
        },
        {
            "source": "MRNF Quebec",
            "spatial_scope": "productive stands covered by MRNF",
            "coverage_ha": f"{MRNF_COVERAGE_HA:.1f}",
            "coverage_pct_aoi": f"{mrnf_coverage_pct:.1f}",
            "pool": "aboveground live-tree biomass/carbon",
            "biomass_Mg_ha": f"{MRNF_BIOMASS_MG_HA:.2f}",
            "carbon_MgC_ha": f"{MRNF_CARBON_MGC_HA:.2f}",
            "reference_type": "provincial inventory-derived productive-stand reference",
            "notes": "Area-weighted over 431 productive MRNF peuplements; not whole-AOI ground truth.",
        },
        {
            "source": "SoilGrids",
            "spatial_scope": "AOI",
            "coverage_ha": f"{box_ha:.1f}",
            "coverage_pct_aoi": "100.0",
            "pool": "soil organic carbon stock, 0-30 cm",
            "biomass_Mg_ha": "",
            "carbon_MgC_ha": f"{soc_mg_ha:.1f}",
            "reference_type": "global soil raster product",
            "notes": "SOC 0-30 cm only; not whole-profile soil carbon and not summed with AGB here.",
        },
    ]
    output_path = write_product_comparison(rows)

    print("=== Carbon, Abitibi analysis zone (~10,000 ha) ===\n")
    print("Wall-to-wall mapped AGB estimates")
    print(f"- ESA CCI Method C ({YEAR}, explicit Hansen forest fraction): "
          f"{esa_biomass_mg_ha:.2f} Mg biomass/ha = {agb_mg_ha:.2f} Mg C/ha")
    print(f"  Forest-fraction area at ESA support: {forest_fraction_ha:,.1f} ha")
    print(f"- SCANFI v1.2: {scanfi_biomass_mg_ha:.2f} Mg biomass/ha = "
          f"{scanfi_carbon_mg_ha:.2f} Mg C/ha")
    print(f"  Valid forested AOI area at SCANFI support: {scanfi_valid_area_ha:,.1f} ha\n")

    print("Provincial productive-stand reference")
    print(f"- MRNF Quebec: {MRNF_BIOMASS_MG_HA:.2f} Mg biomass/ha = "
          f"{MRNF_CARBON_MGC_HA:.2f} Mg C/ha")
    print(f"  Coverage: {MRNF_COVERAGE_HA:,.1f} ha ({mrnf_coverage_pct:.1f}% of AOI), "
          "431 productive peuplements\n")

    print("Soil organic carbon stock, 0-30 cm")
    print(f"Soil organic carbon stock, 0-30 cm (whole zone, SoilGrids): "
          f"{soc_mg_ha:.1f} Mg C/ha\n")

    print("Interpretation")
    print("- AGB estimates are strongly product-dependent in this AOI.")
    print("- ESA CCI is the high estimate; SCANFI is the low estimate.")
    print("- MRNF lies between them on the productive-forest subset.")
    print("- On the identical MRNF footprint, ESA has a strong positive bias relative "
          "to MRNF, while SCANFI has a negative bias and lower absolute bias/RMSE.")
    print("- This is not direct field validation; MRNF is the most locally relevant "
          "provincial inventory-derived reference for productive stands, not ground truth.")
    print("- AGB and SOC are separate pools here; no total ecosystem carbon or total "
          "forest carbon is reported.\n")
    print(f"Saved product comparison: {output_path}")


if __name__ == "__main__":
    main()
