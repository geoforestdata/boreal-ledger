"""
07_agb_product_diagnostics.py

Diagnoses the AGB discrepancy between ESA CCI Biomass and SCANFI over the
Abitibi analysis zone. This does not modify analysis/04_carbon.py.

The script compares biomass against biomass first. Carbon conversion with
0.47 is only reported at the end for reference.

Run: python 07_agb_product_diagnostics.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee
import pandas as pd

from carbon_sources import DEFAULT_CARBON_FRACTION, ESA_CCI_AGB_BAND, ESA_CCI_AGB_COLLECTION
from stand_age_utils import CANOPY_HEIGHT_ASSET
from zones import get_abitibi_analysis_zone

PROJECT = "ibfra2026"  # replace with your own GEE project
YEAR_ESA = 2022
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
SCANFI_ASSET = "projects/gcpm041u-lemur/assets/scanfi_v12/SCANFI_v1_2"
SCANFI_BIOMASS_BAND = "biomass"
SCANFI_LANDCOVER_BAND = "nfiLandCover"
COMMON_SCALE = 100
COMMON_SAMPLE_PIXELS = 5000

NFI_CLASSES = {
    5: "Treed broadleaf",
    6: "Treed conifer",
    7: "Treed mixed",
}


def fmt(value, digits=2):
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{value:.{digits}f}"


def pct_diff(value: float, baseline: float) -> float:
    return 100 * (value - baseline) / baseline if baseline else math.nan


def ee_summary(image: ee.Image, band: str, mask: ee.Image, aoi: ee.Geometry, scale: int, area_scale: int) -> dict:
    masked = image.select(band).updateMask(mask).rename("value")
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10, 25, 75, 90, 95, 99]), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    stats = masked.reduceRegion(
        reducer=reducer,
        geometry=aoi,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()
    valid_area = ee.Image.pixelArea().divide(10000).updateMask(mask).updateMask(masked.mask()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=area_scale,
        maxPixels=1e13,
        bestEffort=True,
    ).get("area").getInfo()
    return {
        "mean": stats.get("value_mean"),
        "median": stats.get("value_median"),
        "sd": stats.get("value_stdDev"),
        "p10": stats.get("value_p10"),
        "p25": stats.get("value_p25"),
        "p75": stats.get("value_p75"),
        "p90": stats.get("value_p90"),
        "p95": stats.get("value_p95"),
        "p99": stats.get("value_p99"),
        "min": stats.get("value_min"),
        "max": stats.get("value_max"),
        "count": stats.get("value_count"),
        "area_ha": valid_area,
    }


def print_same_mask_table(rows: list):
    print("Mask | Source | Support | Mean | Median | SD | p10 | p25 | p75 | p90 | Area ha | Count")
    for row in rows:
        print(
            f"{row['mask']} | {row['source']} | {row['support']} | "
            f"{fmt(row['mean'])} | {fmt(row['median'])} | {fmt(row['sd'])} | "
            f"{fmt(row['p10'])} | {fmt(row['p25'])} | {fmt(row['p75'])} | {fmt(row['p90'])} | "
            f"{fmt(row['area_ha'], 1)} | {row['count']}"
        )


def describe_distribution(df: pd.DataFrame, column: str, label: str):
    thresholds = [100, 150, 200]
    valid = df[column].dropna()
    print(f"{label} | min={fmt(valid.min())} | max={fmt(valid.max())} | "
          f"p95={fmt(valid.quantile(0.95))} | p99={fmt(valid.quantile(0.99))}")
    for threshold in thresholds:
        share = 100 * (valid > threshold).mean() if len(valid) else math.nan
        print(f"{label} area/cell share >{threshold} Mg/ha: {fmt(share)}%")


def main():
    ee.Initialize(project=PROJECT)

    aoi = get_abitibi_analysis_zone()
    hansen = ee.Image(HANSEN_ASSET)
    hansen_mask = hansen.select("treecover2000").gt(50).rename("hansen_forest")

    esa_collection = ee.ImageCollection(ESA_CCI_AGB_COLLECTION).filter(
        ee.Filter.calendarRange(YEAR_ESA, YEAR_ESA, "year")
    )
    esa = ee.Image(esa_collection.first()).select(ESA_CCI_AGB_BAND).rename("esa_biomass").clip(aoi)
    scanfi = ee.Image(SCANFI_ASSET).clip(aoi)
    scanfi_biomass = scanfi.select(SCANFI_BIOMASS_BAND).rename("scanfi_biomass")
    scanfi_lc = scanfi.select(SCANFI_LANDCOVER_BAND)
    scanfi_treed_mask = scanfi_lc.eq(5).Or(scanfi_lc.eq(6)).Or(scanfi_lc.eq(7)).rename("scanfi_treed")

    scanfi_projection = scanfi_biomass.projection()
    common_projection = scanfi_projection.atScale(COMMON_SCALE)

    same_mask_rows = []
    masks = [
        ("Mask 1 Hansen treecover2000 > 50", hansen_mask),
        ("Mask 2 SCANFI treed classes 5/6/7", scanfi_treed_mask),
    ]
    for mask_name, mask in masks:
        esa_stats = ee_summary(esa, "esa_biomass", mask, aoi, scale=100, area_scale=100)
        same_mask_rows.append({"mask": mask_name, "source": "ESA CCI", "support": "100 m", **esa_stats})
        scanfi_stats = ee_summary(scanfi_biomass, "scanfi_biomass", mask, aoi, scale=30, area_scale=30)
        same_mask_rows.append({"mask": mask_name, "source": "SCANFI", "support": "30 m", **scanfi_stats})

    esa_common = esa.resample("bilinear").reproject(crs=common_projection).rename("esa_biomass")
    scanfi_common = (
        scanfi_biomass
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
        .rename("scanfi_biomass")
    )
    hansen_fraction = (
        hansen_mask.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
        .rename("hansen_fraction")
    )
    scanfi_treed_fraction = (
        scanfi_treed_mask.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
        .rename("scanfi_treed_fraction")
    )
    scanfi_height_common = (
        scanfi.select("height")
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
        .rename("scanfi_height")
    )
    eth_height_common = (
        ee.Image(CANOPY_HEIGHT_ASSET)
        .rename("eth_height")
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
        .clip(aoi)
    )
    nfi_mode_common = (
        scanfi_lc.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1024)
        .reproject(crs=common_projection)
        .rename("nfi_class")
    )
    species_common = (
        scanfi.select(["balsamFir", "blackSpruce", "jackPine", "tamarack", "whiteRedPine", "prcB", "prcC"])
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=common_projection)
    )

    common_img = (
        esa_common
        .addBands(scanfi_common)
        .addBands(hansen_fraction)
        .addBands(scanfi_treed_fraction)
        .addBands(scanfi_height_common)
        .addBands(eth_height_common)
        .addBands(nfi_mode_common)
        .addBands(species_common)
    )
    samples = common_img.sample(
        region=aoi,
        projection=common_projection,
        scale=COMMON_SCALE,
        numPixels=COMMON_SAMPLE_PIXELS,
        seed=42,
        geometries=False,
        dropNulls=True,
        tileScale=4,
    ).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in samples])
    df = df[(df["hansen_fraction"] > 0) & (df["scanfi_treed_fraction"] > 0)].copy()
    df["bias"] = df["esa_biomass"] - df["scanfi_biomass"]
    df["ratio"] = df["esa_biomass"] / df["scanfi_biomass"].where(df["scanfi_biomass"] != 0)

    pearson = df[["esa_biomass", "scanfi_biomass"]].corr(method="pearson").iloc[0, 1]
    spearman = df[["esa_biomass", "scanfi_biomass"]].corr(method="spearman").iloc[0, 1]
    mean_bias = df["bias"].mean()
    median_bias = df["bias"].median()
    rmse = math.sqrt((df["bias"] ** 2).mean())
    ratio_mean = df["ratio"].mean()
    ratio_median = df["ratio"].median()

    print("=== AGB product diagnostics: ESA CCI vs SCANFI ===\n")
    print(f"Common support: SCANFI Canada Lambert projection at {COMMON_SCALE} m")
    print(f"Common-cell diagnostics use a deterministic sample capped at {COMMON_SAMPLE_PIXELS} cells")
    print("ESA common-support resampling: bilinear to common projection")
    print("SCANFI common-support aggregation: mean from 30 m to common projection\n")

    print("## Same-mask biomass summaries (Mg biomass/ha)")
    print_same_mask_table(same_mask_rows)

    print("\n## Spatial agreement on common valid cells")
    print(f"Common valid cells: {len(df)}")
    print(f"Pearson correlation: {fmt(pearson, 3)}")
    print(f"Spearman correlation: {fmt(spearman, 3)}")
    print(f"Mean bias ESA - SCANFI: {fmt(mean_bias)} Mg biomass/ha")
    print(f"Median bias ESA - SCANFI: {fmt(median_bias)} Mg biomass/ha")
    print(f"RMSE: {fmt(rmse)} Mg biomass/ha")
    print(f"Mean ratio ESA / SCANFI: {fmt(ratio_mean)}")
    print(f"Median ratio ESA / SCANFI: {fmt(ratio_median)}")

    print("\n## Height diagnostic: SCANFI height")
    height_bins = [-math.inf, 5, 10, 15, 20, math.inf]
    height_labels = ["<5 m", "5-10 m", "10-15 m", "15-20 m", ">=20 m"]
    df["scanfi_height_class"] = pd.cut(df["scanfi_height"], bins=height_bins, labels=height_labels, right=False)
    print("Height class | Cells | ESA Mg/ha | SCANFI Mg/ha | ESA/SCANFI ratio")
    for label, group in df.groupby("scanfi_height_class", observed=False):
        if group.empty:
            print(f"{label} | 0 | NA | NA | NA")
            continue
        print(f"{label} | {len(group)} | {fmt(group['esa_biomass'].mean())} | "
              f"{fmt(group['scanfi_biomass'].mean())} | {fmt(group['esa_biomass'].mean() / group['scanfi_biomass'].mean())}")

    print("\n## Height diagnostic: ETH canopy height")
    df["eth_height_class"] = pd.cut(df["eth_height"], bins=height_bins, labels=height_labels, right=False)
    print("Height class | Cells | ESA Mg/ha | SCANFI Mg/ha | ESA/SCANFI ratio")
    for label, group in df.groupby("eth_height_class", observed=False):
        if group.empty:
            print(f"{label} | 0 | NA | NA | NA")
            continue
        print(f"{label} | {len(group)} | {fmt(group['esa_biomass'].mean())} | "
              f"{fmt(group['scanfi_biomass'].mean())} | {fmt(group['esa_biomass'].mean() / group['scanfi_biomass'].mean())}")

    print("\n## SCANFI forest-type diagnostic")
    print("Forest type | Cells | ESA Mg/ha | SCANFI Mg/ha | Difference | Ratio | Dominant species cover diagnostic")
    species_cols = ["balsamFir", "blackSpruce", "jackPine", "tamarack", "whiteRedPine", "prcB", "prcC"]
    for class_id, name in NFI_CLASSES.items():
        group = df[df["nfi_class"].round().astype("Int64") == class_id]
        if group.empty:
            print(f"{name} | 0 | NA | NA | NA | NA | NA")
            continue
        esa_mean = group["esa_biomass"].mean()
        scanfi_mean = group["scanfi_biomass"].mean()
        species_means = group[species_cols].mean().sort_values(ascending=False).head(3)
        species_diag = ", ".join(f"{idx}={value:.1f}%" for idx, value in species_means.items())
        print(f"{name} | {len(group)} | {fmt(esa_mean)} | {fmt(scanfi_mean)} | "
              f"{fmt(esa_mean - scanfi_mean)} | {fmt(esa_mean / scanfi_mean)} | {species_diag}")

    print("\n## Distribution and outlier diagnostic on common valid cells")
    describe_distribution(df, "esa_biomass", "ESA CCI")
    describe_distribution(df, "scanfi_biomass", "SCANFI")

    esa_mean = df["esa_biomass"].mean()
    scanfi_mean = df["scanfi_biomass"].mean()
    print("\n## Carbon conversion after biomass diagnostics")
    print("Source | Mg biomass/ha | Mg C/ha")
    print(f"ESA CCI common cells | {fmt(esa_mean)} | {fmt(esa_mean * DEFAULT_CARBON_FRACTION)}")
    print(f"SCANFI common cells | {fmt(scanfi_mean)} | {fmt(scanfi_mean * DEFAULT_CARBON_FRACTION)}")


if __name__ == "__main__":
    main()
