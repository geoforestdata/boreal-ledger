"""
10_agb_mrnf_common_footprint.py

Apples-to-apples AGB comparison over the exact MRNF valid productive-stand
footprint inside the Abitibi analysis AOI. This script does not modify the
official ESA method in analysis/04_carbon.py.

Common footprint:
  union of the intersections between MRNF stands that expose both total
  biomass (b_arbv_tot) and total carbon (c_arbv_tot) and the AOI.

Run:
  python analysis/10_agb_mrnf_common_footprint.py
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import ee
import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform, unary_union

from carbon_sources import DEFAULT_CARBON_FRACTION, ESA_CCI_AGB_BAND, ESA_CCI_AGB_COLLECTION
from zones import (
    ABITIBI_10K_CENTER_WEBMERCATOR,
    ABITIBI_10K_HALF_WIDTH_LAT,
    ABITIBI_10K_HALF_WIDTH_LON,
    get_abitibi_analysis_zone,
    point_from_webmercator,
)

MRNF_MODULE_PATH = ROOT / "analysis" / "08_agb_crosscheck_mrnf.py"
mrnf_spec = importlib.util.spec_from_file_location("agb_crosscheck_mrnf", MRNF_MODULE_PATH)
mrnf_module = importlib.util.module_from_spec(mrnf_spec)
mrnf_spec.loader.exec_module(mrnf_module)

AREA_CRS = mrnf_module.AREA_CRS
BIO_FIELD = mrnf_module.BIO_FIELD
BIO_LAYER = mrnf_module.BIO_LAYER
CARB_FIELD = mrnf_module.CARB_FIELD
CARB_LAYER = mrnf_module.CARB_LAYER
WFS_URL = mrnf_module.WFS_URL
WGS84 = mrnf_module.WGS84
fetch_wfs_gml = mrnf_module.fetch_wfs_gml
parse_layer = mrnf_module.parse_layer


PROJECT = "ibfra2026"
YEAR = 2022
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
SCANFI_ASSET = "projects/gcpm041u-lemur/assets/scanfi_v12/SCANFI_v1_2"
SCANFI_BIOMASS_BAND = "biomass"


def fmt(value, digits=2):
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{value:.{digits}f}"


def abitibi_bbox() -> tuple[float, float, float, float]:
    lon, lat = point_from_webmercator(*ABITIBI_10K_CENTER_WEBMERCATOR)
    return (
        lon - ABITIBI_10K_HALF_WIDTH_LON,
        lat - ABITIBI_10K_HALF_WIDTH_LAT,
        lon + ABITIBI_10K_HALF_WIDTH_LON,
        lat + ABITIBI_10K_HALF_WIDTH_LAT,
    )


def geodesic_area_ha(geom) -> float:
    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geom)
    return abs(area_m2) / 10000


def planar_area_ha(geom, crs: str = AREA_CRS) -> float:
    to_area = Transformer.from_crs(WGS84, crs, always_xy=True).transform
    return transform(to_area, geom).area / 10000


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "area_ha") -> float:
    valid = df[[value_col, weight_col]].dropna()
    weights = valid[weight_col]
    if valid.empty or weights.sum() == 0:
        return math.nan
    return float((valid[value_col] * weights).sum() / weights.sum())


def weighted_pearson(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    valid = pd.concat([x, y, w], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 2].sum() == 0:
        return math.nan
    xv = valid.iloc[:, 0]
    yv = valid.iloc[:, 1]
    wv = valid.iloc[:, 2]
    mx = (xv * wv).sum() / wv.sum()
    my = (yv * wv).sum() / wv.sum()
    cov = (wv * (xv - mx) * (yv - my)).sum() / wv.sum()
    vx = (wv * (xv - mx) ** 2).sum() / wv.sum()
    vy = (wv * (yv - my) ** 2).sum() / wv.sum()
    return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else math.nan


def agreement(df: pd.DataFrame, source_col: str, ref_col: str = "mrnf_biomass") -> dict:
    valid = df[[source_col, ref_col, "area_ha"]].dropna().copy()
    valid = valid[valid["area_ha"] > 0]
    bias = valid[source_col] - valid[ref_col]
    weighted_bias = weighted_mean(pd.DataFrame({"bias": bias, "area_ha": valid["area_ha"]}), "bias")
    weighted_rmse = math.sqrt(weighted_mean(pd.DataFrame({"sq": bias ** 2, "area_ha": valid["area_ha"]}), "sq"))
    return {
        "n": len(valid),
        "area_ha": valid["area_ha"].sum(),
        "pearson": valid[[source_col, ref_col]].corr(method="pearson").iloc[0, 1] if len(valid) > 1 else math.nan,
        "spearman": valid[[source_col, ref_col]].corr(method="spearman").iloc[0, 1] if len(valid) > 1 else math.nan,
        "mean_bias": bias.mean(),
        "rmse": math.sqrt((bias ** 2).mean()) if len(valid) else math.nan,
        "weighted_pearson": weighted_pearson(valid[source_col], valid[ref_col], valid["area_ha"]),
        "weighted_spearman": weighted_pearson(
            valid[source_col].rank(method="average"),
            valid[ref_col].rank(method="average"),
            valid["area_ha"],
        ),
        "weighted_bias": weighted_bias,
        "weighted_rmse": weighted_rmse,
    }


def build_mrnf_rows(aoi_lonlat):
    bio_records = parse_layer(fetch_wfs_gml(BIO_LAYER), BIO_LAYER, BIO_FIELD)
    carb_records = parse_layer(fetch_wfs_gml(CARB_LAYER), CARB_LAYER, CARB_FIELD)

    rows = []
    for geocode, bio in bio_records.items():
        if geocode not in carb_records:
            continue
        intersection = bio["geometry"].intersection(aoi_lonlat)
        if intersection.is_empty:
            continue
        area_ha = planar_area_ha(intersection)
        if area_ha <= 0:
            continue
        rows.append(
            {
                "geocode": geocode,
                "geometry": intersection,
                "area_ha": area_ha,
                "mrnf_biomass": bio["value"],
                "mrnf_carbon": carb_records[geocode]["value"],
            }
        )
    return rows


def ee_feature_collection(rows):
    features = []
    for row in rows:
        geom = ee.Geometry(mapping(row["geometry"]), proj=WGS84, geodesic=False)
        features.append(
            ee.Feature(
                geom,
                {
                    "geocode": row["geocode"],
                    "area_ha": row["area_ha"],
                    "mrnf_biomass": row["mrnf_biomass"],
                    "mrnf_carbon": row["mrnf_carbon"],
                },
            )
        )
    return ee.FeatureCollection(features)


def raster_weighted_mean(image: ee.Image, geometry, scale: int, projection=None) -> tuple[float, float]:
    pixel_ha = ee.Image.pixelArea().divide(10000)
    value = image.rename("value")
    weighted = value.multiply(pixel_ha).rename("weighted")
    area = pixel_ha.updateMask(value.mask()).rename("area")
    kwargs = {
        "reducer": ee.Reducer.sum(),
        "geometry": geometry,
        "scale": scale,
        "maxPixels": 1e13,
        "bestEffort": True,
        "tileScale": 4,
    }
    if projection is not None:
        kwargs["crs"] = projection
    stats = weighted.addBands(area).reduceRegion(**kwargs).getInfo()
    total = stats.get("weighted", 0) or 0
    valid_area = stats.get("area", 0) or 0
    return (total / valid_area if valid_area else math.nan, valid_area)


def esa_method_c_on_footprint(esa_biomass: ee.Image, forest_mask: ee.Image, footprint) -> tuple[float, float]:
    pixel_ha = ee.Image.pixelArea().divide(10000)
    esa_projection = esa_biomass.projection()
    forest_fraction = (
        forest_mask.unmask(0)
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=esa_projection, scale=100)
        .rename("forest_fraction")
    )
    weighted_biomass = esa_biomass.multiply(forest_fraction).multiply(pixel_ha).rename("weighted_biomass")
    weighted_area = forest_fraction.multiply(pixel_ha).rename("forest_fraction_area_ha")
    stats = weighted_biomass.addBands(weighted_area).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=footprint,
        crs=esa_projection,
        scale=100,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).getInfo()
    total = stats.get("weighted_biomass", 0) or 0
    valid_area = stats.get("forest_fraction_area_ha", 0) or 0
    return (total / valid_area if valid_area else math.nan, valid_area)


def reduce_stands(fc, image: ee.Image, band_name: str, scale: int) -> pd.DataFrame:
    reduced = image.rename(band_name).reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        maxPixelsPerRegion=1e9,
        tileScale=4,
    )
    features = reduced.getInfo()["features"]
    rows = []
    for feature in features:
        props = feature["properties"]
        rows.append(
            {
                "geocode": props["geocode"],
                band_name: props.get("mean"),
            }
        )
    return pd.DataFrame(rows)


def print_agreement(label: str, stats: dict):
    print(f"{label} | N={stats['n']} | area={fmt(stats['area_ha'], 1)} ha | "
          f"Pearson={fmt(stats['pearson'], 3)} | Spearman={fmt(stats['spearman'], 3)} | "
          f"mean bias={fmt(stats['mean_bias'])} | RMSE={fmt(stats['rmse'])}")
    print(f"{label} area-weighted | Pearson={fmt(stats['weighted_pearson'], 3)} | "
          f"Spearman={fmt(stats['weighted_spearman'], 3)} | "
          f"bias={fmt(stats['weighted_bias'])} | RMSE={fmt(stats['weighted_rmse'])}")


def main():
    ee.Initialize(project=PROJECT)

    aoi = get_abitibi_analysis_zone()
    aoi_bbox = abitibi_bbox()
    aoi_lonlat = box(*aoi_bbox)
    rows = build_mrnf_rows(aoi_lonlat)
    if not rows:
        raise RuntimeError("No MRNF biomass/carbon stands intersected the AOI.")

    footprint = unary_union([row["geometry"] for row in rows])
    footprint_area_ha = planar_area_ha(footprint)
    mrnf_df = pd.DataFrame([{k: v for k, v in row.items() if k != "geometry"} for row in rows])

    # The ~10,009.5 ha value from Earth Engine is geodesic area of the lon/lat
    # rectangle. The MRNF script reports planar area after projection to Quebec
    # Lambert. Both use the same bbox coordinates.
    ee_area_ha = aoi.area().divide(10000).getInfo()
    geod_area = geodesic_area_ha(aoi_lonlat)
    planar_area = planar_area_ha(aoi_lonlat)

    fc = ee_feature_collection(rows)
    footprint_ee = fc.geometry()

    esa = (
        ee.ImageCollection(ESA_CCI_AGB_COLLECTION)
        .filter(ee.Filter.calendarRange(YEAR, YEAR, "year"))
        .select(ESA_CCI_AGB_BAND)
        .mosaic()
        .clip(aoi)
        .rename("esa_biomass")
    )
    scanfi = ee.Image(SCANFI_ASSET).select(SCANFI_BIOMASS_BAND).clip(aoi).rename("scanfi_biomass")
    hansen_forest = ee.Image(HANSEN_ASSET).select("treecover2000").gt(50)

    esa_raw_mean, esa_raw_valid_area = raster_weighted_mean(esa, footprint_ee, 100, projection=esa.projection())
    esa_c_mean, esa_c_area = esa_method_c_on_footprint(esa, hansen_forest, footprint_ee)
    scanfi_mean, scanfi_valid_area = raster_weighted_mean(scanfi, footprint_ee, 30, projection=scanfi.projection())

    mrnf_biomass = weighted_mean(mrnf_df, "mrnf_biomass")
    mrnf_carbon = weighted_mean(mrnf_df, "mrnf_carbon")

    esa_stands = reduce_stands(fc, esa, "esa_biomass", 100)
    scanfi_stands = reduce_stands(fc, scanfi, "scanfi_biomass", 30)
    stand_df = mrnf_df.merge(esa_stands, on="geocode", how="left").merge(scanfi_stands, on="geocode", how="left")

    esa_agreement = agreement(stand_df, "esa_biomass")
    scanfi_agreement = agreement(stand_df, "scanfi_biomass")

    bins = [-math.inf, 50, 75, 100, 125, math.inf]
    labels = ["<50", "50-75", "75-100", "100-125", ">=125"]
    stand_df["mrnf_class"] = pd.cut(stand_df["mrnf_biomass"], bins=bins, labels=labels, right=False)

    print("=== MRNF common-footprint AGB diagnostic ===\n")
    print("## AOI area discrepancy")
    print("Method | CRS | AOI area ha")
    print(f"Earth Engine aoi.area() | geodesic WGS84 geometry | {ee_area_ha:,.1f}")
    print(f"Local pyproj.Geod | geodesic WGS84 polygon | {geod_area:,.1f}")
    print(f"Local shapely projected area | {AREA_CRS} Quebec Lambert | {planar_area:,.1f}")
    print("Same lon/lat bbox coordinates used:")
    print(f"{aoi_bbox}\n")

    print("## Common footprint")
    print(f"MRNF WFS endpoint: {WFS_URL}")
    print(f"Valid productive MRNF stands: {len(rows)}")
    print(f"MRNF_VALID_FOOTPRINT planar area: {footprint_area_ha:,.1f} ha")
    print(f"Weighted MRNF biomass check: {mrnf_biomass:.2f} Mg biomass/ha")
    print(f"Weighted MRNF carbon check: {mrnf_carbon:.2f} Mg C/ha\n")

    print("## Same-footprint comparison")
    print("Source | Common area ha | Biomass Mg/ha | Carbon Mg C/ha")
    print(f"MRNF | {footprint_area_ha:,.1f} | {mrnf_biomass:.2f} | {mrnf_carbon:.2f}")
    print(f"ESA CCI raw footprint | {esa_raw_valid_area:,.1f} | {esa_raw_mean:.2f} | {esa_raw_mean * DEFAULT_CARBON_FRACTION:.2f}")
    print(f"ESA CCI forest-fraction footprint | {esa_c_area:,.1f} | {esa_c_mean:.2f} | {esa_c_mean * DEFAULT_CARBON_FRACTION:.2f}")
    print(f"SCANFI | {scanfi_valid_area:,.1f} | {scanfi_mean:.2f} | {scanfi_mean * DEFAULT_CARBON_FRACTION:.2f}")
    print(f"\nESA raw / MRNF ratio: {esa_raw_mean / mrnf_biomass:.2f}")
    print(f"ESA Method C / MRNF ratio: {esa_c_mean / mrnf_biomass:.2f}")
    print(f"SCANFI / MRNF ratio: {scanfi_mean / mrnf_biomass:.2f}")
    print(f"ESA raw - MRNF bias: {esa_raw_mean - mrnf_biomass:+.2f} Mg biomass/ha")
    print(f"ESA Method C - MRNF bias: {esa_c_mean - mrnf_biomass:+.2f} Mg biomass/ha")
    print(f"SCANFI - MRNF bias: {scanfi_mean - mrnf_biomass:+.2f} Mg biomass/ha\n")

    print("## Stand-level agreement")
    print_agreement("ESA vs MRNF", esa_agreement)
    print_agreement("SCANFI vs MRNF", scanfi_agreement)
    print()

    print("## MRNF biomass-class diagnostic")
    print("MRNF class | Area ha | MRNF | ESA | SCANFI | ESA/MRNF | SCANFI/MRNF")
    for label, group in stand_df.groupby("mrnf_class", observed=False):
        if group.empty:
            print(f"{label} | 0.0 | NA | NA | NA | NA | NA")
            continue
        area = group["area_ha"].sum()
        mrnf_mean = weighted_mean(group, "mrnf_biomass")
        esa_mean = weighted_mean(group, "esa_biomass")
        scanfi_class_mean = weighted_mean(group, "scanfi_biomass")
        print(
            f"{label} | {area:,.1f} | {fmt(mrnf_mean)} | {fmt(esa_mean)} | "
            f"{fmt(scanfi_class_mean)} | {fmt(esa_mean / mrnf_mean)} | "
            f"{fmt(scanfi_class_mean / mrnf_mean)}"
        )

    output_csv = ROOT / "outputs" / "mrnf_common_footprint_stand_diagnostics.csv"
    output_csv.parent.mkdir(exist_ok=True)
    stand_df.to_csv(output_csv, index=False)
    print(f"\nStand diagnostics CSV: {output_csv}")


if __name__ == "__main__":
    main()
