"""
Sentinel-2 NBR baseline for post-disturbance recovery comparison.

Experimental AOI: Lebel-sur-Quevillon temporal-spread candidate selected in
exploration/lebel_aoi_selection/recommended_aoi_temporal_spread.geojson.

Design:
- probable harvest = non-fire residual Hansen canopy loss;
- wildfire = Hansen canopy loss spatially and temporally associated with NBAC;
- recovery condition = Sentinel-2 NBR measured in the common observation year 2024.

The wildfire result is three well-separated post-fire age classes, not a
continuous fire recovery curve. The probable-harvest residual has richer
temporal coverage and can support a descriptive spectral chronosequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recovery_utils import get_sentinel2_nbr_composite


PROJECT = "ibfra2026"
OBSERVATION_YEAR = 2024
MAX_LOSS_YEAR = OBSERVATION_YEAR - 2000
MIN_PIXELS = 100

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NBAC_COLLECTION = "projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530"
AOI_PATH = ROOT / "exploration" / "lebel_aoi_selection" / "recommended_aoi_temporal_spread.geojson"
OUTPUT_CSV = ROOT / "outputs" / "recovery_by_disturbance.csv"
FIGURE_PATH = ROOT / "figures" / "recovery_by_disturbance.png"


def load_aoi() -> ee.Geometry:
    with AOI_PATH.open() as f:
        geojson = json.load(f)
    coords = geojson["geometry"]["coordinates"]
    return ee.Geometry.Polygon(coords, proj="EPSG:4326", geodesic=False)


def grouped_area(area_img: ee.Image, class_img: ee.Image, aoi: ee.Geometry) -> dict[int, dict]:
    groups = (
        area_img.addBands(class_img.rename("class"))
        .reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
            geometry=aoi,
            scale=30,
            maxPixels=1e13,
            tileScale=4,
        )
        .get("groups")
        .getInfo()
        or []
    )
    out = {}
    for item in groups:
        class_id = int(item["class"])
        if class_id <= 0:
            continue
        area_ha = float(item["sum"])
        out[class_id] = {
            "area_ha": area_ha,
            "pixel_count": int(round(area_ha / 0.09)),
        }
    return out


def grouped_nbr_stats(nbr_img: ee.Image, class_img: ee.Image, aoi: ee.Geometry) -> dict[int, dict]:
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)
    )
    groups = (
        nbr_img.addBands(class_img.rename("class"))
        .reduceRegion(
            reducer=reducer.group(groupField=1, groupName="class"),
            geometry=aoi,
            scale=30,
            maxPixels=1e13,
            tileScale=4,
        )
        .get("groups")
        .getInfo()
        or []
    )
    out = {}
    for item in groups:
        class_id = int(item["class"])
        if class_id <= 0:
            continue
        out[class_id] = {
            "mean_NBR": item.get("mean"),
            "median_NBR": item.get("median"),
            "std_NBR": item.get("stdDev"),
            "p25_NBR": item.get("p25"),
            "p75_NBR": item.get("p75"),
        }
    return out


def build_disturbance_images(aoi: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)
    loss = lossyear.gt(0).And(lossyear.lte(MAX_LOSS_YEAR))
    loss_calendar_year = lossyear.add(2000).updateMask(loss).rename("loss_year")

    fires = ee.FeatureCollection(NBAC_COLLECTION).filterBounds(aoi)
    fire_year_img = fires.reduceToImage(
        properties=["YEAR"], reducer=ee.Reducer.first()
    ).rename("fire_year")

    fire_match = loss_calendar_year.eq(fire_year_img).unmask(0)
    wildfire_year = loss_calendar_year.updateMask(fire_match).rename("disturbance_year")
    probable_harvest_year = loss_calendar_year.updateMask(fire_match.Not()).rename("disturbance_year")
    return probable_harvest_year, wildfire_year


def summarize_disturbance_type(
    disturbance_type: str,
    class_img: ee.Image,
    nbr_img: ee.Image,
    aoi: ee.Geometry,
) -> pd.DataFrame:
    pixel_ha = ee.Image.pixelArea().divide(10000).rename("area")
    area_by_year = grouped_area(pixel_ha.updateMask(class_img), class_img, aoi)
    stats_by_year = grouped_nbr_stats(nbr_img.updateMask(class_img), class_img, aoi)
    rows = []
    for year in sorted(area_by_year):
        area_info = area_by_year[year]
        if area_info["pixel_count"] < MIN_PIXELS:
            continue
        stats = stats_by_year.get(year, {})
        rows.append(
            {
                "disturbance_type": disturbance_type,
                "disturbance_year": year,
                "years_since_disturbance": OBSERVATION_YEAR - year,
                "pixel_count": area_info["pixel_count"],
                "area_ha": area_info["area_ha"],
                "mean_NBR": stats.get("mean_NBR"),
                "median_NBR": stats.get("median_NBR"),
                "std_NBR": stats.get("std_NBR"),
                "p25_NBR": stats.get("p25_NBR"),
                "p75_NBR": stats.get("p75_NBR"),
            }
        )
    return pd.DataFrame(rows)


def write_figure(df: pd.DataFrame) -> None:
    harvest = df[df["disturbance_type"] == "probable_harvest"].sort_values("years_since_disturbance")
    fire = df[df["disturbance_type"] == "wildfire"].sort_values("years_since_disturbance")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")

    ax.plot(
        harvest["years_since_disturbance"],
        harvest["mean_NBR"],
        color="#167255",
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Probable harvest: supported age-class trajectory",
    )
    ax.errorbar(
        harvest["years_since_disturbance"],
        harvest["mean_NBR"],
        yerr=[
            harvest["mean_NBR"] - harvest["p25_NBR"],
            harvest["p75_NBR"] - harvest["mean_NBR"],
        ],
        fmt="none",
        ecolor="#167255",
        alpha=0.22,
        linewidth=1.4,
    )

    ax.scatter(
        fire["years_since_disturbance"],
        fire["mean_NBR"],
        s=92,
        color="#c55f3c",
        edgecolor="#4b2418",
        linewidth=0.8,
        zorder=5,
        label="Wildfire: three supported post-fire age classes",
    )
    for _, row in fire.iterrows():
        ax.annotate(
            str(int(row["disturbance_year"])),
            (row["years_since_disturbance"], row["mean_NBR"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            color="#4b2418",
            weight="bold",
        )

    ax.set_title("Sentinel-2 NBR by disturbance type and years since disturbance", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("Years since disturbance in 2024")
    ax.set_ylabel("Sentinel-2 NBR, 2024")
    ax.grid(True, color="#cfc7b9", linewidth=0.6, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.17,
        "NBR is spectral vegetation recovery, not biomass recovery. Wildfire points are not a fitted fire recovery curve.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4f5b53",
    )
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def matched_age_comparison(df: pd.DataFrame) -> pd.DataFrame:
    harvest = df[df["disturbance_type"] == "probable_harvest"].copy()
    fire = df[df["disturbance_type"] == "wildfire"].copy()
    rows = []
    for _, fire_row in fire.sort_values("years_since_disturbance").iterrows():
        age = int(fire_row["years_since_disturbance"])
        candidates = harvest.assign(age_gap=(harvest["years_since_disturbance"] - age).abs())
        harvest_row = candidates.sort_values(["age_gap", "years_since_disturbance"]).iloc[0]
        rows.append(
            {
                "target_age_years": age,
                "fire_year": int(fire_row["disturbance_year"]),
                "harvest_year": int(harvest_row["disturbance_year"]),
                "harvest_age_years": int(harvest_row["years_since_disturbance"]),
                "age_gap_years": int(harvest_row["age_gap"]),
                "fire_mean_NBR": fire_row["mean_NBR"],
                "harvest_mean_NBR": harvest_row["mean_NBR"],
                "mean_difference_fire_minus_harvest": fire_row["mean_NBR"] - harvest_row["mean_NBR"],
                "fire_median_NBR": fire_row["median_NBR"],
                "harvest_median_NBR": harvest_row["median_NBR"],
                "median_difference_fire_minus_harvest": fire_row["median_NBR"] - harvest_row["median_NBR"],
                "fire_std_NBR": fire_row["std_NBR"],
                "harvest_std_NBR": harvest_row["std_NBR"],
                "fire_pixel_count": int(fire_row["pixel_count"]),
                "harvest_pixel_count": int(harvest_row["pixel_count"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ee.Initialize(project=PROJECT)
    aoi = load_aoi()
    harvest_year, wildfire_year = build_disturbance_images(aoi)
    nbr = get_sentinel2_nbr_composite(OBSERVATION_YEAR, aoi)

    harvest_df = summarize_disturbance_type("probable_harvest", harvest_year, nbr, aoi)
    fire_df = summarize_disturbance_type("wildfire", wildfire_year, nbr, aoi)
    df = pd.concat([harvest_df, fire_df], ignore_index=True)
    df = df.sort_values(["disturbance_type", "disturbance_year"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    write_figure(df)

    matched = matched_age_comparison(df)

    print("SUPPORTED HARVEST CLASSES")
    print(
        harvest_df[
            [
                "disturbance_year",
                "years_since_disturbance",
                "pixel_count",
                "area_ha",
                "mean_NBR",
                "median_NBR",
                "std_NBR",
                "p25_NBR",
                "p75_NBR",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nSUPPORTED FIRE CLASSES")
    print(
        fire_df[
            [
                "disturbance_year",
                "years_since_disturbance",
                "pixel_count",
                "area_ha",
                "mean_NBR",
                "median_NBR",
                "std_NBR",
                "p25_NBR",
                "p75_NBR",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nMATCHED-AGE NBR COMPARISON")
    print(matched.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
