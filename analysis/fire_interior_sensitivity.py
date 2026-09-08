"""
Fire-interior sensitivity for AlphaEarth disturbance-type separability.

This test excludes NBAC fire-edge pixels before sampling by eroding fire
polygons by 60, 90, and 150 m. It then repeats one-to-one geographic
matching and the same simple logistic-regression classifiers used in the
strict validation: NBR, Sentinel-2 spectral bands, and AlphaEarth.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from validate_alphaearth_disturbance import spatial_cv


PROJECT = "ibfra2026"
OBSERVATION_YEAR = 2024
YEARS = [2023, 2012, 2005]
BUFFERS_M = [60, 90, 150]
MAX_SAMPLE_POOL = 5000
RANDOM_SEED = 77
LOCAL_CRS = "EPSG:6622"

AOI_PATH = ROOT / "exploration" / "lebel_aoi_selection" / "recommended_aoi_temporal_spread.geojson"
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NBAC_COLLECTION = "projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530"
ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

AVAILABILITY_CSV = ROOT / "outputs" / "fire_interior_sensitivity.csv"
MODEL_CSV = ROOT / "outputs" / "fire_interior_model_comparison.csv"
FIGURE_PATH = ROOT / "figures" / "fire_interior_sensitivity.png"


def load_aoi() -> ee.Geometry:
    with AOI_PATH.open() as f:
        geojson = json.load(f)
    return ee.Geometry.Polygon(geojson["geometry"]["coordinates"], proj="EPSG:4326", geodesic=False)


def sentinel2_image(aoi: ee.Geometry) -> ee.Image:
    def mask_clouds(image):
        scl = image.select("SCL")
        mask = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(mask).divide(10000)

    composite = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2024-06-01", "2024-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
        .median()
        .clip(aoi)
    )
    nbr = composite.normalizedDifference(["B8", "B12"]).rename("NBR_2024")
    return composite.select(["B2", "B3", "B4", "B8", "B11", "B12"]).addBands(nbr)


def alphaearth_image(aoi: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection(ALPHAEARTH_COLLECTION)
        .filterDate("2024-01-01", "2025-01-01")
        .filterBounds(aoi)
        .mosaic()
        .clip(aoi)
    )


def disturbance_masks(aoi: ee.Geometry, year: int, buffer_m: int) -> tuple[ee.Image, ee.Image]:
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)
    loss_year = lossyear.add(2000).updateMask(lossyear.gt(0).And(lossyear.lte(24)))
    all_fire_year = (
        ee.FeatureCollection(NBAC_COLLECTION)
        .filterBounds(aoi)
        .reduceToImage(properties=["YEAR"], reducer=ee.Reducer.first())
        .rename("fire_year")
    )
    harvest = loss_year.eq(year).And(loss_year.eq(all_fire_year).unmask(0).Not()).selfMask()

    fires = ee.FeatureCollection(NBAC_COLLECTION).filterBounds(aoi).filter(ee.Filter.eq("YEAR", year))
    fire_mask = ee.Image().byte().paint(fires, 1).selfMask()
    fire_interior_img = fire_mask.focal_min(radius=buffer_m, units="meters").selfMask()
    wildfire = loss_year.eq(year).And(fire_interior_img).selfMask()
    return harvest.rename("class_mask"), wildfire.rename("class_mask")


def count_pixels(mask: ee.Image, aoi: ee.Geometry) -> dict:
    pixel_ha = ee.Image.pixelArea().divide(10000).rename("area")
    result = (
        pixel_ha.updateMask(mask)
        .reduceRegion(
            reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=aoi,
            scale=30,
            maxPixels=1e13,
            tileScale=4,
        )
        .getInfo()
    )
    area = float(result.get("area_sum") or result.get("constant_sum") or 0)
    count = int(round(area / 0.09))
    if result.get("area_count") is not None:
        count = int(result["area_count"])
    elif result.get("constant_count") is not None:
        count = int(result["constant_count"])
    return {"area_ha": area, "pixel_count": count}


def sample_group(image: ee.Image, mask: ee.Image, aoi: ee.Geometry, dtype: str, year: int, buffer_m: int, n: int) -> pd.DataFrame:
    seed = RANDOM_SEED + year + buffer_m + (0 if dtype == "wildfire" else 10000)
    fc = (
        image.updateMask(mask)
        .sample(region=aoi, scale=30, factor=1, seed=seed, geometries=True, tileScale=4)
        .randomColumn("random", seed)
        .sort("random")
        .limit(n)
    )
    rows = []
    for i, feature in enumerate(fc.getInfo()["features"]):
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"]
        row = {
            "sample_id": f"{dtype}_{year}_{buffer_m}_{i:05d}",
            "disturbance_type": dtype,
            "disturbance_year": year,
            "years_since_disturbance": OBSERVATION_YEAR - year,
            "interior_buffer_m": buffer_m,
            "lon": lon,
            "lat": lat,
            "NBR_2024": props.get("NBR_2024"),
            "B2": props.get("B2"),
            "B3": props.get("B3"),
            "B4": props.get("B4"),
            "B8": props.get("B8"),
            "B11": props.get("B11"),
            "B12": props.get("B12"),
        }
        for j in range(64):
            band = f"A{j:02d}"
            row[f"embedding_{band}"] = props.get(band)
        rows.append(row)
    return pd.DataFrame(rows)


def add_xy(df: pd.DataFrame) -> pd.DataFrame:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x_m"] = x
    out["y_m"] = y
    return out


def distance_matrix(fire: pd.DataFrame, harvest: pd.DataFrame) -> np.ndarray:
    f = fire[["x_m", "y_m"]].to_numpy(dtype=float)
    h = harvest[["x_m", "y_m"]].to_numpy(dtype=float)
    return np.sqrt(((f[:, None, :] - h[None, :, :]) ** 2).sum(axis=2))


def greedy_match(fire: pd.DataFrame, harvest: pd.DataFrame, max_distance_m: int) -> tuple[pd.DataFrame, dict]:
    dmat = distance_matrix(fire, harvest)
    candidates = np.argwhere(dmat <= max_distance_m)
    order = np.argsort(dmat[candidates[:, 0], candidates[:, 1]]) if len(candidates) else []
    used_fire = set()
    used_harvest = set()
    rows = []
    pair_distances = []
    pair_id = 0
    for item in order:
        i, j = map(int, candidates[item])
        if i in used_fire or j in used_harvest:
            continue
        used_fire.add(i)
        used_harvest.add(j)
        dist = float(dmat[i, j])
        pair_distances.append(dist)
        midpoint_x = (fire.iloc[i]["x_m"] + harvest.iloc[j]["x_m"]) / 2
        midpoint_y = (fire.iloc[i]["y_m"] + harvest.iloc[j]["y_m"]) / 2
        for source_df, idx in [(fire, i), (harvest, j)]:
            row = source_df.iloc[idx].copy()
            row["pair_id"] = f"pair_{pair_id:05d}"
            row["pair_distance_km"] = dist / 1000
            row["pair_midpoint_x_m"] = midpoint_x
            row["pair_midpoint_y_m"] = midpoint_y
            rows.append(row)
        pair_id += 1
    matched = pd.DataFrame(rows)
    if not matched.empty:
        block_m = 5000
        bx = np.floor(matched["pair_midpoint_x_m"].to_numpy() / block_m).astype(int)
        by = np.floor(matched["pair_midpoint_y_m"].to_numpy() / block_m).astype(int)
        matched["fold_5km"] = (bx * 73856093 + by * 19349663) % 5
    summary = {
        "n_matched_pairs": int(len(pair_distances)),
        "mean_pair_distance_km": float(np.mean(pair_distances) / 1000) if pair_distances else np.nan,
        "median_pair_distance_km": float(np.median(pair_distances) / 1000) if pair_distances else np.nan,
        "max_pair_distance_km": float(np.max(pair_distances) / 1000) if pair_distances else np.nan,
    }
    return matched, summary


def run_models(matched: pd.DataFrame) -> list[dict]:
    model_specs = {
        "NBR": ["NBR_2024"],
        "Sentinel-2": ["B2", "B3", "B4", "B8", "B11", "B12", "NBR_2024"],
        "AlphaEarth": [f"embedding_A{i:02d}" for i in range(64)],
    }
    rows = []
    if matched.empty or matched["pair_id"].nunique() < 50:
        return rows
    age = int(matched["years_since_disturbance"].iloc[0])
    for model, cols in model_specs.items():
        clean = matched.dropna(subset=cols).copy()
        if clean["pair_id"].nunique() < 50:
            continue
        result = spatial_cv(clean, cols, age=age, block_km=5)
        rows.append(
            {
                "years_since_disturbance": age,
                "disturbance_year": int(clean["disturbance_year"].iloc[0]),
                "interior_buffer_m": int(clean["interior_buffer_m"].iloc[0]),
                "model": model,
                "n_matched_pairs": int(clean["pair_id"].nunique()),
                "spatial_block_km": 5,
                "valid_folds": int(result["spatial_folds"]),
                "balanced_accuracy": result["balanced_accuracy"],
                "roc_auc": result["roc_auc"],
                "precision": result["precision"],
                "recall": result["recall"],
            }
        )
    return rows


def plot_results(model_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=True)
    fig.patch.set_facecolor("#fbf7ef")
    colors = {"NBR": "#7b8d87", "Sentinel-2": "#167255", "AlphaEarth": "#375f76"}
    for ax, age in zip(axes, [1, 12, 19]):
        sub = model_df[model_df["years_since_disturbance"] == age]
        ax.set_facecolor("#fbf7ef")
        for model, mdf in sub.groupby("model"):
            mdf = mdf.sort_values("interior_buffer_m")
            ax.plot(mdf["interior_buffer_m"], mdf["roc_auc"], marker="o", linewidth=2, color=colors[model], label=model)
        ax.axhline(0.5, color="#a69d91", linewidth=0.8, linestyle="--")
        ax.set_title(f"{age} years", fontsize=10, weight="bold")
        ax.set_xlabel("Fire-interior buffer (m)")
        ax.set_ylim(0, 1)
        ax.grid(color="#d8d0c2", linewidth=0.55, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ROC AUC")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Fire-interior sensitivity: disturbance-type classification", x=0.02, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ee.Initialize(project=PROJECT)
    aoi = load_aoi()
    image = alphaearth_image(aoi).addBands(sentinel2_image(aoi))
    availability_rows = []
    model_rows = []

    for year in YEARS:
        for buffer_m in BUFFERS_M:
            harvest_mask, fire_mask = disturbance_masks(aoi, year, buffer_m)
            fire_available = count_pixels(fire_mask, aoi)
            harvest_available = count_pixels(harvest_mask, aoi)
            n = min(MAX_SAMPLE_POOL, fire_available["pixel_count"], harvest_available["pixel_count"])
            fire = add_xy(sample_group(image, fire_mask, aoi, "wildfire", year, buffer_m, n))
            harvest = add_xy(sample_group(image, harvest_mask, aoi, "probable_harvest", year, buffer_m, n))
            fire = fire.dropna().reset_index(drop=True)
            harvest = harvest.dropna().reset_index(drop=True)

            matched_2km, summary_2km = greedy_match(fire, harvest, 2000)
            chosen = matched_2km
            chosen_summary = summary_2km
            chosen_distance = 2
            summary_5km = {"n_matched_pairs": np.nan, "mean_pair_distance_km": np.nan, "median_pair_distance_km": np.nan, "max_pair_distance_km": np.nan}
            summary_10km = {"n_matched_pairs": np.nan, "mean_pair_distance_km": np.nan, "median_pair_distance_km": np.nan, "max_pair_distance_km": np.nan}
            if summary_2km["n_matched_pairs"] < 100:
                matched_5km, summary_5km = greedy_match(fire, harvest, 5000)
                chosen = matched_5km
                chosen_summary = summary_5km
                chosen_distance = 5
            if summary_2km["n_matched_pairs"] < 100 and year == 2012:
                matched_10km, summary_10km = greedy_match(fire, harvest, 10000)

            availability_rows.append(
                {
                    "disturbance_year": year,
                    "years_since_disturbance": OBSERVATION_YEAR - year,
                    "interior_buffer_m": buffer_m,
                    "available_fire_pixels": fire_available["pixel_count"],
                    "available_fire_area_ha": fire_available["area_ha"],
                    "available_probable_harvest_pixels": harvest_available["pixel_count"],
                    "available_probable_harvest_area_ha": harvest_available["area_ha"],
                    "matched_pairs_2km": summary_2km["n_matched_pairs"],
                    "mean_pair_distance_2km": summary_2km["mean_pair_distance_km"],
                    "median_pair_distance_2km": summary_2km["median_pair_distance_km"],
                    "max_pair_distance_2km": summary_2km["max_pair_distance_km"],
                    "matched_pairs_5km": summary_5km["n_matched_pairs"],
                    "mean_pair_distance_5km": summary_5km["mean_pair_distance_km"],
                    "median_pair_distance_5km": summary_5km["median_pair_distance_km"],
                    "max_pair_distance_5km": summary_5km["max_pair_distance_km"],
                    "matched_pairs_10km_diagnostic": summary_10km["n_matched_pairs"],
                    "chosen_matching_distance_km": chosen_distance,
                    "chosen_matched_pairs": chosen_summary["n_matched_pairs"],
                    "chosen_mean_pair_distance_km": chosen_summary["mean_pair_distance_km"],
                    "chosen_median_pair_distance_km": chosen_summary["median_pair_distance_km"],
                    "chosen_max_pair_distance_km": chosen_summary["max_pair_distance_km"],
                }
            )
            if chosen_summary["n_matched_pairs"] >= 50:
                model_rows.extend(run_models(chosen))

    availability = pd.DataFrame(availability_rows)
    models = pd.DataFrame(model_rows)
    AVAILABILITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(AVAILABILITY_CSV, index=False, float_format="%.6f")
    models.to_csv(MODEL_CSV, index=False, float_format="%.6f")
    plot_results(models)

    print("INTERIOR SAMPLE AVAILABILITY")
    print(availability.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMODEL PERFORMANCE")
    print(models.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFILES")
    print(AVAILABILITY_CSV)
    print(MODEL_CSV)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
