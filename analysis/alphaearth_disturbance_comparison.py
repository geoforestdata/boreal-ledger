"""
AlphaEarth vs Sentinel-2 NBR disturbance-type comparison.

Experimental AOI: Lebel-sur-Quevillon temporal-spread candidate.

This script tests whether AlphaEarth satellite embeddings distinguish
wildfire and non-fire residual Hansen canopy loss at matched disturbance
ages better than a conventional Sentinel-2 NBR baseline. It is a
descriptive observational analysis, not causal inference.
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
sys.path.insert(0, str(ROOT / "src"))

from recovery_utils import get_sentinel2_nbr_composite


PROJECT = "ibfra2026"
OBSERVATION_YEAR = 2024
AOI_PATH = ROOT / "exploration" / "lebel_aoi_selection" / "recommended_aoi_temporal_spread.geojson"

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NBAC_COLLECTION = "projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530"
ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

MATCHED_YEARS = [2023, 2012, 2005]
MIN_PIXELS = 100
MAX_SAMPLES_PER_GROUP = 2000
RANDOM_SEED = 42
SPATIAL_BLOCK_DEGREES = 0.025
N_FOLDS = 5
RIDGE = 0.01

OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
SAMPLES_CSV = OUTPUTS / "alphaearth_samples.csv"
CENTROIDS_CSV = OUTPUTS / "alphaearth_centroid_distances.csv"
DISPERSION_CSV = OUTPUTS / "alphaearth_group_dispersion.csv"
AE_CLASS_CSV = OUTPUTS / "alphaearth_classification.csv"
NBR_CLASS_CSV = OUTPUTS / "nbr_classification_baseline.csv"
AE_VS_NBR_CSV = OUTPUTS / "alphaearth_vs_nbr.csv"
PCA_FIG = FIGURES / "alphaearth_pca.png"
COMPARE_FIG = FIGURES / "alphaearth_vs_nbr.png"


def load_aoi() -> ee.Geometry:
    with AOI_PATH.open() as f:
        geojson = json.load(f)
    return ee.Geometry.Polygon(geojson["geometry"]["coordinates"], proj="EPSG:4326", geodesic=False)


def alphaearth_info(aoi: ee.Geometry) -> dict:
    collection = ee.ImageCollection(ALPHAEARTH_COLLECTION)
    years = sorted(
        {
            int(pd.Timestamp(t, unit="ms", tz="UTC").year)
            for t in collection.aggregate_array("system:time_start").getInfo()
        }
    )
    image_2024 = collection.filterDate("2024-01-01", "2025-01-01").filterBounds(aoi)
    first = ee.Image(image_2024.first())
    band_names = first.bandNames().getInfo()
    projection = first.select(0).projection().getInfo()
    properties = first.toDictionary(
        [
            "system:id",
            "system:time_start",
            "system:time_end",
            "DATASET_VERSION",
            "MODEL_VERSION",
            "PROCESSING_SOFTWARE_VERSION",
            "UTM_ZONE",
        ]
    ).getInfo()
    return {
        "asset": ALPHAEARTH_COLLECTION,
        "years": years,
        "resolution_m": abs(float(projection["transform"][0])),
        "dimensions": len(band_names),
        "bands": band_names,
        "aoi_2024_tile_count": image_2024.size().getInfo(),
        "projection_crs": projection["crs"],
        "properties": properties,
    }


def disturbance_year_images(aoi: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)
    loss_calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0).And(lossyear.lte(24)))
    fire_year = (
        ee.FeatureCollection(NBAC_COLLECTION)
        .filterBounds(aoi)
        .reduceToImage(properties=["YEAR"], reducer=ee.Reducer.first())
        .rename("fire_year")
    )
    fire_match = loss_calendar_year.eq(fire_year).unmask(0)
    wildfire_year = loss_calendar_year.updateMask(fire_match).rename("disturbance_year")
    harvest_year = loss_calendar_year.updateMask(fire_match.Not()).rename("disturbance_year")
    return harvest_year, wildfire_year


def area_counts(class_img: ee.Image, aoi: ee.Geometry) -> dict[int, dict]:
    pixel_ha = ee.Image.pixelArea().divide(10000).rename("area")
    groups = (
        pixel_ha.updateMask(class_img)
        .addBands(class_img.rename("class"))
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
        year = int(item["class"])
        area_ha = float(item["sum"])
        out[year] = {"area_ha": area_ha, "pixel_count": int(round(area_ha / 0.09))}
    return out


def alphaearth_image(aoi: ee.Geometry) -> ee.Image:
    collection = (
        ee.ImageCollection(ALPHAEARTH_COLLECTION)
        .filterDate(f"{OBSERVATION_YEAR}-01-01", f"{OBSERVATION_YEAR + 1}-01-01")
        .filterBounds(aoi)
    )
    return collection.mosaic().clip(aoi)


def sample_group(
    image: ee.Image,
    class_img: ee.Image,
    aoi: ee.Geometry,
    disturbance_type: str,
    year: int,
    n_samples: int,
) -> pd.DataFrame:
    mask = class_img.eq(year)
    sample_img = image.updateMask(mask)
    seed = RANDOM_SEED + year + (0 if disturbance_type == "wildfire" else 10000)
    fc = (
        sample_img.sample(
            region=aoi,
            scale=30,
            factor=1,
            seed=seed,
            geometries=True,
            tileScale=4,
        )
        .randomColumn("random", seed)
        .sort("random")
        .limit(n_samples)
    )
    features = fc.getInfo()["features"]
    rows = []
    for idx, feature in enumerate(features):
        props = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"]
        row = {
            "sample_id": f"{disturbance_type}_{year}_{idx:04d}",
            "disturbance_type": disturbance_type,
            "disturbance_year": year,
            "years_since_disturbance": OBSERVATION_YEAR - year,
            "lon": lon,
            "lat": lat,
            "NBR_2024": props.get("NBR"),
        }
        for band in [f"A{i:02d}" for i in range(64)]:
            row[f"embedding_{band}"] = props.get(band)
        rows.append(row)
    return pd.DataFrame(rows)


def spatial_fold(lon: float, lat: float) -> int:
    ix = math.floor((lon + 180) / SPATIAL_BLOCK_DEGREES)
    iy = math.floor((lat + 90) / SPATIAL_BLOCK_DEGREES)
    return int((ix * 73856093 + iy * 19349663) % N_FOLDS)


def standardize_global(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    return (x - mean) / sd, mean, sd


def pca_svd(x: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    eigenvalues = (s**2) / (x.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()
    return scores, explained[:n_components], vt[:n_components]


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, ranks)
    avg_ranks = rank_sums / counts
    ranks = avg_ranks[inverse]
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = RIDGE) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    w = np.zeros(x_aug.shape[1])
    lr = 0.2
    for _ in range(900):
        z = np.clip(x_aug @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        grad = (x_aug.T @ (p - y)) / len(y)
        grad[1:] += l2 * w[1:] / len(y)
        w -= lr * grad
    return w


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    z = np.clip(x_aug @ w, -35, 35)
    return 1 / (1 + np.exp(-z))


def classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    balanced_accuracy = (recall + specificity) / 2
    return {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc_score(y_true, scores),
        "precision": precision,
        "recall": recall,
    }


def spatial_cv(df: pd.DataFrame, feature_cols: list[str], age: int) -> dict:
    sub = df[df["years_since_disturbance"] == age].copy()
    y = (sub["disturbance_type"] == "wildfire").astype(int).to_numpy()
    x = sub[feature_cols].to_numpy(dtype=float)
    folds = sub["spatial_fold"].to_numpy()
    scores = np.full(len(sub), np.nan)
    used_folds = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        mean = x[train].mean(axis=0)
        sd = x[train].std(axis=0, ddof=0)
        sd[sd == 0] = 1
        x_train = (x[train] - mean) / sd
        x_test = (x[test] - mean) / sd
        w = fit_logistic(x_train, y[train])
        scores[test] = predict_logistic(w, x_test)
        used_folds.append(int(fold))
    valid = ~np.isnan(scores)
    metrics = classification_metrics(y[valid], scores[valid])
    metrics.update(
        {
            "years_since_disturbance": age,
            "fire_year": OBSERVATION_YEAR - age,
            "harvest_year": OBSERVATION_YEAR - age,
            "n_fire": int((sub["disturbance_type"] == "wildfire").sum()),
            "n_harvest": int((sub["disturbance_type"] == "probable_harvest").sum()),
            "n_used": int(valid.sum()),
            "spatial_folds": len(used_folds),
        }
    )
    return metrics


def centroid_distances(df: pd.DataFrame, embed_cols: list[str], pc_cols: list[str]) -> pd.DataFrame:
    rows = []
    for age in sorted(df["years_since_disturbance"].unique()):
        sub = df[df["years_since_disturbance"] == age]
        fire = sub[sub["disturbance_type"] == "wildfire"]
        harvest = sub[sub["disturbance_type"] == "probable_harvest"]
        rows.append(
            {
                "years_since_disturbance": age,
                "fire_year": OBSERVATION_YEAR - age,
                "harvest_year": OBSERVATION_YEAR - age,
                "n_fire": len(fire),
                "n_harvest": len(harvest),
                "distance_full_embedding": float(np.linalg.norm(fire[embed_cols].mean().to_numpy() - harvest[embed_cols].mean().to_numpy())),
                "distance_pc2": float(np.linalg.norm(fire[pc_cols[:2]].mean().to_numpy() - harvest[pc_cols[:2]].mean().to_numpy())),
                "distance_pc3": float(np.linalg.norm(fire[pc_cols[:3]].mean().to_numpy() - harvest[pc_cols[:3]].mean().to_numpy())),
            }
        )
    return pd.DataFrame(rows)


def group_dispersion(df: pd.DataFrame, embed_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (dtype, year), sub in df.groupby(["disturbance_type", "disturbance_year"]):
        x = sub[embed_cols].to_numpy(dtype=float)
        centroid = x.mean(axis=0)
        d = np.linalg.norm(x - centroid, axis=1)
        rows.append(
            {
                "disturbance_type": dtype,
                "disturbance_year": year,
                "years_since_disturbance": OBSERVATION_YEAR - year,
                "n": len(sub),
                "mean_distance_to_centroid": float(d.mean()),
                "median_distance_to_centroid": float(np.median(d)),
                "sd_distance_to_centroid": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["years_since_disturbance", "disturbance_type"])


def plot_pca(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    colors = {"wildfire": "#c55f3c", "probable_harvest": "#167255"}
    markers = {1: "o", 12: "s", 19: "^"}
    for (dtype, age), sub in df.groupby(["disturbance_type", "years_since_disturbance"]):
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=16,
            alpha=0.34,
            c=colors[dtype],
            marker=markers[int(age)],
            linewidth=0,
            label=f"{dtype.replace('_', ' ')} · {int(age)} yr",
        )
    ax.set_title("AlphaEarth embedding PCA by disturbance type and matched age", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#d8d0c2", linewidth=0.55, alpha=0.55)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(PCA_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(comp: pd.DataFrame) -> None:
    x = np.arange(len(comp))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    ax.bar(x - width / 2, comp["alphaearth_balanced_accuracy"], width, color="#375f76", label="AlphaEarth embeddings")
    ax.bar(x + width / 2, comp["nbr_balanced_accuracy"], width, color="#167255", label="NBR only")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)} yr" for v in comp["years_since_disturbance"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spatial-CV balanced accuracy")
    ax.set_title("Fire vs probable-harvest distinguishability by matched age", loc="left", fontsize=13, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8d0c2", linewidth=0.55, alpha=0.55)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(COMPARE_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ee.Initialize(project=PROJECT)
    aoi = load_aoi()
    product = alphaearth_info(aoi)
    bands = product["bands"]
    embed_cols = [f"embedding_{band}" for band in bands]

    harvest_year, wildfire_year = disturbance_year_images(aoi)
    counts_harvest = area_counts(harvest_year, aoi)
    counts_fire = area_counts(wildfire_year, aoi)
    ae = alphaearth_image(aoi)
    nbr = get_sentinel2_nbr_composite(OBSERVATION_YEAR, aoi)
    sample_img = ae.addBands(nbr)

    samples = []
    for year in MATCHED_YEARS:
        n = min(MAX_SAMPLES_PER_GROUP, counts_fire[year]["pixel_count"], counts_harvest[year]["pixel_count"])
        samples.append(sample_group(sample_img, wildfire_year, aoi, "wildfire", year, n))
        samples.append(sample_group(sample_img, harvest_year, aoi, "probable_harvest", year, n))
    df = pd.concat(samples, ignore_index=True)
    df = df.dropna(subset=embed_cols + ["NBR_2024"]).reset_index(drop=True)
    df["spatial_fold"] = [spatial_fold(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]

    x_std, _, _ = standardize_global(df[embed_cols].to_numpy(dtype=float))
    std_cols = [f"std_{band}" for band in bands]
    for i, col in enumerate(std_cols):
        df[col] = x_std[:, i]
    pc_scores, explained, _ = pca_svd(x_std, 3)
    for i in range(3):
        df[f"PC{i + 1}"] = pc_scores[:, i]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLES_CSV, index=False, float_format="%.6f")

    centroid_df = centroid_distances(df, std_cols, ["PC1", "PC2", "PC3"])
    dispersion_df = group_dispersion(df, std_cols)
    ae_class = pd.DataFrame([spatial_cv(df, std_cols, age) for age in sorted(df["years_since_disturbance"].unique())])
    nbr_class = pd.DataFrame([spatial_cv(df, ["NBR_2024"], age) for age in sorted(df["years_since_disturbance"].unique())])

    comp = ae_class[
        ["years_since_disturbance", "balanced_accuracy", "roc_auc"]
    ].rename(
        columns={
            "balanced_accuracy": "alphaearth_balanced_accuracy",
            "roc_auc": "alphaearth_auc",
        }
    ).merge(
        nbr_class[["years_since_disturbance", "balanced_accuracy", "roc_auc"]].rename(
            columns={"balanced_accuracy": "nbr_balanced_accuracy", "roc_auc": "nbr_auc"}
        ),
        on="years_since_disturbance",
    )
    comp["difference_balanced_accuracy"] = comp["alphaearth_balanced_accuracy"] - comp["nbr_balanced_accuracy"]
    comp["difference_auc"] = comp["alphaearth_auc"] - comp["nbr_auc"]

    centroid_df.to_csv(CENTROIDS_CSV, index=False, float_format="%.6f")
    dispersion_df.to_csv(DISPERSION_CSV, index=False, float_format="%.6f")
    ae_class.to_csv(AE_CLASS_CSV, index=False, float_format="%.6f")
    nbr_class.to_csv(NBR_CLASS_CSV, index=False, float_format="%.6f")
    comp.to_csv(AE_VS_NBR_CSV, index=False, float_format="%.6f")
    plot_pca(df)
    plot_comparison(comp)

    sample_size = (
        df.groupby(["disturbance_type", "disturbance_year", "years_since_disturbance"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["years_since_disturbance", "disturbance_type"])
    )

    print("ALPHAEARTH PRODUCT")
    print(f"asset: {product['asset']}")
    print(f"years: {product['years']}")
    print(f"resolution: {product['resolution_m']:.0f} m")
    print(f"dimensions: {product['dimensions']}")
    print(f"observation year used: {OBSERVATION_YEAR}")
    print(f"AOI 2024 tiles: {product['aoi_2024_tile_count']}")
    print(f"projection CRS: {product['projection_crs']}")
    print(f"temporal image start/end ms: {product['properties'].get('system:time_start')} / {product['properties'].get('system:time_end')}")
    print("\nSAMPLE SIZE")
    print(sample_size.to_string(index=False))
    print("\nPCA")
    print(f"PC1 variance: {explained[0]:.6f}")
    print(f"PC2 variance: {explained[1]:.6f}")
    print(f"PC3 variance: {explained[2]:.6f}")
    print("\nCENTROID DISTANCES")
    print(centroid_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCLASSIFICATION: ALPHAEARTH")
    print(ae_class.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCLASSIFICATION: NBR")
    print(nbr_class.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nALPHAEARTH VS NBR")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFILES")
    for path in [SAMPLES_CSV, CENTROIDS_CSV, DISPERSION_CSV, AE_CLASS_CSV, NBR_CLASS_CSV, AE_VS_NBR_CSV, PCA_FIG, COMPARE_FIG]:
        print(path)


if __name__ == "__main__":
    main()
