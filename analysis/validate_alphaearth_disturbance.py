"""
Validation checks for the AlphaEarth disturbance-type result.

This script audits whether AlphaEarth is separating disturbance type or
mainly geographic/site differences. It uses the already selected Lebel AOI
and the existing balanced AlphaEarth sample table, adds Sentinel-2 spectral
bands at those same sample locations, then repeats classification after
geographic one-to-one matching and stricter spatial-block CV.
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
from pyproj import Geod, Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT = "ibfra2026"
OBSERVATION_YEAR = 2024
RANDOM_SEED = 42
MATCH_THRESHOLDS_M = [2000, 5000, 10000]
MIN_DEFENSIBLE_PAIRS = 100
FOLD_COUNT = 5
RIDGE = 0.01
PERMUTATIONS = 100

SAMPLES_CSV = ROOT / "outputs" / "alphaearth_samples.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
SPECTRAL_SAMPLES_CSV = ROOT / "outputs" / "alphaearth_samples_with_sentinel2.csv"
SPATIAL_DIAGNOSTICS_CSV = ROOT / "outputs" / "spatial_separation_diagnostics.csv"
MATCHED_PAIRS_CSV = ROOT / "outputs" / "alphaearth_spatially_matched_samples.csv"
AE_MATCHED_CSV = ROOT / "outputs" / "alphaearth_spatially_matched_classification.csv"
NBR_MATCHED_CSV = ROOT / "outputs" / "nbr_spatially_matched_classification.csv"
S2_BASELINE_CSV = ROOT / "outputs" / "sentinel2_spectral_baseline.csv"
AE_VS_NBR_MATCHED_CSV = ROOT / "outputs" / "alphaearth_vs_nbr_spatially_matched.csv"
PERMUTATION_CSV = ROOT / "outputs" / "alphaearth_permutation_test.csv"
FINAL_COMPARISON_CSV = ROOT / "outputs" / "final_model_comparison.csv"
SAMPLE_MAP = ROOT / "figures" / "alphaearth_sample_locations.png"
FINAL_FIG = ROOT / "figures" / "final_alphaearth_validation.png"


def logistic_fit(x: np.ndarray, y: np.ndarray, iterations: int = 450) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    w = np.zeros(x_aug.shape[1])
    lr = 0.18
    for _ in range(iterations):
        z = np.clip(x_aug @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        grad = (x_aug.T @ (p - y)) / len(y)
        grad[1:] += RIDGE * w[1:] / len(y)
        w -= lr * grad
    return w


def logistic_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    z = np.clip(x_aug @ w, -35, 35)
    return 1 / (1 + np.exp(-z))


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
    avg = np.bincount(inverse, ranks) / counts
    ranks = avg[inverse]
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "roc_auc": auc_score(y_true, scores),
        "precision": precision,
        "recall": recall,
    }


def spatial_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    age: int,
    block_km: int,
    y_override: np.ndarray | None = None,
    iterations: int = 450,
) -> dict:
    sub = df[df["years_since_disturbance"] == age].copy()
    y = (sub["disturbance_type"] == "wildfire").astype(int).to_numpy()
    if y_override is not None:
        y = y_override.copy()
    x = sub[feature_cols].to_numpy(dtype=float)
    folds = sub[f"fold_{block_km}km"].to_numpy()
    scores = np.full(len(sub), np.nan)
    used = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        mean = x[train].mean(axis=0)
        sd = x[train].std(axis=0, ddof=0)
        sd[sd == 0] = 1
        w = logistic_fit((x[train] - mean) / sd, y[train], iterations=iterations)
        scores[test] = logistic_predict(w, (x[test] - mean) / sd)
        used.append(int(fold))
    valid = ~np.isnan(scores)
    out = metrics(y[valid], scores[valid]) if valid.any() else {
        "balanced_accuracy": float("nan"),
        "roc_auc": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
    }
    out.update(
        {
            "years_since_disturbance": age,
            "n_fire": int((sub["disturbance_type"] == "wildfire").sum()),
            "n_harvest": int((sub["disturbance_type"] == "probable_harvest").sum()),
            "n_used": int(valid.sum()),
            "spatial_block_km": block_km,
            "spatial_folds": len(used),
        }
    )
    return out


def add_local_xy(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6622", always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x_m"] = x
    out["y_m"] = y
    return out


def spatial_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    geod = Geod(ellps="WGS84")
    rows = []
    for age, sub in df.groupby("years_since_disturbance"):
        fire = sub[sub["disturbance_type"] == "wildfire"]
        harvest = sub[sub["disturbance_type"] == "probable_harvest"]
        fire_lon = fire["lon"].mean()
        fire_lat = fire["lat"].mean()
        harvest_lon = harvest["lon"].mean()
        harvest_lat = harvest["lat"].mean()
        _, _, centroid_m = geod.inv(fire_lon, fire_lat, harvest_lon, harvest_lat)
        dmat = distance_matrix_m(fire, harvest)
        nearest = dmat.min(axis=1)
        rows.append(
            {
                "years_since_disturbance": int(age),
                "fire_centroid_lon": fire_lon,
                "fire_centroid_lat": fire_lat,
                "harvest_centroid_lon": harvest_lon,
                "harvest_centroid_lat": harvest_lat,
                "centroid_distance_km": centroid_m / 1000,
                "median_fire_to_harvest_nearest_neighbor_km": float(np.median(nearest) / 1000),
                "n_fire": len(fire),
                "n_harvest": len(harvest),
            }
        )
    return pd.DataFrame(rows).sort_values("years_since_disturbance")


def distance_matrix_m(fire: pd.DataFrame, harvest: pd.DataFrame) -> np.ndarray:
    f = fire[["x_m", "y_m"]].to_numpy(dtype=float)
    h = harvest[["x_m", "y_m"]].to_numpy(dtype=float)
    return np.sqrt(((f[:, None, :] - h[None, :, :]) ** 2).sum(axis=2))


def greedy_pairs(dmat: np.ndarray, threshold_m: int) -> list[tuple[int, int, float]]:
    pairs_under_threshold = np.argwhere(dmat <= threshold_m)
    order = np.argsort(dmat[pairs_under_threshold[:, 0], pairs_under_threshold[:, 1]])
    used_fire = set()
    used_harvest = set()
    pairs = []
    for item in order:
        i, j = map(int, pairs_under_threshold[item])
        if i in used_fire or j in used_harvest:
            continue
        used_fire.add(i)
        used_harvest.add(j)
        pairs.append((i, j, float(dmat[i, j])))
    return pairs


def matched_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    summary = []
    pair_counter = 0
    for age, sub in df.groupby("years_since_disturbance"):
        fire = sub[sub["disturbance_type"] == "wildfire"].reset_index(drop=True)
        harvest = sub[sub["disturbance_type"] == "probable_harvest"].reset_index(drop=True)
        dmat = distance_matrix_m(fire, harvest)
        threshold_results = []
        for threshold in MATCH_THRESHOLDS_M:
            pairs = greedy_pairs(dmat, threshold)
            threshold_results.append((threshold, pairs))
        chosen_threshold, chosen_pairs = threshold_results[-1]
        for threshold, pairs in threshold_results:
            if len(pairs) >= MIN_DEFENSIBLE_PAIRS:
                chosen_threshold, chosen_pairs = threshold, pairs
                break
        order = np.arange(len(chosen_pairs))
        rng.shuffle(order)
        chosen_pairs = [chosen_pairs[i] for i in order]
        distances = [p[2] for p in chosen_pairs]
        summary.append(
            {
                "years_since_disturbance": int(age),
                "maximum_matching_distance_km": chosen_threshold / 1000,
                "n_matched_pairs": len(chosen_pairs),
                "mean_pair_distance_km": float(np.mean(distances) / 1000) if distances else float("nan"),
                "median_pair_distance_km": float(np.median(distances) / 1000) if distances else float("nan"),
                "threshold_note": "smallest threshold with >=100 pairs" if len(chosen_pairs) >= MIN_DEFENSIBLE_PAIRS else "10 km maximum used; <100 pairs available",
            }
        )
        for fire_idx, harvest_idx, dist_m in chosen_pairs:
            midpoint_x = (fire.loc[fire_idx, "x_m"] + harvest.loc[harvest_idx, "x_m"]) / 2
            midpoint_y = (fire.loc[fire_idx, "y_m"] + harvest.loc[harvest_idx, "y_m"]) / 2
            for source, idx in [("wildfire", fire_idx), ("probable_harvest", harvest_idx)]:
                row = (fire if source == "wildfire" else harvest).loc[idx].copy()
                row["pair_id"] = f"pair_{pair_counter:05d}"
                row["pair_distance_km"] = dist_m / 1000
                row["pair_midpoint_x_m"] = midpoint_x
                row["pair_midpoint_y_m"] = midpoint_y
                all_rows.append(row)
            pair_counter += 1
    matched = pd.DataFrame(all_rows).reset_index(drop=True)
    for block_km in [5, 10]:
        block_m = block_km * 1000
        bx = np.floor(matched["pair_midpoint_x_m"].to_numpy() / block_m).astype(int)
        by = np.floor(matched["pair_midpoint_y_m"].to_numpy() / block_m).astype(int)
        matched[f"fold_{block_km}km"] = (bx * 73856093 + by * 19349663) % FOLD_COUNT
    return matched, pd.DataFrame(summary).sort_values("years_since_disturbance")


def add_sentinel2_bands(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in ["B2", "B3", "B4", "B8", "B11", "B12"]):
        return df
    ee.Initialize(project=PROJECT)
    point_rows = df[["sample_id", "lon", "lat"]].to_dict("records")
    fc_all = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                {"sample_id": row["sample_id"]},
            )
            for row in point_rows
        ]
    )
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(fc_all.geometry())
        .filterDate("2024-06-01", "2024-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )

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

    composite = collection.map(mask_clouds).median().select(["B2", "B3", "B4", "B8", "B11", "B12"])
    rows = []
    chunk_size = 1000
    for start in range(0, len(point_rows), chunk_size):
        chunk = point_rows[start : start + chunk_size]
        fc = ee.FeatureCollection(
            [
                ee.Feature(
                    ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                    {"sample_id": row["sample_id"]},
                )
                for row in chunk
            ]
        )
        sampled = composite.sampleRegions(
            collection=fc, properties=["sample_id"], scale=10, tileScale=4
        ).getInfo()["features"]
        for feature in sampled:
            props = feature["properties"]
            rows.append({"sample_id": props["sample_id"], **{band: props.get(band) for band in ["B2", "B3", "B4", "B8", "B11", "B12"]}})
    band_df = pd.DataFrame(rows)
    out = df.merge(band_df, on="sample_id", how="left")
    return out.dropna(subset=["B2", "B3", "B4", "B8", "B11", "B12"]).reset_index(drop=True)


def run_models(matched: pd.DataFrame, block_km: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embed_cols = [f"embedding_A{i:02d}" for i in range(64)]
    nbr_cols = ["NBR_2024"]
    s2_cols = ["B2", "B3", "B4", "B8", "B11", "B12", "NBR_2024"]
    ages = sorted(matched["years_since_disturbance"].unique())
    ae = pd.DataFrame([spatial_cv(matched, embed_cols, int(age), block_km) for age in ages])
    nbr = pd.DataFrame([spatial_cv(matched, nbr_cols, int(age), block_km) for age in ages])
    s2 = pd.DataFrame([spatial_cv(matched, s2_cols, int(age), block_km) for age in ages])
    return ae, nbr, s2


def permutation_test(matched: pd.DataFrame, block_km: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    embed_cols = [f"embedding_A{i:02d}" for i in range(64)]
    rows = []
    for age in sorted(matched["years_since_disturbance"].unique()):
        observed = spatial_cv(matched, embed_cols, int(age), block_km)
        sub = matched[matched["years_since_disturbance"] == age]
        y = (sub["disturbance_type"] == "wildfire").astype(int).to_numpy()
        perm_acc = []
        for _ in range(PERMUTATIONS):
            yp = rng.permutation(y)
            result = spatial_cv(matched, embed_cols, int(age), block_km, y_override=yp, iterations=260)
            perm_acc.append(result["balanced_accuracy"])
        perm_acc = np.array(perm_acc, dtype=float)
        rows.append(
            {
                "years_since_disturbance": int(age),
                "observed_balanced_accuracy": observed["balanced_accuracy"],
                "mean_permuted_balanced_accuracy": float(np.nanmean(perm_acc)),
                "p95_permuted_accuracy": float(np.nanpercentile(perm_acc, 95)),
                "empirical_p_value": float((np.sum(perm_acc >= observed["balanced_accuracy"]) + 1) / (np.sum(~np.isnan(perm_acc)) + 1)),
                "permutations": PERMUTATIONS,
                "spatial_block_km": block_km,
            }
        )
    return pd.DataFrame(rows)


def final_comparison(ae: pd.DataFrame, nbr: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    out = nbr[["years_since_disturbance", "n_fire", "spatial_block_km", "balanced_accuracy", "roc_auc"]].rename(
        columns={"n_fire": "n_matched_per_class", "balanced_accuracy": "nbr_balanced_accuracy", "roc_auc": "nbr_auc"}
    )
    out = out.merge(
        s2[["years_since_disturbance", "balanced_accuracy", "roc_auc"]].rename(
            columns={"balanced_accuracy": "sentinel2_balanced_accuracy", "roc_auc": "sentinel2_auc"}
        ),
        on="years_since_disturbance",
    )
    out = out.merge(
        ae[["years_since_disturbance", "balanced_accuracy", "roc_auc"]].rename(
            columns={"balanced_accuracy": "alphaearth_balanced_accuracy", "roc_auc": "alphaearth_auc"}
        ),
        on="years_since_disturbance",
    )
    return out.sort_values("years_since_disturbance")


def plot_sample_locations(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor("#fbf7ef")
    colors = {"wildfire": "#c55f3c", "probable_harvest": "#167255"}
    for ax, age in zip(axes, [1, 12, 19]):
        sub = df[df["years_since_disturbance"] == age]
        ax.set_facecolor("#fbf7ef")
        for dtype, group in sub.groupby("disturbance_type"):
            ax.scatter(group["lon"], group["lat"], s=10, alpha=0.55, linewidth=0, c=colors[dtype], label=dtype.replace("_", " "))
        ax.set_title(f"{age} years since disturbance", fontsize=10, weight="bold")
        ax.grid(color="#d8d0c2", linewidth=0.5, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Latitude")
    for ax in axes:
        ax.set_xlabel("Longitude")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("AlphaEarth sample locations by matched disturbance age", x=0.02, ha="left", fontsize=13, weight="bold")
    fig.tight_layout()
    SAMPLE_MAP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLE_MAP, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final(comp: pd.DataFrame) -> None:
    ages = comp["years_since_disturbance"].to_numpy()
    x = np.arange(len(ages))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    ax.bar(x - width, comp["nbr_balanced_accuracy"], width, color="#7b8d87", label="NBR")
    ax.bar(x, comp["sentinel2_balanced_accuracy"], width, color="#167255", label="Sentinel-2 spectral")
    ax.bar(x + width, comp["alphaearth_balanced_accuracy"], width, color="#375f76", label="AlphaEarth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{age} yr" for age in ages])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy, spatial CV")
    ax.set_title("Strict geographically matched validation", loc="left", fontsize=13, weight="bold")
    ax.grid(axis="y", color="#d8d0c2", linewidth=0.55, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    FINAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SAMPLES_CSV)
    df = add_local_xy(df)
    df = add_sentinel2_bands(df)
    df.to_csv(SPECTRAL_SAMPLES_CSV, index=False, float_format="%.6f")

    sep = spatial_diagnostics(df)
    sep.to_csv(SPATIAL_DIAGNOSTICS_CSV, index=False, float_format="%.6f")
    plot_sample_locations(df)

    matched, match_summary = matched_samples(df)
    matched.to_csv(MATCHED_PAIRS_CSV, index=False, float_format="%.6f")
    match_summary.to_csv(ROOT / "outputs" / "alphaearth_matching_summary.csv", index=False, float_format="%.6f")

    ae5, nbr5, s25 = run_models(matched, 5)
    ae10, nbr10, s210 = run_models(matched, 10)
    ae = pd.concat([ae5, ae10], ignore_index=True)
    nbr = pd.concat([nbr5, nbr10], ignore_index=True)
    s2 = pd.concat([s25, s210], ignore_index=True)
    ae.to_csv(AE_MATCHED_CSV, index=False, float_format="%.6f")
    nbr.to_csv(NBR_MATCHED_CSV, index=False, float_format="%.6f")
    s2.to_csv(S2_BASELINE_CSV, index=False, float_format="%.6f")

    comp5 = final_comparison(ae5, nbr5, s25)
    comp5.to_csv(FINAL_COMPARISON_CSV, index=False, float_format="%.6f")
    comp5[[
        "years_since_disturbance",
        "nbr_balanced_accuracy",
        "nbr_auc",
        "alphaearth_balanced_accuracy",
        "alphaearth_auc",
    ]].to_csv(AE_VS_NBR_MATCHED_CSV, index=False, float_format="%.6f")
    plot_final(comp5)

    perm = permutation_test(matched, 5)
    perm.to_csv(PERMUTATION_CSV, index=False, float_format="%.6f")

    print("SPATIAL SEPARATION")
    print(sep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nMATCHED SAMPLE")
    print(match_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSTRICT MATCHED CLASSIFICATION 5 KM")
    print(comp5.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSTRICT MATCHED CLASSIFICATION 10 KM")
    print(final_comparison(ae10, nbr10, s210).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPERMUTATION TEST")
    print(perm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFILES")
    for path in [
        SAMPLE_MAP,
        SPATIAL_DIAGNOSTICS_CSV,
        MATCHED_PAIRS_CSV,
        AE_MATCHED_CSV,
        NBR_MATCHED_CSV,
        S2_BASELINE_CSV,
        AE_VS_NBR_MATCHED_CSV,
        PERMUTATION_CSV,
        FINAL_COMPARISON_CSV,
        FINAL_FIG,
    ]:
        print(path)


if __name__ == "__main__":
    main()
