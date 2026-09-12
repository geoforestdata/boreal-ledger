#!/usr/bin/env python3
"""Extract Sentinel-2 and AlphaEarth paired recovery time series.

Phase 10 is data extraction plus feasibility audit only. It preserves the
existing disturbed-reference design and does not fit recovery models.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "multivariate_recovery"
FIG_DIR = ROOT / "figures" / "multivariate_recovery"
MPL_DIR = OUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

FINAL_REFS = ROOT / "outputs" / "reference_timeseries" / "final_reference_sites.gpkg"
FIRE_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_fire_sites.gpkg"
HARVEST_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_harvest_sites.gpkg"
PAIR_COMPLETENESS = ROOT / "outputs" / "reference_timeseries" / "reference_pair_completeness.csv"
LANDSAT_EVENT_YEAR = ROOT / "outputs" / "landsat_recovery_model" / "event_recovery_longitudinal.csv"

PROJECT = "ibfra2026"
YEARS = list(range(2017, 2026))
START_MONTH_DAY = "07-01"
END_MONTH_DAY = "08-31"
S2_SCALE_M = 20
AE_SCALE_M = 10
CHUNK_SIZE = 400
S2_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def initialize_ee():
    venv_site = ROOT / ".venv" / "lib" / "python3.14" / "site-packages"
    if venv_site.exists():
        sys.path.insert(0, str(venv_site))
    import ee

    ee.Initialize(project=PROJECT)
    return ee


def read_design() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    refs = gpd.read_file(FINAL_REFS, layer="final_reference_sites", engine="pyogrio").to_crs("EPSG:32198")
    fire = gpd.read_file(FIRE_SITES, layer="candidate_fire_sites", engine="pyogrio").to_crs("EPSG:32198")
    harvest = gpd.read_file(HARVEST_SITES, layer="candidate_harvest_sites", engine="pyogrio").to_crs("EPSG:32198")
    disturbed = pd.concat([fire, harvest], ignore_index=True)
    disturbed = gpd.GeoDataFrame(disturbed, geometry="geometry", crs="EPSG:32198")
    disturbed = disturbed.rename(columns={"site_id": "associated_site_id", "geometry": "disturbed_geometry"})
    refs = refs.merge(
        disturbed[["associated_site_id", "disturbed_geometry"]],
        on="associated_site_id",
        how="left",
    )
    refs = gpd.GeoDataFrame(refs, geometry="geometry", crs="EPSG:32198")
    if refs["disturbed_geometry"].isna().any():
        raise ValueError("Some final reference rows could not be matched to disturbed site coordinates.")
    point_rows = []
    for _, row in refs.iterrows():
        base = {
            "pair_id": row["pair_id"],
            "event_id": row["event_id"],
            "site_id": row["associated_site_id"],
            "reference_id": row["reference_id"],
            "disturbance_type": row["disturbance_type"],
            "disturbance_year": int(row["disturbance_year"]),
            "censor_year": row.get("censor_year"),
            "censor_reason": row.get("censor_reason", ""),
            "reference_censor_year": row.get("reference_censor_year"),
            "reference_censor_reason": row.get("reference_censor_reason", ""),
        }
        point_rows.append(base | {"point_role": "disturbed", "point_id": f"{row['pair_id']}_disturbed", "geometry": row["disturbed_geometry"]})
        point_rows.append(base | {"point_role": "reference", "point_id": f"{row['pair_id']}_reference", "geometry": row.geometry})
    points = gpd.GeoDataFrame(point_rows, geometry="geometry", crs="EPSG:32198").to_crs("EPSG:4326")
    return refs, points


def ee_features(ee, points: gpd.GeoDataFrame):
    features = []
    for _, row in points.iterrows():
        props = row.drop(labels="geometry").to_dict()
        props = {k: (None if pd.isna(v) else v) for k, v in props.items()}
        features.append(ee.Feature(ee.Geometry.Point([float(row.geometry.x), float(row.geometry.y)]), props))
    return features


def s2_prep(ee, image):
    scl = image.select("SCL")
    mask = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    qa = image.select("QA60")
    mask = mask.And(qa.bitwiseAnd(1 << 10).eq(0)).And(qa.bitwiseAnd(1 << 11).eq(0))
    scaled = image.select(["B2", "B3", "B4", "B8", "B11", "B12"], S2_BANDS).multiply(0.0001)
    nbr = scaled.normalizedDifference(["nir", "swir2"]).rename("nbr")
    ndvi = scaled.normalizedDifference(["nir", "red"]).rename("ndvi")
    return scaled.addBands([nbr, ndvi]).updateMask(mask)


def s2_annual_image(ee, year: int, region):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(f"{year}-{START_MONTH_DAY}", f"{year}-{END_MONTH_DAY}")
        .map(lambda img: s2_prep(ee, img))
    )
    median = coll.select(S2_BANDS + ["nbr", "ndvi"]).median()
    count = coll.select("blue").count().rename("s2_valid_observation_count")
    return median.addBands(count).set("observation_year", year)


def alphaearth_band_names_and_scale(ee) -> tuple[list[str], float]:
    coll = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    image = ee.Image(coll.first())
    bands = image.bandNames().getInfo()
    scale = image.select([bands[0]]).projection().nominalScale().getInfo()
    return bands, float(scale)


def alphaearth_annual_image(ee, year: int, band_names: list[str], region):
    coll = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    )
    image = coll.mosaic().select(band_names)
    valid = image.select([band_names[0]]).mask().rename("ae_valid_mask")
    return image.addBands(valid).set("observation_year", year)


def sample_image_by_year(ee, image, features, bands: list[str], year: int, scale: int) -> pd.DataFrame:
    rows = []
    for start in range(0, len(features), CHUNK_SIZE):
        fc = ee.FeatureCollection(features[start : start + CHUNK_SIZE])
        sampled = image.sampleRegions(collection=fc, scale=scale, geometries=False, tileScale=4)
        data = sampled.getInfo().get("features", [])
        for feat in data:
            props = feat["properties"]
            row = {k: props.get(k) for k in props.keys()}
            for band in bands:
                row.setdefault(band, None)
            row["observation_year"] = year
            rows.append(row)
        time.sleep(0.03)
    return pd.DataFrame(rows)


def extract_s2(points: gpd.GeoDataFrame) -> pd.DataFrame:
    ee = initialize_ee()
    features = ee_features(ee, points)
    coords = [[float(g.x), float(g.y)] for g in points.geometry]
    region = ee.Geometry.MultiPoint(coords).bounds().buffer(15000)
    bands = S2_BANDS + ["nbr", "ndvi", "s2_valid_observation_count"]
    frames = []
    for year in YEARS:
        print(f"Extracting Sentinel-2 {year}...", flush=True)
        frames.append(sample_image_by_year(ee, s2_annual_image(ee, year, region), features, bands, year, S2_SCALE_M))
    return pd.concat(frames, ignore_index=True)


def extract_alphaearth(points: gpd.GeoDataFrame) -> tuple[pd.DataFrame, list[str], float]:
    ee = initialize_ee()
    band_names, nominal_scale = alphaearth_band_names_and_scale(ee)
    features = ee_features(ee, points)
    coords = [[float(g.x), float(g.y)] for g in points.geometry]
    region = ee.Geometry.MultiPoint(coords).bounds().buffer(15000)
    frames = []
    for year in YEARS:
        print(f"Extracting AlphaEarth {year}...", flush=True)
        frames.append(sample_image_by_year(ee, alphaearth_annual_image(ee, year, band_names, region), features, band_names + ["ae_valid_mask"], year, AE_SCALE_M))
    return pd.concat(frames, ignore_index=True), band_names, nominal_scale


def pivot_roles(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    keys = ["pair_id", "site_id", "reference_id", "event_id", "disturbance_type", "disturbance_year", "observation_year"]
    roles = []
    for role, group in df.groupby("point_role"):
        keep = keys + value_cols
        part = group[keep].copy()
        rename = {col: f"{role}_{col}" for col in value_cols}
        roles.append(part.rename(columns=rename))
    out = roles[0]
    for part in roles[1:]:
        out = out.merge(part, on=keys, how="outer")
    out["years_since_disturbance"] = out["observation_year"] - out["disturbance_year"]
    out["product"] = prefix
    return out.sort_values(["disturbance_type", "event_id", "site_id", "observation_year"])


def robust_reference_standardization(s2_site: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    out = s2_site.copy()
    for band in S2_BANDS:
        ref = pd.to_numeric(out[f"reference_{band}"], errors="coerce").dropna()
        med = float(ref.median())
        iqr = float(ref.quantile(0.75) - ref.quantile(0.25))
        scale = iqr / 1.349 if iqr > 0 else float(ref.mad() if hasattr(ref, "mad") else (ref - med).abs().median())
        if not np.isfinite(scale) or scale == 0:
            scale = float(ref.std()) if ref.std() > 0 else 1.0
        rows.append({"band": band, "reference_median": med, "reference_iqr": iqr, "robust_scale_iqr_over_1_349": scale, "n_reference_observations": int(len(ref))})
        out[f"{band}_diff"] = out[f"disturbed_{band}"] - out[f"reference_{band}"]
        out[f"{band}_diff_zrobust"] = out[f"{band}_diff"] / scale
    zcols = [f"{band}_diff_zrobust" for band in S2_BANDS]
    out["s2_reference_distance"] = np.sqrt((out[zcols] ** 2).sum(axis=1))
    valid = out[[f"disturbed_{b}" for b in S2_BANDS] + [f"reference_{b}" for b in S2_BANDS]].notna().all(axis=1)
    out.loc[~valid, "s2_reference_distance"] = np.nan
    out["s2_nbr_gap"] = out["reference_nbr"] - out["disturbed_nbr"]
    out["s2_ndvi_gap"] = out["reference_ndvi"] - out["disturbed_ndvi"]
    out["s2_pair_available"] = valid
    return out, pd.DataFrame(rows)


def alphaearth_distances(ae_site: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    out = ae_site.copy()
    dcols = [f"disturbed_{b}" for b in bands]
    rcols = [f"reference_{b}" for b in bands]
    d = out[dcols].to_numpy(dtype=float)
    r = out[rcols].to_numpy(dtype=float)
    valid = np.isfinite(d).all(axis=1) & np.isfinite(r).all(axis=1)
    diff = d - r
    out["ae_euclidean_distance"] = np.nan
    out["ae_cosine_distance"] = np.nan
    out["ae_disturbed_norm"] = np.nan
    out["ae_reference_norm"] = np.nan
    out.loc[valid, "ae_euclidean_distance"] = np.sqrt((diff[valid] ** 2).sum(axis=1))
    dn = np.linalg.norm(d[valid], axis=1)
    rn = np.linalg.norm(r[valid], axis=1)
    cos = np.sum(d[valid] * r[valid], axis=1) / np.where((dn * rn) == 0, np.nan, dn * rn)
    out.loc[valid, "ae_cosine_distance"] = 1 - cos
    out.loc[valid, "ae_disturbed_norm"] = dn
    out.loc[valid, "ae_reference_norm"] = rn
    out["ae_pair_available"] = valid
    return out


def aggregate_event(site: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    valid_metrics = [m for m in metrics if m in site.columns]
    rows = []
    for keys, group in site.groupby(["disturbance_type", "event_id", "disturbance_year", "observation_year", "years_since_disturbance"]):
        row = {
            "disturbance_type": keys[0],
            "event_id": keys[1],
            "disturbance_year": int(keys[2]),
            "observation_year": int(keys[3]),
            "years_since_disturbance": int(keys[4]),
            "n_valid_sites": int(group[valid_metrics[0]].notna().sum()) if valid_metrics else int(len(group)),
        }
        for metric in valid_metrics:
            row[metric] = float(group[metric].median(skipna=True)) if group[metric].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["disturbance_type", "event_id", "observation_year"])


def combine_event_years(s2_event: pd.DataFrame, ae_event: pd.DataFrame, landsat: pd.DataFrame) -> pd.DataFrame:
    keys = ["disturbance_type", "event_id", "disturbance_year", "observation_year", "years_since_disturbance"]
    out = s2_event.merge(ae_event, on=keys, how="outer", suffixes=("_s2", "_ae"))
    ls = landsat[keys + ["nbr_gap_median", "nbr_gap_change", "dist_nbr_median", "ref_nbr_median"]].copy()
    out = out.merge(ls, on=keys, how="left")
    out["both_products_available"] = out["s2_reference_distance"].notna() & out["ae_euclidean_distance"].notna()
    return out


def age_bin(y: pd.Series) -> pd.Series:
    return pd.cut(y, bins=[-999, -1, 5, 10, 15, 20, 999], labels=["pre_disturbance", "0-5", "6-10", "11-15", "16-20", "21+"])


def pair_support(event: pd.DataFrame, pairs: pd.DataFrame, product_col: str) -> pd.DataFrame:
    rows = []
    if pd.api.types.is_bool_dtype(event[product_col]):
        valid_metric = event[product_col].fillna(False)
    else:
        valid_metric = event[product_col].notna()
    fire = event[event["disturbance_type"].eq("fire_total") & valid_metric]
    harvest = event[event["disturbance_type"].eq("harvest_total") & valid_metric]
    ages = sorted(set(event["years_since_disturbance"].dropna().astype(int)))
    for y in ages:
        f_events = set(fire.loc[fire["years_since_disturbance"].eq(y), "event_id"])
        h_events = set(harvest.loc[harvest["years_since_disturbance"].eq(y), "event_id"])
        complete = pairs[pairs["fire_event_id"].isin(f_events) & pairs["harvest_event_id"].isin(h_events)]
        rows.append({"years_since_disturbance": int(y), "n_complete_fire_harvest_pairs": int(len(complete))})
    return pd.DataFrame(rows)


def temporal_support(event: pd.DataFrame, site: pd.DataFrame, pairs: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    if pd.api.types.is_bool_dtype(event[metric]):
        valid_event = event[event[metric].fillna(False)].copy()
    else:
        valid_event = event[event[metric].notna()].copy()
    if pd.api.types.is_bool_dtype(site[metric]):
        valid_site = site[site[metric].fillna(False)].copy()
    else:
        valid_site = site[site[metric].notna()].copy()
    rows = []
    pair_counts = pair_support(event, pairs, metric)
    for keys, group in valid_event.groupby(["disturbance_type", "years_since_disturbance"]):
        dtype, y = keys
        site_group = valid_site[(valid_site["disturbance_type"].eq(dtype)) & (valid_site["years_since_disturbance"].eq(y))]
        rows.append(
            {
                "product": product,
                "disturbance_type": dtype,
                "years_since_disturbance": int(y),
                "n_unique_events": int(group["event_id"].nunique()),
                "n_event_years": int(len(group)),
                "n_sites": int(site_group["site_id"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(pair_counts, on="years_since_disturbance", how="left")
    return out


def temporal_support_bins(support: pd.DataFrame) -> pd.DataFrame:
    df = support.copy()
    df["age_bin"] = age_bin(df["years_since_disturbance"])
    return df.groupby(["product", "disturbance_type", "age_bin"], observed=False, as_index=False).agg(
        n_unique_events=("n_unique_events", "max"),
        n_event_years=("n_event_years", "sum"),
        n_sites=("n_sites", "max"),
        max_complete_fire_harvest_pairs=("n_complete_fire_harvest_pairs", "max"),
    )


def event_longitudinal_support(event: pd.DataFrame, product: str, metric: str) -> pd.DataFrame:
    valid = event[event[metric].notna()].copy()
    return valid.groupby(["disturbance_type", "event_id"], as_index=False).agg(
        first_observation_year=("observation_year", "min"),
        last_observation_year=("observation_year", "max"),
        n_valid_annual_observations=("observation_year", "nunique"),
        min_years_since_disturbance=("years_since_disturbance", "min"),
        max_years_since_disturbance=("years_since_disturbance", "max"),
    ).assign(product=product, temporal_span=lambda d: d["last_observation_year"] - d["first_observation_year"])


def pre_disturbance_audit(multi: pd.DataFrame) -> pd.DataFrame:
    pre = multi[multi["years_since_disturbance"].lt(0)].copy()
    metrics = ["s2_reference_distance", "s2_nbr_gap", "ae_euclidean_distance", "ae_cosine_distance"]
    rows = []
    for metric in metrics:
        for dtype, group in pre.groupby("disturbance_type"):
            vals = group[metric].dropna()
            rows.append(
                {
                    "metric": metric,
                    "disturbance_type": dtype,
                    "event_count": int(group.loc[group[metric].notna(), "event_id"].nunique()),
                    "event_year_count": int(vals.shape[0]),
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "p25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                    "p75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                    "strong_outlier_count_p99_rule": int((vals > vals.quantile(0.99)).sum()) if len(vals) else 0,
                }
            )
    return pd.DataFrame(rows)


def product_agreement(multi: pd.DataFrame) -> pd.DataFrame:
    metrics = ["nbr_gap_median", "nbr_gap_change", "s2_nbr_gap", "s2_reference_distance", "ae_euclidean_distance", "ae_cosine_distance"]
    rows = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            pair = multi[[a, b]].dropna()
            if len(pair) < 5:
                continue
            rows.append({"level": "event_year", "metric_a": a, "metric_b": b, "n": int(len(pair)), "pearson": float(pair[a].corr(pair[b])), "spearman": float(pair[a].corr(pair[b], method="spearman"))})
    return pd.DataFrame(rows)


def quality_audit(s2_site: pd.DataFrame, ae_site: pd.DataFrame, multi: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    rows = []
    for product, df, metric in [("sentinel2", s2_site, "s2_reference_distance"), ("alphaearth", ae_site, "ae_euclidean_distance")]:
        rows.append({"product": product, "check": "site_year_rows", "value": int(len(df)), "note": ""})
        rows.append({"product": product, "check": "missing_pair_years", "value": int(df[metric].isna().sum()) if metric in df else int(len(df)), "note": "metric missing"})
        if metric in df and df[metric].notna().any():
            rows.append({"product": product, "check": "extreme_distance_p99", "value": float(df[metric].quantile(0.99)), "note": metric})
        by_year = df.groupby("observation_year")[metric].apply(lambda s: s.notna().mean()) if metric in df else pd.Series(dtype=float)
        for year, value in by_year.items():
            rows.append({"product": product, "check": "annual_availability_fraction", "value": float(value), "note": str(year)})
    if "ae_disturbed_norm" in ae_site:
        rows.extend(
            [
                {"product": "alphaearth", "check": "disturbed_norm_median", "value": float(ae_site["ae_disturbed_norm"].median()), "note": ""},
                {"product": "alphaearth", "check": "reference_norm_median", "value": float(ae_site["ae_reference_norm"].median()), "note": ""},
                {"product": "alphaearth", "check": "embedding_band_count", "value": int(len(bands)), "note": "programmatically verified"},
            ]
        )
    one_site = multi.groupby(["disturbance_type", "event_id", "observation_year"], as_index=False).agg(n_valid_sites=("n_valid_sites_s2", "max"))
    rows.append({"product": "combined", "check": "event_years_with_one_or_fewer_valid_s2_sites", "value": int(one_site["n_valid_sites"].le(1).sum()), "note": ""})
    return pd.DataFrame(rows)


def make_figures(support: pd.DataFrame, longitudinal: pd.DataFrame, s2_event: pd.DataFrame, ae_event: pd.DataFrame, agreement: pd.DataFrame) -> None:
    colors = {"fire_total": "#c95f3f", "harvest_total": "#2f7f73"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for (product, dtype), group in support.groupby(["product", "disturbance_type"]):
        if product == "both":
            ax.plot(group["years_since_disturbance"], group["n_unique_events"], marker="o", color=colors[dtype], linestyle="--", label=f"{dtype} both")
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("events")
    ax.set_title("Temporal support by age, both products")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_support_by_age.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    longitudinal.groupby(["product", "disturbance_type"])["temporal_span"].median().unstack().plot(kind="bar", ax=ax)
    ax.set_ylabel("median calendar-year span")
    ax.set_title("Event longitudinal span")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_longitudinal_span.png", dpi=220)
    plt.close(fig)

    for metric, event, fname, title in [
        ("s2_reference_distance", s2_event, "s2_reference_distance_by_age.png", "Sentinel-2 standardized distance by age"),
        ("ae_euclidean_distance", ae_event, "alphaearth_euclidean_distance_by_age.png", "AlphaEarth Euclidean distance by age"),
        ("ae_cosine_distance", ae_event, "alphaearth_cosine_distance_by_age.png", "AlphaEarth cosine distance by age"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for dtype, group in event.groupby("disturbance_type"):
            desc = group.groupby("years_since_disturbance")[metric].median().reset_index()
            ax.plot(desc["years_since_disturbance"], desc[metric], marker="o", color=colors[dtype], label=dtype)
        ax.set_xlabel("years since disturbance")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    subset = agreement.sort_values("spearman", key=lambda s: s.abs(), ascending=False).head(12)
    ax.barh(subset["metric_a"] + " vs " + subset["metric_b"], subset["spearman"], color="#526760")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman correlation")
    ax.set_title("Product agreement audit")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "product_agreement.png", dpi=220)
    plt.close(fig)


def write_readme(summary: dict, ae_bands: list[str], ae_scale: float) -> None:
    text = f"""# Multivariate recovery extraction

This Phase 10 output preserves the existing disturbed-reference design and
extracts paired Sentinel-2 SR Harmonized and AlphaEarth annual embedding
observations for calendar years 2017-2025.

Sentinel-2 uses July 1-August 31 seasonal medians with SCL/QA60 cloud, shadow,
snow, and cirrus masking. Sentinel-2 distances are Euclidean distances in a
robustly standardized six-band reference-forest feature space.

AlphaEarth uses `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. Band count was verified
programmatically as {len(ae_bands)} bands. Nominal scale reported by Earth Engine:
{ae_scale} m.

No recovery model, classifier, PCA, UMAP, or new sampling frame is created here.

```json
{json.dumps(summary, indent=2)}
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    refs, points = read_design()
    pairs = pd.read_csv(PAIR_COMPLETENESS)
    landsat = pd.read_csv(LANDSAT_EVENT_YEAR)

    s2_raw_csv = OUT_DIR / "s2_point_year_raw.csv"
    ae_raw_csv = OUT_DIR / "alphaearth_point_year_raw.csv"
    if s2_raw_csv.exists():
        s2_raw = pd.read_csv(s2_raw_csv)
    else:
        s2_raw = extract_s2(points)
        s2_raw.to_csv(s2_raw_csv, index=False, float_format="%.6f")

    ae_meta_path = OUT_DIR / "alphaearth_metadata.json"
    if ae_raw_csv.exists() and ae_meta_path.exists():
        ae_raw = pd.read_csv(ae_raw_csv)
        meta = json.loads(ae_meta_path.read_text())
        ae_bands = meta["band_names"]
        ae_scale = float(meta["nominal_scale_m"])
    else:
        ae_raw, ae_bands, ae_scale = extract_alphaearth(points)
        ae_raw.to_csv(ae_raw_csv, index=False, float_format="%.8f")
        ae_meta_path.write_text(json.dumps({"band_names": ae_bands, "nominal_scale_m": ae_scale}, indent=2), encoding="utf-8")

    s2_site = pivot_roles(s2_raw, S2_BANDS + ["nbr", "ndvi", "s2_valid_observation_count"], "sentinel2")
    s2_site, std = robust_reference_standardization(s2_site)
    ae_site = pivot_roles(ae_raw, ae_bands + ["ae_valid_mask"], "alphaearth")
    ae_site = alphaearth_distances(ae_site, ae_bands)

    s2_metrics = ["s2_reference_distance", "s2_nbr_gap", "s2_ndvi_gap", "disturbed_nbr", "reference_nbr", "disturbed_ndvi", "reference_ndvi"] + [f"{b}_diff" for b in S2_BANDS]
    ae_metrics = ["ae_euclidean_distance", "ae_cosine_distance", "ae_disturbed_norm", "ae_reference_norm"]
    s2_event = aggregate_event(s2_site, s2_metrics)
    ae_event = aggregate_event(ae_site, ae_metrics)
    multi = combine_event_years(s2_event, ae_event, landsat)

    both_site = s2_site[["pair_id", "site_id", "event_id", "disturbance_type", "disturbance_year", "observation_year", "years_since_disturbance", "s2_reference_distance"]].merge(
        ae_site[["pair_id", "site_id", "observation_year", "ae_euclidean_distance"]],
        on=["pair_id", "site_id", "observation_year"],
        how="outer",
    )
    both_site["both_products_available"] = both_site["s2_reference_distance"].notna() & both_site["ae_euclidean_distance"].notna()
    both_event = multi.copy()

    support = pd.concat(
        [
            temporal_support(s2_event, s2_site, pairs, "sentinel2", "s2_reference_distance"),
            temporal_support(ae_event, ae_site, pairs, "alphaearth", "ae_euclidean_distance"),
            temporal_support(both_event, both_site, pairs, "both", "both_products_available"),
        ],
        ignore_index=True,
    )
    support_bins = temporal_support_bins(support)
    longitudinal = pd.concat(
        [
            event_longitudinal_support(s2_event, "sentinel2", "s2_reference_distance"),
            event_longitudinal_support(ae_event, "alphaearth", "ae_euclidean_distance"),
        ],
        ignore_index=True,
    )
    pre = pre_disturbance_audit(multi)
    agreement = product_agreement(multi)
    quality = quality_audit(s2_site, ae_site, multi, ae_bands)

    s2_site.to_csv(OUT_DIR / "s2_site_year.csv", index=False, float_format="%.6f")
    s2_event.to_csv(OUT_DIR / "s2_event_year.csv", index=False, float_format="%.6f")
    ae_site.to_csv(OUT_DIR / "alphaearth_site_year.csv", index=False, float_format="%.8f")
    ae_event.to_csv(OUT_DIR / "alphaearth_event_year.csv", index=False, float_format="%.8f")
    multi.to_csv(OUT_DIR / "multivariate_event_year.csv", index=False, float_format="%.8f")
    std.to_csv(OUT_DIR / "s2_reference_standardization.csv", index=False, float_format="%.8f")
    support.to_csv(OUT_DIR / "temporal_support_by_age.csv", index=False)
    support_bins.to_csv(OUT_DIR / "temporal_support_by_age_bin.csv", index=False)
    longitudinal.to_csv(OUT_DIR / "event_longitudinal_support.csv", index=False)
    pre.to_csv(OUT_DIR / "pre_disturbance_multivariate_audit.csv", index=False, float_format="%.8f")
    agreement.to_csv(OUT_DIR / "product_agreement_audit.csv", index=False, float_format="%.8f")
    quality.to_csv(OUT_DIR / "multivariate_quality_audit.csv", index=False, float_format="%.8f")

    def event_counts(event, metric):
        valid = event[event[metric].notna()]
        return valid.groupby("disturbance_type")["event_id"].nunique().to_dict()

    both_valid = multi[multi["both_products_available"]]
    ae_corr = ae_site[["ae_euclidean_distance", "ae_cosine_distance"]].dropna()
    age_support_fixed = support[support["years_since_disturbance"].isin([0, 1, 5, 10, 15, 20])].to_dict(orient="records")
    bins_summary = support_bins.to_dict(orient="records")
    long_summary = longitudinal.groupby(["product", "disturbance_type"]).agg(
        median_valid_years=("n_valid_annual_observations", "median"),
        median_temporal_span=("temporal_span", "median"),
        median_age_span=("max_years_since_disturbance", "median"),
    ).reset_index().to_dict(orient="records")

    summary = {
        "sentinel2_site_year_rows": int(len(s2_site)),
        "sentinel2_site_year_valid_pairs": int(s2_site["s2_reference_distance"].notna().sum()),
        "sentinel2_event_year_rows": int(len(s2_event)),
        "alphaearth_site_year_rows": int(len(ae_site)),
        "alphaearth_site_year_valid_pairs": int(ae_site["ae_euclidean_distance"].notna().sum()),
        "alphaearth_event_year_rows": int(len(ae_event)),
        "sentinel2_events_by_type": event_counts(s2_event, "s2_reference_distance"),
        "alphaearth_events_by_type": event_counts(ae_event, "ae_euclidean_distance"),
        "both_products_events_by_type": both_valid.groupby("disturbance_type")["event_id"].nunique().to_dict(),
        "exact_pairs_total_design": int(len(pairs)),
        "age_support_fixed": age_support_fixed,
        "age_bin_support": bins_summary,
        "longitudinal_support_summary": long_summary,
        "s2_standardization": "reference-only robust median and IQR/1.349 scale over selected reference observations; common transform for fire and harvest",
        "alphaearth_band_count": int(len(ae_bands)),
        "alphaearth_nominal_scale_m": ae_scale,
        "alphaearth_euclidean_cosine_spearman": float(ae_corr.corr(method="spearman").iloc[0, 1]) if len(ae_corr) else None,
        "phase11_s2_supported": bool(support[(support["product"].eq("sentinel2")) & (support["years_since_disturbance"].between(0, 15))]["n_unique_events"].median() >= 20),
        "phase11_alphaearth_supported": bool(support[(support["product"].eq("alphaearth")) & (support["years_since_disturbance"].between(0, 15))]["n_unique_events"].median() >= 20),
        "recommended_s2_age_range": "0-15 primary; 16-20 cautious; 21+ descriptive only",
        "recommended_alphaearth_age_range": "0-15 primary; 16-20 cautious; 21+ descriptive only",
        "main_limitation": "2017-2025 product window means older recovery ages are mostly cohort substitution, not fully longitudinal within-event trajectories.",
    }
    (OUT_DIR / "multivariate_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary, ae_bands, ae_scale)
    make_figures(support, longitudinal, s2_event, ae_event, agreement)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
