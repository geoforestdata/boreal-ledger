#!/usr/bin/env python3
"""Extract and QA annual Landsat recovery time series for approved sites.

This phase builds longitudinal spectral observations only. It does not fit
recovery curves, extract Sentinel-2, extract AlphaEarth, or validate reference
forests.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
MPL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "landsat_timeseries" / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VENV_SITE_PACKAGES = Path(__file__).resolve().parents[1] / ".venv" / "lib" / "python3.14" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import ee


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "landsat_timeseries"
FIG_DIR = ROOT / "figures" / "landsat_timeseries"

FIRE_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_fire_sites.gpkg"
HARVEST_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_harvest_sites.gpkg"
MATCHES = ROOT / "outputs" / "sampling_frame" / "matching_strategy_comparison.csv"
CANDIDATE_PAIRS = ROOT / "outputs" / "sampling_frame" / "fire_harvest_candidate_pairs.csv"
SELECTED_FIRE = ROOT / "outputs" / "sampling_frame" / "selected_fire_events.csv"
SELECTED_HARVEST = ROOT / "outputs" / "sampling_frame" / "selected_harvest_events.csv"

PROJECT = "ibfra2026"
START_MONTH_DAY = "07-01"
END_MONTH_DAY = "08-31"
FINAL_OBSERVATION_YEAR = 2025
SCALE_M = 30
CHUNK_SIZE = 180

SENSORS = {
    "LT05": {
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    },
    "LE07": {
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    },
    "LC08": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    },
    "LC09": {
        "collection": "LANDSAT/LC09/C02/T1_L2",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    },
}
COMMON_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
SPECTRAL_FIELDS = COMMON_BANDS + ["nbr", "ndvi"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def initialize_ee() -> None:
    ee.Initialize(project=PROJECT)


def qa_mask(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    radsat = image.select("QA_RADSAT")
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
        .And(radsat.eq(0))
    )
    return image.updateMask(mask)


def prep_landsat(image: ee.Image, sensor: str) -> ee.Image:
    spec = SENSORS[sensor]
    scaled = qa_mask(image).select(spec["bands"], COMMON_BANDS).multiply(0.0000275).add(-0.2)
    valid = scaled.reduce(ee.Reducer.min()).gt(-0.2).And(scaled.reduce(ee.Reducer.max()).lt(1.6))
    nbr = scaled.normalizedDifference(["nir", "swir2"]).rename("nbr")
    ndvi = scaled.normalizedDifference(["nir", "red"]).rename("ndvi")
    return scaled.addBands([nbr, ndvi]).updateMask(valid).copyProperties(image, ["system:time_start"]).set("sensor", sensor)


def sensor_collection(sensor: str, year: int, region: ee.Geometry) -> ee.ImageCollection:
    start = f"{year}-{START_MONTH_DAY}"
    end = f"{year}-{END_MONTH_DAY}"
    return (
        ee.ImageCollection(SENSORS[sensor]["collection"])
        .filterBounds(region)
        .filterDate(start, end)
        .map(lambda img: prep_landsat(img, sensor))
    )


def annual_image(year: int, region: ee.Geometry) -> ee.Image:
    collections = [sensor_collection(sensor, year, region) for sensor in SENSORS]
    merged = ee.ImageCollection(collections[0])
    for coll in collections[1:]:
        merged = merged.merge(coll)
    composite = merged.select(SPECTRAL_FIELDS).median()
    counts = []
    for sensor, coll in zip(SENSORS.keys(), collections):
        counts.append(coll.select("blue").count().rename(f"count_{sensor}"))
    total_count = merged.select("blue").count().rename("valid_observation_count")
    return composite.addBands(counts).addBands(total_count).set("observation_year", year)


def read_sites() -> gpd.GeoDataFrame:
    fire = gpd.read_file(FIRE_SITES, layer="candidate_fire_sites", engine="pyogrio")
    harvest = gpd.read_file(HARVEST_SITES, layer="candidate_harvest_sites", engine="pyogrio")
    sites = pd.concat([fire, harvest], ignore_index=True)
    sites = gpd.GeoDataFrame(sites, geometry="geometry", crs=fire.crs).to_crs("EPSG:4326")
    if not all(sites.geometry.geom_type.eq("Point")):
        raise ValueError("Sampling-frame site geometries must be Points.")
    return sites


def site_features(sites: gpd.GeoDataFrame) -> list[ee.Feature]:
    features = []
    for _, row in sites.iterrows():
        props = row.drop(labels="geometry").to_dict()
        props = {k: (None if pd.isna(v) else v) for k, v in props.items()}
        point = ee.Geometry.Point([float(row.geometry.x), float(row.geometry.y)])
        features.append(ee.Feature(point, props))
    return features


def expected_site_years(sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in sites.iterrows():
        start = int(row["disturbance_year"]) - 3
        for year in range(start, FINAL_OBSERVATION_YEAR + 1):
            censor_year = row.get("censor_year")
            censor_value = pd.to_numeric(censor_year, errors="coerce")
            post_censor = pd.notna(censor_value) and year >= int(censor_value)
            rows.append(
                {
                    "disturbance_type": row["disturbance_type"],
                    "event_id": row["event_id"],
                    "site_id": row["site_id"],
                    "disturbance_year": int(row["disturbance_year"]),
                    "observation_year": year,
                    "years_since_disturbance": year - int(row["disturbance_year"]),
                    "censor_year": censor_value if pd.notna(censor_value) else pd.NA,
                    "censor_reason": row.get("censor_reason", ""),
                    "post_censor": bool(post_censor),
                }
            )
    return pd.DataFrame(rows)


def extract_year(year: int, features: list[ee.Feature], region: ee.Geometry) -> pd.DataFrame:
    image = annual_image(year, region)
    rows = []
    bands = SPECTRAL_FIELDS + [f"count_{sensor}" for sensor in SENSORS] + ["valid_observation_count"]
    for i in range(0, len(features), CHUNK_SIZE):
        fc = ee.FeatureCollection(features[i : i + CHUNK_SIZE])
        sampled = image.sampleRegions(collection=fc, scale=SCALE_M, geometries=False, tileScale=4)
        data = sampled.getInfo().get("features", [])
        for feat in data:
            props = feat["properties"]
            rows.append({k: props.get(k) for k in props.keys() if k not in bands} | {b: props.get(b) for b in bands})
        time.sleep(0.05)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["observation_year"] = year
    count_cols = [f"count_{sensor}" for sensor in SENSORS]
    for col in count_cols + ["valid_observation_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["sensor_information"] = out[count_cols].apply(
        lambda r: ";".join([f"{sensor}:{int(r[f'count_{sensor}'])}" for sensor in SENSORS if int(r[f"count_{sensor}"]) > 0]),
        axis=1,
    )
    out["data_available"] = out["valid_observation_count"] > 0
    return out


def build_site_year_table(sites: gpd.GeoDataFrame) -> pd.DataFrame:
    expected = expected_site_years(sites.drop(columns="geometry"))
    years = sorted(expected["observation_year"].unique())
    features = site_features(sites)
    region = ee.Geometry.MultiPoint([[float(g.x), float(g.y)] for g in sites.geometry]).bounds().buffer(15_000)
    observed = []
    for year in years:
        print(f"Extracting Landsat seasonal composite samples for {year}...", flush=True)
        observed.append(extract_year(int(year), features, region))
    obs = pd.concat([df for df in observed if not df.empty], ignore_index=True) if observed else pd.DataFrame()
    keys = ["disturbance_type", "event_id", "site_id", "disturbance_year", "observation_year"]
    keep = keys + SPECTRAL_FIELDS + [f"count_{sensor}" for sensor in SENSORS] + ["valid_observation_count", "sensor_information", "data_available"]
    obs = obs[[c for c in keep if c in obs.columns]].copy() if not obs.empty else pd.DataFrame(columns=keep)
    site_year = expected.merge(obs, on=keys, how="left")
    for col in SPECTRAL_FIELDS:
        site_year[col] = pd.to_numeric(site_year[col], errors="coerce")
    for col in [f"count_{sensor}" for sensor in SENSORS] + ["valid_observation_count"]:
        site_year[col] = pd.to_numeric(site_year[col], errors="coerce").fillna(0).astype(int)
    site_year["sensor_information"] = site_year["sensor_information"].fillna("")
    site_year["data_available"] = site_year["valid_observation_count"] > 0
    return site_year.sort_values(["disturbance_type", "event_id", "site_id", "observation_year"])


def aggregate_event_year(site_year: pd.DataFrame) -> pd.DataFrame:
    primary = site_year[~site_year["post_censor"] & site_year["data_available"]].copy()
    agg = primary.groupby(["disturbance_type", "event_id", "disturbance_year", "observation_year", "years_since_disturbance"], as_index=False).agg(
        **{f"{field}_median": (field, "median") for field in SPECTRAL_FIELDS},
        **{f"{field}_mean": (field, "mean") for field in SPECTRAL_FIELDS},
        **{f"{field}_sd": (field, "std") for field in SPECTRAL_FIELDS},
        valid_site_count=("site_id", "nunique"),
        median_valid_observation_count=("valid_observation_count", "median"),
        mean_valid_observation_count=("valid_observation_count", "mean"),
    )
    return agg.sort_values(["disturbance_type", "event_id", "observation_year"])


def matched_pair_year(event_year: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(CANDIDATE_PAIRS)
    selected_fire = set(pd.read_csv(SELECTED_FIRE)["event_id"].astype(str))
    selected_harvest = set(pd.read_csv(SELECTED_HARVEST)["event_id"].astype(str))
    pairs = pairs[pairs["fire_event_id"].astype(str).isin(selected_fire) & pairs["harvest_event_id"].astype(str).isin(selected_harvest)]
    pairs = pairs.sort_values(["age_difference", "centroid_distance_m"]).drop_duplicates("fire_event_id").copy()
    fire = event_year[event_year["disturbance_type"] == "fire_total"].copy()
    harvest = event_year[event_year["disturbance_type"] == "harvest_total"].copy()
    fire = fire.rename(columns={"event_id": "fire_event_id"})
    harvest = harvest.rename(columns={"event_id": "harvest_event_id"})
    rows = []
    for idx, pair in pairs.reset_index(drop=True).iterrows():
        f = fire[fire["fire_event_id"] == pair["fire_event_id"]]
        h = harvest[harvest["harvest_event_id"] == pair["harvest_event_id"]]
        merged = f.merge(h, on=["observation_year", "years_since_disturbance"], suffixes=("_fire", "_harvest"))
        for _, row in merged.iterrows():
            item = {
                "pair_id": f"pair_{idx + 1:03d}",
                "fire_event_id": pair["fire_event_id"],
                "harvest_event_id": pair["harvest_event_id"],
                "observation_year": int(row["observation_year"]),
                "years_since_disturbance": int(row["years_since_disturbance"]),
            }
            for field in SPECTRAL_FIELDS:
                item[f"fire_{field}"] = row.get(f"{field}_median_fire")
                item[f"harvest_{field}"] = row.get(f"{field}_median_harvest")
            rows.append(item)
    return pd.DataFrame(rows)


def coverage_tables(site_year: pd.DataFrame, event_year: pd.DataFrame) -> dict[str, pd.DataFrame]:
    expected = site_year.copy()
    expected["period"] = pd.cut(
        expected["years_since_disturbance"],
        bins=[-999, -1, 0, 5, 10, 15, 20, 999],
        labels=["pre_disturbance", "year_0", "years_1_5", "years_6_10", "years_11_15", "years_16_20", "years_21plus"],
    )
    by_year = expected.groupby(["observation_year", "disturbance_type"], as_index=False).agg(
        expected_site_years=("site_id", "count"),
        observed_site_years=("data_available", "sum"),
    )
    by_year["missing_site_years"] = by_year["expected_site_years"] - by_year["observed_site_years"]
    by_year["coverage_percent"] = by_year["observed_site_years"] / by_year["expected_site_years"] * 100
    by_event = expected.groupby(["disturbance_type", "event_id"], as_index=False).agg(
        expected_site_years=("site_id", "count"),
        observed_site_years=("data_available", "sum"),
    )
    by_event["coverage_percent"] = by_event["observed_site_years"] / by_event["expected_site_years"] * 100
    by_site = expected.groupby(["disturbance_type", "event_id", "site_id"], as_index=False).agg(
        expected_site_years=("site_id", "count"),
        observed_site_years=("data_available", "sum"),
    )
    by_site["coverage_percent"] = by_site["observed_site_years"] / by_site["expected_site_years"] * 100
    pre = expected[expected["years_since_disturbance"].isin([-3, -2, -1])].groupby(
        ["disturbance_type", "event_id", "site_id"], as_index=False
    ).agg(pre_years_available=("data_available", "sum"))
    pre["pre_disturbance_class"] = pre["pre_years_available"].astype(int).astype(str) + "_years"
    sensor_cols = [f"count_{sensor}" for sensor in SENSORS]
    sensor_rows = []
    for _, row in expected.iterrows():
        for sensor in SENSORS:
            sensor_rows.append(
                {
                    "observation_year": row["observation_year"],
                    "disturbance_type": row["disturbance_type"],
                    "sensor": sensor,
                    "valid_observations": int(row[f"count_{sensor}"]),
                    "site_id": row["site_id"],
                    "data_available": bool(row[f"count_{sensor}"] > 0),
                }
            )
    sensor = pd.DataFrame(sensor_rows).groupby(["observation_year", "disturbance_type", "sensor"], as_index=False).agg(
        site_years_with_sensor=("data_available", "sum"),
        total_valid_observations=("valid_observations", "sum"),
    )
    by_ysd = expected.groupby(["years_since_disturbance", "disturbance_type"], as_index=False).agg(
        expected_site_years=("site_id", "count"),
        observed_site_years=("data_available", "sum"),
    )
    by_ysd["coverage_percent"] = by_ysd["observed_site_years"] / by_ysd["expected_site_years"] * 100
    return {
        "landsat_coverage_by_year.csv": by_year,
        "landsat_coverage_by_event.csv": by_event,
        "landsat_coverage_by_site.csv": by_site,
        "landsat_pre_disturbance_coverage.csv": pre,
        "landsat_sensor_audit.csv": sensor,
        "coverage_by_years_since_disturbance_internal": by_ysd,
        "period_internal": expected.groupby(["period", "disturbance_type"], observed=False, as_index=False).agg(
            expected_site_years=("site_id", "count"),
            observed_site_years=("data_available", "sum"),
        ),
    }


def sensor_artifact_audit(site_year: pd.DataFrame) -> dict:
    annual = site_year[site_year["data_available"]].groupby(["observation_year", "disturbance_type"], as_index=False).agg(
        nbr_median=("nbr", "median"), nbr_mean=("nbr", "mean"), valid_count=("site_id", "count")
    )
    transition_notes = []
    for year in [2003, 2012, 2013, 2021, 2022]:
        before = annual[annual["observation_year"] == year - 1]["nbr_median"]
        after = annual[annual["observation_year"] == year]["nbr_median"]
        if not before.empty and not after.empty:
            transition_notes.append({"transition_year": year, "median_nbr_step": float(after.median() - before.median())})
    slc = site_year[(site_year["observation_year"] >= 2004) & (site_year["observation_year"] <= 2012)]
    non_slc = site_year[(site_year["observation_year"] < 2004) | (site_year["observation_year"] > 2012)]
    return {
        "sensor_transition_steps_checked": transition_notes,
        "landsat7_slc_off_period_coverage_percent": float(slc["data_available"].mean() * 100) if len(slc) else None,
        "non_slc_off_period_coverage_percent": float(non_slc["data_available"].mean() * 100) if len(non_slc) else None,
    }


def write_figures(site_year: pd.DataFrame, event_year: pd.DataFrame, tables: dict[str, pd.DataFrame], pair_year: pd.DataFrame) -> None:
    by_year = tables["landsat_coverage_by_year.csv"]
    fig, ax = plt.subplots(figsize=(11, 5))
    by_year.pivot(index="observation_year", columns="disturbance_type", values="coverage_percent").plot(ax=ax)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Annual Landsat site-year coverage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_by_year.png", dpi=220)
    plt.close(fig)

    by_ysd = tables["coverage_by_years_since_disturbance_internal"]
    fig, ax = plt.subplots(figsize=(11, 5))
    by_ysd.pivot(index="years_since_disturbance", columns="disturbance_type", values="coverage_percent").plot(ax=ax)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage by years since disturbance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_by_years_since_disturbance.png", dpi=220)
    plt.close(fig)

    pre = tables["landsat_pre_disturbance_coverage.csv"]
    fig, ax = plt.subplots(figsize=(8, 5))
    pre.groupby(["disturbance_type", "pre_years_available"]).size().unstack(fill_value=0).plot(kind="bar", ax=ax)
    ax.set_ylabel("Site count")
    ax.set_title("Pre-disturbance availability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pre_disturbance_coverage.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    site_year.loc[site_year["data_available"], "valid_observation_count"].plot(kind="hist", bins=20, ax=ax, color="#4c78a8")
    ax.set_xlabel("Valid source observations per seasonal composite")
    ax.set_title("Valid Landsat observations per composite")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "valid_observations_per_composite.png", dpi=220)
    plt.close(fig)

    sensor = tables["landsat_sensor_audit.csv"]
    fig, ax = plt.subplots(figsize=(12, 5))
    sensor.groupby(["observation_year", "sensor"])["site_years_with_sensor"].sum().unstack(fill_value=0).plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Site-years with sensor contribution")
    ax.set_title("Sensor contributions through time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensor_contribution_by_year.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    event_year.groupby(["observation_year", "disturbance_type"])["valid_site_count"].median().unstack().plot(ax=ax)
    ax.set_ylabel("Median valid sites per event-year")
    ax.set_title("Event-level site support through time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_site_support_by_year.png", dpi=220)
    plt.close(fig)

    rng = np.random.default_rng(42)
    fire_events = event_year.loc[event_year["disturbance_type"] == "fire_total", "event_id"].drop_duplicates().to_numpy()
    chosen = set(rng.choice(fire_events, size=min(8, len(fire_events)), replace=False)) if len(fire_events) else set()
    matched_harvest = set()
    if not pair_year.empty:
        matched_harvest = set(pair_year.loc[pair_year["fire_event_id"].isin(chosen), "harvest_event_id"])
    plot_df = event_year[event_year["event_id"].isin(chosen | matched_harvest)].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, part in plot_df.groupby("event_id"):
        color = "#b55239" if part["disturbance_type"].iloc[0] == "fire_total" else "#327c73"
        ax.plot(part["years_since_disturbance"], part["nbr_median"], alpha=0.55, linewidth=1.2, color=color)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Years since disturbance")
    ax.set_ylabel("NBR median across valid sites")
    ax.set_title("Example raw matched NBR trajectories, unsmoothed QA view")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_raw_matched_trajectories.png", dpi=220)
    plt.close(fig)


def write_readme(summary: dict) -> None:
    lines = [
        "# Landsat recovery time series QA",
        "",
        "This phase extracts annual July-August Landsat Collection 2 Level 2 surface-reflectance composites for approved disturbed sampling sites.",
        "",
        "Rows in `landsat_site_year.csv` are site-year observations. The intended primary modelling input is `landsat_event_year.csv`, where valid sites are aggregated to event-year observations so that disturbance events, not sites, remain the inferential unit.",
        "",
        "NBR and NDVI are spectral indicators. They are not interpreted here as biomass, structural recovery, or ecological recovery.",
        "",
        "Reference-search areas are not extracted in this phase because they are candidate-only and not validated undisturbed controls.",
        "",
        "## Summary",
        "",
        json.dumps(summary, indent=2),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    initialize_ee()
    sites = read_sites()
    print("SITE GEOMETRY AUDIT")
    print(f"site_count={len(sites)} geometry_types={sites.geometry.geom_type.value_counts().to_dict()} crs={sites.crs}")

    site_year = build_site_year_table(sites)
    event_year = aggregate_event_year(site_year)
    pair_year = matched_pair_year(event_year)
    tables = coverage_tables(site_year, event_year)
    sensor_audit = sensor_artifact_audit(site_year)

    site_year.to_csv(OUT_DIR / "landsat_site_year.csv", index=False, float_format="%.6f")
    event_year.to_csv(OUT_DIR / "landsat_event_year.csv", index=False, float_format="%.6f")
    pair_year.to_csv(OUT_DIR / "landsat_matched_pair_year.csv", index=False, float_format="%.6f")
    for filename, df in tables.items():
        if filename.endswith(".csv"):
            df.to_csv(OUT_DIR / filename, index=False, float_format="%.6f")

    expected = int(len(site_year))
    observed = int(site_year["data_available"].sum())
    by_type = site_year.groupby("disturbance_type").agg(expected_site_years=("site_id", "count"), observed_site_years=("data_available", "sum"))
    by_type["coverage_percent"] = by_type["observed_site_years"] / by_type["expected_site_years"] * 100
    event_completeness = event_year.groupby("disturbance_type")["event_id"].nunique().to_dict()
    pair_expected = int(pd.read_csv(SELECTED_FIRE)["event_id"].nunique())
    pair_observed = int(pair_year["pair_id"].nunique()) if not pair_year.empty else 0
    pre = tables["landsat_pre_disturbance_coverage.csv"]
    summary = {
        "total_expected_site_years": expected,
        "observed_site_years": observed,
        "missing_site_years": expected - observed,
        "overall_coverage_percent": observed / expected * 100 if expected else None,
        "coverage_by_disturbance_type": by_type.reset_index().to_dict(orient="records"),
        "pre_disturbance_site_counts": pre.groupby(["disturbance_type", "pre_years_available"]).size().reset_index(name="site_count").to_dict(orient="records"),
        "events_with_any_event_year_observation": event_completeness,
        "matched_pairs_with_any_longitudinal_overlap": pair_observed,
        "selected_pairs_expected": pair_expected,
        "sensor_audit": sensor_audit,
        "landsat_sampling_support": "Point sample at 30 m scale; represents the Landsat pixel intersecting the site point.",
        "seasonal_window": "July 1-August 31, annual median composite; missing years are not interpolated.",
        "adequacy_note": "Adequacy should be judged from coverage diagnostics; no recovery interpretation is made in this phase.",
    }
    (OUT_DIR / "landsat_timeseries_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    write_figures(site_year, event_year, tables, pair_year)

    print("\nLANDSAT TIMESERIES SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nCOVERAGE BY YEARS SINCE DISTURBANCE")
    print(tables["coverage_by_years_since_disturbance_internal"].to_string(index=False))
    print("\nEVENT-YEAR ROWS")
    print(len(event_year))
    print("\nMATCHED-PAIR YEAR ROWS")
    print(len(pair_year))


if __name__ == "__main__":
    main()
