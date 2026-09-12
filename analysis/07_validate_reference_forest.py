#!/usr/bin/env python3
"""Validate candidate reference-forest support for the recovery design.

This script stops at the sampling-frame validation stage. It does not extract
reference Landsat trajectories, Sentinel-2, AlphaEarth, recovery gaps, or fitted
recovery models.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reference_forest"
FIG_DIR = ROOT / "figures" / "reference_forest"
MPL_DIR = OUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt

REFERENCE_AREAS = ROOT / "outputs" / "sampling_frame" / "candidate_reference_search_areas.gpkg"
FIRE_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_fire_sites.gpkg"
HARVEST_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_harvest_sites.gpkg"
DISTURBANCE_GPKG = ROOT / "outputs" / "event_histories" / "disturbance_events.gpkg"
FORESTRY_GPKG = ROOT / "data" / "raw" / "forestry_interventions" / "interv_fores_lebel_100km_circle.gpkg"
LANDSAT_SITE_YEAR = ROOT / "outputs" / "landsat_timeseries" / "landsat_site_year.csv"
LANDSAT_EVENT_YEAR = ROOT / "outputs" / "landsat_timeseries" / "landsat_event_year.csv"

CRS_PROJECTED = "EPSG:32198"
CRS_WGS84 = "EPSG:4326"
OBSERVATION_YEAR = 2024
PROJECT = "ibfra2026"
CELL_SIZE_M = 30
CELL_AREA_HA = CELL_SIZE_M * CELL_SIZE_M / 10_000
PRIMARY_TREECOVER_THRESHOLD = 50
TREECOVER_THRESHOLDS = [30, 50, 70]
INTERIOR_DISTANCES_M = [30, 60]
PRIMARY_INTERIOR_M = 60
INITIAL_RINGS_M = list(range(600, 3001, 200))
INITIAL_BEARINGS_DEG = list(range(0, 360, 15))
MAX_GEE_CANDIDATES_PER_SITE = 90
MAX_SELECTED_REFERENCES_PER_SITE = 10
MIN_REFERENCE_SPACING_M = 150
CHUNK_SIZE = 450


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


def clean_geom(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if not geom.is_empty else None


def read_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ref = gpd.read_file(REFERENCE_AREAS, layer="candidate_reference_search_areas", engine="pyogrio").to_crs(CRS_PROJECTED)
    fire_sites = gpd.read_file(FIRE_SITES, layer="candidate_fire_sites", engine="pyogrio").to_crs(CRS_PROJECTED)
    harvest_sites = gpd.read_file(HARVEST_SITES, layer="candidate_harvest_sites", engine="pyogrio").to_crs(CRS_PROJECTED)
    sites = pd.concat([fire_sites, harvest_sites], ignore_index=True)
    sites = gpd.GeoDataFrame(sites, geometry="geometry", crs=fire_sites.crs).to_crs(CRS_PROJECTED)
    fires = gpd.read_file(DISTURBANCE_GPKG, layer="fire_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    harvest = gpd.read_file(DISTURBANCE_GPKG, layer="harvest_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    forestry = gpd.read_file(FORESTRY_GPKG, engine="pyogrio").to_crs(CRS_PROJECTED)
    official = pd.concat(
        [
            fires.assign(exclusion_source="mapped_wildfire")[["exclusion_source", "geometry"]],
            harvest.assign(exclusion_source="mapped_harvest_event")[["exclusion_source", "geometry"]],
            forestry.assign(exclusion_source="mapped_forestry_intervention")[["exclusion_source", "geometry"]],
        ],
        ignore_index=True,
    )
    official = gpd.GeoDataFrame(official, geometry="geometry", crs=CRS_PROJECTED)
    official = official[official.geometry.notna() & ~official.geometry.is_empty].copy()
    return ref, sites, official, forestry


def generate_raw_candidates(ref: gpd.GeoDataFrame, sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    site_lookup = sites.set_index("site_id")
    rows = []
    for _, annulus in ref.iterrows():
        site_id = annulus["associated_site_id"]
        if site_id not in site_lookup.index:
            continue
        site = site_lookup.loc[site_id]
        sx, sy = float(site.geometry.x), float(site.geometry.y)
        annulus_geom = clean_geom(annulus.geometry)
        if annulus_geom is None:
            continue
        prepared = prep(annulus_geom)
        candidate_number = 0
        for radius in INITIAL_RINGS_M:
            for bearing in INITIAL_BEARINGS_DEG:
                angle = math.radians(bearing)
                point = Point(sx + radius * math.cos(angle), sy + radius * math.sin(angle))
                if not prepared.contains(point):
                    continue
                candidate_number += 1
                rows.append(
                    {
                        "candidate_id": f"{site_id}_refcand_{candidate_number:04d}",
                        "reference_search_id": annulus["reference_search_id"],
                        "associated_site_id": site_id,
                        "associated_event_id": annulus["associated_event_id"],
                        "disturbance_type": site["disturbance_type"],
                        "disturbance_year": int(site["disturbance_year"]),
                        "years_since_disturbance": int(site["years_since_disturbance"]),
                        "age_bin": site["age_bin"],
                        "distance_m": float(point.distance(site.geometry)),
                        "bearing_deg": int(bearing),
                        "stage": "candidate",
                        "geometry": point,
                    }
                )
    candidates = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_PROJECTED)
    return candidates


def official_filter(candidates: gpd.GeoDataFrame, official: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = candidates.copy()
    out["candidate_row_id"] = np.arange(len(out))
    official_geom = official[["geometry"]].copy()

    hits = gpd.sjoin(
        out[["candidate_row_id", "geometry"]],
        official_geom,
        how="left",
        predicate="within",
    )
    hit_ids = set(hits.loc[hits["index_right"].notna(), "candidate_row_id"].astype(int))
    out["official_disturbance_overlap"] = out["candidate_row_id"].isin(hit_ids)
    out["official_disturbance_distance_m"] = np.inf
    out.loc[out["official_disturbance_overlap"], "official_disturbance_distance_m"] = 0.0

    clean = out[~out["official_disturbance_overlap"]].copy()
    if not clean.empty:
        nearest = gpd.sjoin_nearest(
            clean[["candidate_row_id", "geometry"]],
            official_geom,
            how="left",
            max_distance=max(INTERIOR_DISTANCES_M),
            distance_col="nearest_official_disturbance_m",
        )
        nearest = nearest.dropna(subset=["nearest_official_disturbance_m"])
        if not nearest.empty:
            nearest_min = nearest.groupby("candidate_row_id")["nearest_official_disturbance_m"].min()
            out.loc[out["candidate_row_id"].isin(nearest_min.index), "official_disturbance_distance_m"] = out.loc[
                out["candidate_row_id"].isin(nearest_min.index), "candidate_row_id"
            ].map(nearest_min)

    out["passes_official_disturbance_exclusion"] = ~out["official_disturbance_overlap"]
    out["official_interior_30m"] = out["passes_official_disturbance_exclusion"] & out["official_disturbance_distance_m"].ge(30)
    out["official_interior_60m"] = out["passes_official_disturbance_exclusion"] & out["official_disturbance_distance_m"].ge(60)
    out["stage"] = np.where(out["passes_official_disturbance_exclusion"], "mapped_clean", "candidate")
    return gpd.GeoDataFrame(out.drop(columns=["candidate_row_id"]), geometry="geometry", crs=CRS_PROJECTED)


def keep_nearest_per_site(candidates: gpd.GeoDataFrame, limit: int) -> gpd.GeoDataFrame:
    clean = candidates[candidates["passes_official_disturbance_exclusion"]].copy()
    clean = clean.sort_values(["associated_site_id", "distance_m", "bearing_deg", "candidate_id"])
    return clean.groupby("associated_site_id", group_keys=False).head(limit).copy()


def ee_feature_collection(ee, gdf: gpd.GeoDataFrame):
    features = []
    wgs = gdf.to_crs(CRS_WGS84)
    for _, row in wgs.iterrows():
        props = {
            "candidate_id": row["candidate_id"],
            "associated_site_id": row["associated_site_id"],
            "associated_event_id": row["associated_event_id"],
        }
        features.append(ee.Feature(ee.Geometry.Point([float(row.geometry.x), float(row.geometry.y)]), props))
    return ee.FeatureCollection(features)


def reference_screen_image(ee):
    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    tree = gfc.select("treecover2000").unmask(0)
    loss = gfc.select("lossyear").unmask(0)
    datamask = gfc.select("datamask").unmask(0)
    bands = [
        tree.rename("treecover2000"),
        loss.rename("hansen_lossyear"),
        datamask.rename("hansen_datamask"),
    ]
    for threshold in TREECOVER_THRESHOLDS:
        forest = tree.gte(threshold).And(loss.eq(0)).And(datamask.eq(1)).unmask(0).rename(f"forest_screen_tcc{threshold}")
        bands.append(forest)
    primary = tree.gte(PRIMARY_TREECOVER_THRESHOLD).And(loss.eq(0)).And(datamask.eq(1))
    for interior in INTERIOR_DISTANCES_M:
        bands.append(primary.focal_min(radius=interior, units="meters").unmask(0).rename(f"forest_screen_tcc50_interior_{interior}m"))
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().unmask(-999).rename("elevation_m")
    slope = ee.Terrain.slope(dem).unmask(-999).rename("slope_deg")
    bands.extend([dem, slope])
    return ee.Image.cat(bands)


def sample_ee_candidate_screen(candidates: gpd.GeoDataFrame) -> pd.DataFrame:
    ee = initialize_ee()
    image = reference_screen_image(ee)
    rows = []
    for start in range(0, len(candidates), CHUNK_SIZE):
        chunk = candidates.iloc[start : start + CHUNK_SIZE].copy()
        fc = ee_feature_collection(ee, chunk)
        sampled = image.sampleRegions(collection=fc, scale=30, geometries=False, tileScale=4)
        data = sampled.getInfo().get("features", [])
        for feat in data:
            rows.append(feat["properties"])
        time.sleep(0.05)
    out = pd.DataFrame(rows)
    return out


def sample_ee_sites(sites: gpd.GeoDataFrame) -> pd.DataFrame:
    ee = initialize_ee()
    image = reference_screen_image(ee).select(["elevation_m", "slope_deg"])
    features = []
    wgs = sites.to_crs(CRS_WGS84)
    for _, row in wgs.iterrows():
        features.append(ee.Feature(ee.Geometry.Point([float(row.geometry.x), float(row.geometry.y)]), {"associated_site_id": row["site_id"]}))
    rows = []
    for start in range(0, len(features), CHUNK_SIZE):
        sampled = image.sampleRegions(collection=ee.FeatureCollection(features[start : start + CHUNK_SIZE]), scale=30, geometries=False, tileScale=4)
        rows.extend([feat["properties"] for feat in sampled.getInfo().get("features", [])])
    return pd.DataFrame(rows)


def select_reference_candidates(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    selected_rows = []
    for _, group in candidates[candidates["valid_reference_60m"]].groupby("associated_site_id"):
        group = group.sort_values(["distance_m", "bearing_deg", "candidate_id"])
        selected = []
        for _, cand in group.iterrows():
            if selected:
                distances = [cand.geometry.distance(prev.geometry) for prev in selected]
                if min(distances) < MIN_REFERENCE_SPACING_M:
                    continue
            selected.append(cand)
            if len(selected) >= MAX_SELECTED_REFERENCES_PER_SITE:
                break
        if selected:
            selected_rows.extend(selected)
    if not selected_rows:
        return candidates.iloc[0:0].copy()
    selected = gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=candidates.crs).copy()
    selected["stage"] = "selected_reference"
    return selected


def availability_tables(
    all_candidates: gpd.GeoDataFrame, screened: gpd.GeoDataFrame, selected: gpd.GeoDataFrame, sites: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_counts = all_candidates.groupby("associated_site_id").size().rename("candidate_count")
    mapped_clean_counts = all_candidates[all_candidates["passes_official_disturbance_exclusion"]].groupby("associated_site_id").size().rename("mapped_clean_count")
    forest_counts = screened[screened["forest_screen_tcc50"]].groupby("associated_site_id").size().rename("forest_screened_count")
    valid30 = screened[screened["valid_reference_30m"]].groupby("associated_site_id").size().rename("valid_reference_30m_count")
    valid60 = screened[screened["valid_reference_60m"]].groupby("associated_site_id").size().rename("valid_reference_60m_count")
    selected_counts = selected.groupby("associated_site_id").size().rename("selected_reference_count")
    site_cols = [
        "disturbance_type",
        "event_id",
        "site_id",
        "disturbance_year",
        "years_since_disturbance",
        "age_bin",
    ]
    availability = sites[site_cols].copy().rename(columns={"site_id": "associated_site_id"})
    for series in [candidate_counts, mapped_clean_counts, forest_counts, valid30, valid60, selected_counts]:
        availability = availability.merge(series, left_on="associated_site_id", right_index=True, how="left")
    count_cols = [c for c in availability.columns if c.endswith("_count") or c == "candidate_count"]
    availability[count_cols] = availability[count_cols].fillna(0).astype(int)
    for n in [1, 3, 5, 10]:
        availability[f"has_ge_{n}_selected_candidates"] = availability["selected_reference_count"] >= n
    availability["has_zero_selected_candidates"] = availability["selected_reference_count"] == 0

    event = availability.groupby(["disturbance_type", "event_id", "disturbance_year", "age_bin"], as_index=False).agg(
        disturbed_site_count=("associated_site_id", "nunique"),
        sites_ge1=("has_ge_1_selected_candidates", "sum"),
        sites_ge3=("has_ge_3_selected_candidates", "sum"),
        sites_ge5=("has_ge_5_selected_candidates", "sum"),
        sites_ge10=("has_ge_10_selected_candidates", "sum"),
        zero_candidate_sites=("has_zero_selected_candidates", "sum"),
        total_selected_candidates=("selected_reference_count", "sum"),
    )
    event["event_reference_status"] = np.select(
        [
            event["zero_candidate_sites"].eq(0),
            event["zero_candidate_sites"].eq(event["disturbed_site_count"]),
        ],
        ["all_sites_have_valid_candidates", "no_viable_reference_candidates"],
        default="partial_valid_candidates",
    )

    filter_rows = []
    total = len(all_candidates)
    official_pass = int(all_candidates["passes_official_disturbance_exclusion"].sum())
    sampled = len(screened)
    hansen_loss_free = int((screened["hansen_lossyear"].fillna(99).eq(0)).sum())
    forest50 = int(screened["forest_screen_tcc50"].sum())
    valid60_count = int(screened["valid_reference_60m"].sum())
    stages = [
        ("candidate", total, np.nan),
        ("mapped_clean", official_pass, total - official_pass),
        ("hansen_loss_excluded", hansen_loss_free, sampled - hansen_loss_free),
        ("forest_screened_tcc50", forest50, sampled - forest50),
        ("selected_reference_support_60m", valid60_count, sampled - valid60_count),
    ]
    for stage, retained, removed in stages:
        filter_rows.append(
            {
                "stage": stage,
                "candidate_count_retained": int(retained),
                "candidate_count_removed_at_stage": None if pd.isna(removed) else int(removed),
                "approx_candidate_support_area_ha_retained": float(retained * CELL_AREA_HA),
            }
        )
    filter_accounting = pd.DataFrame(filter_rows)

    boundary_rows = []
    for threshold in TREECOVER_THRESHOLDS:
        col = f"forest_screen_tcc{threshold}"
        boundary_rows.append(
            {
                "screen": f"treecover2000>={threshold}, Hansen lossyear=0, datamask=land",
                "candidate_count": int(screened[col].sum()) if col in screened else 0,
                "site_count_ge1": int(screened[screened[col]].groupby("associated_site_id").size().ge(1).sum()) if col in screened else 0,
            }
        )
    for interior in INTERIOR_DISTANCES_M:
        col = f"valid_reference_{interior}m"
        boundary_rows.append(
            {
                "screen": f"primary forest screen plus >= {interior} m interior support",
                "candidate_count": int(screened[col].sum()),
                "site_count_ge1": int(screened[screened[col]].groupby("associated_site_id").size().ge(1).sum()),
            }
        )
    boundary = pd.DataFrame(boundary_rows)
    return availability, event, filter_accounting, boundary


def enrich_environmental_matching(
    screened: gpd.GeoDataFrame, sites: gpd.GeoDataFrame, site_env: pd.DataFrame
) -> gpd.GeoDataFrame:
    site_env = site_env.rename(columns={"associated_site_id": "associated_site_id", "elevation_m": "site_elevation_m", "slope_deg": "site_slope_deg"})
    out = screened.merge(site_env, on="associated_site_id", how="left")
    out["elevation_difference_m"] = out["elevation_m"] - out["site_elevation_m"]
    out["slope_difference_deg"] = out["slope_deg"] - out["site_slope_deg"]
    transformer = Transformer.from_crs(CRS_PROJECTED, CRS_WGS84, always_xy=True)
    lon, lat = transformer.transform(out.geometry.x.to_numpy(), out.geometry.y.to_numpy())
    out["longitude"] = lon
    out["latitude"] = lat
    return out


def landsat_sensor_diagnostic() -> pd.DataFrame:
    site_year = pd.read_csv(LANDSAT_SITE_YEAR)
    site_year["data_available"] = site_year["data_available"].astype(bool)
    valid = site_year[site_year["data_available"] & ~site_year["post_censor"].astype(bool)].copy()
    annual = valid[valid["observation_year"].isin([2021, 2022])].copy()
    rows = []
    for year, group in annual.groupby("observation_year"):
        rows.append(
            {
                "diagnostic": "annual_all_sites",
                "observation_year": int(year),
                "site_count": int(group["site_id"].nunique()),
                "median_nbr": float(group["nbr"].median()),
                "mean_nbr": float(group["nbr"].mean()),
                "lc08_site_years": int((group["count_LC08"] > 0).sum()),
                "lc09_site_years": int((group["count_LC09"] > 0).sum()),
                "lc08_observations": int(group["count_LC08"].sum()),
                "lc09_observations": int(group["count_LC09"].sum()),
                "note": "Merged seasonal Landsat composite; sensor counts available, sensor-specific NBR not retained.",
            }
        )
    paired = annual.pivot_table(index=["disturbance_type", "event_id", "site_id"], columns="observation_year", values="nbr", aggfunc="first")
    paired = paired.dropna(subset=[2021, 2022])
    if not paired.empty:
        delta = paired[2022] - paired[2021]
        rows.append(
            {
                "diagnostic": "same_site_2022_minus_2021",
                "observation_year": 2022,
                "site_count": int(len(delta)),
                "median_nbr": float(delta.median()),
                "mean_nbr": float(delta.mean()),
                "lc08_site_years": np.nan,
                "lc09_site_years": np.nan,
                "lc08_observations": np.nan,
                "lc09_observations": np.nan,
                "note": "Same-site paired change tests whether the step is explained by site composition.",
            }
        )
    for label, mask in [
        ("2022_lc08_and_lc09", (annual["observation_year"].eq(2022) & annual["count_LC08"].gt(0) & annual["count_LC09"].gt(0))),
        ("2022_lc08_only", (annual["observation_year"].eq(2022) & annual["count_LC08"].gt(0) & annual["count_LC09"].eq(0))),
        ("2022_lc09_only", (annual["observation_year"].eq(2022) & annual["count_LC08"].eq(0) & annual["count_LC09"].gt(0))),
    ]:
        group = annual[mask]
        if group.empty:
            continue
        rows.append(
            {
                "diagnostic": label,
                "observation_year": 2022,
                "site_count": int(group["site_id"].nunique()),
                "median_nbr": float(group["nbr"].median()),
                "mean_nbr": float(group["nbr"].mean()),
                "lc08_site_years": int((group["count_LC08"] > 0).sum()),
                "lc09_site_years": int((group["count_LC09"] > 0).sum()),
                "lc08_observations": int(group["count_LC08"].sum()),
                "lc09_observations": int(group["count_LC09"].sum()),
                "note": "Sensor-presence association within the merged 2022 composite.",
            }
        )
    return pd.DataFrame(rows)


def pre_disturbance_support(sites: gpd.GeoDataFrame) -> pd.DataFrame:
    event_year = pd.read_csv(LANDSAT_EVENT_YEAR)
    pre = event_year[event_year["years_since_disturbance"].isin([-3, -2, -1])].copy()
    all_events = (
        sites[["disturbance_type", "event_id", "disturbance_year"]]
        .drop_duplicates()
        .sort_values(["disturbance_type", "disturbance_year", "event_id"])
    )
    rows = []
    grouped = {
        keys: group
        for keys, group in pre.groupby(["disturbance_type", "event_id", "disturbance_year"])
    }
    for _, event in all_events.iterrows():
        keys = (event["disturbance_type"], event["event_id"], event["disturbance_year"])
        group = grouped.get(keys, pre.iloc[0:0])
        row = {
            "disturbance_type": event["disturbance_type"],
            "event_id": event["event_id"],
            "disturbance_year": int(event["disturbance_year"]),
        }
        usable = 0
        for rel in [-3, -2, -1]:
            hit = group[group["years_since_disturbance"].eq(rel)]
            has = (not hit.empty) and hit["nbr_median"].notna().any() and hit["valid_site_count"].fillna(0).max() > 0
            row[f"has_pre_year_{rel}"] = bool(has)
            row[f"valid_site_count_pre_year_{rel}"] = int(hit["valid_site_count"].fillna(0).max()) if not hit.empty else 0
            usable += int(has)
        row["usable_pre_year_count"] = usable
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["disturbance_type", "disturbance_year", "event_id"])


def make_figures(
    availability: pd.DataFrame,
    selected: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    filter_accounting: pd.DataFrame,
    pre_support: pd.DataFrame,
    sensor_diag: pd.DataFrame,
) -> None:
    plt.style.use("default")
    colors = {"fire_total": "#c95f3f", "harvest_total": "#2f7f73"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, MAX_SELECTED_REFERENCES_PER_SITE + 2) - 0.5
    for dtype, group in availability.groupby("disturbance_type"):
        ax.hist(group["selected_reference_count"], bins=bins, alpha=0.72, label=dtype.replace("_", " "), color=colors.get(dtype))
    ax.set_xlabel("selected candidate reference points per disturbed site")
    ax.set_ylabel("disturbed sites")
    ax.legend(frameon=False)
    ax.set_title("Reference candidate availability per disturbed site")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reference_candidate_availability_per_site.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    sites.plot(ax=ax, markersize=10, color=sites["disturbance_type"].map(colors), alpha=0.8)
    if not selected.empty:
        selected.plot(ax=ax, markersize=4, color="#222222", alpha=0.45)
    ax.set_axis_off()
    ax.set_title("Disturbed sites and selected candidate reference points")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "disturbed_sites_and_reference_candidates_map.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if not selected.empty:
        ax.hist(selected["distance_m"], bins=24, color="#577590", edgecolor="white")
    ax.set_xlabel("distance from disturbed site to reference candidate (m)")
    ax.set_ylabel("candidate references")
    ax.set_title("Reference distance distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reference_distance_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = filter_accounting.copy()
    ax.barh(plot_df["stage"], plot_df["candidate_count_retained"], color="#526760")
    ax.set_xlabel("candidate points retained")
    ax.set_title("Candidate accounting by filter stage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "candidate_loss_accounting_by_filter.png", dpi=180)
    plt.close(fig)

    age = availability.groupby(["age_bin", "disturbance_type"], as_index=False).agg(
        site_count=("associated_site_id", "count"),
        sites_ge1=("has_ge_1_selected_candidates", "sum"),
    )
    age["availability_pct"] = 100 * age["sites_ge1"] / age["site_count"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for dtype, group in age.groupby("disturbance_type"):
        group = group.sort_values("age_bin")
        ax.plot(group["age_bin"], group["availability_pct"], marker="o", label=dtype.replace("_", " "), color=colors.get(dtype))
    ax.set_ylim(0, 105)
    ax.set_ylabel("sites with >=1 selected reference (%)")
    ax.set_xlabel("2024 age bin")
    ax.legend(frameon=False)
    ax.set_title("Reference availability by age bin")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reference_availability_by_age_bin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = pre_support.groupby(["disturbance_type", "usable_pre_year_count"]).size().reset_index(name="events")
    x = np.arange(4)
    width = 0.36
    for i, dtype in enumerate(["fire_total", "harvest_total"]):
        vals = counts[counts["disturbance_type"].eq(dtype)].set_index("usable_pre_year_count")["events"].reindex(x, fill_value=0)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=dtype.replace("_", " "), color=colors[dtype])
    ax.set_xticks(x)
    ax.set_xlabel("usable pre-disturbance years among -3, -2, -1")
    ax.set_ylabel("events")
    ax.legend(frameon=False)
    ax.set_title("Event-level pre-disturbance Landsat support")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_level_pre_disturbance_support.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    annual = sensor_diag[sensor_diag["diagnostic"].eq("annual_all_sites")]
    ax.plot(annual["observation_year"], annual["median_nbr"], marker="o", color="#143d36", label="all valid site-years")
    same = sensor_diag[sensor_diag["diagnostic"].eq("same_site_2022_minus_2021")]
    if not same.empty:
        text = f"same-site median 2022-2021: {same.iloc[0]['median_nbr']:+.3f}"
        ax.text(0.03, 0.08, text, transform=ax.transAxes, fontsize=9)
    ax.set_xticks([2021, 2022])
    ax.set_ylabel("median NBR")
    ax.set_title("Landsat 2021-2022 merged-composite diagnostic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landsat_2021_2022_sensor_diagnostic.png", dpi=180)
    plt.close(fig)


def write_readme(summary: dict) -> None:
    readme = f"""# Reference-forest validation

This folder validates the candidate reference-forest sampling frame for the
Lebel-sur-Quevillon recovery design. It does not extract full reference Landsat
trajectories and does not estimate recovery gaps.

## Definitions

- `candidate`: deterministic points generated inside the existing 500-3000 m
  reference-search annuli with the associated disturbed event geometry already
  removed by the sampling-frame stage.
- `mapped_clean`: candidates outside existing official mapped wildfire,
  harvest-event, and forestry-intervention geometries.
- `forest_screened`: candidates passing the Hansen Global Forest Change
  independent screen: `treecover2000 >= 50`, `lossyear == 0`, and land
  `datamask == 1`.
- `selected_reference`: up to 10 spatially separated candidate reference
  points per disturbed site passing the 60 m interior version of the forest
  screen.

Candidate references are not interpreted as undisturbed forest until they pass
these mapped-disturbance and forest-condition filters. Absence of mapped
disturbance is only interpreted within the temporal and thematic coverage of
the official records available here.

## Official disturbance exclusions

- NBAC/MRNF fire-event geometries already retained in
  `outputs/event_histories/disturbance_events.gpkg`.
- Mapped harvest events retained in
  `outputs/event_histories/disturbance_events.gpkg`.
- All mapped forestry intervention polygons in
  `data/raw/forestry_interventions/interv_fores_lebel_100km_circle.gpkg`.

Any mapped forestry intervention is treated as a reference exclusion. Hansen
loss is used only as an independent disturbance screen, not as fire/harvest
attribution.

## Summary

- Disturbed sites with at least one selected reference candidate:
  {summary['sites_with_reference']} of {summary['disturbed_site_count']}
  ({summary['sites_with_reference_pct']:.1f}%).
- Events with all disturbed sites supported:
  {summary['events_all_sites_supported']} of {summary['event_count']}.
- Recommended interior distance: {summary['recommended_interior_m']} m.
- 2021-2022 Landsat diagnostic: {summary['landsat_2021_2022_conclusion']}

"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Reading local sampling frame and official disturbance layers...")
    ref, sites, official, forestry = read_inputs()
    print(f"Reference annuli: {len(ref):,}; disturbed sites: {len(sites):,}; official exclusion polygons: {len(official):,}")

    print("Generating deterministic candidate reference points...")
    all_candidates = generate_raw_candidates(ref, sites)
    print(f"Raw candidates: {len(all_candidates):,}")

    print("Applying official mapped-disturbance exclusion...")
    all_candidates = official_filter(all_candidates, official)
    gee_candidates = keep_nearest_per_site(all_candidates, MAX_GEE_CANDIDATES_PER_SITE)
    print(f"Candidates sent to GEE forest/Hansen screening: {len(gee_candidates):,}")

    print("Sampling Hansen forest/loss screens and DEM from Earth Engine...")
    ee_screen = sample_ee_candidate_screen(gee_candidates)
    site_env = sample_ee_sites(sites)
    screened = gee_candidates.merge(ee_screen, on=["candidate_id", "associated_site_id", "associated_event_id"], how="left")
    for col in ["hansen_lossyear", "treecover2000", "hansen_datamask", "elevation_m", "slope_deg"]:
        screened[col] = pd.to_numeric(screened[col], errors="coerce")
    for col in [f"forest_screen_tcc{t}" for t in TREECOVER_THRESHOLDS] + [f"forest_screen_tcc50_interior_{m}m" for m in INTERIOR_DISTANCES_M]:
        screened[col] = pd.to_numeric(screened[col], errors="coerce").fillna(0).astype(int).astype(bool)
    screened["hansen_loss_present"] = screened["hansen_lossyear"].fillna(0).gt(0)
    screened["valid_reference_30m"] = screened["official_interior_30m"] & screened["forest_screen_tcc50_interior_30m"]
    screened["valid_reference_60m"] = screened["official_interior_60m"] & screened["forest_screen_tcc50_interior_60m"]
    screened["stage"] = np.where(screened["forest_screen_tcc50"], "forest_screened", "mapped_clean")
    screened = enrich_environmental_matching(gpd.GeoDataFrame(screened, geometry="geometry", crs=CRS_PROJECTED), sites, site_env)

    print("Selecting up to 10 reference candidates per disturbed site...")
    selected = select_reference_candidates(screened)
    availability, event_coverage, filter_accounting, boundary = availability_tables(all_candidates, screened, selected, sites)

    print("Running Landsat pre-disturbance and 2021-2022 diagnostics from existing tables...")
    pre_support = pre_disturbance_support(sites)
    sensor_diag = landsat_sensor_diagnostic()

    print("Writing outputs and figures...")
    availability.to_csv(OUT_DIR / "reference_candidate_availability.csv", index=False)
    event_coverage.to_csv(OUT_DIR / "reference_event_coverage.csv", index=False)
    filter_accounting.to_csv(OUT_DIR / "reference_filter_accounting.csv", index=False)
    boundary.to_csv(OUT_DIR / "reference_boundary_sensitivity.csv", index=False)
    pre_support.to_csv(OUT_DIR / "pre_disturbance_event_support.csv", index=False)
    sensor_diag.to_csv(OUT_DIR / "landsat_2021_2022_sensor_diagnostic.csv", index=False)
    candidate_layers = pd.concat([screened, selected], ignore_index=True)
    candidate_layers = gpd.GeoDataFrame(candidate_layers, geometry="geometry", crs=CRS_PROJECTED)
    gpkg_path = OUT_DIR / "reference_candidates.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    candidate_layers.to_file(gpkg_path, layer="reference_candidates", driver="GPKG")

    make_figures(availability, selected, sites, filter_accounting, pre_support, sensor_diag)

    sites_with_reference = int(availability["has_ge_1_selected_candidates"].sum())
    events_all = int(event_coverage["event_reference_status"].eq("all_sites_have_valid_candidates").sum())
    same_site = sensor_diag[sensor_diag["diagnostic"].eq("same_site_2022_minus_2021")]
    annual = sensor_diag[sensor_diag["diagnostic"].eq("annual_all_sites")].set_index("observation_year")
    annual_step = float(annual.loc[2022, "median_nbr"] - annual.loc[2021, "median_nbr"]) if {2021, 2022}.issubset(annual.index) else None
    same_site_step = float(same_site.iloc[0]["median_nbr"]) if not same_site.empty else None
    if same_site_step is not None and annual_step is not None and abs(same_site_step - annual_step) < 0.015:
        landsat_conclusion = (
            "the 2021-2022 NBR step persists in same-site comparisons, so site composition alone does not explain it; "
            "the retained tables cannot isolate LC08-vs-LC09 spectral offsets because sensor-specific NBR was not stored."
        )
    else:
        landsat_conclusion = (
            "the existing merged-composite tables show a 2021-2022 step, but sensor-specific NBR was not retained, "
            "so a correction should not be estimated from these outputs alone."
        )
    summary = {
        "disturbed_site_count": int(len(availability)),
        "sites_with_reference": sites_with_reference,
        "sites_with_reference_pct": float(100 * sites_with_reference / len(availability)),
        "event_count": int(len(event_coverage)),
        "events_all_sites_supported": events_all,
        "recommended_interior_m": PRIMARY_INTERIOR_M,
        "forest_screen": {
            "source": "UMD/Hansen Global Forest Change 2025 v1.13",
            "threshold": "treecover2000 >= 50, lossyear == 0, datamask == 1",
            "role": "independent disturbance and forest-condition screen; not fire/harvest attribution",
        },
        "official_disturbance_exclusion": [
            "outputs/event_histories/disturbance_events.gpkg: fire_events",
            "outputs/event_histories/disturbance_events.gpkg: harvest_events",
            "data/raw/forestry_interventions/interv_fores_lebel_100km_circle.gpkg: all mapped forestry interventions",
        ],
        "landsat_annual_median_step_2022_minus_2021": annual_step,
        "landsat_same_site_median_step_2022_minus_2021": same_site_step,
        "landsat_2021_2022_conclusion": landsat_conclusion,
    }
    (OUT_DIR / "reference_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
