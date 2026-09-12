#!/usr/bin/env python3
"""Select final reference sites and extract reference Landsat time series.

This stage builds reference-relative observational tables only. It does not fit
recovery trajectories, extract Sentinel-2, extract AlphaEarth, or estimate
causal treatment effects.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "reference_timeseries"
FIG_DIR = ROOT / "figures" / "reference_timeseries"
MPL_DIR = OUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt


REFERENCE_CANDIDATES = ROOT / "outputs" / "reference_forest" / "reference_candidates.gpkg"
REFERENCE_EVENT_COVERAGE = ROOT / "outputs" / "reference_forest" / "reference_event_coverage.csv"
FIRE_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_fire_sites.gpkg"
HARVEST_SITES = ROOT / "outputs" / "sampling_frame" / "candidate_harvest_sites.gpkg"
SELECTED_FIRE_EVENTS = ROOT / "outputs" / "sampling_frame" / "selected_fire_events.csv"
SELECTED_HARVEST_EVENTS = ROOT / "outputs" / "sampling_frame" / "selected_harvest_events.csv"
FIRE_HARVEST_PAIRS = ROOT / "outputs" / "sampling_frame" / "fire_harvest_candidate_pairs.csv"
DISTURBED_SITE_YEAR = ROOT / "outputs" / "landsat_timeseries" / "landsat_site_year.csv"
SCRIPT_06 = ROOT / "analysis" / "06_extract_landsat_recovery_timeseries.py"
SCRIPT_07 = ROOT / "analysis" / "07_validate_reference_forest.py"

CRS_PROJECTED = "EPSG:32198"
CRS_WGS84 = "EPSG:4326"
OBSERVATION_END_YEAR = 2025
PRIMARY_INTERIOR_M = 60
RESCUE_OUTER_RADIUS_M = 5000
RESCUE_RINGS_M = list(range(3200, RESCUE_OUTER_RADIUS_M + 1, 200))
RESCUE_BEARINGS_DEG = list(range(0, 360, 15))
MAX_GEE_CANDIDATES_PER_SITE = 90
MAX_SELECTED_PER_SITE = 1
MIN_UNIQUE_REFERENCE_SPACING_M = 1.0
SPECTRAL_FIELDS = ["blue", "green", "red", "nir", "swir1", "swir2", "nbr", "ndvi"]
COUNT_FIELDS = ["count_LT05", "count_LE07", "count_LC08", "count_LC09", "valid_observation_count"]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    spec.loader.exec_module(module)
    return module


def read_sites() -> gpd.GeoDataFrame:
    fire = gpd.read_file(FIRE_SITES, layer="candidate_fire_sites", engine="pyogrio").to_crs(CRS_PROJECTED)
    harvest = gpd.read_file(HARVEST_SITES, layer="candidate_harvest_sites", engine="pyogrio").to_crs(CRS_PROJECTED)
    sites = pd.concat([fire, harvest], ignore_index=True)
    sites = gpd.GeoDataFrame(sites, geometry="geometry", crs=CRS_PROJECTED)
    return sites


def read_valid_candidates() -> gpd.GeoDataFrame:
    candidates = gpd.read_file(REFERENCE_CANDIDATES, engine="pyogrio").to_crs(CRS_PROJECTED)
    candidates = candidates.drop_duplicates("candidate_id").copy()
    candidates = candidates[candidates["valid_reference_60m"].astype(bool)].copy()
    candidates["search_radius_max_m"] = 3000
    candidates["rescue_candidate"] = False
    return candidates


def generate_rescue_candidates(
    zero_event_sites: gpd.GeoDataFrame, existing_candidates: gpd.GeoDataFrame, official: gpd.GeoDataFrame, module07
) -> gpd.GeoDataFrame:
    if zero_event_sites.empty:
        return existing_candidates.iloc[0:0].copy()
    rows = []
    for _, site in zero_event_sites.iterrows():
        sx, sy = float(site.geometry.x), float(site.geometry.y)
        candidate_number = 0
        for radius in RESCUE_RINGS_M:
            for bearing in RESCUE_BEARINGS_DEG:
                angle = math.radians(bearing)
                point = Point(sx + radius * math.cos(angle), sy + radius * math.sin(angle))
                candidate_number += 1
                rows.append(
                    {
                        "candidate_id": f"{site['site_id']}_rescue_refcand_{candidate_number:04d}",
                        "reference_search_id": f"rescue_500_5000_{site['site_id']}",
                        "associated_site_id": site["site_id"],
                        "associated_event_id": site["event_id"],
                        "disturbance_type": site["disturbance_type"],
                        "disturbance_year": int(site["disturbance_year"]),
                        "years_since_disturbance": int(site["years_since_disturbance"]),
                        "age_bin": site["age_bin"],
                        "distance_m": float(radius),
                        "bearing_deg": int(bearing),
                        "stage": "candidate",
                        "search_radius_max_m": RESCUE_OUTER_RADIUS_M,
                        "rescue_candidate": True,
                        "geometry": point,
                    }
                )
    rescue = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_PROJECTED)
    if rescue.empty:
        return rescue
    print(f"Applying official exclusion to {len(rescue):,} rescue candidates...", flush=True)
    rescue = module07.official_filter(rescue, official)
    rescue = rescue[rescue["passes_official_disturbance_exclusion"]].copy()
    rescue = rescue.sort_values(["associated_site_id", "distance_m", "bearing_deg"])
    rescue = rescue.groupby("associated_site_id", group_keys=False).head(MAX_GEE_CANDIDATES_PER_SITE).copy()
    if rescue.empty:
        return rescue
    print(f"Sampling Hansen/DEM for {len(rescue):,} rescue candidates...", flush=True)
    ee_screen = module07.sample_ee_candidate_screen(rescue)
    rescue = rescue.merge(ee_screen, on=["candidate_id", "associated_site_id", "associated_event_id"], how="left")
    for col in ["hansen_lossyear", "treecover2000", "hansen_datamask", "elevation_m", "slope_deg"]:
        rescue[col] = pd.to_numeric(rescue[col], errors="coerce")
    bool_cols = [
        "forest_screen_tcc30",
        "forest_screen_tcc50",
        "forest_screen_tcc70",
        "forest_screen_tcc50_interior_30m",
        "forest_screen_tcc50_interior_60m",
    ]
    for col in bool_cols:
        rescue[col] = pd.to_numeric(rescue[col], errors="coerce").fillna(0).astype(int).astype(bool)
    site_env = module07.sample_ee_sites(zero_event_sites)
    rescue = module07.enrich_environmental_matching(gpd.GeoDataFrame(rescue, geometry="geometry", crs=CRS_PROJECTED), zero_event_sites, site_env)
    rescue["hansen_loss_present"] = rescue["hansen_lossyear"].fillna(0).gt(0)
    rescue["valid_reference_30m"] = rescue["official_interior_30m"] & rescue["forest_screen_tcc50_interior_30m"]
    rescue["valid_reference_60m"] = rescue["official_interior_60m"] & rescue["forest_screen_tcc50_interior_60m"]
    rescue["stage"] = np.where(rescue["valid_reference_60m"], "forest_screened", "mapped_clean")
    rescue["search_radius_max_m"] = RESCUE_OUTER_RADIUS_M
    rescue["rescue_candidate"] = True
    return rescue[rescue["valid_reference_60m"]].copy()


def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    scale = values.mad() if hasattr(values, "mad") else (values - values.median()).abs().median()
    if not np.isfinite(scale) or scale == 0:
        scale = values.std()
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (values - values.median()).abs() / scale


def select_final_references(candidates: gpd.GeoDataFrame, sites: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["abs_elevation_difference_m"] = candidates["elevation_difference_m"].abs()
    candidates["abs_slope_difference_deg"] = candidates["slope_difference_deg"].abs()
    candidates["elevation_z"] = robust_z(candidates["abs_elevation_difference_m"])
    candidates["slope_z"] = robust_z(candidates["abs_slope_difference_deg"])
    candidates["environmental_score"] = candidates["elevation_z"] + candidates["slope_z"]
    candidates["rounded_reference_xy"] = (
        candidates.geometry.x.round(3).astype(str) + "_" + candidates.geometry.y.round(3).astype(str)
    )

    nearest = (
        candidates.sort_values(["associated_site_id", "distance_m", "environmental_score", "candidate_id"])
        .groupby("associated_site_id", as_index=False)
        .head(1)[["associated_site_id", "candidate_id"]]
        .rename(columns={"candidate_id": "nearest_candidate_id"})
    )

    selected_rows = []
    used_xy: set[str] = set()
    for site_id, group in candidates.sort_values(
        ["associated_site_id", "environmental_score", "distance_m", "candidate_id"]
    ).groupby("associated_site_id"):
        choice = None
        for _, cand in group.iterrows():
            xy = cand["rounded_reference_xy"]
            if xy not in used_xy:
                choice = cand
                used_xy.add(xy)
                break
        if choice is None:
            choice = group.iloc[0]
        selected_rows.append(choice)

    final = gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=candidates.crs).copy()
    final = final.reset_index(drop=True)
    final["reference_id"] = [f"reference_{i + 1:04d}" for i in range(len(final))]
    final["pair_id"] = [f"site_ref_pair_{i + 1:04d}" for i in range(len(final))]
    final["selection_method"] = "environmental_score_then_distance_unique_where_feasible"
    final["stage"] = "selected_reference"
    final["event_id"] = final["associated_event_id"]
    final["reference_censor_year"] = pd.NA
    final["reference_censor_reason"] = ""
    final = final.merge(nearest, on="associated_site_id", how="left")
    final["same_as_nearest_geographic"] = final["candidate_id"].eq(final["nearest_candidate_id"])
    final["reference_reuse_count"] = final.groupby("rounded_reference_xy")["rounded_reference_xy"].transform("count")

    site_meta = sites.drop(columns="geometry").rename(columns={"site_id": "associated_site_id"})
    final = final.merge(site_meta, on=["associated_site_id", "event_id", "disturbance_type", "disturbance_year", "years_since_disturbance", "age_bin"], how="left")

    audit_cols = [
        "pair_id",
        "disturbance_type",
        "event_id",
        "associated_site_id",
        "reference_id",
        "candidate_id",
        "selection_method",
        "environmental_score",
        "abs_elevation_difference_m",
        "abs_slope_difference_deg",
        "distance_m",
        "same_as_nearest_geographic",
        "search_radius_max_m",
        "rescue_candidate",
    ]
    selection_audit = final[audit_cols].copy()
    reuse = final.groupby("rounded_reference_xy", as_index=False).agg(
        reference_reuse_count=("reference_id", "count"),
        reference_ids=("reference_id", lambda s: ";".join(s)),
        site_ids=("associated_site_id", lambda s: ";".join(s)),
        x=("geometry", lambda s: float(s.iloc[0].x)),
        y=("geometry", lambda s: float(s.iloc[0].y)),
    )
    return final, selection_audit, reuse


def reference_sites_for_landsat(final_refs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for _, ref in final_refs.iterrows():
        rows.append(
            {
                "disturbance_type": ref["disturbance_type"],
                "event_id": ref["event_id"],
                "site_id": ref["reference_id"],
                "reference_id": ref["reference_id"],
                "paired_disturbed_site_id": ref["associated_site_id"],
                "pair_id": ref["pair_id"],
                "disturbance_year": int(ref["disturbance_year"]),
                "associated_disturbance_year": int(ref["disturbance_year"]),
                "years_since_disturbance": int(ref["years_since_disturbance"]),
                "age_bin": ref["age_bin"],
                "censor_year": pd.NA,
                "censor_reason": "",
                "reference_censor_year": pd.NA,
                "reference_censor_reason": "",
                "geometry": ref.geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=final_refs.crs).to_crs(CRS_WGS84)


def extract_reference_landsat(reference_sites: gpd.GeoDataFrame) -> pd.DataFrame:
    module06 = load_module(SCRIPT_06, "landsat_timeseries")
    module06.ensure_dirs = lambda: None
    module06.OUT_DIR = OUT_DIR
    module06.FIG_DIR = FIG_DIR
    module06.initialize_ee()
    site_year = module06.build_site_year_table(reference_sites)
    site_year = site_year.rename(columns={"disturbance_year": "associated_disturbance_year"})
    site_year["reference_id"] = site_year["site_id"]
    meta = reference_sites.drop(columns="geometry")
    site_year = site_year.merge(
        meta[["reference_id", "paired_disturbed_site_id", "pair_id", "reference_censor_year", "reference_censor_reason"]],
        on="reference_id",
        how="left",
    )
    return site_year


def build_disturbed_reference_site_year(reference_site_year: pd.DataFrame, final_refs: gpd.GeoDataFrame) -> pd.DataFrame:
    disturbed = pd.read_csv(DISTURBED_SITE_YEAR)
    meta = final_refs[
        [
            "pair_id",
            "disturbance_type",
            "event_id",
            "associated_site_id",
            "reference_id",
            "reference_censor_year",
            "reference_censor_reason",
        ]
    ].rename(columns={"associated_site_id": "site_id"})
    disturbed = disturbed.merge(meta, on=["disturbance_type", "event_id", "site_id"], how="inner")

    ref = reference_site_year.copy()
    ref = ref.rename(columns={"associated_disturbance_year": "disturbance_year"})
    join_keys = ["pair_id", "reference_id", "observation_year", "years_since_disturbance"]
    ref_keep = join_keys + SPECTRAL_FIELDS + COUNT_FIELDS + ["sensor_information", "data_available", "reference_censor_year", "reference_censor_reason"]
    ref = ref[ref_keep].copy()
    rename_ref = {field: f"ref_{field}" for field in SPECTRAL_FIELDS + COUNT_FIELDS}
    rename_ref.update({"sensor_information": "ref_sensor_information", "data_available": "ref_data_available"})
    ref = ref.rename(columns=rename_ref)

    dist_keep = [
        "pair_id",
        "disturbance_type",
        "event_id",
        "site_id",
        "reference_id",
        "disturbance_year",
        "observation_year",
        "years_since_disturbance",
        "censor_year",
        "censor_reason",
        "post_censor",
    ] + SPECTRAL_FIELDS + COUNT_FIELDS + ["sensor_information", "data_available"]
    dist = disturbed[dist_keep].copy()
    rename_dist = {field: f"dist_{field}" for field in SPECTRAL_FIELDS + COUNT_FIELDS}
    rename_dist.update({"sensor_information": "dist_sensor_information", "data_available": "dist_data_available"})
    dist = dist.rename(columns=rename_dist)

    paired = dist.merge(ref, on=join_keys, how="inner")
    paired["reference_post_censor"] = False
    paired["valid_disturbed_reference_pair"] = (
        paired["dist_data_available"].astype(bool)
        & paired["ref_data_available"].astype(bool)
        & ~paired["post_censor"].astype(bool)
        & ~paired["reference_post_censor"].astype(bool)
    )
    for field in SPECTRAL_FIELDS:
        paired[f"{field}_gap"] = paired[f"ref_{field}"] - paired[f"dist_{field}"]
    return paired.sort_values(["disturbance_type", "event_id", "site_id", "observation_year"])


def aggregate_event_year(paired: pd.DataFrame) -> pd.DataFrame:
    valid = paired[paired["valid_disturbed_reference_pair"]].copy()
    agg_spec = {
        "supported_site_reference_pairs": ("pair_id", "nunique"),
    }
    for field in SPECTRAL_FIELDS:
        agg_spec[f"{field}_gap_median"] = (f"{field}_gap", "median")
        agg_spec[f"{field}_gap_mean"] = (f"{field}_gap", "mean")
        agg_spec[f"{field}_gap_sd"] = (f"{field}_gap", "std")
        agg_spec[f"dist_{field}_median"] = (f"dist_{field}", "median")
        agg_spec[f"ref_{field}_median"] = (f"ref_{field}", "median")
    out = valid.groupby(
        ["disturbance_type", "event_id", "disturbance_year", "observation_year", "years_since_disturbance"],
        as_index=False,
    ).agg(**agg_spec)
    return out.sort_values(["disturbance_type", "event_id", "observation_year"])


def temporal_period(years_since: pd.Series) -> pd.Series:
    return pd.cut(
        years_since,
        bins=[-999, -1, 0, 5, 10, 15, 20, 999],
        labels=["pre_disturbance", "year_0", "years_1_5", "years_6_10", "years_11_15", "years_16_20", "years_21plus"],
    )


def reference_temporal_coverage(paired: pd.DataFrame, event_year: pd.DataFrame) -> pd.DataFrame:
    expected = paired.drop_duplicates(["disturbance_type", "event_id", "observation_year", "years_since_disturbance"]).copy()
    expected["period"] = temporal_period(expected["years_since_disturbance"])
    observed = event_year.copy()
    observed["period"] = temporal_period(observed["years_since_disturbance"])
    exp = expected.groupby(["disturbance_type", "period"], observed=False, as_index=False).agg(
        expected_supported_event_years=("event_id", "count")
    )
    obs = observed.groupby(["disturbance_type", "period"], observed=False, as_index=False).agg(
        observed_disturbed_reference_event_years=("event_id", "count")
    )
    out = exp.merge(obs, on=["disturbance_type", "period"], how="left")
    out["observed_disturbed_reference_event_years"] = out["observed_disturbed_reference_event_years"].fillna(0).astype(int)
    out["coverage_percent"] = 100 * out["observed_disturbed_reference_event_years"] / out["expected_supported_event_years"]
    return out


def reference_pair_completeness(final_refs: gpd.GeoDataFrame, event_year: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(FIRE_HARVEST_PAIRS)
    selected_fire = set(pd.read_csv(SELECTED_FIRE_EVENTS)["event_id"].astype(str))
    selected_harvest = set(pd.read_csv(SELECTED_HARVEST_EVENTS)["event_id"].astype(str))
    pairs = pairs[pairs["fire_event_id"].astype(str).isin(selected_fire) & pairs["harvest_event_id"].astype(str).isin(selected_harvest)]
    pairs = pairs.sort_values(["age_difference", "centroid_distance_m"]).drop_duplicates("fire_event_id").copy()
    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = [f"event_pair_{i + 1:03d}" for i in range(len(pairs))]
    supported = event_year.groupby(["disturbance_type", "event_id"]).size().rename("supported_event_years").reset_index()
    fire_support = supported[supported["disturbance_type"].eq("fire_total")].rename(columns={"event_id": "fire_event_id", "supported_event_years": "fire_supported_event_years"})
    harvest_support = supported[supported["disturbance_type"].eq("harvest_total")].rename(columns={"event_id": "harvest_event_id", "supported_event_years": "harvest_supported_event_years"})
    out = pairs.merge(fire_support[["fire_event_id", "fire_supported_event_years"]], on="fire_event_id", how="left")
    out = out.merge(harvest_support[["harvest_event_id", "harvest_supported_event_years"]], on="harvest_event_id", how="left")
    out[["fire_supported_event_years", "harvest_supported_event_years"]] = out[["fire_supported_event_years", "harvest_supported_event_years"]].fillna(0).astype(int)
    out["fire_has_reference_support"] = out["fire_supported_event_years"] > 0
    out["harvest_has_reference_support"] = out["harvest_supported_event_years"] > 0
    out["support_class"] = np.select(
        [
            out["fire_has_reference_support"] & out["harvest_has_reference_support"],
            out["fire_has_reference_support"] & ~out["harvest_has_reference_support"],
            ~out["fire_has_reference_support"] & out["harvest_has_reference_support"],
        ],
        ["both_fire_and_harvest_supported", "only_fire_supported", "only_harvest_supported"],
        default="neither_supported",
    )
    return out


def pre_disturbance_gap_audit(event_year: pd.DataFrame) -> pd.DataFrame:
    pre = event_year[event_year["years_since_disturbance"].isin([-3, -2, -1])].copy()
    rows = []
    for dtype, group in pre.groupby("disturbance_type"):
        for rel, part in group.groupby("years_since_disturbance"):
            rows.append(
                {
                    "disturbance_type": dtype,
                    "years_since_disturbance": int(rel),
                    "event_year_count": int(len(part)),
                    "median_nbr_gap": float(part["nbr_gap_median"].median()),
                    "p25_nbr_gap": float(part["nbr_gap_median"].quantile(0.25)),
                    "p75_nbr_gap": float(part["nbr_gap_median"].quantile(0.75)),
                    "median_ndvi_gap": float(part["ndvi_gap_median"].median()),
                    "p25_ndvi_gap": float(part["ndvi_gap_median"].quantile(0.25)),
                    "p75_ndvi_gap": float(part["ndvi_gap_median"].quantile(0.75)),
                    "strong_outlier_count_abs_nbr_gap_gt_0_2": int(part["nbr_gap_median"].abs().gt(0.2).sum()),
                }
            )
    overall = pre.groupby("disturbance_type").agg(
        event_year_count=("event_id", "count"),
        median_nbr_gap=("nbr_gap_median", "median"),
        p25_nbr_gap=("nbr_gap_median", lambda s: s.quantile(0.25)),
        p75_nbr_gap=("nbr_gap_median", lambda s: s.quantile(0.75)),
        median_ndvi_gap=("ndvi_gap_median", "median"),
        p25_ndvi_gap=("ndvi_gap_median", lambda s: s.quantile(0.25)),
        p75_ndvi_gap=("ndvi_gap_median", lambda s: s.quantile(0.75)),
        strong_outlier_count_abs_nbr_gap_gt_0_2=("nbr_gap_median", lambda s: int(s.abs().gt(0.2).sum())),
    ).reset_index()
    overall["years_since_disturbance"] = "all_pre"
    return pd.concat([overall, pd.DataFrame(rows)], ignore_index=True)


def common_mode_audit(paired: pd.DataFrame) -> pd.DataFrame:
    valid = paired[paired["valid_disturbed_reference_pair"] & paired["observation_year"].isin([2021, 2022])].copy()
    rows = []
    for dtype, group in valid.groupby("disturbance_type"):
        pivot = group.pivot_table(index=["event_id", "site_id", "reference_id"], columns="observation_year", values=["dist_nbr", "ref_nbr", "nbr_gap"], aggfunc="first")
        if pivot.empty or 2021 not in pivot["dist_nbr"].columns or 2022 not in pivot["dist_nbr"].columns:
            continue
        dist_step = pivot["dist_nbr"][2022] - pivot["dist_nbr"][2021]
        ref_step = pivot["ref_nbr"][2022] - pivot["ref_nbr"][2021]
        gap_step = pivot["nbr_gap"][2022] - pivot["nbr_gap"][2021]
        common = pd.DataFrame({"disturbed": dist_step, "reference": ref_step, "gap": gap_step}).dropna()
        if common.empty:
            continue
        rows.append(
            {
                "disturbance_type": dtype,
                "paired_site_count": int(len(common)),
                "median_disturbed_nbr_step_2022_minus_2021": float(common["disturbed"].median()),
                "median_reference_nbr_step_2022_minus_2021": float(common["reference"].median()),
                "median_nbr_gap_step_2022_minus_2021": float(common["gap"].median()),
                "mean_disturbed_nbr_step_2022_minus_2021": float(common["disturbed"].mean()),
                "mean_reference_nbr_step_2022_minus_2021": float(common["reference"].mean()),
                "mean_nbr_gap_step_2022_minus_2021": float(common["gap"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    final_refs: gpd.GeoDataFrame,
    selection_audit: pd.DataFrame,
    paired: pd.DataFrame,
    event_year: pd.DataFrame,
    temporal_coverage: pd.DataFrame,
    pre_gap: pd.DataFrame,
    common_mode: pd.DataFrame,
    sites: gpd.GeoDataFrame,
) -> None:
    colors = {"fire_total": "#c95f3f", "harvest_total": "#2f7f73"}
    fig, ax = plt.subplots(figsize=(9, 9))
    sites[sites["site_id"].isin(final_refs["associated_site_id"])].plot(ax=ax, markersize=10, color=sites["disturbance_type"].map(colors), alpha=0.75)
    final_refs.plot(ax=ax, markersize=5, color="#202020", alpha=0.5)
    ax.set_axis_off()
    ax.set_title("Final disturbed/reference site pairs")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_disturbed_reference_site_pairs_map.png", dpi=220)
    plt.close(fig)

    for col, name, xlabel in [
        ("distance_m", "final_reference_distance_distribution.png", "distance to final reference (m)"),
        ("abs_elevation_difference_m", "elevation_difference_distribution.png", "absolute elevation difference (m)"),
        ("abs_slope_difference_deg", "slope_difference_distribution.png", "absolute slope difference (degrees)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        selection_audit[col].plot(kind="hist", bins=28, ax=ax, color="#526760", edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_title(xlabel)
        fig.tight_layout()
        fig.savefig(FIG_DIR / name, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = temporal_coverage.pivot(index="period", columns="disturbance_type", values="coverage_percent")
    pivot.plot(kind="bar", ax=ax, color=[colors.get(c, "#777") for c in pivot.columns])
    ax.set_ylabel("event-year coverage (%)")
    ax.set_title("Disturbed-reference temporal coverage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "disturbed_reference_temporal_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pre = event_year[event_year["years_since_disturbance"].isin([-3, -2, -1])]
    if not pre.empty:
        pre.boxplot(column="nbr_gap_median", by="disturbance_type", ax=ax)
    ax.set_title("Pre-disturbance event-level NBR gap")
    ax.set_xlabel("")
    ax.set_ylabel("NBR gap (reference - disturbed)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pre_disturbance_nbr_gap_distribution.png", dpi=180)
    plt.close(fig)

    examples = event_year.groupby("event_id")["observation_year"].count().sort_values(ascending=False).head(6).index
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for event_id, group in event_year[event_year["event_id"].isin(examples)].groupby("event_id"):
        color = colors.get(group["disturbance_type"].iloc[0], "#777")
        ax.plot(group["years_since_disturbance"], group["dist_nbr_median"], color=color, alpha=0.55)
        ax.plot(group["years_since_disturbance"], group["ref_nbr_median"], color=color, linestyle="--", alpha=0.55)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("NBR")
    ax.set_title("Example raw disturbed/reference NBR trajectories")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_raw_disturbed_reference_nbr_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for event_id, group in event_year[event_year["event_id"].isin(examples)].groupby("event_id"):
        color = colors.get(group["disturbance_type"].iloc[0], "#777")
        ax.plot(group["years_since_disturbance"], group["nbr_gap_median"], color=color, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("years since disturbance")
    ax.set_ylabel("NBR gap (reference - disturbed)")
    ax.set_title("Example NBR-gap trajectories")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_nbr_gap_trajectories.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not common_mode.empty:
        labels = common_mode["disturbance_type"].str.replace("_", " ")
        x = np.arange(len(labels))
        width = 0.25
        ax.bar(x - width, common_mode["median_disturbed_nbr_step_2022_minus_2021"], width, label="disturbed", color="#c95f3f")
        ax.bar(x, common_mode["median_reference_nbr_step_2022_minus_2021"], width, label="reference", color="#526760")
        ax.bar(x + width, common_mode["median_nbr_gap_step_2022_minus_2021"], width, label="NBR gap", color="#2f7f73")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("median 2022-2021 NBR step")
    ax.set_title("2021-2022 disturbed/reference/common-mode diagnostic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landsat_2021_2022_common_mode_diagnostic.png", dpi=180)
    plt.close(fig)


def write_readme(summary: dict) -> None:
    text = f"""# Reference time series

This folder contains the final one-to-one reference-site selection and annual
Landsat reference extraction for the Lebel-sur-Quevillon recovery design.

Controls were selected only from design variables independent of recovery
outcomes: standardized absolute elevation difference, standardized absolute
slope difference, and geographic distance as a secondary criterion. NBR, NDVI,
reflectance, Sentinel-2, and AlphaEarth were not used for control selection.

`NBR_gap = NBR_reference - NBR_disturbed`. A gap near zero means similar annual
spectral NBR condition relative to the selected reference; it is not interpreted
as complete ecological recovery.

Reference sites are not independent disturbance replicates. The inferential unit
remains the disturbance event.

## Summary

```json
{json.dumps(summary, indent=2)}
```
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Reading validated candidates and disturbed sites...", flush=True)
    sites = read_sites()
    candidates = read_valid_candidates()
    coverage = pd.read_csv(REFERENCE_EVENT_COVERAGE)
    zero_events = coverage.loc[coverage["event_reference_status"].eq("no_viable_reference_candidates"), "event_id"].astype(str)

    print("Performing targeted 3-5 km rescue for zero-reference events only...", flush=True)
    module07 = load_module(SCRIPT_07, "reference_validation")
    _, _, official, _ = module07.read_inputs()
    zero_sites = sites[sites["event_id"].astype(str).isin(set(zero_events))].copy()
    rescue = generate_rescue_candidates(zero_sites, candidates, official, module07)
    if not rescue.empty:
        candidates = pd.concat([candidates, rescue], ignore_index=True)
        candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=CRS_PROJECTED)
    rescue_summary = {
        "zero_reference_event_count_before_rescue": int(len(set(zero_events))),
        "zero_reference_site_count_before_rescue": int(len(zero_sites)),
        "rescue_valid_candidate_count": int(len(rescue)),
        "rescue_site_count_with_valid_candidate": int(rescue["associated_site_id"].nunique()) if not rescue.empty else 0,
        "rescue_event_count_with_valid_candidate": int(rescue["associated_event_id"].nunique()) if not rescue.empty else 0,
    }

    print("Selecting one final reference per supported disturbed site...", flush=True)
    final_refs, selection_audit, reuse = select_final_references(candidates, sites)
    final_gpkg = OUT_DIR / "final_reference_sites.gpkg"
    if final_gpkg.exists():
        final_gpkg.unlink()
    final_refs.to_file(final_gpkg, layer="final_reference_sites", driver="GPKG")
    selection_audit.to_csv(OUT_DIR / "reference_selection_audit.csv", index=False, float_format="%.6f")
    reuse.to_csv(OUT_DIR / "reference_reuse_audit.csv", index=False, float_format="%.6f")

    reference_csv = OUT_DIR / "landsat_reference_site_year.csv"
    expected_reference_ids = set(final_refs["reference_id"].astype(str))
    if reference_csv.exists():
        cached = pd.read_csv(reference_csv)
        cached_ids = set(cached["reference_id"].astype(str)) if "reference_id" in cached.columns else set()
        if expected_reference_ids.issubset(cached_ids):
            print(f"Reusing existing Landsat reference extraction from {reference_csv}...", flush=True)
            ref_site_year = cached[cached["reference_id"].astype(str).isin(expected_reference_ids)].copy()
        else:
            print(f"Extracting Landsat for {len(final_refs):,} final reference sites...", flush=True)
            reference_sites = reference_sites_for_landsat(final_refs)
            ref_site_year = extract_reference_landsat(reference_sites)
            ref_site_year.to_csv(reference_csv, index=False, float_format="%.6f")
    else:
        print(f"Extracting Landsat for {len(final_refs):,} final reference sites...", flush=True)
        reference_sites = reference_sites_for_landsat(final_refs)
        ref_site_year = extract_reference_landsat(reference_sites)
        ref_site_year.to_csv(reference_csv, index=False, float_format="%.6f")

    print("Building disturbed-reference site-year and event-year tables...", flush=True)
    paired = build_disturbed_reference_site_year(ref_site_year, final_refs)
    event_year = aggregate_event_year(paired)
    temporal = reference_temporal_coverage(paired, event_year)
    pair_complete = reference_pair_completeness(final_refs, event_year)
    pre_gap = pre_disturbance_gap_audit(event_year)
    common_mode = common_mode_audit(paired)

    paired.to_csv(OUT_DIR / "disturbed_reference_site_year.csv", index=False, float_format="%.6f")
    event_year.to_csv(OUT_DIR / "disturbed_reference_event_year.csv", index=False, float_format="%.6f")
    temporal.to_csv(OUT_DIR / "reference_temporal_coverage.csv", index=False, float_format="%.6f")
    pair_complete.to_csv(OUT_DIR / "reference_pair_completeness.csv", index=False, float_format="%.6f")
    pre_gap.to_csv(OUT_DIR / "pre_disturbance_gap_audit.csv", index=False, float_format="%.6f")
    common_mode.to_csv(OUT_DIR / "landsat_2021_2022_common_mode_audit.csv", index=False, float_format="%.6f")

    make_figures(final_refs, selection_audit, paired, event_year, temporal, pre_gap, common_mode, sites)

    supported_events = event_year.groupby("disturbance_type")["event_id"].nunique().to_dict()
    pair_counts = pair_complete["support_class"].value_counts().to_dict()
    valid_pairs = paired["valid_disturbed_reference_pair"].sum()
    expected_pairs = len(paired)
    pre_overall = pre_gap[pre_gap["years_since_disturbance"].astype(str).eq("all_pre")]
    common_note = "not enough paired 2021-2022 observations"
    if not common_mode.empty:
        max_gap_step = common_mode["median_nbr_gap_step_2022_minus_2021"].abs().max()
        common_note = (
            "NBR gap is comparatively stable; annual shift is mostly common-mode"
            if max_gap_step < 0.02
            else "NBR gap also shifts; flag this transition for later modelling"
        )
    summary = {
        "final_disturbed_sites_with_selected_references": int(len(final_refs)),
        "rescue_summary": rescue_summary,
        "events_with_any_reference_support": supported_events,
        "fire_harvest_event_pair_support": pair_counts,
        "complete_fire_harvest_event_pairs_for_primary_reference_relative_analysis": int(pair_counts.get("both_fire_and_harvest_supported", 0)),
        "reference_distance_m": selection_audit["distance_m"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "absolute_elevation_difference_m": selection_audit["abs_elevation_difference_m"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "absolute_slope_difference_deg": selection_audit["abs_slope_difference_deg"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict(),
        "reference_reuse": {
            "unique_reference_coordinates": int(reuse.shape[0]),
            "max_reuse_count": int(reuse["reference_reuse_count"].max()) if not reuse.empty else 0,
            "coordinates_reused_more_than_once": int(reuse["reference_reuse_count"].gt(1).sum()) if not reuse.empty else 0,
        },
        "landsat_reference_site_years": {
            "rows": int(len(ref_site_year)),
            "available_rows": int(ref_site_year["data_available"].sum()),
            "coverage_percent": float(100 * ref_site_year["data_available"].mean()),
        },
        "disturbed_reference_site_years": {
            "rows": int(expected_pairs),
            "valid_pairs": int(valid_pairs),
            "coverage_percent": float(100 * valid_pairs / expected_pairs) if expected_pairs else None,
        },
        "pre_disturbance_gap_summary": pre_overall.to_dict(orient="records"),
        "landsat_2021_2022_common_mode_result": common_note,
        "adequacy_for_recovery_modelling": "adequate for event-level reference-relative modelling with incomplete-pair flags; do not force unsupported controls.",
    }
    (OUT_DIR / "reference_timeseries_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
