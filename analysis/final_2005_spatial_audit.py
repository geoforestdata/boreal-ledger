"""
Final spatial audit for the 2005 / 19-year matched disturbance result.

The goal is to check obvious spatial artifacts before freezing the public
AlphaEarth disturbance-type story.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROJECT = "ibfra2026"
OBSERVATION_YEAR = 2024
TARGET_YEAR = 2005
TARGET_AGE = 19
LOCAL_CRS = "EPSG:6622"
AOI_PATH = ROOT / "exploration" / "lebel_aoi_selection" / "recommended_aoi_temporal_spread.geojson"
MATCHED_CSV = ROOT / "outputs" / "alphaearth_spatially_matched_samples.csv"
PAIR_DIAGNOSTICS_CSV = ROOT / "outputs" / "final_2005_pair_diagnostics.csv"
AUDIT_CSV = ROOT / "outputs" / "final_2005_spatial_audit.csv"
MAP_PATH = ROOT / "figures" / "final_2005_matched_pairs_map.png"

HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
NBAC_COLLECTION = "projects/sat-io/open-datasets/CA_FOREST/NBAC/nbac_1972_2023_20240530"


def load_aoi_local() -> tuple[Polygon, object]:
    geojson = json.load(open(AOI_PATH))
    aoi = shape(geojson["geometry"])
    to_local = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True).transform
    return aoi, transform(to_local, aoi)


def point_local(lon: float, lat: float) -> Point:
    to_local = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True).transform
    x, y = to_local(lon, lat)
    return Point(x, y)


def ee_aoi() -> ee.Geometry:
    geojson = json.load(open(AOI_PATH))
    return ee.Geometry.Polygon(geojson["geometry"]["coordinates"], proj="EPSG:4326", geodesic=False)


def fire_geometry_local(aoi: ee.Geometry):
    fc = ee.FeatureCollection(NBAC_COLLECTION).filterBounds(aoi).filter(ee.Filter.eq("YEAR", TARGET_YEAR))
    features = fc.getInfo()["features"]
    to_local = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True).transform
    geoms = []
    for feature in features:
        geom = shape(feature["geometry"])
        if not geom.is_empty:
            geoms.append(transform(to_local, geom))
    return unary_union(geoms) if geoms else None


def mask_overlap_pixels(aoi: ee.Geometry) -> int:
    lossyear = ee.Image(HANSEN_ASSET).select("lossyear").clip(aoi)
    loss_calendar_year = lossyear.add(2000).updateMask(lossyear.gt(0).And(lossyear.lte(24)))
    fire_year = (
        ee.FeatureCollection(NBAC_COLLECTION)
        .filterBounds(aoi)
        .reduceToImage(properties=["YEAR"], reducer=ee.Reducer.first())
        .rename("fire_year")
    )
    fire_match = loss_calendar_year.eq(fire_year).unmask(0)
    wildfire_2005 = loss_calendar_year.eq(TARGET_YEAR).And(fire_match).selfMask()
    harvest_2005 = loss_calendar_year.eq(TARGET_YEAR).And(fire_match.Not()).selfMask()
    overlap = wildfire_2005.And(harvest_2005).selfMask()
    count = (
        overlap.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=30,
            maxPixels=1e13,
            tileScale=4,
        )
        .get("lossyear")
        .getInfo()
    )
    return int(count or 0)


def add_pair_diagnostics(df: pd.DataFrame, aoi_local: Polygon, fire_local) -> pd.DataFrame:
    rows = []
    for pair_id, pair in df.groupby("pair_id"):
        if len(pair) != 2:
            continue
        fire = pair[pair["disturbance_type"] == "wildfire"].iloc[0]
        harvest = pair[pair["disturbance_type"] == "probable_harvest"].iloc[0]
        fire_point = point_local(fire["lon"], fire["lat"])
        harvest_point = point_local(harvest["lon"], harvest["lat"])
        fire_boundary_distance = fire_point.distance(fire_local.boundary) if fire_local is not None else np.nan
        rows.append(
            {
                "pair_id": pair_id,
                "fire_lon": fire["lon"],
                "fire_lat": fire["lat"],
                "harvest_lon": harvest["lon"],
                "harvest_lat": harvest["lat"],
                "pair_distance_km": fire_point.distance(harvest_point) / 1000,
                "fire_distance_to_aoi_boundary_m": fire_point.distance(aoi_local.boundary),
                "harvest_distance_to_aoi_boundary_m": harvest_point.distance(aoi_local.boundary),
                "fire_distance_to_nbac_fire_edge_m": fire_boundary_distance,
                "fire_inside_nbac_2005": bool(fire_local.contains(fire_point)) if fire_local is not None else False,
            }
        )
    return pd.DataFrame(rows)


def nearest_duplicate_distance_m(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.nan
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    distances[distances == 0] = np.nan
    return float(np.nanmin(distances))


def cluster_warning(df: pd.DataFrame) -> tuple[str, int, int]:
    block_size = 1000
    blocks = {
        (int(row["x_m"] // block_size), int(row["y_m"] // block_size))
        for _, row in df.iterrows()
    }
    dominant = (
        df.assign(block=list(zip((df["x_m"] // block_size).astype(int), (df["y_m"] // block_size).astype(int))))
        .groupby("block")
        .size()
        .max()
    )
    warning = "yes" if len(blocks) <= 2 or dominant / len(df) > 0.6 else "no"
    return warning, len(blocks), int(dominant)


def write_map(df: pd.DataFrame, aoi_wgs: Polygon, fire_local, aoi_local: Polygon) -> None:
    to_wgs = Transformer.from_crs(LOCAL_CRS, "EPSG:4326", always_xy=True).transform
    fire_wgs = transform(to_wgs, fire_local) if fire_local is not None else None
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#edf0e7")
    if fire_wgs is not None:
        geoms = [fire_wgs] if fire_wgs.geom_type == "Polygon" else list(fire_wgs.geoms)
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, color="#c55f3c", alpha=0.18, label="NBAC 2005 wildfire area")
            ax.plot(x, y, color="#8d3d25", linewidth=0.8)
    aoi_x, aoi_y = aoi_wgs.exterior.xy
    ax.plot(aoi_x, aoi_y, color="#0b2b22", linewidth=1.8, label="AOI boundary")
    for pair_id, pair in df.groupby("pair_id"):
        fire = pair[pair["disturbance_type"] == "wildfire"].iloc[0]
        harvest = pair[pair["disturbance_type"] == "probable_harvest"].iloc[0]
        ax.plot([fire["lon"], harvest["lon"]], [fire["lat"], harvest["lat"]], color="#6a716c", linewidth=0.35, alpha=0.45)
    fire_df = df[df["disturbance_type"] == "wildfire"]
    harvest_df = df[df["disturbance_type"] == "probable_harvest"]
    ax.scatter(harvest_df["lon"], harvest_df["lat"], s=18, c="#167255", alpha=0.85, linewidth=0, label="Matched probable-harvest samples")
    ax.scatter(fire_df["lon"], fire_df["lat"], s=18, c="#c55f3c", alpha=0.9, linewidth=0, label="Matched wildfire samples")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale_lon0 = x0 + (x1 - x0) * 0.08
    scale_lat0 = y0 + (y1 - y0) * 0.08
    transformer = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True)
    inv = Transformer.from_crs(LOCAL_CRS, "EPSG:4326", always_xy=True)
    sx, sy = transformer.transform(scale_lon0, scale_lat0)
    slon1, slat1 = inv.transform(sx + 5000, sy)
    ax.plot([scale_lon0, slon1], [scale_lat0, slat1], color="#0b2b22", linewidth=2.2)
    ax.text((scale_lon0 + slon1) / 2, scale_lat0 + 0.006, "5 km", ha="center", fontsize=8, color="#0b2b22")
    ax.set_title("2005 matched fire-harvest pairs, 19 years since disturbance", loc="left", fontsize=12, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#c9c2b6", linewidth=0.5, alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True, facecolor="#fbf7ef", edgecolor="#c9c2b6", fontsize=8)
    fig.tight_layout()
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAP_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ee.Initialize(project=PROJECT)
    aoi_wgs, aoi_local = load_aoi_local()
    aoi = ee_aoi()
    fire_local = fire_geometry_local(aoi)
    df = pd.read_csv(MATCHED_CSV)
    df = df[(df["years_since_disturbance"] == TARGET_AGE) & (df["disturbance_year"] == TARGET_YEAR)].copy()
    transformer = Transformer.from_crs("EPSG:4326", LOCAL_CRS, always_xy=True)
    df["x_m"], df["y_m"] = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    pair_diag = add_pair_diagnostics(df, aoi_local, fire_local)
    pair_diag.to_csv(PAIR_DIAGNOSTICS_CSV, index=False, float_format="%.6f")
    write_map(df, aoi_wgs, fire_local, aoi_local)

    fire_df = df[df["disturbance_type"] == "wildfire"]
    harvest_df = df[df["disturbance_type"] == "probable_harvest"]
    fire_points = fire_df[["x_m", "y_m"]].to_numpy(dtype=float)
    harvest_points = harvest_df[["x_m", "y_m"]].to_numpy(dtype=float)
    cluster_fire, fire_blocks, fire_dominant = cluster_warning(fire_df)
    cluster_harvest, harvest_blocks, harvest_dominant = cluster_warning(harvest_df)
    fire_extent_x = (fire_df["x_m"].max() - fire_df["x_m"].min()) / 1000
    fire_extent_y = (fire_df["y_m"].max() - fire_df["y_m"].min()) / 1000
    harvest_extent_x = (harvest_df["x_m"].max() - harvest_df["x_m"].min()) / 1000
    harvest_extent_y = (harvest_df["y_m"].max() - harvest_df["y_m"].min()) / 1000
    edge_close_share = float((pair_diag["fire_distance_to_nbac_fire_edge_m"] < 60).mean())
    audit = pd.DataFrame(
        [
            {
                "target_year": TARGET_YEAR,
                "years_since_disturbance": TARGET_AGE,
                "n_matched_pairs": int(pair_diag.shape[0]),
                "mean_pair_distance_km": pair_diag["pair_distance_km"].mean(),
                "median_pair_distance_km": pair_diag["pair_distance_km"].median(),
                "max_pair_distance_km": pair_diag["pair_distance_km"].max(),
                "distinct_5km_blocks": int(df["fold_5km"].nunique()),
                "distinct_10km_blocks": int(df["fold_10km"].nunique()),
                "fire_extent_x_km": fire_extent_x,
                "fire_extent_y_km": fire_extent_y,
                "harvest_extent_x_km": harvest_extent_x,
                "harvest_extent_y_km": harvest_extent_y,
                "minimum_fire_duplicate_distance_m": nearest_duplicate_distance_m(fire_points),
                "minimum_harvest_duplicate_distance_m": nearest_duplicate_distance_m(harvest_points),
                "median_fire_distance_to_aoi_boundary_m": pair_diag["fire_distance_to_aoi_boundary_m"].median(),
                "median_harvest_distance_to_aoi_boundary_m": pair_diag["harvest_distance_to_aoi_boundary_m"].median(),
                "median_fire_distance_to_nbac_edge_m": pair_diag["fire_distance_to_nbac_fire_edge_m"].median(),
                "share_fire_samples_within_60m_of_nbac_edge": edge_close_share,
                "mask_overlap_pixels": mask_overlap_pixels(aoi),
                "fire_cluster_warning": cluster_fire,
                "fire_1km_blocks": fire_blocks,
                "fire_largest_1km_block_sample_count": fire_dominant,
                "harvest_cluster_warning": cluster_harvest,
                "harvest_1km_blocks": harvest_blocks,
                "harvest_largest_1km_block_sample_count": harvest_dominant,
                "fire_boundary_sampling_warning": "yes" if edge_close_share > 0.5 else "no",
                "mask_overlap_warning": "yes" if mask_overlap_pixels(aoi) > 0 else "no",
                "linear_clearing_warning": "not assessed without road vectors; no new external road data acquired",
                "water_proximity_warning": "not assessed with external water vectors; no new water data acquired",
                "overall_pass": "pass" if edge_close_share <= 0.5 and mask_overlap_pixels(aoi) == 0 and cluster_fire == "no" and cluster_harvest == "no" else "review",
            }
        ]
    )
    audit.to_csv(AUDIT_CSV, index=False, float_format="%.6f")
    print(audit.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(PAIR_DIAGNOSTICS_CSV)
    print(MAP_PATH)


if __name__ == "__main__":
    main()
