#!/usr/bin/env python3
"""Audit recovery-cohort definitions before remote-sensing extraction.

This script uses only local official Quebec fire and forestry-intervention
vectors prepared by the preceding event-history workflow. It does not download
or process Hansen, Landsat, Sentinel-2, AlphaEarth, NBR, or embeddings.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_GPKG = REPO_ROOT / "outputs" / "event_histories" / "disturbance_events.gpkg"
COMPOUND_CSV = REPO_ROOT / "outputs" / "event_histories" / "compound_disturbance_histories.csv"
FORESTRY_GPKG = REPO_ROOT / "data" / "raw" / "forestry_interventions" / "interv_fores_lebel_100km_circle.gpkg"
SCRIPT_03 = REPO_ROOT / "analysis" / "03_build_disturbance_event_histories.py"

OUT_DIR = REPO_ROOT / "outputs" / "cohort_definition_audit"
FIG_DIR = REPO_ROOT / "figures" / "cohort_definition_audit"
MPL_DIR = OUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt
from scipy import ndimage
from shapely import contains_xy


CRS_PROJECTED = "EPSG:32198"
OBSERVATION_YEAR = 2024
BUFFER_DISTANCES_M = [60, 90, 150]
MAJOR_HARVEST_CLASS = "harvest_total"
CELL_SIZE_M = 30
CELL_AREA_HA = CELL_SIZE_M * CELL_SIZE_M / 10_000
CLASS_TO_ID = {
    "unknown": 0,
    "harvest_total": 1,
    "harvest_partial": 2,
    "reforestation": 3,
    "thinning": 4,
    "other_silviculture": 5,
    "salvage_after_fire": 6,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def load_event_module():
    spec = importlib.util.spec_from_file_location("event_histories", SCRIPT_03)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_03}")
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def area_ha(geom: BaseGeometry | None) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area / 10_000)


def clean_geom(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if not geom.is_empty else None


def union_geoms(geoms) -> BaseGeometry | None:
    geoms = [clean_geom(g) for g in geoms if g is not None and not g.is_empty]
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    return gpd.GeoSeries(geoms, crs=CRS_PROJECTED).union_all()


def clipped_union(fire_geom: BaseGeometry, records: gpd.GeoDataFrame) -> BaseGeometry | None:
    if records.empty:
        return None
    return union_geoms(records.geometry.intersection(fire_geom))


def subtract_union(base: BaseGeometry, cutters: list[BaseGeometry | None]) -> BaseGeometry | None:
    out = clean_geom(base)
    if out is None:
        return None
    for cutter in cutters:
        cutter = clean_geom(cutter)
        if cutter is not None and not cutter.is_empty:
            out = clean_geom(out.difference(cutter))
            if out is None:
                return None
    return out


def lag_bucket(lag: float | None) -> str:
    if lag is None or pd.isna(lag):
        return "no mapped prior intervention"
    lag = int(lag)
    if lag == 0:
        return "same year"
    if 1 <= lag <= 5:
        return "1-5 years before fire"
    if 6 <= lag <= 10:
        return "6-10 years before fire"
    if 11 <= lag <= 20:
        return "11-20 years before fire"
    if 21 <= lag <= 30:
        return "21-30 years before fire"
    return ">30 years before fire"


def largest_component_ha(geom: BaseGeometry | None) -> float:
    geom = clean_geom(geom)
    if geom is None:
        return 0.0
    if geom.geom_type == "Polygon":
        return area_ha(geom)
    if geom.geom_type == "MultiPolygon":
        return max((area_ha(g) for g in geom.geoms), default=0.0)
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in {"Polygon", "MultiPolygon"}]
    return max((area_ha(g) for g in parts), default=0.0)


def read_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    module03 = load_event_module()
    fires = gpd.read_file(EVENT_GPKG, layer="fire_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    fires = fires[fires["disturbance_type"] == "fire_total"].copy()
    harvest_events = gpd.read_file(EVENT_GPKG, layer="harvest_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    forestry_raw = module03.norm_cols(gpd.read_file(FORESTRY_GPKG, engine="pyogrio")).to_crs(CRS_PROJECTED)
    forestry = module03.classify_forestry_records(forestry_raw)
    forestry = forestry.dropna(subset=["disturbance_year"]).copy()
    forestry["disturbance_year"] = forestry["disturbance_year"].astype(int)
    compounds = pd.read_csv(COMPOUND_CSV)
    return fires, harvest_events, forestry, compounds


def prior_lag_partition(fire: pd.Series, candidates: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, BaseGeometry | None]]:
    fire_geom = fire.geometry
    fire_year = int(fire["disturbance_year"])
    prior = candidates[candidates["disturbance_year"] < fire_year].copy()
    remaining = clean_geom(fire_geom)
    rows = []
    bucket_geoms: dict[str, list[BaseGeometry]] = {}
    year_geoms: dict[int, BaseGeometry | None] = {}

    for year in sorted(prior["disturbance_year"].dropna().unique(), reverse=True):
        year_part = prior[prior["disturbance_year"] == year]
        year_union = clipped_union(fire_geom, year_part)
        year_geoms[int(year)] = year_union
        if remaining is None or year_union is None:
            continue
        assigned = clean_geom(remaining.intersection(year_union))
        if assigned is None:
            continue
        classes = ";".join(sorted(set(year_part["project_class"].dropna().astype(str))))
        lag = fire_year - int(year)
        bucket = lag_bucket(lag)
        rows.append(
            {
                "event_id": fire["event_id"],
                "fire_year": fire_year,
                "previous_intervention_year": int(year),
                "years_between_previous_intervention_and_fire": int(lag),
                "previous_intervention_class": classes,
                "lag_bucket": bucket,
                "area_ha": area_ha(assigned),
                "percent_of_fire_area": area_ha(assigned) / area_ha(fire_geom) * 100,
            }
        )
        bucket_geoms.setdefault(bucket, []).append(assigned)
        remaining = clean_geom(remaining.difference(year_union))

    if remaining is not None and not remaining.is_empty:
        rows.append(
            {
                "event_id": fire["event_id"],
                "fire_year": fire_year,
                "previous_intervention_year": pd.NA,
                "years_between_previous_intervention_and_fire": pd.NA,
                "previous_intervention_class": "none",
                "lag_bucket": "no mapped prior intervention",
                "area_ha": area_ha(remaining),
                "percent_of_fire_area": area_ha(remaining) / area_ha(fire_geom) * 100,
            }
        )
        bucket_geoms.setdefault("no mapped prior intervention", []).append(remaining)

    return pd.DataFrame(rows), {k: union_geoms(v) for k, v in bucket_geoms.items()} | {"prior_any": union_geoms(year_geoms.values())}


def build_fire_audit(
    fires: gpd.GeoDataFrame, forestry: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    sindex = forestry.sindex
    partition_rows = []
    eligibility_rows = []
    lag_rows = []
    buffer_rows = []
    map_parts = []

    for _, fire in fires.iterrows():
        fire_geom = clean_geom(fire.geometry)
        if fire_geom is None:
            continue
        fire_year = int(fire["disturbance_year"])
        total_area = area_ha(fire_geom)
        candidates = forestry.iloc[list(sindex.query(fire_geom, predicate="intersects"))].copy()
        candidates = candidates[candidates.intersects(fire_geom)].copy()

        same_year = clipped_union(fire_geom, candidates[candidates["disturbance_year"] == fire_year])
        salvage_after = clipped_union(
            fire_geom,
            candidates[(candidates["project_class"] == "salvage_after_fire") & (candidates["disturbance_year"] >= fire_year)],
        )
        later_harvest = clipped_union(
            fire_geom,
            candidates[(candidates["project_class"] == MAJOR_HARVEST_CLASS) & (candidates["disturbance_year"] > fire_year)],
        )
        prior_any = clipped_union(fire_geom, candidates[candidates["disturbance_year"] < fire_year])
        prior_within_10 = clipped_union(
            fire_geom,
            candidates[(candidates["disturbance_year"] < fire_year) & ((fire_year - candidates["disturbance_year"]) <= 10)],
        )
        prior_within_20 = clipped_union(
            fire_geom,
            candidates[(candidates["disturbance_year"] < fire_year) & ((fire_year - candidates["disturbance_year"]) <= 20)],
        )
        prior_major_20 = clipped_union(
            fire_geom,
            candidates[
                (candidates["project_class"] == MAJOR_HARVEST_CLASS)
                & (candidates["disturbance_year"] < fire_year)
                & ((fire_year - candidates["disturbance_year"]) <= 20)
            ],
        )

        lag_df, lag_geoms = prior_lag_partition(fire, candidates)
        lag_rows.append(lag_df)

        precedence = [
            ("same_year_fire_forestry", same_year),
            ("salvage_after_fire", salvage_after),
            ("later_harvest_after_fire", later_harvest),
            ("prior_intervention", lag_geoms.get("prior_any")),
            ("fire_eligible_no_recent_harvest", None),
        ]
        remaining = fire_geom
        for rank, (category, geom) in enumerate(precedence, start=1):
            if category == "fire_eligible_no_recent_harvest":
                piece = remaining
            else:
                piece = clean_geom(remaining.intersection(geom)) if remaining is not None and geom is not None else None
            if piece is not None and not piece.is_empty:
                partition_rows.append(
                    {
                        "event_id": fire["event_id"],
                        "fire_year": fire_year,
                        "category": category,
                        "precedence_rank": rank,
                        "area_ha": area_ha(piece),
                        "percent_of_fire_area": area_ha(piece) / total_area * 100,
                    }
                )
                if total_area == fires["area_ha"].max() or fire["event_id"] == fires.sort_values("area_ha", ascending=False).iloc[0]["event_id"]:
                    map_parts.append(
                        {
                            "event_id": fire["event_id"],
                            "fire_year": fire_year,
                            "category": category,
                            "area_ha": area_ha(piece),
                            "geometry": piece,
                        }
                    )
            if category != "fire_eligible_no_recent_harvest" and geom is not None and remaining is not None:
                remaining = clean_geom(remaining.difference(geom))

        definitions = {
            "A_no_mapped_previous_intervention": [same_year, salvage_after, prior_any],
            "B_no_intervention_within_10y_before_fire": [same_year, salvage_after, prior_within_10],
            "C_no_intervention_within_20y_before_fire": [same_year, salvage_after, prior_within_20],
            "D_no_major_harvest_within_20y_before_fire": [same_year, salvage_after, prior_major_20],
        }
        interiors = {
            buffer_m: clean_geom(fire_geom.buffer(-buffer_m, quad_segs=1))
            for buffer_m in BUFFER_DISTANCES_M
        }
        event_metrics = {
            "event_id": fire["event_id"],
            "fire_year": fire_year,
            "total_fire_area_ha": total_area,
            "area_no_prior_intervention_ha": float(lag_df.loc[lag_df["lag_bucket"].eq("no mapped prior intervention"), "area_ha"].sum()),
            "area_prior_intervention_1_5y_ha": float(lag_df.loc[lag_df["lag_bucket"].eq("1-5 years before fire"), "area_ha"].sum()),
            "area_prior_intervention_6_10y_ha": float(lag_df.loc[lag_df["lag_bucket"].eq("6-10 years before fire"), "area_ha"].sum()),
            "area_prior_intervention_11_20y_ha": float(lag_df.loc[lag_df["lag_bucket"].eq("11-20 years before fire"), "area_ha"].sum()),
            "area_prior_intervention_21plus_ha": float(lag_df.loc[lag_df["lag_bucket"].isin(["21-30 years before fire", ">30 years before fire"]), "area_ha"].sum()),
            "area_same_year_forestry_ha": area_ha(clean_geom(fire_geom.intersection(same_year))) if same_year is not None else 0.0,
            "area_salvage_after_fire_ha": area_ha(clean_geom(fire_geom.intersection(salvage_after))) if salvage_after is not None else 0.0,
            "area_later_harvest_ha": area_ha(clean_geom(fire_geom.intersection(later_harvest))) if later_harvest is not None else 0.0,
        }
        for def_name, cutters in definitions.items():
            eligible = subtract_union(fire_geom, cutters)
            event_metrics[f"eligible_area_ha__{def_name}"] = area_ha(eligible)
            event_metrics[f"eligible_percent__{def_name}"] = area_ha(eligible) / total_area * 100
            event_metrics[f"largest_patch_ha__{def_name}"] = largest_component_ha(eligible)
            event_metrics[f"has_25ha_patch__{def_name}"] = largest_component_ha(eligible) >= 25
            for buffer_m in BUFFER_DISTANCES_M:
                interior = interiors[buffer_m]
                eligible_buffered = subtract_union(interior, cutters) if interior is not None else None
                buffer_rows.append(
                    {
                        "event_id": fire["event_id"],
                        "fire_year": fire_year,
                        "definition": def_name,
                        "inward_buffer_m": buffer_m,
                        "eligible_area_ha": area_ha(eligible_buffered),
                        "largest_patch_ha": largest_component_ha(eligible_buffered),
                    }
                )
        eligibility_rows.append(event_metrics)

    partition = pd.DataFrame(partition_rows)
    eligibility = pd.DataFrame(eligibility_rows)
    lag = pd.concat(lag_rows, ignore_index=True) if lag_rows else pd.DataFrame()
    buffers = pd.DataFrame(buffer_rows)
    map_gdf = gpd.GeoDataFrame(map_parts, geometry="geometry", crs=CRS_PROJECTED)
    return partition, eligibility, lag, buffers, map_gdf


def summarize_buffer_sensitivity(buffers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (definition, buffer_m), part in buffers.groupby(["definition", "inward_buffer_m"]):
        areas = part["eligible_area_ha"]
        patches = part["largest_patch_ha"]
        rows.append(
            {
                "definition": definition,
                "inward_buffer_m": int(buffer_m),
                "events_retaining_usable_area": int((areas > 0).sum()),
                "total_eligible_area_ha": float(areas.sum()),
                "median_eligible_area_per_event_ha": float(areas.median()),
                "events_with_largest_patch_ge_1ha": int((patches >= 1).sum()),
                "events_with_largest_patch_ge_5ha": int((patches >= 5).sum()),
                "events_with_largest_patch_ge_10ha": int((patches >= 10).sum()),
                "events_with_largest_patch_ge_25ha": int((patches >= 25).sum()),
            }
        )
    return pd.DataFrame(rows)


def compound_level_counts(compounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway, part in compounds.groupby("pathway", dropna=False):
        rows.append(
            {
                "pathway": pathway,
                "source_polygon_rows": pd.NA,
                "spatial_intersection_rows": int(len(part)),
                "unique_fire_events": int(part["fire_event_id"].dropna().nunique()) if "fire_event_id" in part else 0,
                "unique_harvest_events": int(part["harvest_event_id"].dropna().nunique()) if "harvest_event_id" in part else 0,
                "unique_forestry_events": int(part["forestry_event_id"].dropna().nunique()) if "forestry_event_id" in part else 0,
                "note": "Rows are event-event spatial relationships from aggregated event geometries, not independent compound events.",
            }
        )
    return pd.DataFrame(rows)


def harvest_size_audit(harvest_events: gpd.GeoDataFrame) -> pd.DataFrame:
    areas = harvest_events["area_ha"].astype(float)
    rows = [
        {"metric": "event_count", "value": float(len(areas))},
        {"metric": "min", "value": float(areas.min())},
        {"metric": "p10", "value": float(areas.quantile(0.10))},
        {"metric": "median", "value": float(areas.median())},
        {"metric": "p90", "value": float(areas.quantile(0.90))},
        {"metric": "p95", "value": float(areas.quantile(0.95))},
        {"metric": "p99", "value": float(areas.quantile(0.99))},
        {"metric": "max", "value": float(areas.max())},
        {"metric": "events_lt_1ha", "value": float((areas < 1).sum())},
        {"metric": "events_1_5ha", "value": float(((areas >= 1) & (areas < 5)).sum())},
        {"metric": "events_5_25ha", "value": float(((areas >= 5) & (areas < 25)).sum())},
        {"metric": "events_25_100ha", "value": float(((areas >= 25) & (areas < 100)).sum())},
        {"metric": "events_gt_100ha", "value": float((areas > 100).sum())},
        {"metric": "events_with_multiple_source_polygons", "value": float((harvest_events["polygon_count"] > 1).sum())},
    ]
    return pd.DataFrame(rows)


def raster_grid_for_geom(geom: BaseGeometry) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = geom.bounds
    minx = np.floor(minx / CELL_SIZE_M) * CELL_SIZE_M
    miny = np.floor(miny / CELL_SIZE_M) * CELL_SIZE_M
    maxx = np.ceil(maxx / CELL_SIZE_M) * CELL_SIZE_M
    maxy = np.ceil(maxy / CELL_SIZE_M) * CELL_SIZE_M
    xs = np.arange(minx + CELL_SIZE_M / 2, maxx, CELL_SIZE_M)
    ys = np.arange(miny + CELL_SIZE_M / 2, maxy, CELL_SIZE_M)
    return xs, ys, (minx, maxx, miny, maxy)


def mark_geometry(mask: np.ndarray, geom: BaseGeometry, xs: np.ndarray, ys: np.ndarray, value=True) -> None:
    hit_slice = geometry_hit_slice(geom, xs, ys)
    if hit_slice is None:
        return
    y0, y1, x0, x1, hit = hit_slice
    mask[y0:y1, x0:x1][hit] = value


def geometry_hit_slice(
    geom: BaseGeometry, xs: np.ndarray, ys: np.ndarray
) -> tuple[int, int, int, int, np.ndarray] | None:
    geom = clean_geom(geom)
    if geom is None:
        return None
    minx, miny, maxx, maxy = geom.bounds
    x0 = max(0, int(np.floor((minx - xs[0]) / CELL_SIZE_M)))
    x1 = min(len(xs), int(np.ceil((maxx - xs[0]) / CELL_SIZE_M)) + 1)
    y0 = max(0, int(np.floor((miny - ys[0]) / CELL_SIZE_M)))
    y1 = min(len(ys), int(np.ceil((maxy - ys[0]) / CELL_SIZE_M)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    xx, yy = np.meshgrid(xs[x0:x1], ys[y0:y1])
    hit = contains_xy(geom, xx, yy)
    if not hit.any():
        return None
    return y0, y1, x0, x1, hit


def largest_patch_from_mask(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() * CELL_AREA_HA) if len(sizes) else 0.0


def build_fire_audit_raster(
    fires: gpd.GeoDataFrame, forestry: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Partition fire surfaces on an explicit 30 m local grid.

    The audit is about sampling feasibility, so a 30 m grid is an appropriate
    approximation and avoids dropping entire fire events because of tiny vector
    overlaps. Reported original all/clean areas remain the vector values from
    the event-history output.
    """
    sindex = forestry.sindex
    partition_rows = []
    eligibility_rows = []
    lag_rows = []
    buffer_rows = []
    example_map = {}
    largest_event_id = fires.sort_values("area_ha", ascending=False).iloc[0]["event_id"]

    for n, (_, fire) in enumerate(fires.iterrows(), start=1):
        if n % 10 == 0:
            print(f"  processed {n}/{len(fires)} fire events", flush=True)
        fire_geom = clean_geom(fire.geometry)
        if fire_geom is None:
            continue
        fire_year = int(fire["disturbance_year"])
        xs, ys, extent = raster_grid_for_geom(fire_geom)
        xx, yy = np.meshgrid(xs, ys)
        fire_mask = contains_xy(fire_geom, xx, yy)
        if not fire_mask.any():
            continue
        total_raster_area = float(fire_mask.sum() * CELL_AREA_HA)
        total_vector_area = area_ha(fire_geom)

        candidates = forestry.iloc[list(sindex.query(fire_geom, predicate="intersects"))].copy()
        candidates = candidates[candidates.intersects(fire_geom)].copy()
        candidates = candidates.sort_values("disturbance_year")

        prev_year = np.full(fire_mask.shape, -1, dtype=np.int16)
        prev_class = np.zeros(fire_mask.shape, dtype=np.int8)
        same_year = np.zeros(fire_mask.shape, dtype=bool)
        salvage_after = np.zeros(fire_mask.shape, dtype=bool)
        later_harvest = np.zeros(fire_mask.shape, dtype=bool)
        prior_major_20 = np.zeros(fire_mask.shape, dtype=bool)

        for _, cand in candidates.iterrows():
            year = int(cand["disturbance_year"])
            cls = str(cand["project_class"])
            hit_slice = geometry_hit_slice(cand.geometry, xs, ys)
            if hit_slice is None:
                continue
            y0, y1, x0, x1, hit = hit_slice
            hit = hit & fire_mask[y0:y1, x0:x1]
            if not hit.any():
                continue
            prev_year_view = prev_year[y0:y1, x0:x1]
            prev_class_view = prev_class[y0:y1, x0:x1]
            same_year_view = same_year[y0:y1, x0:x1]
            salvage_after_view = salvage_after[y0:y1, x0:x1]
            later_harvest_view = later_harvest[y0:y1, x0:x1]
            prior_major_20_view = prior_major_20[y0:y1, x0:x1]
            if year < fire_year:
                prev_year_view[hit] = year
                prev_class_view[hit] = CLASS_TO_ID.get(cls, 0)
                if cls == MAJOR_HARVEST_CLASS and fire_year - year <= 20:
                    prior_major_20_view[hit] = True
            elif year == fire_year:
                same_year_view[hit] = True
            if cls == "salvage_after_fire" and year >= fire_year:
                salvage_after_view[hit] = True
            if cls == MAJOR_HARVEST_CLASS and year > fire_year:
                later_harvest_view[hit] = True

        prior_any = prev_year >= 0
        lag_arr = np.where(prior_any, fire_year - prev_year, -1)
        prior_within_10 = prior_any & (lag_arr <= 10)
        prior_within_20 = prior_any & (lag_arr <= 20)
        no_prior = fire_mask & ~prior_any

        for year in sorted(np.unique(prev_year[prior_any]), reverse=True):
            ymask = fire_mask & (prev_year == year)
            for class_id in sorted(np.unique(prev_class[ymask])):
                cmask = ymask & (prev_class == class_id)
                if not cmask.any():
                    continue
                lag = fire_year - int(year)
                lag_rows.append(
                    {
                        "event_id": fire["event_id"],
                        "fire_year": fire_year,
                        "previous_intervention_year": int(year),
                        "years_between_previous_intervention_and_fire": int(lag),
                        "previous_intervention_class": ID_TO_CLASS.get(int(class_id), "unknown"),
                        "lag_bucket": lag_bucket(lag),
                        "pixel_count_30m": int(cmask.sum()),
                        "area_ha": float(cmask.sum() * CELL_AREA_HA),
                        "percent_of_fire_area": float(cmask.sum() * CELL_AREA_HA / total_raster_area * 100),
                    }
                )
        if no_prior.any():
            lag_rows.append(
                {
                    "event_id": fire["event_id"],
                    "fire_year": fire_year,
                    "previous_intervention_year": pd.NA,
                    "years_between_previous_intervention_and_fire": pd.NA,
                    "previous_intervention_class": "none",
                    "lag_bucket": "no mapped prior intervention",
                    "pixel_count_30m": int(no_prior.sum()),
                    "area_ha": float(no_prior.sum() * CELL_AREA_HA),
                    "percent_of_fire_area": float(no_prior.sum() * CELL_AREA_HA / total_raster_area * 100),
                }
            )

        remaining = fire_mask.copy()
        category_masks = [
            ("same_year_fire_forestry", same_year),
            ("salvage_after_fire", salvage_after),
            ("later_harvest_after_fire", later_harvest),
            ("harvest_before_fire", prior_any),
            ("fire_eligible_no_recent_harvest", fire_mask),
        ]
        category_image = np.zeros(fire_mask.shape, dtype=np.int8)
        for rank, (category, raw_mask) in enumerate(category_masks, start=1):
            piece = remaining & raw_mask
            if category == "fire_eligible_no_recent_harvest":
                piece = remaining.copy()
            if piece.any():
                partition_rows.append(
                    {
                        "event_id": fire["event_id"],
                        "fire_year": fire_year,
                        "category": category,
                        "precedence_rank": rank,
                        "pixel_count_30m": int(piece.sum()),
                        "area_ha": float(piece.sum() * CELL_AREA_HA),
                        "percent_of_fire_area": float(piece.sum() * CELL_AREA_HA / total_raster_area * 100),
                    }
                )
                category_image[piece] = rank
            if category != "fire_eligible_no_recent_harvest":
                remaining &= ~raw_mask

        definitions = {
            "A_no_mapped_previous_intervention": prior_any,
            "B_no_intervention_within_10y_before_fire": prior_within_10,
            "C_no_intervention_within_20y_before_fire": prior_within_20,
            "D_no_major_harvest_within_20y_before_fire": prior_major_20,
        }
        event_metrics = {
            "event_id": fire["event_id"],
            "fire_year": fire_year,
            "total_fire_area_ha": total_vector_area,
            "rasterized_fire_area_ha": total_raster_area,
            "area_no_prior_intervention_ha": float(no_prior.sum() * CELL_AREA_HA),
            "area_prior_intervention_1_5y_ha": float((fire_mask & prior_any & (lag_arr >= 1) & (lag_arr <= 5)).sum() * CELL_AREA_HA),
            "area_prior_intervention_6_10y_ha": float((fire_mask & prior_any & (lag_arr >= 6) & (lag_arr <= 10)).sum() * CELL_AREA_HA),
            "area_prior_intervention_11_20y_ha": float((fire_mask & prior_any & (lag_arr >= 11) & (lag_arr <= 20)).sum() * CELL_AREA_HA),
            "area_prior_intervention_21plus_ha": float((fire_mask & prior_any & (lag_arr >= 21)).sum() * CELL_AREA_HA),
            "area_same_year_forestry_ha": float((fire_mask & same_year).sum() * CELL_AREA_HA),
            "area_salvage_after_fire_ha": float((fire_mask & salvage_after).sum() * CELL_AREA_HA),
            "area_later_harvest_ha": float((fire_mask & later_harvest).sum() * CELL_AREA_HA),
        }
        distance_to_edge_m = ndimage.distance_transform_edt(fire_mask) * CELL_SIZE_M
        for def_name, prior_exclusion in definitions.items():
            eligible = fire_mask & ~same_year & ~salvage_after & ~prior_exclusion
            event_metrics[f"eligible_area_ha__{def_name}"] = float(eligible.sum() * CELL_AREA_HA)
            event_metrics[f"eligible_percent__{def_name}"] = float(eligible.sum() * CELL_AREA_HA / total_raster_area * 100)
            event_metrics[f"largest_patch_ha__{def_name}"] = largest_patch_from_mask(eligible)
            event_metrics[f"has_25ha_patch__{def_name}"] = largest_patch_from_mask(eligible) >= 25
            for buffer_m in BUFFER_DISTANCES_M:
                interior = fire_mask & (distance_to_edge_m >= buffer_m)
                eligible_buffered = interior & ~same_year & ~salvage_after & ~prior_exclusion
                buffer_rows.append(
                    {
                        "event_id": fire["event_id"],
                        "fire_year": fire_year,
                        "definition": def_name,
                        "inward_buffer_m": buffer_m,
                        "eligible_area_ha": float(eligible_buffered.sum() * CELL_AREA_HA),
                        "largest_patch_ha": largest_patch_from_mask(eligible_buffered),
                    }
                )
        eligibility_rows.append(event_metrics)

        if fire["event_id"] == largest_event_id:
            example_map = {
                "event_id": fire["event_id"],
                "fire_year": fire_year,
                "category_image": np.where(fire_mask, category_image, np.nan),
                "extent": extent,
            }

    return (
        pd.DataFrame(partition_rows),
        pd.DataFrame(eligibility_rows),
        pd.DataFrame(lag_rows),
        pd.DataFrame(buffer_rows),
        example_map,
    )


def write_figures(
    eligibility: pd.DataFrame,
    lag: pd.DataFrame,
    buffers: pd.DataFrame,
    example_map: dict,
) -> None:
    def_col = "A_no_mapped_previous_intervention"
    elig_col = f"eligible_area_ha__{def_col}"
    ordered = eligibility.sort_values("total_fire_area_ha", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ordered))
    ax.bar(x, ordered["total_fire_area_ha"], color="#b24a33", alpha=0.35, label="total fire area")
    ax.bar(x, ordered[elig_col], color="#1f7a5a", alpha=0.85, label="eligible area, definition A")
    ax.set_xticks([])
    ax.set_ylabel("Area ha")
    ax.set_title("Total fire area vs eligible fire area by event")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "total_vs_eligible_fire_area_by_event.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    frac = eligibility[f"eligible_percent__{def_col}"].sort_values()
    ax.plot(np.arange(len(frac)), frac, color="#1f7a5a", linewidth=2)
    ax.set_xlabel("Fire event, sorted")
    ax.set_ylabel("Eligible fraction (%)")
    ax.set_title("Eligible-area fraction by fire event")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eligible_area_fraction_by_fire_event.png", dpi=220)
    plt.close(fig)

    lag_summary = lag.groupby("lag_bucket")["area_ha"].sum().reindex(
        [
            "no mapped prior intervention",
            "1-5 years before fire",
            "6-10 years before fire",
            "11-20 years before fire",
            "21-30 years before fire",
            ">30 years before fire",
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    lag_summary.dropna().plot(kind="bar", ax=ax, color="#6a8f5a")
    ax.set_ylabel("Area ha")
    ax.set_title("Prior-intervention lag distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prior_intervention_lag_distribution.png", dpi=220)
    plt.close(fig)

    retention = []
    for col in [c for c in eligibility.columns if c.startswith("eligible_area_ha__")]:
        definition = col.replace("eligible_area_ha__", "")
        retention.append({"definition": definition, "events": int((eligibility[col] > 0).sum()), "area_ha": float(eligibility[col].sum())})
    retention_df = pd.DataFrame(retention)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(retention_df["definition"], retention_df["events"], color="#457b74")
    ax.set_xlabel("Events retaining eligible area")
    ax.set_title("Event retention under alternative lookback definitions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_retention_by_definition.png", dpi=220)
    plt.close(fig)

    if example_map:
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#f7f7f7", "#756bb1", "#de2d26", "#3182bd", "#fdae6b", "#2ca25f"])
        labels = [
            "outside fire",
            "same-year/compound",
            "salvage after fire",
            "later harvest after fire",
            "prior intervention",
            "eligible fire area",
        ]
        fig, ax = plt.subplots(figsize=(9, 9))
        img = ax.imshow(
            example_map["category_image"],
            extent=example_map["extent"],
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"Example large fire partitioned by treatment history\n{example_map['event_id']}")
        handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=cmap(i), markersize=9, label=label) for i, label in enumerate(labels[1:], start=1)]
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "example_large_fire_partition.png", dpi=220)
        plt.close(fig)


def write_readme(summary: dict) -> None:
    lines = [
        "# Recovery cohort definition audit",
        "",
        "This audit stops before remote-sensing extraction. It uses local official Quebec fire and forestry-intervention vectors only.",
        "",
        "The previous `clean_fire_total` cohort was event-level: if a fire event intersected same-year forestry or fire-salvage pathways, the whole fire event was removed. This audit separates the fire event from eligible recovery surface within that event.",
        "",
        "## Area precedence",
        "",
        "The mutually exclusive partition applies this order within each fire geometry: same-year fire/forestry, salvage after fire, later harvest after fire, prior intervention, then remaining eligible fire area. Prior-intervention lag tables separately assign each burned surface to the most recent mapped intervention before fire.",
        "",
        "## Candidate definitions",
        "",
        "- A: no mapped previous forestry intervention at all.",
        "- B: no forestry intervention within 10 years before fire.",
        "- C: no forestry intervention within 20 years before fire.",
        "- D: no major harvest within 20 years before fire; minor treatments retained separately.",
        "",
        "## Summary",
        "",
        json.dumps(summary, indent=2),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Reading event-history outputs and local official forestry vectors...", flush=True)
    fires, harvest_events, forestry, compounds = read_inputs()
    print("Partitioning fire events by prior and subsequent treatment history...", flush=True)
    partition, eligibility, lag, buffers_raw, example_map = build_fire_audit_raster(fires, forestry)
    buffer_summary = summarize_buffer_sensitivity(buffers_raw)
    compound_counts = compound_level_counts(compounds)
    harvest_audit = harvest_size_audit(harvest_events)

    partition.to_csv(OUT_DIR / "fire_area_history_partition.csv", index=False)
    eligibility.to_csv(OUT_DIR / "fire_event_eligibility.csv", index=False)
    lag.to_csv(OUT_DIR / "fire_prior_intervention_lag.csv", index=False)
    buffer_summary.to_csv(OUT_DIR / "fire_buffer_sensitivity.csv", index=False)
    compound_counts.to_csv(OUT_DIR / "compound_pathway_level_counts.csv", index=False)
    harvest_audit.to_csv(OUT_DIR / "harvest_event_size_audit.csv", index=False)

    write_figures(eligibility, lag, buffers_raw, example_map)

    definitions = {
        col.replace("eligible_area_ha__", ""): {
            "events_with_area_gt_0": int((eligibility[col] > 0).sum()),
            "total_eligible_area_ha": float(eligibility[col].sum()),
            "events_with_largest_patch_ge_25ha": int((eligibility[col.replace("eligible_area_ha__", "largest_patch_ha__")] >= 25).sum()),
        }
        for col in eligibility.columns
        if col.startswith("eligible_area_ha__")
    }
    previous_clean_area = 4295.440383
    total_fire_area = float(eligibility["total_fire_area_ha"].sum())
    summary = {
        "current_algorithm_diagnosis": {
            "exclusion_level": "event level",
            "clean_primary_rule": "fire events are removed if their event_id appears in fire_then_salvage or ambiguous_same_year; fire_then_harvest is treated as censoring, not clean-fire exclusion",
            "all_fire_total_area_ha": total_fire_area,
            "previous_clean_fire_area_ha": previous_clean_area,
            "removed_area_ha": total_fire_area - previous_clean_area,
            "removed_area_percent": (total_fire_area - previous_clean_area) / total_fire_area * 100,
            "reason": "large fire events were dropped in full when any qualifying compound relationship touched the event geometry",
        },
        "definition_retention": definitions,
        "compound_rows_are": "event-event spatial relationship rows from aggregated geometries, not independent compound events",
        "harvest_grouping_check": "harvest_total events are grouped by disturbance year plus spatial contiguity/intersection; event-size distribution is written separately",
        "decision_needed_before_sampling": "choose a pre-fire lookback/exclusion rule and decide whether later post-fire harvest/salvage censors or excludes candidate sampling surfaces",
    }
    (OUT_DIR / "cohort_definition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)

    print("CURRENT CLEAN-FIRE LOGIC")
    print(json.dumps(summary["current_algorithm_diagnosis"], indent=2))
    print("\nEVENT RETENTION BY DEFINITION")
    print(pd.DataFrame(definitions).T.to_string())
    print("\nBUFFER SENSITIVITY")
    print(buffer_summary.to_string(index=False))
    print("\nCOMPOUND PATHWAY LEVEL COUNTS")
    print(compound_counts.to_string(index=False))
    print("\nHARVEST EVENT SIZE AUDIT")
    print(harvest_audit.to_string(index=False))
    print("\nFILES CREATED/MODIFIED")
    for path in [
        Path(__file__),
        OUT_DIR / "fire_area_history_partition.csv",
        OUT_DIR / "fire_event_eligibility.csv",
        OUT_DIR / "fire_prior_intervention_lag.csv",
        OUT_DIR / "fire_buffer_sensitivity.csv",
        OUT_DIR / "compound_pathway_level_counts.csv",
        OUT_DIR / "harvest_event_size_audit.csv",
        OUT_DIR / "cohort_definition_summary.json",
        OUT_DIR / "README.md",
        FIG_DIR / "total_vs_eligible_fire_area_by_event.png",
        FIG_DIR / "eligible_area_fraction_by_fire_event.png",
        FIG_DIR / "prior_intervention_lag_distribution.png",
        FIG_DIR / "event_retention_by_definition.png",
        FIG_DIR / "example_large_fire_partition.png",
    ]:
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
