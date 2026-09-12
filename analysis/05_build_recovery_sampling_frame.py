#!/usr/bin/env python3
"""Build an event-balanced recovery sampling frame.

This stage stops before remote-sensing extraction. It uses only local official
fire/intervention vectors and the cohort-definition audit outputs.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_GPKG = REPO_ROOT / "outputs" / "event_histories" / "disturbance_events.gpkg"
FIRE_ELIGIBILITY = REPO_ROOT / "outputs" / "cohort_definition_audit" / "fire_event_eligibility.csv"
FIRE_BUFFER_AUDIT = REPO_ROOT / "outputs" / "cohort_definition_audit" / "fire_buffer_sensitivity.csv"
FORESTRY_GPKG = REPO_ROOT / "data" / "raw" / "forestry_interventions" / "interv_fores_lebel_100km_circle.gpkg"
SCRIPT_03 = REPO_ROOT / "analysis" / "03_build_disturbance_event_histories.py"

OUT_DIR = REPO_ROOT / "outputs" / "sampling_frame"
FIG_DIR = REPO_ROOT / "figures" / "sampling_frame"
MPL_DIR = OUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt


CRS_PROJECTED = "EPSG:32198"
OBSERVATION_YEAR = 2024
CELL_SIZE_M = 30
CELL_AREA_HA = CELL_SIZE_M * CELL_SIZE_M / 10_000
FIRE_BUFFERS_M = [60, 90, 150]
HARVEST_BUFFERS_M = [30, 60, 90]
PRIMARY_FIRE_BUFFER_M = 60
PRIMARY_HARVEST_BUFFER_M = 30
PRIMARY_HARVESTS_PER_FIRE = 1
PRIMARY_SITE_CAP = 10
SITE_CAPS = [5, 10, 20]
SITE_SPACING_M = 150
REFERENCE_INNER_RADIUS_M = 500
REFERENCE_OUTER_RADIUS_M = 3000
DEFINITION_A = "A_no_mapped_previous_intervention"
DEFINITION_D = "D_no_major_harvest_within_20y_before_fire"


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


def age_bin(year: int) -> str:
    age = OBSERVATION_YEAR - int(year)
    if 0 <= age <= 5:
        return "0-5"
    if 6 <= age <= 10:
        return "6-10"
    if 11 <= age <= 15:
        return "11-15"
    if 16 <= age <= 20:
        return "16-20"
    if 21 <= age <= 23:
        return "21-23"
    return "outside"


def clean_geom(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if not geom.is_empty else None


def area_ha(geom: BaseGeometry | None) -> float:
    geom = clean_geom(geom)
    return 0.0 if geom is None else float(geom.area / 10_000)


def raster_grid_for_geom(geom: BaseGeometry) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = geom.bounds
    minx = np.floor(minx / CELL_SIZE_M) * CELL_SIZE_M
    miny = np.floor(miny / CELL_SIZE_M) * CELL_SIZE_M
    maxx = np.ceil(maxx / CELL_SIZE_M) * CELL_SIZE_M
    maxy = np.ceil(maxy / CELL_SIZE_M) * CELL_SIZE_M
    xs = np.arange(minx + CELL_SIZE_M / 2, maxx, CELL_SIZE_M)
    ys = np.arange(miny + CELL_SIZE_M / 2, maxy, CELL_SIZE_M)
    return xs, ys


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


def largest_patch_ha(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() * CELL_AREA_HA) if len(sizes) else 0.0


def greedy_points_from_mask(
    mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, cap: int, spacing_m: float, event_id: str
) -> list[dict]:
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return []
    coords = np.column_stack([xs[x_idx], ys[y_idx]])
    centroid = coords.mean(axis=0)
    order = np.argsort(np.linalg.norm(coords - centroid, axis=1))
    selected: list[np.ndarray] = []
    rows = []
    for idx in order:
        coord = coords[idx]
        if selected and np.min(np.linalg.norm(np.vstack(selected) - coord, axis=1)) < spacing_m:
            continue
        selected.append(coord)
        rows.append({"event_id": event_id, "x": float(coord[0]), "y": float(coord[1])})
        if len(rows) >= cap:
            break
    return rows


def greedy_points_in_geom(geom: BaseGeometry, cap: int, spacing_m: float, event_id: str) -> list[dict]:
    geom = clean_geom(geom)
    if geom is None:
        return []
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(np.floor(minx / spacing_m) * spacing_m + spacing_m / 2, maxx, spacing_m)
    ys = np.arange(np.floor(miny / spacing_m) * spacing_m + spacing_m / 2, maxy, spacing_m)
    if len(xs) == 0 or len(ys) == 0:
        rp = geom.representative_point()
        return [{"event_id": event_id, "x": float(rp.x), "y": float(rp.y)}]
    xx, yy = np.meshgrid(xs, ys)
    hit = contains_xy(geom, xx, yy)
    y_idx, x_idx = np.where(hit)
    if len(x_idx) == 0:
        rp = geom.representative_point()
        return [{"event_id": event_id, "x": float(rp.x), "y": float(rp.y)}]
    coords = np.column_stack([xs[x_idx], ys[y_idx]])
    c = np.array([geom.centroid.x, geom.centroid.y])
    order = np.argsort(np.linalg.norm(coords - c, axis=1))
    return [{"event_id": event_id, "x": float(coords[i, 0]), "y": float(coords[i, 1])} for i in order[:cap]]


def read_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    module03 = load_event_module()
    fires = gpd.read_file(EVENT_GPKG, layer="fire_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    fires = fires[fires["disturbance_type"] == "fire_total"].copy()
    harvest = gpd.read_file(EVENT_GPKG, layer="harvest_events", engine="pyogrio").to_crs(CRS_PROJECTED)
    forestry_raw = module03.norm_cols(gpd.read_file(FORESTRY_GPKG, engine="pyogrio")).to_crs(CRS_PROJECTED)
    forestry = module03.classify_forestry_records(forestry_raw).dropna(subset=["disturbance_year"]).copy()
    forestry["disturbance_year"] = forestry["disturbance_year"].astype(int)
    fire_eligibility = pd.read_csv(FIRE_ELIGIBILITY)
    return fires, harvest, forestry, fire_eligibility


def fire_masks_for_event(fire: pd.Series, forestry: gpd.GeoDataFrame, sindex) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom = clean_geom(fire.geometry)
    xs, ys = raster_grid_for_geom(geom)
    xx, yy = np.meshgrid(xs, ys)
    fire_mask = contains_xy(geom, xx, yy)
    candidates = forestry.iloc[list(sindex.query(geom, predicate="intersects"))].copy()
    candidates = candidates[candidates.intersects(geom)].copy()
    fire_year = int(fire["disturbance_year"])
    prior_any = np.zeros(fire_mask.shape, dtype=bool)
    prior_major_20 = np.zeros(fire_mask.shape, dtype=bool)
    same_year = np.zeros(fire_mask.shape, dtype=bool)
    salvage_after = np.zeros(fire_mask.shape, dtype=bool)
    for _, cand in candidates.iterrows():
        hit_slice = geometry_hit_slice(cand.geometry, xs, ys)
        if hit_slice is None:
            continue
        y0, y1, x0, x1, hit = hit_slice
        hit = hit & fire_mask[y0:y1, x0:x1]
        if not hit.any():
            continue
        year = int(cand["disturbance_year"])
        cls = str(cand["project_class"])
        if year < fire_year:
            prior_any[y0:y1, x0:x1][hit] = True
            if cls == "harvest_total" and fire_year - year <= 20:
                prior_major_20[y0:y1, x0:x1][hit] = True
        elif year == fire_year:
            same_year[y0:y1, x0:x1][hit] = True
        if cls == "salvage_after_fire" and year >= fire_year:
            salvage_after[y0:y1, x0:x1][hit] = True
    definition_a = fire_mask & ~prior_any & ~same_year & ~salvage_after
    definition_d = fire_mask & ~prior_major_20 & ~same_year & ~salvage_after
    return xs, ys, fire_mask, definition_a, definition_d


def compute_fire_buffer_comparison(
    fires: gpd.GeoDataFrame, forestry: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows = []
    masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sindex = forestry.sindex
    for n, (_, fire) in enumerate(fires.iterrows(), start=1):
        if n % 20 == 0:
            print(f"  fire buffer metrics {n}/{len(fires)}", flush=True)
        xs, ys, fire_mask, definition_a, definition_d = fire_masks_for_event(fire, forestry, sindex)
        edge_distance = ndimage.distance_transform_edt(fire_mask) * CELL_SIZE_M
        for buffer_m in FIRE_BUFFERS_M:
            eligible = definition_a & (edge_distance >= buffer_m)
            rows.append(
                {
                    "event_id": fire["event_id"],
                    "disturbance_year": int(fire["disturbance_year"]),
                    "age_bin": age_bin(int(fire["disturbance_year"])),
                    "definition": DEFINITION_A,
                    "fire_buffer_m": buffer_m,
                    "eligible_area_ha": float(eligible.sum() * CELL_AREA_HA),
                    "largest_patch_ha": largest_patch_ha(eligible),
                    "retained": bool(eligible.any()),
                    "ge_1ha": largest_patch_ha(eligible) >= 1,
                    "ge_5ha": largest_patch_ha(eligible) >= 5,
                    "ge_10ha": largest_patch_ha(eligible) >= 10,
                    "ge_25ha": largest_patch_ha(eligible) >= 25,
                }
            )
            if buffer_m == PRIMARY_FIRE_BUFFER_M:
                masks[str(fire["event_id"])] = (xs, ys, eligible)
    return pd.DataFrame(rows), masks


def summarize_buffer_rows(df: pd.DataFrame, buffer_col: str) -> pd.DataFrame:
    return (
        df.groupby(buffer_col)
        .agg(
            events_retained=("retained", "sum"),
            eligible_area_ha=("eligible_area_ha", "sum"),
            median_eligible_area_per_event_ha=("eligible_area_ha", "median"),
            events_ge_1ha=("ge_1ha", "sum"),
            events_ge_5ha=("ge_5ha", "sum"),
            events_ge_10ha=("ge_10ha", "sum"),
            events_ge_25ha=("ge_25ha", "sum"),
        )
        .reset_index()
    )


def harvest_buffer_metrics(harvest: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, BaseGeometry]]:
    rows = []
    primary_geoms: dict[str, BaseGeometry] = {}
    for _, row in harvest.iterrows():
        geom = clean_geom(row.geometry)
        if geom is None:
            continue
        for buffer_m in HARVEST_BUFFERS_M:
            interior = clean_geom(geom.buffer(-buffer_m, quad_segs=1))
            usable = area_ha(interior)
            largest = usable if interior is not None else 0.0
            rows.append(
                {
                    "event_id": row["event_id"],
                    "disturbance_year": int(row["disturbance_year"]),
                    "age_bin": age_bin(int(row["disturbance_year"])),
                    "harvest_buffer_m": buffer_m,
                    "usable_area_ha": usable,
                    "largest_patch_ha": largest,
                    "retained": usable > 0,
                    "ge_1ha": largest >= 1,
                    "ge_5ha": largest >= 5,
                    "ge_10ha": largest >= 10,
                    "ge_25ha": largest >= 25,
                }
            )
            if buffer_m == PRIMARY_HARVEST_BUFFER_M and usable > 0:
                primary_geoms[str(row["event_id"])] = interior
    return pd.DataFrame(rows), primary_geoms


def candidate_pairs(
    fires: gpd.GeoDataFrame,
    harvest: gpd.GeoDataFrame,
    fire_buffer_rows: pd.DataFrame,
    harvest_buffer_rows: pd.DataFrame,
) -> pd.DataFrame:
    fire_usable = fire_buffer_rows[
        (fire_buffer_rows["fire_buffer_m"] == PRIMARY_FIRE_BUFFER_M)
        & (fire_buffer_rows["definition"] == DEFINITION_A)
        & (fire_buffer_rows["eligible_area_ha"] > 0)
    ][["event_id", "eligible_area_ha", "largest_patch_ha"]].rename(
        columns={"eligible_area_ha": "fire_usable_area_ha", "largest_patch_ha": "fire_largest_patch_ha"}
    )
    harvest_usable = harvest_buffer_rows[
        (harvest_buffer_rows["harvest_buffer_m"] == PRIMARY_HARVEST_BUFFER_M)
        & (harvest_buffer_rows["usable_area_ha"] > 0)
    ][["event_id", "usable_area_ha", "largest_patch_ha"]].rename(
        columns={"event_id": "harvest_event_id", "usable_area_ha": "harvest_usable_area_ha", "largest_patch_ha": "harvest_largest_patch_ha"}
    )
    fire_sel = fires.merge(fire_usable, on="event_id", how="inner").copy()
    harvest_sel = harvest.merge(harvest_usable, left_on="event_id", right_on="harvest_event_id", how="inner").copy()
    h_by_year = {int(y): g.copy() for y, g in harvest_sel.groupby("disturbance_year")}
    rows = []
    for _, fire in fire_sel.iterrows():
        fire_year = int(fire["disturbance_year"])
        fire_cent = fire.geometry.centroid
        for tolerance in [0, 1, 2]:
            years = [y for y in range(fire_year - tolerance, fire_year + tolerance + 1) if y in h_by_year]
            candidates = pd.concat([h_by_year[y] for y in years], ignore_index=True) if years else harvest_sel.iloc[[]].copy()
            if candidates.empty:
                continue
            for _, hrow in candidates.iterrows():
                age_diff = abs(int(hrow["disturbance_year"]) - fire_year)
                if age_diff > tolerance:
                    continue
                rows.append(
                    {
                        "fire_event_id": fire["event_id"],
                        "harvest_event_id": hrow["event_id"],
                        "fire_year": fire_year,
                        "harvest_year": int(hrow["disturbance_year"]),
                        "age_difference": int(age_diff),
                        "temporal_quality": "exact_year" if age_diff == 0 else f"pm{age_diff}_year",
                        "centroid_distance_m": float(fire_cent.distance(hrow.geometry.centroid)),
                        "fire_usable_area_ha": float(fire["fire_usable_area_ha"]),
                        "harvest_usable_area_ha": float(hrow["harvest_usable_area_ha"]),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["fire_event_id", "harvest_event_id"])


def select_matches(pairs: pd.DataFrame, harvests_per_fire: int) -> pd.DataFrame:
    selected = []
    used_harvest: set[str] = set()
    ranked = pairs.sort_values(["age_difference", "centroid_distance_m"]).copy()
    for fire_id, part in ranked.groupby("fire_event_id", sort=False):
        count = 0
        for _, row in part.iterrows():
            hid = str(row["harvest_event_id"])
            if hid in used_harvest:
                continue
            selected.append(row.to_dict() | {"strategy_harvests_per_fire": harvests_per_fire})
            used_harvest.add(hid)
            count += 1
            if count >= harvests_per_fire:
                break
    return pd.DataFrame(selected)


def matching_strategy_comparison(pairs: pd.DataFrame, selected_by_strategy: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ratio, sel in selected_by_strategy.items():
        if sel.empty:
            continue
        rows.append(
            {
                "harvests_per_fire": ratio,
                "fire_events_represented": int(sel["fire_event_id"].nunique()),
                "harvest_events_selected": int(sel["harvest_event_id"].nunique()),
                "exact_year_matches": int((sel["age_difference"] == 0).sum()),
                "pm1_year_matches": int((sel["age_difference"] == 1).sum()),
                "pm2_year_matches": int((sel["age_difference"] == 2).sum()),
                "median_pair_distance_m": float(sel["centroid_distance_m"].median()),
                "p90_pair_distance_m": float(sel["centroid_distance_m"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def make_sites(
    selected_matches: pd.DataFrame,
    fires: gpd.GeoDataFrame,
    harvest: gpd.GeoDataFrame,
    fire_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    harvest_geoms: dict[str, BaseGeometry],
    cap: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    selected_fire_ids = sorted(set(selected_matches["fire_event_id"]))
    selected_harvest_ids = sorted(set(selected_matches["harvest_event_id"]))
    fire_meta = fires.set_index("event_id")
    harvest_meta = harvest.set_index("event_id")
    fire_rows = []
    harvest_rows = []
    for event_id in selected_fire_ids:
        xs, ys, mask = fire_masks[event_id]
        pts = greedy_points_from_mask(mask, xs, ys, cap, SITE_SPACING_M, event_id)
        event = fire_meta.loc[event_id]
        for i, pt in enumerate(pts, start=1):
            censor_year = event.get("censor_year", pd.NA)
            fire_rows.append(
                {
                    "disturbance_type": "fire_total",
                    "event_id": event_id,
                    "site_id": f"fire_{event_id}_site_{i:03d}",
                    "disturbance_year": int(event["disturbance_year"]),
                    "years_since_disturbance": OBSERVATION_YEAR - int(event["disturbance_year"]),
                    "age_bin": age_bin(int(event["disturbance_year"])),
                    "site_cap": cap,
                    "spacing_m": SITE_SPACING_M,
                    "censor_year": censor_year,
                    "censor_reason": "later_major_disturbance_or_salvage" if pd.notna(censor_year) else "",
                    "geometry": Point(pt["x"], pt["y"]),
                }
            )
    for event_id in selected_harvest_ids:
        geom = harvest_geoms.get(event_id)
        pts = greedy_points_in_geom(geom, cap, SITE_SPACING_M, event_id)
        event = harvest_meta.loc[event_id]
        for i, pt in enumerate(pts, start=1):
            censor_year = event.get("censor_year", pd.NA)
            harvest_rows.append(
                {
                    "disturbance_type": "harvest_total",
                    "event_id": event_id,
                    "site_id": f"harvest_{event_id}_site_{i:03d}",
                    "disturbance_year": int(event["disturbance_year"]),
                    "years_since_disturbance": OBSERVATION_YEAR - int(event["disturbance_year"]),
                    "age_bin": age_bin(int(event["disturbance_year"])),
                    "site_cap": cap,
                    "spacing_m": SITE_SPACING_M,
                    "censor_year": censor_year,
                    "censor_reason": "second_total_harvest" if pd.notna(censor_year) else "",
                    "replanted_after_harvest": event.get("replanted_after_harvest", False),
                    "thinning_after_harvest": event.get("thinning_after_harvest", False),
                    "geometry": Point(pt["x"], pt["y"]),
                }
            )
    return (
        gpd.GeoDataFrame(fire_rows, geometry="geometry", crs=CRS_PROJECTED),
        gpd.GeoDataFrame(harvest_rows, geometry="geometry", crs=CRS_PROJECTED),
    )


def reference_search_areas(sites: gpd.GeoDataFrame, event_geoms: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    event_lookup = event_geoms.set_index("event_id")
    rows = []
    for _, site in sites.iterrows():
        annulus = site.geometry.buffer(REFERENCE_OUTER_RADIUS_M).difference(site.geometry.buffer(REFERENCE_INNER_RADIUS_M))
        event_geom = event_lookup.loc[site["event_id"]].geometry
        search = clean_geom(annulus.difference(event_geom))
        if search is None:
            continue
        rows.append(
            {
                "reference_search_id": f"ref_{site['site_id']}",
                "associated_site_id": site["site_id"],
                "associated_event_id": site["event_id"],
                "candidate_only": True,
                "inner_radius_m": REFERENCE_INNER_RADIUS_M,
                "outer_radius_m": REFERENCE_OUTER_RADIUS_M,
                "geometry": search,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_PROJECTED)


def site_cap_sensitivity(
    selected_matches: pd.DataFrame,
    fires: gpd.GeoDataFrame,
    harvest: gpd.GeoDataFrame,
    fire_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    harvest_geoms: dict[str, BaseGeometry],
) -> pd.DataFrame:
    rows = []
    for cap in SITE_CAPS:
        fire_sites, harvest_sites = make_sites(selected_matches, fires, harvest, fire_masks, harvest_geoms, cap)
        for design in ["equal_max_sites_per_event", "area_informed_strict_cap"]:
            if design == "equal_max_sites_per_event":
                fs, hs = fire_sites, harvest_sites
            else:
                fs = fire_sites.groupby("event_id").head(cap).copy()
                hs = harvest_sites.groupby("event_id").head(cap).copy()
            rows.append(
                {
                    "site_cap": cap,
                    "design": design,
                    "fire_events": int(fs["event_id"].nunique()) if not fs.empty else 0,
                    "harvest_events": int(hs["event_id"].nunique()) if not hs.empty else 0,
                    "fire_sites": int(len(fs)),
                    "harvest_sites": int(len(hs)),
                    "site_ratio_harvest_to_fire": float(len(hs) / len(fs)) if len(fs) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def temporal_balance(events_or_sites: pd.DataFrame, unit_col: str, output_kind: str) -> pd.DataFrame:
    if events_or_sites.empty:
        return pd.DataFrame()
    return (
        events_or_sites.groupby(["disturbance_type", "disturbance_year", "age_bin"])
        .agg(count=(unit_col, "nunique"))
        .reset_index()
        .assign(output_kind=output_kind)
    )


def write_figures(
    fire_buffer: pd.DataFrame,
    harvest_buffer: pd.DataFrame,
    selected_matches: pd.DataFrame,
    fires: gpd.GeoDataFrame,
    harvest: gpd.GeoDataFrame,
    temporal_events: pd.DataFrame,
    temporal_sites: pd.DataFrame,
    age_events: pd.DataFrame,
) -> None:
    fire_summary = summarize_buffer_rows(fire_buffer, "fire_buffer_m")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fire_summary["fire_buffer_m"].astype(str), fire_summary["events_retained"], color="#b55239")
    ax.set_xlabel("Fire interior buffer (m)")
    ax.set_ylabel("Retained fire events")
    ax.set_title("Fire-event retention by interior buffer")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fire_event_retention_by_buffer.png", dpi=220)
    plt.close(fig)

    harvest_summary = summarize_buffer_rows(harvest_buffer.rename(columns={"harvest_buffer_m": "buffer_m", "usable_area_ha": "eligible_area_ha"}), "buffer_m")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(harvest_summary["buffer_m"].astype(str), harvest_summary["events_retained"], color="#327c73")
    ax.set_xlabel("Harvest interior buffer (m)")
    ax.set_ylabel("Retained harvest events")
    ax.set_title("Harvest-event retention by interior buffer")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "harvest_event_retention_by_buffer.png", dpi=220)
    plt.close(fig)

    fire_sel = fires[fires["event_id"].isin(selected_matches["fire_event_id"])]
    harvest_sel = harvest[harvest["event_id"].isin(selected_matches["harvest_event_id"])]
    fig, ax = plt.subplots(figsize=(9, 9))
    if not fire_sel.empty:
        fire_sel.boundary.plot(ax=ax, color="#c44e33", linewidth=0.7, alpha=0.8, label="selected fire events")
    if not harvest_sel.empty:
        harvest_sel.boundary.plot(ax=ax, color="#2b7a78", linewidth=0.5, alpha=0.7, label="selected harvest events")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("Fire-anchored matched event frame")
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fire_harvest_matched_event_map.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    selected_matches["centroid_distance_m"].plot(kind="hist", bins=20, ax=ax, color="#59656f")
    ax.set_xlabel("Centroid distance (m)")
    ax.set_title("Pair-distance distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pair_distance_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    temporal_events.pivot_table(index="disturbance_year", columns="disturbance_type", values="count", fill_value=0).plot(kind="bar", ax=ax)
    ax.set_xlabel("Disturbance year")
    ax.set_ylabel("Event count")
    ax.set_title("Selected events by disturbance year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_counts_by_disturbance_year.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    temporal_sites.pivot_table(index="disturbance_year", columns="disturbance_type", values="count", fill_value=0).plot(kind="bar", ax=ax)
    ax.set_xlabel("Disturbance year")
    ax.set_ylabel("Site count")
    ax.set_title("Candidate sites by disturbance year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "site_counts_by_disturbance_year.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    age_events.pivot_table(index="age_bin", columns="disturbance_type", values="count", fill_value=0).plot(kind="bar", ax=ax)
    ax.set_xlabel("Age bin")
    ax.set_ylabel("Event count")
    ax.set_title("Age-bin balance by disturbance type")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_bin_balance_by_disturbance_type.png", dpi=220)
    plt.close(fig)


def write_readme(summary: dict) -> None:
    text = [
        "# Recovery sampling frame",
        "",
        "This directory contains an event-balanced sampling frame for a future recovery experiment. It stops before remote-sensing extraction.",
        "",
        "The primary design is fire-anchored: wildfire events define the limiting replication, and harvest events are selected as temporally and geographically comparable candidates rather than using all available harvest events.",
        "",
        "Primary fire eligibility uses Definition A from the cohort audit: no mapped forestry intervention before wildfire. Definition D is retained as a sensitivity concept only.",
        "",
        "Reference-search areas are candidate-only annuli around selected sites. They are not claimed to be undisturbed forest until future imagery/forest-condition filtering is performed.",
        "",
        "## Summary",
        "",
        json.dumps(summary, indent=2),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Reading local event histories and forestry vectors...", flush=True)
    fires, harvest, forestry, fire_eligibility = read_inputs()

    print("Computing fire interior-buffer comparison for Definition A...", flush=True)
    fire_buffer, fire_masks = compute_fire_buffer_comparison(fires, forestry)
    fire_buffer.to_csv(OUT_DIR / "fire_buffer_comparison.csv", index=False)

    print("Computing harvest interior-buffer feasibility...", flush=True)
    harvest_buffer, harvest_geoms = harvest_buffer_metrics(harvest)
    harvest_buffer.to_csv(OUT_DIR / "harvest_buffer_comparison.csv", index=False)

    fire_buffer_summary = summarize_buffer_rows(fire_buffer, "fire_buffer_m")
    harvest_buffer_summary = summarize_buffer_rows(
        harvest_buffer.rename(columns={"harvest_buffer_m": "buffer_m", "usable_area_ha": "eligible_area_ha"}), "buffer_m"
    )
    fire_recommendation = {
        "recommended_fire_buffer_m": PRIMARY_FIRE_BUFFER_M,
        "reason": "60 m retains substantially more temporal/event replication than 90 m while still excluding immediate fire-boundary pixels; 150 m is retained as sensitivity only.",
        "comparison": fire_buffer_summary.to_dict(orient="records"),
    }
    harvest_recommendation = {
        "recommended_harvest_buffer_m": PRIMARY_HARVEST_BUFFER_M,
        "reason": "Harvest events are small, with median event area near 5 ha; 30 m preserves representation while still avoiding the mapped event edge.",
        "comparison": harvest_buffer_summary.to_dict(orient="records"),
    }
    (OUT_DIR / "fire_buffer_recommendation.json").write_text(json.dumps(fire_recommendation, indent=2), encoding="utf-8")
    (OUT_DIR / "harvest_buffer_recommendation.json").write_text(json.dumps(harvest_recommendation, indent=2), encoding="utf-8")

    print("Building fire-anchored harvest candidate pairs...", flush=True)
    pairs = candidate_pairs(fires, harvest, fire_buffer, harvest_buffer)
    pairs.to_csv(OUT_DIR / "fire_harvest_candidate_pairs.csv", index=False)
    selected_by_strategy = {ratio: select_matches(pairs, ratio) for ratio in [1, 3, 5]}
    strategy = matching_strategy_comparison(pairs, selected_by_strategy)
    strategy.to_csv(OUT_DIR / "matching_strategy_comparison.csv", index=False)
    selected = selected_by_strategy[PRIMARY_HARVESTS_PER_FIRE]

    selected_fires = fires[fires["event_id"].isin(selected["fire_event_id"])].copy()
    selected_harvest = harvest[harvest["event_id"].isin(selected["harvest_event_id"])].copy()
    selected_fires.drop(columns="geometry").to_csv(OUT_DIR / "selected_fire_events.csv", index=False)
    selected_harvest.drop(columns="geometry").to_csv(OUT_DIR / "selected_harvest_events.csv", index=False)

    print("Generating candidate sampling sites for primary design...", flush=True)
    fire_sites, harvest_sites = make_sites(selected, fires, harvest, fire_masks, harvest_geoms, PRIMARY_SITE_CAP)
    fire_sites.to_file(OUT_DIR / "candidate_fire_sites.gpkg", layer="candidate_fire_sites", driver="GPKG")
    harvest_sites.to_file(OUT_DIR / "candidate_harvest_sites.gpkg", layer="candidate_harvest_sites", driver="GPKG")
    all_sites = pd.concat([fire_sites, harvest_sites], ignore_index=True)
    event_geoms = pd.concat([fires[["event_id", "geometry"]], harvest[["event_id", "geometry"]]], ignore_index=True)
    event_geoms = gpd.GeoDataFrame(event_geoms, geometry="geometry", crs=CRS_PROJECTED)
    refs = reference_search_areas(gpd.GeoDataFrame(all_sites, geometry="geometry", crs=CRS_PROJECTED), event_geoms)
    refs.to_file(OUT_DIR / "candidate_reference_search_areas.gpkg", layer="candidate_reference_search_areas", driver="GPKG")

    sensitivity = site_cap_sensitivity(selected, fires, harvest, fire_masks, harvest_geoms)
    sensitivity.to_csv(OUT_DIR / "site_cap_sensitivity.csv", index=False)

    selected_event_table = pd.concat(
        [
            selected_fires.assign(disturbance_type="fire_total")[["disturbance_type", "event_id", "disturbance_year"]],
            selected_harvest.assign(disturbance_type="harvest_total")[["disturbance_type", "event_id", "disturbance_year"]],
        ],
        ignore_index=True,
    )
    selected_event_table["age_bin"] = selected_event_table["disturbance_year"].apply(age_bin)
    temporal_events = temporal_balance(selected_event_table, "event_id", "events")
    temporal_events.to_csv(OUT_DIR / "temporal_balance_events.csv", index=False)
    site_table = pd.concat(
        [
            fire_sites.drop(columns="geometry"),
            harvest_sites.drop(columns="geometry"),
        ],
        ignore_index=True,
    )
    temporal_sites = temporal_balance(site_table, "site_id", "sites")
    temporal_sites.to_csv(OUT_DIR / "temporal_balance_sites.csv", index=False)
    age_events = selected_event_table.groupby(["disturbance_type", "age_bin"]).agg(count=("event_id", "nunique")).reset_index()
    age_sites = site_table.groupby(["disturbance_type", "age_bin"]).agg(count=("site_id", "nunique")).reset_index()

    summary = {
        "recommended_primary_design": {
            "fire_interior_buffer_m": PRIMARY_FIRE_BUFFER_M,
            "harvest_interior_buffer_m": PRIMARY_HARVEST_BUFFER_M,
            "harvests_per_fire_event": PRIMARY_HARVESTS_PER_FIRE,
            "maximum_sites_per_event": PRIMARY_SITE_CAP,
            "minimum_spatial_separation_between_sites_m": SITE_SPACING_M,
            "temporal_tolerance": "same year first, ±1 year if needed, ±2 years as fallback/sensitivity",
            "reference_search_geometry": f"{REFERENCE_INNER_RADIUS_M}-{REFERENCE_OUTER_RADIUS_M} m annulus around each selected site, with the disturbed event geometry removed; candidate_only=True",
        },
        "fire_events_retained_primary_buffer": int(fire_buffer[(fire_buffer["fire_buffer_m"] == PRIMARY_FIRE_BUFFER_M) & (fire_buffer["eligible_area_ha"] > 0)]["event_id"].nunique()),
        "fire_events_represented_in_primary_matching": int(selected["fire_event_id"].nunique()),
        "harvest_events_selected_primary": int(selected["harvest_event_id"].nunique()),
        "candidate_fire_sites": int(len(fire_sites)),
        "candidate_harvest_sites": int(len(harvest_sites)),
        "candidate_reference_search_areas": int(len(refs)),
        "matching_strategy_summary": strategy.to_dict(orient="records"),
        "remaining_uncertainties": [
            "Fire eligibility is based on official mapped forestry history and does not yet include Hansen or spectral confirmation.",
            "Harvest events are provisional connected harvest_total components because NO_SEC_INT is not exposed by the WFS subset.",
            "Reference-search areas are candidate-only until future forest-condition and disturbance-screening layers are applied.",
            "The inferential unit remains the event; candidate sites are subsamples.",
        ],
    }
    (OUT_DIR / "sampling_frame_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)

    write_figures(fire_buffer, harvest_buffer, selected, fires, harvest, temporal_events, temporal_sites, age_events)

    print("FIRE BUFFER SUMMARY")
    print(fire_buffer_summary.to_string(index=False))
    print("\nHARVEST BUFFER SUMMARY")
    print(harvest_buffer_summary.to_string(index=False))
    print("\nMATCHING STRATEGY COMPARISON")
    print(strategy.to_string(index=False))
    print("\nEVENT BALANCE BY AGE BIN")
    print(age_events.to_string(index=False))
    print("\nSITE BALANCE BY AGE BIN")
    print(age_sites.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
