#!/usr/bin/env python3
"""Select a defensible Lebel-sur-Quevillon study extent for fire recovery work.

This is an extent-screening analysis only. It uses the local Quebec fire
GeoPackage already present under data/raw/fire and does not download forestry,
Sentinel-2, Landsat, AlphaEarth, or any new remote-sensing product.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parents[1]
FIRE_GPKG = REPO_ROOT / "data" / "raw" / "fire" / "FEUX_PROV.gpkg"
OUT_DIR = REPO_ROOT / "outputs" / "study_extent_selection"
FIG_DIR = REPO_ROOT / "figures" / "study_extent_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt

OBSERVATION_YEAR = 2024
START_YEAR = 2001
END_YEAR = 2023
CRS_PROJECTED = "EPSG:32198"
CRS_GEOGRAPHIC = "EPSG:4269"
RADII_KM = [25, 50, 75, 100, 125, 150, 200]

# Natural Resources Canada / Canadian Geographical Names Database, key EGYQT:
# Lebel-sur-Quevillon, official, decimal latitude-longitude 49.052083, -76.976052.
ANCHOR_SOURCE = "Natural Resources Canada Canadian Geographical Names Database, key EGYQT; provider Quebec-Commission de Toponymie."
ANCHOR_LON = -76.976052
ANCHOR_LAT = 49.052083

AGE_BINS = [
    ("0-5", 0, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16-20", 16, 20),
    ("21-23", 21, 23),
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def anchor_point_projected() -> Point:
    transformer = Transformer.from_crs(CRS_GEOGRAPHIC, CRS_PROJECTED, always_xy=True)
    x, y = transformer.transform(ANCHOR_LON, ANCHOR_LAT)
    return Point(x, y)


def primary_year(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in ["an_origine", "an_perturb", "exercice"]:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def read_fire_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not FIRE_GPKG.exists():
        raise FileNotFoundError(f"Missing required local fire GeoPackage: {FIRE_GPKG}")
    feux = normalize_columns(gpd.read_file(FIRE_GPKG, layer="feux_prov", engine="pyogrio"))
    meta = normalize_columns(gpd.read_file(FIRE_GPKG, layer="meta_feux_prov", engine="pyogrio"))
    if feux.crs is None or feux.crs.to_string() != CRS_PROJECTED:
        feux = feux.to_crs(CRS_PROJECTED)
    if meta.crs is None or meta.crs.to_string() != CRS_PROJECTED:
        meta = meta.to_crs(CRS_PROJECTED)
    return feux, meta


def grouping_diagnostics(feux: gpd.GeoDataFrame, meta: gpd.GeoDataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    for keys in [["geoc_fmj"], ["geoc_fmj", "exercice"]]:
        feux_dupes = int(feux.duplicated(keys).sum())
        meta_dupes = int(meta.duplicated(keys).sum())
        joined = feux.drop(columns="geometry").merge(
            meta.drop(columns="geometry")[keys + ["nopert_pee"]],
            on=keys,
            how="left",
            indicator=True,
        )
        rows.append(
            {
                "join_keys": " + ".join(keys),
                "feux_duplicate_key_rows": feux_dupes,
                "meta_duplicate_key_rows": meta_dupes,
                "joined_rows": int(len(joined)),
                "left_only_rows": int((joined["_merge"] == "left_only").sum()),
                "nopert_pee_null_rows": int(joined["nopert_pee"].isna().sum()),
                "unique_nopert_pee": int(joined["nopert_pee"].nunique(dropna=True)),
                "usable_for_event_transfer": bool(
                    len(joined) == len(feux)
                    and feux_dupes == 0
                    and meta_dupes == 0
                    and (joined["_merge"] == "left_only").sum() == 0
                    and joined["nopert_pee"].isna().sum() == 0
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    usable = bool(diagnostics.loc[diagnostics["join_keys"] == "geoc_fmj + exercice", "usable_for_event_transfer"].iloc[0])
    return diagnostics, usable


def attach_event_id(feux: gpd.GeoDataFrame, meta: gpd.GeoDataFrame, use_official_join: bool) -> gpd.GeoDataFrame:
    gdf = feux.copy()
    if use_official_join:
        meta_cols = meta.drop(columns="geometry")[["geoc_fmj", "exercice", "nopert_pee"]]
        gdf = gdf.merge(meta_cols, on=["geoc_fmj", "exercice"], how="left")
        gdf["event_grouping_method"] = "NOPERT_PEE transferred by exact GEOC_FMJ + EXERCICE join"
        gdf["event_id"] = gdf["nopert_pee"].astype(str)
    else:
        gdf["event_grouping_method"] = "provisional polygon-level grouping; no official event identifier transferred"
        gdf["event_id"] = "polygon_" + gdf.index.astype(str)
    gdf["disturbance_year"] = primary_year(gdf).astype("Int64")
    gdf["years_since_disturbance"] = OBSERVATION_YEAR - gdf["disturbance_year"].astype(float)
    return gdf


def age_bin(age: float) -> str | None:
    if pd.isna(age):
        return None
    for label, lo, hi in AGE_BINS:
        if lo <= age <= hi:
            return label
    return None


def event_inventory_for_radius(fires: gpd.GeoDataFrame, anchor: Point, radius_km: int) -> gpd.GeoDataFrame:
    extent = anchor.buffer(radius_km * 1000)
    idx = list(fires.sindex.query(extent, predicate="intersects"))
    sel = fires.iloc[idx].copy()
    if sel.empty:
        return gpd.GeoDataFrame(columns=["event_id", "disturbance_year", "geometry"], geometry="geometry", crs=CRS_PROJECTED)
    sel["clip_geom"] = sel.geometry.intersection(extent)
    sel = sel[~sel["clip_geom"].is_empty].copy()
    sel["clip_area_ha"] = sel["clip_geom"].area / 10_000
    sel = sel[sel["clip_area_ha"] > 0].copy()
    records = []
    grouped = sel.groupby(["event_id", "disturbance_year"], dropna=False)
    for (event_id, year), part in grouped:
        geom = unary_union(list(part["clip_geom"]))
        area_ha = geom.area / 10_000
        if area_ha <= 0:
            continue
        records.append(
            {
                "candidate_radius_km": radius_km,
                "event_id": str(event_id),
                "disturbance_year": int(year) if not pd.isna(year) else None,
                "years_since_disturbance": OBSERVATION_YEAR - int(year) if not pd.isna(year) else None,
                "age_bin": age_bin(OBSERVATION_YEAR - int(year)) if not pd.isna(year) else None,
                "polygon_count": int(len(part)),
                "burned_area_ha": float(area_ha),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)


def summarize_candidate(events: gpd.GeoDataFrame, radius_km: int, anchor: Point) -> dict:
    extent_area_ha = anchor.buffer(radius_km * 1000).area / 10_000
    if events.empty:
        return {
            "radius_km": radius_km,
            "extent_area_ha": extent_area_ha,
            "extent_area_km2": extent_area_ha / 100,
            "fire_polygons": 0,
            "independent_fire_events": 0,
            "total_burned_ha": 0.0,
            "distinct_fire_years": 0,
            "earliest_fire_year": np.nan,
            "latest_fire_year": np.nan,
            "age_min_years": np.nan,
            "age_max_years": np.nan,
            "largest_event_area_ha": 0.0,
            "largest_event_pct_burned_area": np.nan,
            "top3_event_pct_burned_area": np.nan,
            "balanced_age_bins_with_3_events": 0,
            "age_bins_represented": 0,
            "period_span_years": 0,
            "meets_screening_target": False,
            "dominance_flag": "none",
        }
    event_areas = events["burned_area_ha"].sort_values(ascending=False)
    total = float(event_areas.sum())
    age_counts = events.groupby("age_bin")["event_id"].nunique()
    years = events["disturbance_year"].dropna().astype(int)
    largest_pct = float(event_areas.iloc[0] / total * 100) if total else np.nan
    top3_pct = float(event_areas.head(3).sum() / total * 100) if total else np.nan
    all_bins_have_3 = all(age_counts.get(label, 0) >= 3 for label, _, _ in AGE_BINS)
    meets = bool(
        len(events) >= 10
        and years.nunique() >= 10
        and int(years.min()) <= 2003
        and int(years.max()) >= 2021
        and all_bins_have_3
    )
    return {
        "radius_km": radius_km,
        "extent_area_ha": extent_area_ha,
        "extent_area_km2": extent_area_ha / 100,
        "fire_polygons": int(events["polygon_count"].sum()),
        "independent_fire_events": int(events["event_id"].nunique()),
        "event_year_units": int(len(events)),
        "total_burned_ha": total,
        "distinct_fire_years": int(years.nunique()),
        "earliest_fire_year": int(years.min()) if not years.empty else np.nan,
        "latest_fire_year": int(years.max()) if not years.empty else np.nan,
        "age_min_years": float(events["years_since_disturbance"].min()),
        "age_max_years": float(events["years_since_disturbance"].max()),
        "largest_event_area_ha": float(event_areas.iloc[0]) if len(event_areas) else 0.0,
        "largest_event_pct_burned_area": largest_pct,
        "top3_event_pct_burned_area": top3_pct,
        "balanced_age_bins_with_3_events": int(sum(age_counts.get(label, 0) >= 3 for label, _, _ in AGE_BINS)),
        "age_bins_represented": int(sum(age_counts.get(label, 0) > 0 for label, _, _ in AGE_BINS)),
        "period_span_years": int(years.max() - years.min()) if not years.empty else 0,
        "meets_screening_target": meets,
        "dominance_flag": "dominated_by_top3" if top3_pct >= 75 else ("dominated_by_largest" if largest_pct >= 50 else "not_strongly_dominated"),
    }


def age_bin_table(events_by_radius: list[gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    for events in events_by_radius:
        if events.empty:
            continue
        radius = int(events["candidate_radius_km"].iloc[0])
        for label, _, _ in AGE_BINS:
            part = events[events["age_bin"] == label]
            rows.append(
                {
                    "radius_km": radius,
                    "age_bin": label,
                    "event_count": int(part["event_id"].nunique()),
                    "distinct_disturbance_years": int(part["disturbance_year"].nunique()),
                    "burned_hectares": float(part["burned_area_ha"].sum()),
                }
            )
    return pd.DataFrame(rows)


def select_radius(summary: pd.DataFrame) -> int:
    meets = summary[summary["meets_screening_target"]].sort_values("radius_km")
    if not meets.empty:
        return int(meets.iloc[0]["radius_km"])
    ranked = summary.copy()
    ranked["score"] = (
        ranked["balanced_age_bins_with_3_events"] * 100
        + ranked["age_bins_represented"] * 20
        + ranked["distinct_fire_years"] * 2
        + ranked["independent_fire_events"].clip(upper=20)
        + ranked["period_span_years"]
        - (ranked["top3_event_pct_burned_area"].fillna(100) / 10)
        - (ranked["radius_km"] / 100)
    )
    return int(ranked.sort_values(["score", "radius_km"], ascending=[False, True]).iloc[0]["radius_km"])


def bounds_tables(geom_projected) -> dict:
    g = gpd.GeoDataFrame([{"geometry": geom_projected}], geometry="geometry", crs=CRS_PROJECTED)
    wgs = g.to_crs(CRS_GEOGRAPHIC)
    return {
        "epsg_32198_bounds": [float(x) for x in g.total_bounds],
        "wgs84_bounds": [float(x) for x in wgs.total_bounds],
        "geojson_geometry_wgs84": mapping(wgs.geometry.iloc[0]),
    }


def write_recommended_geojson(anchor: Point, radius_km: int) -> dict:
    geom = anchor.buffer(radius_km * 1000)
    gdf = gpd.GeoDataFrame(
        [
            {
                "name": f"lebel_recovery_study_extent_{radius_km}km",
                "anchor": "Lebel-sur-Quevillon",
                "radius_km": radius_km,
                "area_ha_epsg_32198": geom.area / 10_000,
                "anchor_lon": ANCHOR_LON,
                "anchor_lat": ANCHOR_LAT,
                "anchor_source": ANCHOR_SOURCE,
                "geometry": geom,
            }
        ],
        geometry="geometry",
        crs=CRS_PROJECTED,
    )
    path = OUT_DIR / "recommended_study_extent.geojson"
    gdf.to_crs(CRS_GEOGRAPHIC).to_file(path, driver="GeoJSON")
    return bounds_tables(geom)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Small Markdown table writer to avoid adding tabulate as a dependency."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}" if np.isfinite(value) else "")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_figures(fires_200: gpd.GeoDataFrame, events_all: list[gpd.GeoDataFrame], summary: pd.DataFrame, age_bins: pd.DataFrame, anchor: Point, selected_radius: int) -> None:
    # 1. Map with fires and candidate extents.
    fig, ax = plt.subplots(figsize=(10, 10))
    if not fires_200.empty:
        fires_200.plot(
            ax=ax,
            column="disturbance_year",
            cmap="inferno_r",
            linewidth=0.05,
            edgecolor="none",
            alpha=0.65,
            legend=True,
            legend_kwds={"label": "Fire year"},
        )
    for radius in RADII_KM:
        ring = gpd.GeoSeries([anchor.buffer(radius * 1000)], crs=CRS_PROJECTED)
        ring.boundary.plot(ax=ax, color="#234f3a" if radius != selected_radius else "#00a6a6", linewidth=0.8 if radius != selected_radius else 2.0)
        ax.text(anchor.x + radius * 1000, anchor.y, f"{radius} km", fontsize=7, color="#234f3a")
    ax.scatter([anchor.x], [anchor.y], marker="*", s=120, color="#111111", zorder=5)
    ax.set_title("Lebel-sur-Quevillon candidate recovery-study extents\nQuebec fire polygons, 2001-2023")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "candidate_extents_fire_map.png", dpi=220)
    plt.close(fig)

    # 2. Temporal coverage by year and radius.
    inv = pd.concat([e.drop(columns="geometry") for e in events_all if not e.empty], ignore_index=True)
    year_counts = inv.groupby(["candidate_radius_km", "disturbance_year"])["event_id"].nunique().reset_index(name="events")
    fig, ax = plt.subplots(figsize=(12, 6))
    for radius in RADII_KM:
        part = year_counts[year_counts["candidate_radius_km"] == radius]
        ax.plot(part["disturbance_year"], part["events"], marker="o", linewidth=1.4, label=f"{radius} km")
    ax.set_xlabel("Disturbance year")
    ax.set_ylabel("Independent fire events")
    ax.set_title("Temporal coverage of independent wildfire events by candidate radius")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_coverage_by_radius.png", dpi=220)
    plt.close(fig)

    # 3. Age-bin replication.
    pivot = age_bins.pivot(index="radius_km", columns="age_bin", values="event_count").reindex(RADII_KM)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot[[label for label, _, _ in AGE_BINS]].plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(3, color="#111111", linestyle="--", linewidth=1, label="screening target: 3 events")
    ax.set_xlabel("Candidate radius")
    ax.set_ylabel("Independent fire events")
    ax.set_title("Fire-event replication by recovery-age bin")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_bin_replication.png", dpi=220)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, age_bins: pd.DataFrame, selected: int, bounds: dict, diagnostics: pd.DataFrame) -> None:
    selected_row = summary[summary["radius_km"] == selected].iloc[0].to_dict()
    selected_bins = age_bins[age_bins["radius_km"] == selected]
    lines = [
        "# Study extent selection around Lebel-sur-Quevillon",
        "",
        "This directory contains a fire-only screening analysis for choosing the smallest defensible study extent before retrieving the large Quebec forestry-intervention dataset.",
        "",
        "## Anchor",
        "",
        f"- Coordinates: {ANCHOR_LAT}, {ANCHOR_LON}",
        f"- Source: {ANCHOR_SOURCE}",
        f"- Working CRS: {CRS_PROJECTED}",
        "",
        "## Event grouping result",
        "",
        "The fire polygon layer `FEUX_PROV` and metadata layer `META_FEUX_PROV` both contain `GEOC_FMJ` and `EXERCICE`.",
        "`GEOC_FMJ` alone is not unique, but `GEOC_FMJ + EXERCICE` is a complete one-to-one join in the inspected GeoPackage.",
        "Therefore `NOPERT_PEE` can be transferred to fire polygons for event-level screening.",
        "Events with multiple associated years remain flagged as a limitation and should be reviewed before final inference.",
        "",
        df_to_markdown(diagnostics),
        "",
        "## Recommendation",
        "",
        f"Selected radius: **{selected} km**.",
        f"Area: **{selected_row['extent_area_ha']:.1f} ha** ({selected_row['extent_area_km2']:.1f} km2).",
        f"Independent/provisional wildfire events: **{int(selected_row['independent_fire_events'])}**.",
        f"Distinct fire years: **{int(selected_row['distinct_fire_years'])}**.",
        f"Age range: **{selected_row['age_min_years']:.0f}-{selected_row['age_max_years']:.0f} years since disturbance in 2024**.",
        f"Largest-event dominance: **{selected_row['largest_event_pct_burned_area']:.1f}%** of burned area; top three events: **{selected_row['top3_event_pct_burned_area']:.1f}%**.",
        "",
        "## Age-bin replication for selected radius",
        "",
        df_to_markdown(selected_bins),
        "",
        "## Bounds for subsequent forestry-data query",
        "",
        f"- EPSG:32198 bounds: `{bounds['epsg_32198_bounds']}`",
        f"- WGS84 bounds: `{bounds['wgs84_bounds']}`",
        "",
        "## Limitations",
        "",
        "- This phase uses fire data only and does not retrieve or interpret forestry interventions.",
        "- The age bins are screening diagnostics, not final statistical model bins.",
        "- Fire-event grouping uses `NOPERT_PEE` transferred through the validated `GEOC_FMJ + EXERCICE` join; final inference should still document this relationship.",
        "- No forest-cover stratification was performed in this phase; landscape sanity is limited to geographic extent and fire chronology.",
        "- The recommended extent is a circular search/study extent, not a final ecological stratum.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feux, meta = read_fire_layers()
    diagnostics, official_join = grouping_diagnostics(feux, meta)
    diagnostics.to_csv(OUT_DIR / "fire_event_grouping_diagnostics.csv", index=False)

    fires = attach_event_id(feux, meta, official_join)
    fires = fires[(fires["disturbance_year"] >= START_YEAR) & (fires["disturbance_year"] <= END_YEAR)].copy()
    anchor = anchor_point_projected()
    max_extent = anchor.buffer(max(RADII_KM) * 1000)
    fires_200 = fires.iloc[list(fires.sindex.query(max_extent, predicate="intersects"))].copy()

    events_all = []
    summary_rows = []
    for radius in RADII_KM:
        events = event_inventory_for_radius(fires, anchor, radius)
        events_all.append(events)
        summary_rows.append(summarize_candidate(events, radius, anchor))

    summary = pd.DataFrame(summary_rows)
    age_bins = age_bin_table(events_all)
    selected = select_radius(summary)
    selected_events = [e for e in events_all if not e.empty and int(e["candidate_radius_km"].iloc[0]) == selected][0]

    event_inventory = pd.concat([e.drop(columns="geometry") for e in events_all if not e.empty], ignore_index=True)
    event_inventory.to_csv(OUT_DIR / "fire_event_inventory.csv", index=False)
    summary.to_csv(OUT_DIR / "candidate_extent_summary.csv", index=False)
    age_bins.to_csv(OUT_DIR / "candidate_extent_age_bins.csv", index=False)
    bounds = write_recommended_geojson(anchor, selected)
    make_figures(fires_200, events_all, summary, age_bins, anchor, selected)
    write_readme(summary, age_bins, selected, bounds, diagnostics)

    selected_years = (
        selected_events.groupby("disturbance_year")
        .agg(events=("event_id", "nunique"), burned_hectares=("burned_area_ha", "sum"))
        .reset_index()
        .sort_values("disturbance_year")
    )
    selected_bins = age_bins[age_bins["radius_km"] == selected].copy()

    report = {
        "event_grouping_result": {
            "official_nopert_pee_transfer": official_join,
            "join_basis": "GEOC_FMJ + EXERCICE" if official_join else "none; provisional polygon grouping",
            "diagnostics_path": str(OUT_DIR / "fire_event_grouping_diagnostics.csv"),
        },
        "recommended_radius_km": selected,
        "recommended_extent": summary[summary["radius_km"] == selected].iloc[0].to_dict(),
        "epsg_32198_bounds": bounds["epsg_32198_bounds"],
        "wgs84_bounds": bounds["wgs84_bounds"],
        "anchor": {
            "lon": ANCHOR_LON,
            "lat": ANCHOR_LAT,
            "source": ANCHOR_SOURCE,
        },
    }
    (OUT_DIR / "selection_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("EVENT-GROUPING RESULT")
    print(diagnostics.to_string(index=False))
    print(f"Official NOPERT_PEE transfer used: {official_join}")
    print("\nCANDIDATE-RADIUS COMPARISON")
    cols = [
        "radius_km",
        "extent_area_ha",
        "independent_fire_events",
        "distinct_fire_years",
        "earliest_fire_year",
        "latest_fire_year",
        "total_burned_ha",
        "largest_event_pct_burned_area",
        "top3_event_pct_burned_area",
        "balanced_age_bins_with_3_events",
        "meets_screening_target",
    ]
    print(summary[cols].to_string(index=False, formatters={"extent_area_ha": "{:.1f}".format, "total_burned_ha": "{:.1f}".format, "largest_event_pct_burned_area": "{:.1f}".format, "top3_event_pct_burned_area": "{:.1f}".format}))
    print("\nRECOMMENDED RADIUS")
    selected_row = summary[summary["radius_km"] == selected].iloc[0]
    print(f"{selected} km ({selected_row['extent_area_ha']:.1f} ha / {selected_row['extent_area_km2']:.1f} km2)")
    print("\nTEMPORAL REPLICATION")
    print(selected_years.to_string(index=False, formatters={"burned_hectares": "{:.1f}".format}))
    print("\nAGE-BIN REPLICATION")
    print(selected_bins.to_string(index=False, formatters={"burned_hectares": "{:.1f}".format}))
    print("\nGEOGRAPHIC BOUNDS FOR SUBSEQUENT FORESTRY-DATA QUERY")
    print(f"EPSG:32198 bounds: {bounds['epsg_32198_bounds']}")
    print(f"WGS84 bounds: {bounds['wgs84_bounds']}")
    print("\nLIMITATIONS")
    print("- Fire-only screening; forestry data was not downloaded or analyzed.")
    print("- Age bins are diagnostics, not inferential strata.")
    print("- NOPERT_PEE transfer is supported by GEOC_FMJ + EXERCICE one-to-one matching in this GeoPackage, but final analysis should still document the official relationship.")
    print("- No forest-cover/ecological stratification was performed in this phase.")
    print("\nFILES CREATED")
    for path in [
        OUT_DIR / "candidate_extent_summary.csv",
        OUT_DIR / "candidate_extent_age_bins.csv",
        OUT_DIR / "fire_event_inventory.csv",
        OUT_DIR / "fire_event_grouping_diagnostics.csv",
        OUT_DIR / "recommended_study_extent.geojson",
        OUT_DIR / "README.md",
        OUT_DIR / "selection_summary.json",
        FIG_DIR / "candidate_extents_fire_map.png",
        FIG_DIR / "temporal_coverage_by_radius.png",
        FIG_DIR / "age_bin_replication.png",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
