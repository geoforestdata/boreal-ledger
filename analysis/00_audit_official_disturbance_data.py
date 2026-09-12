#!/usr/bin/env python3
"""Audit official Quebec fire and forestry-intervention disturbance datasets.

This script is intentionally an ingestion/data-quality audit only. It does not
classify disturbance pixels, run recovery analysis, or alter production outputs.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/Users/jano/Downloads")
OUT_DIR = REPO_ROOT / "outputs" / "data_audit"
RAW_DIRS = [
    REPO_ROOT / "data" / "raw" / "fire",
    REPO_ROOT / "data" / "raw" / "forestry_interventions",
    REPO_ROOT / "data" / "interim",
    REPO_ROOT / "data" / "processed",
]

YEAR_MIN = 2001
YEAR_MAX = 2024
AREA_COL = "superficie"

FIRE_EXPECTED = "FEUX_PROV_GPKG.zip"
FORESTRY_EXPECTED = "INTERV_FORES_PROV_GPKG.zip"
FORESTRY_TEMP = "INTERV_FORES_PROV_GPKG.duckload"

FIRE_LAYER = "feux_prov"
FIRE_META_LAYER = "meta_feux_prov"
FORESTRY_LAYER = "interv_fores_prov"

FIRE_INTERPRETATION = {
    "origin_definition": "ORIGINE + AN_ORIGINE identify originating/total disturbances; official documentation describes total burns as >75% mortality.",
    "partial_definition": "PERTURB + AN_PERTURB identify partial disturbances; official documentation describes partial burns as 25-75% mortality.",
    "area_definition": "SUPERFICIE is the mapped polygon area in hectares in the source attribute table.",
}

FORESTRY_INTERPRETATION = {
    "origin_definition": "ORIGINE + AN_ORIGINE identify total interventions/disturbances (>75% basal area affected, per provided documentation).",
    "partial_definition": "PERTURB + AN_PERTURB identify partial/intermediate interventions/disturbances (25-75% basal area affected, per provided documentation).",
    "important_caution": "Do not interpret every polygon as harvest; the layer can contain multiple silvicultural or disturbance/intervention codes and overlapping treatment histories.",
}

UNKNOWN_CODE_NOTE = "UNKNOWN in this audit unless official code-domain documentation is supplied in the dataset or repository."


@dataclass
class DatasetLocation:
    kind: str
    archive: Path | None
    gpkg_name: str | None
    uri: str | None
    status: str
    size_bytes: int | None = None
    zip_contents: list[str] | None = None
    error: str | None = None


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in RAW_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def zip_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def find_zip(expected: str, fallback_globs: Iterable[str]) -> Path | None:
    candidate = DOWNLOADS / expected
    if candidate.exists():
        return candidate
    matches: list[Path] = []
    for pattern in fallback_globs:
        matches.extend(DOWNLOADS.glob(pattern))
    return sorted(matches, key=lambda p: p.stat().st_size if p.exists() else -1, reverse=True)[0] if matches else None


def locate_dataset(kind: str, expected_zip: str, fallback_globs: Iterable[str]) -> DatasetLocation:
    archive = find_zip(expected_zip, fallback_globs)
    if archive is None:
        return DatasetLocation(kind, None, None, None, "missing", error="No matching archive found in /Users/jano/Downloads.")

    size = archive.stat().st_size
    if size == 0:
        return DatasetLocation(kind, archive, None, None, "unusable", size, error="Archive exists but is 0 bytes.")

    try:
        members = zip_members(archive)
    except Exception as exc:
        return DatasetLocation(kind, archive, None, None, "unusable", size, error=f"Archive cannot be read as complete ZIP: {exc}")

    gpkg_members = [m for m in members if m.lower().endswith(".gpkg")]
    if not gpkg_members:
        return DatasetLocation(kind, archive, None, None, "unusable", size, members, "No .gpkg member found in archive.")

    gpkg_name = gpkg_members[0]
    uri = f"zip://{archive}!{gpkg_name}"
    return DatasetLocation(kind, archive, gpkg_name, uri, "available", size, members)


def materialize_gpkg(location: DatasetLocation, target_dir: Path) -> DatasetLocation:
    """Extract the GPKG to ignored data/raw storage for faster repeated reads."""
    if location.status != "available" or not location.archive or not location.gpkg_name:
        return location
    target = target_dir / Path(location.gpkg_name).name
    if not target.exists() or target.stat().st_size == 0:
        with zipfile.ZipFile(location.archive) as zf:
            with zf.open(location.gpkg_name) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    dst.write(chunk)
    location.uri = str(target)
    return location


def read_layer(location: DatasetLocation, layer: str) -> gpd.GeoDataFrame | None:
    if not location.uri:
        return None
    return gpd.read_file(location.uri, layer=layer, engine="pyogrio")


def list_layers(location: DatasetLocation) -> list[dict]:
    if not location.uri:
        return []
    rows = []
    for name, geom in pyogrio.list_layers(location.uri):
        info = pyogrio.read_info(location.uri, layer=name)
        rows.append(
            {
                "layer_name": name,
                "geometry_type": geom,
                "feature_count": info.get("features"),
                "crs": info.get("crs"),
                "fields": list(info.get("fields", [])),
            }
        )
    return rows


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf.columns = [str(c).lower() for c in gdf.columns]
    return gdf


def area_series_ha(gdf: gpd.GeoDataFrame) -> pd.Series:
    if AREA_COL in gdf.columns:
        vals = pd.to_numeric(gdf[AREA_COL], errors="coerce")
        if vals.notna().any():
            return vals
    return gdf.geometry.area / 10_000


def year_series(gdf: gpd.GeoDataFrame, preferred: str) -> pd.Series:
    cols = [preferred, "exercice"]
    out = pd.Series(pd.NA, index=gdf.index, dtype="Float64")
    for col in cols:
        if col in gdf.columns:
            out = out.fillna(pd.to_numeric(gdf[col], errors="coerce"))
    return out


def primary_year_series(gdf: gpd.GeoDataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=gdf.index, dtype="Float64")
    for col in ["an_origine", "an_perturb", "exercice"]:
        if col in gdf.columns:
            out = out.fillna(pd.to_numeric(gdf[col], errors="coerce"))
    return out


def schema_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    n = len(gdf)
    for col in gdf.columns:
        if col == "geometry":
            dtype = str(gdf.geometry.dtype)
            nulls = int(gdf.geometry.isna().sum())
        else:
            dtype = str(gdf[col].dtype)
            nulls = int(gdf[col].isna().sum())
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "null_count": nulls,
                "null_pct": (nulls / n * 100) if n else np.nan,
                "non_null_count": n - nulls,
            }
        )
    return pd.DataFrame(rows)


def code_frequency(gdf: gpd.GeoDataFrame, code_col: str, year_col: str) -> pd.DataFrame:
    cols = ["code", "feature_count", "hectares", "first_year", "last_year", "meaning"]
    if code_col not in gdf.columns:
        return pd.DataFrame(columns=cols)
    tmp = pd.DataFrame(
        {
            "code": gdf[code_col].fillna("<NULL>").astype(str),
            "area": area_series_ha(gdf),
            "year": year_series(gdf, year_col),
        }
    )
    out = (
        tmp.groupby("code", dropna=False)
        .agg(feature_count=("code", "size"), hectares=("area", "sum"), first_year=("year", "min"), last_year=("year", "max"))
        .reset_index()
    )
    out["meaning"] = UNKNOWN_CODE_NOTE
    return out.sort_values(["hectares", "feature_count"], ascending=False)


def attribute_frequencies(gdf: gpd.GeoDataFrame, attrs: list[str]) -> pd.DataFrame:
    frames = []
    years = primary_year_series(gdf)
    area = area_series_ha(gdf)
    for attr in attrs:
        if attr not in gdf.columns:
            continue
        tmp = pd.DataFrame({"attribute": attr, "code": gdf[attr].fillna("<NULL>").astype(str), "area": area, "year": years})
        freq = (
            tmp.groupby(["attribute", "code"], dropna=False)
            .agg(feature_count=("code", "size"), hectares=("area", "sum"), first_year=("year", "min"), last_year=("year", "max"))
            .reset_index()
        )
        freq["meaning"] = UNKNOWN_CODE_NOTE
        frames.append(freq)
    if not frames:
        return pd.DataFrame(columns=["attribute", "code", "feature_count", "hectares", "first_year", "last_year", "meaning"])
    return pd.concat(frames, ignore_index=True).sort_values(["attribute", "hectares"], ascending=[True, False])


def year_summary(gdf: gpd.GeoDataFrame, label: str) -> pd.DataFrame:
    rows = []
    area = area_series_ha(gdf)
    for source, y_col in [("origin", "an_origine"), ("partial", "an_perturb"), ("exercise", "exercice")]:
        if y_col not in gdf.columns:
            continue
        years = pd.to_numeric(gdf[y_col], errors="coerce")
        tmp = pd.DataFrame({"year": years, "area": area}).dropna(subset=["year"])
        tmp = tmp[(tmp["year"] >= YEAR_MIN) & (tmp["year"] <= YEAR_MAX)]
        if tmp.empty:
            continue
        part = (
            tmp.groupby("year")
            .agg(polygons=("year", "size"), hectares=("area", "sum"), median_polygon_area_ha=("area", "median"))
            .reset_index()
        )
        part["dataset"] = label
        part["year_basis"] = source
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["dataset", "year_basis", "year", "polygons", "hectares", "median_polygon_area_ha"])
    return pd.concat(rows, ignore_index=True).sort_values(["year_basis", "year"])


def numeric_stats(gdf: gpd.GeoDataFrame, cols: list[str]) -> dict:
    stats = {}
    for col in cols:
        if col not in gdf.columns:
            continue
        vals = pd.to_numeric(gdf[col], errors="coerce")
        stats[col] = {
            "non_null": int(vals.notna().sum()),
            "min": float(vals.min()) if vals.notna().any() else None,
            "median": float(vals.median()) if vals.notna().any() else None,
            "mean": float(vals.mean()) if vals.notna().any() else None,
            "max": float(vals.max()) if vals.notna().any() else None,
        }
    return stats


def dataset_profile(gdf: gpd.GeoDataFrame) -> dict:
    geom = gdf.geometry
    area = area_series_ha(gdf)
    dup_wkb = int(geom.to_wkb().duplicated().sum()) if len(gdf) else 0
    return {
        "feature_count": int(len(gdf)),
        "geometry_type_counts": {str(k): int(v) for k, v in geom.geom_type.value_counts(dropna=False).to_dict().items()},
        "crs": str(gdf.crs),
        "epsg": gdf.crs.to_epsg() if gdf.crs else None,
        "total_bounds": [float(x) for x in gdf.total_bounds] if len(gdf) else None,
        "valid_geometry_count": int(geom.is_valid.sum()),
        "invalid_geometry_count": int((~geom.is_valid).sum()),
        "duplicate_geometry_count": dup_wkb,
        "superficie_stats_ha": {
            "min": float(area.min()) if len(area) else None,
            "median": float(area.median()) if len(area) else None,
            "mean": float(area.mean()) if len(area) else None,
            "max": float(area.max()) if len(area) else None,
            "sum": float(area.sum()) if len(area) else None,
        },
        "year_stats": numeric_stats(gdf, ["exercice", "an_origine", "an_perturb"]),
    }


def filter_2001_2024(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    years = primary_year_series(gdf)
    return gdf[(years >= YEAR_MIN) & (years <= YEAR_MAX)].copy()


def safe_union(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    return gdf.geometry.make_valid().union_all()


def overlap_summary(fire: gpd.GeoDataFrame | None, forestry: gpd.GeoDataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cols = ["metric", "area_ha", "notes"]
    by_cols = ["fire_year", "forestry_year", "temporal_relation", "pair_count", "pairwise_overlap_area_ha"]
    if fire is None or forestry is None:
        reason = "Forestry dataset unavailable; spatial overlap audit could not be completed."
        return pd.DataFrame([{"metric": "blocked", "area_ha": np.nan, "notes": reason}]), pd.DataFrame(columns=by_cols), {"status": "blocked", "reason": reason}

    fire_y = filter_2001_2024(fire)
    forest_y = filter_2001_2024(forestry)
    if fire_y.empty or forest_y.empty:
        return pd.DataFrame([{"metric": "no_overlap_inputs", "area_ha": 0.0, "notes": "One filtered 2001-2024 layer is empty."}]), pd.DataFrame(columns=by_cols), {"status": "empty"}

    if fire_y.crs != forest_y.crs:
        forest_y = forest_y.to_crs(fire_y.crs)

    fire_union = safe_union(fire_y)
    forestry_union = safe_union(forest_y)
    unique_overlap_ha = fire_union.intersection(forestry_union).area / 10_000 if fire_union and forestry_union else 0.0
    fire_area = fire_union.area / 10_000 if fire_union else 0.0
    forestry_area = forestry_union.area / 10_000 if forestry_union else 0.0

    joined = gpd.sjoin(
        fire_y[["geometry"]].assign(fire_year=primary_year_series(fire_y).astype("Int64")),
        forest_y[["geometry"]].assign(forestry_year=primary_year_series(forest_y).astype("Int64")),
        predicate="intersects",
        how="inner",
    )
    pair_rows = []
    if not joined.empty:
        fire_geoms = fire_y.geometry
        forestry_geoms = forest_y.geometry
        chunks = []
        for idx, row in joined.iterrows():
            inter_area = fire_geoms.loc[idx].intersection(forestry_geoms.loc[row["index_right"]]).area / 10_000
            if inter_area > 0:
                chunks.append(
                    {
                        "fire_year": int(row["fire_year"]) if not pd.isna(row["fire_year"]) else np.nan,
                        "forestry_year": int(row["forestry_year"]) if not pd.isna(row["forestry_year"]) else np.nan,
                        "pairwise_overlap_area_ha": inter_area,
                    }
                )
        if chunks:
            pairs = pd.DataFrame(chunks)
            pairs["temporal_relation"] = np.where(pairs["fire_year"] == pairs["forestry_year"], "same_year", "different_year")
            pair_rows = (
                pairs.groupby(["fire_year", "forestry_year", "temporal_relation"], dropna=False)
                .agg(pair_count=("pairwise_overlap_area_ha", "size"), pairwise_overlap_area_ha=("pairwise_overlap_area_ha", "sum"))
                .reset_index()
            )

    summary = pd.DataFrame(
        [
            {"metric": "unique_fire_area_2001_2024_ha", "area_ha": fire_area, "notes": "Unioned fire geometry; avoids double-counting overlapping polygons."},
            {"metric": "unique_intervention_area_2001_2024_ha", "area_ha": forestry_area, "notes": "Unioned forestry-intervention geometry; avoids double-counting overlapping polygons."},
            {"metric": "unique_spatial_overlap_ha", "area_ha": unique_overlap_ha, "notes": "Area of intersection between unioned fire and forestry-intervention geometries."},
            {"metric": "overlap_pct_of_fire_area", "area_ha": (unique_overlap_ha / fire_area * 100) if fire_area else np.nan, "notes": "Percent, not hectares."},
            {"metric": "overlap_pct_of_intervention_area", "area_ha": (unique_overlap_ha / forestry_area * 100) if forestry_area else np.nan, "notes": "Percent, not hectares."},
        ]
    )
    return summary, pd.DataFrame(pair_rows, columns=by_cols), {"status": "completed"}


def repeated_intervention_evidence(forestry: gpd.GeoDataFrame | None) -> dict:
    if forestry is None:
        return {"status": "blocked", "reason": "Forestry dataset unavailable."}
    gdf = filter_2001_2024(forestry)
    years = primary_year_series(gdf)
    result = {"status": "exploratory", "rows_2001_2024": int(len(gdf))}
    for key in ["no_sec_int", "territoire", "geoc_fmj"]:
        if key in gdf.columns:
            tmp = pd.DataFrame({"key": gdf[key].astype(str), "year": years, "code": gdf.get("origine", pd.Series(index=gdf.index, dtype=object)).astype(str)})
            grouped = tmp.groupby("key").agg(row_count=("key", "size"), distinct_years=("year", "nunique"), distinct_origin_codes=("code", "nunique"))
            result[f"{key}_groups"] = int(len(grouped))
            result[f"{key}_groups_multiple_years"] = int((grouped["distinct_years"] > 1).sum())
            result[f"{key}_groups_multiple_origin_codes"] = int((grouped["distinct_origin_codes"] > 1).sum())
    if len(gdf):
        result["exact_duplicate_geometry_count"] = int(gdf.geometry.to_wkb().duplicated().sum())
    result["limitation"] = "This is not a treatment-history reconstruction; overlapping and repeated interventions require an official grouping/key interpretation before classification."
    return result


def write_empty_outputs(prefix: str) -> None:
    pd.DataFrame(columns=["column", "dtype", "null_count", "null_pct", "non_null_count"]).to_csv(OUT_DIR / f"{prefix}_schema.csv", index=False)
    pd.DataFrame(columns=["code", "feature_count", "hectares", "first_year", "last_year", "meaning"]).to_csv(OUT_DIR / f"{prefix}_origin_codes.csv", index=False)
    pd.DataFrame(columns=["code", "feature_count", "hectares", "first_year", "last_year", "meaning"]).to_csv(OUT_DIR / f"{prefix}_partial_codes.csv", index=False)
    pd.DataFrame(columns=["dataset", "year_basis", "year", "polygons", "hectares", "median_polygon_area_ha"]).to_csv(OUT_DIR / f"{prefix}_year_summary.csv", index=False)


def audit_dataset(location: DatasetLocation, layer: str, prefix: str, attrs: list[str]) -> tuple[gpd.GeoDataFrame | None, dict]:
    info = {"location": location.__dict__, "layers": [], "profile": None}
    if location.status != "available":
        write_empty_outputs(prefix)
        if prefix == "forestry":
            pd.DataFrame(
                columns=["attribute", "code", "feature_count", "hectares", "first_year", "last_year", "meaning"]
            ).to_csv(OUT_DIR / "forestry_attribute_frequencies.csv", index=False)
        return None, info

    info["layers"] = list_layers(location)
    layer_names = {row["layer_name"].lower(): row["layer_name"] for row in info["layers"]}
    if layer not in layer_names:
        info["location"]["status"] = "unusable"
        info["location"]["error"] = f"Expected layer {layer!r} not found."
        write_empty_outputs(prefix)
        return None, info

    gdf = normalize_columns(read_layer(location, layer_names[layer]))
    info["profile"] = dataset_profile(gdf)
    schema_table(gdf).to_csv(OUT_DIR / f"{prefix}_schema.csv", index=False)
    code_frequency(gdf, "origine", "an_origine").to_csv(OUT_DIR / f"{prefix}_origin_codes.csv", index=False)
    code_frequency(gdf, "perturb", "an_perturb").to_csv(OUT_DIR / f"{prefix}_partial_codes.csv", index=False)
    year_summary(gdf, prefix).to_csv(OUT_DIR / f"{prefix}_year_summary.csv", index=False)
    if prefix == "forestry":
        attribute_frequencies(gdf, attrs).to_csv(OUT_DIR / "forestry_attribute_frequencies.csv", index=False)
    return gdf, info


def audit_fire_meta(location: DatasetLocation) -> dict:
    result = {"status": "not_available"}
    if location.status != "available":
        return result
    layers = {row["layer_name"].lower(): row["layer_name"] for row in list_layers(location)}
    if FIRE_META_LAYER not in layers:
        return {"status": "missing_layer"}
    meta = normalize_columns(read_layer(location, layers[FIRE_META_LAYER]))
    result = {
        "status": "available",
        "feature_count": int(len(meta)),
        "columns": [c for c in meta.columns if c != "geometry"],
        "crs": str(meta.crs),
        "nopert_pee_non_null": int(meta["nopert_pee"].notna().sum()) if "nopert_pee" in meta.columns else None,
        "nopert_pee_unique": int(meta["nopert_pee"].nunique(dropna=True)) if "nopert_pee" in meta.columns else None,
    }
    return result


def write_readme(summary: dict) -> None:
    fire_loc = summary["fire"]["location"]
    forestry_loc = summary["forestry"]["location"]
    lines = [
        "# Official disturbance data audit",
        "",
        "This directory was generated by `analysis/00_audit_official_disturbance_data.py`.",
        "It is an ingestion and data-quality audit only; it does not classify recovery, harvest, or fire effects.",
        "",
        "## Files found",
        "",
        f"- Fire archive: `{fire_loc.get('archive')}`; status: `{fire_loc.get('status')}`; size: `{fire_loc.get('size_bytes')}` bytes.",
        f"- Forestry-intervention archive: `{forestry_loc.get('archive')}`; status: `{forestry_loc.get('status')}`; size: `{forestry_loc.get('size_bytes')}` bytes.",
        "",
        "## Official interpretation constraints used",
        "",
        f"- Fire ORIGINE/AN_ORIGINE: {FIRE_INTERPRETATION['origin_definition']}",
        f"- Fire PERTURB/AN_PERTURB: {FIRE_INTERPRETATION['partial_definition']}",
        f"- Forestry ORIGINE/AN_ORIGINE: {FORESTRY_INTERPRETATION['origin_definition']}",
        f"- Forestry PERTURB/AN_PERTURB: {FORESTRY_INTERPRETATION['partial_definition']}",
        "- Code meanings are not inferred from abbreviations in this audit. Values without an official code-domain table are flagged as UNKNOWN.",
        "",
        "## Blocking issues",
        "",
    ]
    if forestry_loc.get("status") != "available":
        lines.append(f"- Forestry-intervention audit is blocked: {forestry_loc.get('error')}")
    else:
        lines.append("- No blocking issue detected for forestry-intervention ingestion.")
    lines += [
        "",
        "## Decisions required before recovery analysis",
        "",
        "- Confirm official code-domain meanings for fire and forestry `ORIGINE` / `PERTURB` values.",
        "- Decide which forestry-intervention codes represent harvest, partial harvest, regeneration, thinning, planting/reforestation, site preparation, and other silviculture.",
        "- Decide how to handle overlapping forestry interventions and repeated treatments through time.",
        "- Decide whether spatial/temporal overlap between fire and interventions should be excluded, separated, or used only as an uncertainty flag.",
        "- Decide the event-grain key if any (`NOPERT_PEE`, `NO_SEC_INT`, `TERRITOIRE`, or another official identifier).",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def finite_fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}"
    return str(value)


def main() -> None:
    ensure_dirs()

    fire_loc = locate_dataset("fire", FIRE_EXPECTED, ["*FEUX*GPKG*.zip", "*feux*gpkg*.zip"])
    forestry_loc = locate_dataset("forestry", FORESTRY_EXPECTED, ["*INTERV*FORES*GPKG*.zip", "*interv*fores*gpkg*.zip"])
    if forestry_loc.status != "available":
        temp = DOWNLOADS / FORESTRY_TEMP
        if temp.exists():
            try:
                _ = zip_members(temp)
            except Exception as exc:
                forestry_loc.error = f"{forestry_loc.error or ''} Temporary download file also unusable: {temp} ({exc})".strip()

    fire_loc = materialize_gpkg(fire_loc, REPO_ROOT / "data" / "raw" / "fire")
    forestry_loc = materialize_gpkg(forestry_loc, REPO_ROOT / "data" / "raw" / "forestry_interventions")

    fire_gdf, fire_info = audit_dataset(fire_loc, FIRE_LAYER, "fire", [])
    forestry_attrs = ["type_couv", "gr_ess", "part_str", "reb_ess1", "reb_ess2", "reb_ess3", "et_domi", "cl_dens", "cl_haut", "cl_age", "etagement", "couv_gaule"]
    forestry_gdf, forestry_info = audit_dataset(forestry_loc, FORESTRY_LAYER, "forestry", forestry_attrs)

    fire_meta = audit_fire_meta(fire_loc)
    if fire_gdf is not None and fire_meta.get("status") == "available":
        fire_cols = [c for c in fire_gdf.columns if c != "geometry"]
        fire_meta["join_key_check"] = {
            "fire_has_nopert_pee": "nopert_pee" in fire_cols,
            "fire_has_geoc_fmj": "geoc_fmj" in fire_cols,
            "meta_has_geoc_fmj": "geoc_fmj" in fire_meta.get("columns", []),
            "meaningful_direct_nopert_pee_join_to_fire": False,
            "note": "FEUX_PROV does not expose NOPERT_PEE in the inspected schema; META_FEUX_PROV has GEOC_FMJ and NOPERT_PEE, so any event grouping requires explicit documentation before use.",
        }

    overlap_table, overlap_by_year, overlap_info = overlap_summary(fire_gdf, forestry_gdf)
    overlap_table.to_csv(OUT_DIR / "fire_forestry_overlap_summary.csv", index=False)
    overlap_by_year.to_csv(OUT_DIR / "fire_forestry_overlap_by_year.csv", index=False)

    summary = {
        "downloads_dir": str(DOWNLOADS),
        "output_dir": str(OUT_DIR),
        "fire": fire_info,
        "fire_meta": fire_meta,
        "fire_interpretation": FIRE_INTERPRETATION,
        "forestry": forestry_info,
        "forestry_interpretation": FORESTRY_INTERPRETATION,
        "overlap": overlap_info,
        "repeated_forestry_interventions": repeated_intervention_evidence(forestry_gdf),
        "code_meaning_policy": UNKNOWN_CODE_NOTE,
    }
    (OUT_DIR / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_readme(summary)

    print("FILES FOUND")
    for loc in [fire_loc, forestry_loc]:
        print(f"- {loc.kind}: {loc.archive} | status={loc.status} | size={finite_fmt(loc.size_bytes)} bytes | gpkg={loc.gpkg_name}")
        if loc.error:
            print(f"  error: {loc.error}")

    print("\nLAYERS USED")
    for name, info in [("fire", fire_info), ("forestry", forestry_info)]:
        if info["layers"]:
            for layer in info["layers"]:
                print(f"- {name}: {layer['layer_name']} | {layer['geometry_type']} | {finite_fmt(layer['feature_count'])} features | {layer['crs']}")
        else:
            print(f"- {name}: no layers available")

    print("\nFEATURE COUNTS")
    for name, info in [("fire", fire_info), ("forestry", forestry_info)]:
        profile = info.get("profile")
        print(f"- {name}: {finite_fmt(profile.get('feature_count') if profile else None)}")

    print("\nYEAR COVERAGE")
    for label, path in [("fire", OUT_DIR / "fire_year_summary.csv"), ("forestry", OUT_DIR / "forestry_year_summary.csv")]:
        df = pd.read_csv(path)
        if df.empty:
            print(f"- {label}: no usable year summary")
        else:
            print(f"- {label}: {int(df['year'].min())}-{int(df['year'].max())}; rows={len(df)}")

    print("\nFIRE CODES FOUND")
    for label, path in [("ORIGINE", OUT_DIR / "fire_origin_codes.csv"), ("PERTURB", OUT_DIR / "fire_partial_codes.csv")]:
        df = pd.read_csv(path)
        print(f"- {label}: {', '.join(df['code'].head(20).astype(str).tolist()) if not df.empty else 'none'}")

    print("\nFORESTRY CODES FOUND")
    for label, path in [("ORIGINE", OUT_DIR / "forestry_origin_codes.csv"), ("PERTURB", OUT_DIR / "forestry_partial_codes.csv")]:
        df = pd.read_csv(path)
        print(f"- {label}: {', '.join(df['code'].head(20).astype(str).tolist()) if not df.empty else 'none'}")

    print("\nCODE MEANINGS DOCUMENTED/UNRESOLVED")
    print(f"- Fire high-level ORIGINE/PERTURB roles documented from provided request; individual code-domain meanings unresolved and marked UNKNOWN.")
    print(f"- Forestry high-level ORIGINE/PERTURB roles documented from provided request; individual code-domain meanings unresolved and marked UNKNOWN.")

    print("\nFIRE-FORESTRY OVERLAPS")
    print(overlap_table.to_string(index=False))

    print("\nREPEATED FORESTRY INTERVENTIONS")
    print(json.dumps(summary["repeated_forestry_interventions"], indent=2, default=str))

    print("\nTEMPORAL REPLICATION 2001-2024")
    for label, path in [("fire", OUT_DIR / "fire_year_summary.csv"), ("forestry", OUT_DIR / "forestry_year_summary.csv")]:
        df = pd.read_csv(path)
        supported = df[(df["polygons"] > 0) & (df["hectares"] >= 9.0)] if not df.empty else df
        print(f"- {label}: {len(supported) if supported is not None else 0} year-basis rows with >=9 ha; interpret only after code meanings and event grain are resolved.")

    print("\nBLOCKING ISSUES")
    if forestry_loc.status != "available":
        print(f"- Forestry-intervention dataset unavailable: {forestry_loc.error}")
    print("- Individual disturbance/intervention code meanings require official domain documentation before mapping to harvest/silviculture classes.")
    print("- Overlapping forestry polygons and repeated treatments require an explicit event-history rule before recovery analysis.")

    print("\nFILES CREATED/MODIFIED")
    for path in sorted(OUT_DIR.glob("*")):
        print(f"- {path}")
    print(f"- {REPO_ROOT / 'analysis' / '00_audit_official_disturbance_data.py'}")

    print("\nDECISIONS REQUIRED BEFORE RECOVERY ANALYSIS")
    print("- Which `ORIGINE` and `PERTURB` codes are total harvest, partial harvest, regeneration harvest, thinning, planting/reforestation, site preparation, or other silviculture?")
    print("- Should fire/intervention overlaps be excluded, stratified, or used as uncertainty flags?")
    print("- What official key defines a disturbance/intervention event, if any?")
    print("- How should repeated forestry interventions at the same location be converted into a single recovery clock?")


if __name__ == "__main__":
    main()
