#!/usr/bin/env python3
"""Build disturbance event and treatment-history datasets for recovery design.

This script uses only local official Quebec fire/intervention vector data. It
does not download or process Hansen, Sentinel-2, Landsat, AlphaEarth, or NBR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FIRE_GPKG = REPO_ROOT / "data" / "raw" / "fire" / "FEUX_PROV.gpkg"
FORESTRY_GPKG = REPO_ROOT / "data" / "raw" / "forestry_interventions" / "interv_fores_lebel_100km_circle.gpkg"
STUDY_GEOJSON = REPO_ROOT / "outputs" / "study_extent_selection" / "recommended_study_extent.geojson"
FORESTRY_CANDIDATES = REPO_ROOT / "outputs" / "forestry_subset_audit" / "intervention_code_candidates.csv"

CONFIG_DIR = REPO_ROOT / "config"
OUT_DIR = REPO_ROOT / "outputs" / "event_histories"
FIG_DIR = REPO_ROOT / "figures" / "event_histories"
MPL_DIR = OUT_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib.pyplot as plt

CRS_PROJECTED = "EPSG:32198"
OBSERVATION_YEAR = 2024
START_YEAR = 2001
END_YEAR = 2023

AGE_BINS = [
    ("0-5", 0, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16-20", 16, 20),
    ("21-23", 21, 23),
]

# Crosswalk decisions are based on official descriptions exposed in the WFS and
# parsed from the dictionary audit. They are deliberately conservative.
FIRE_CODE_RULES = {
    ("fire", "origine", "BR"): ("Brûlis total / total burn; ORIGINE denotes total disturbance in the official fire layer.", "fire_total", True),
    ("fire", "perturb", "BRP"): ("Brûlis partiel / partial burn; PERTURB denotes partial disturbance in the official fire layer.", "fire_partial", False),
}

HARVEST_TOTAL_CODES = {
    "CT",
    "CPR",
    "CPRS_U",
    "CPRS_DA",
    "CBA",
    "CRR",
    "CBT",
    "CPHRS",
    "CPT",
    "CPPTM_U",
    "CPPTM_DIS",
    "CRS",
    "CTSP_U",
    "CS",
}
HARVEST_PARTIAL_CODES = {
    "CP",
    "EC",
    "CPI_RL",
    "CPI_CP",
    "CPI_TA",
    "CPR_U",
    "CPR_T",
    "EC_SEL",
    "CIP",
    "RPLB",
    "ECL",
    "RBV",
    "CB",
}
REFORESTATION_CODES = {"P", "PL", "PLR", "PLN", "PRR", "PL_REG", "ENS", "ENM", "RR", "RRP", "RRR", "RRN", "ENP"}
THINNING_CODES = {"EPC", "EPC_SYS", "EPC_PUITS", "EC", "EC_SEL"}
SITE_PREP_CODES = set()
OTHER_SILVICULTURE_CODES = {"DEG", "NET", "DRM", "REA", "ETR", "RPS", "RECUP_C-T", "CPC", "CPS", "CEF", "CPE"}
SALVAGE_AFTER_FIRE_CODES = {"RECUP_F-T"}


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def norm_cols(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def read_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    for path in [FIRE_GPKG, FORESTRY_GPKG, STUDY_GEOJSON, FORESTRY_CANDIDATES]:
        if not path.exists():
            raise FileNotFoundError(path)
    study = gpd.read_file(STUDY_GEOJSON, engine="pyogrio").to_crs(CRS_PROJECTED)
    fire = norm_cols(gpd.read_file(FIRE_GPKG, layer="feux_prov", engine="pyogrio")).to_crs(CRS_PROJECTED)
    meta = norm_cols(gpd.read_file(FIRE_GPKG, layer="meta_feux_prov", engine="pyogrio")).to_crs(CRS_PROJECTED)
    forestry = norm_cols(gpd.read_file(FORESTRY_GPKG, engine="pyogrio")).to_crs(CRS_PROJECTED)
    return study, fire, meta, forestry


def coalesce_year(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in ["an_origine", "an_perturb", "exercice"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
            out = out.fillna(vals)
    return out


def age_bin(age: float) -> str | None:
    if pd.isna(age):
        return None
    for label, lo, hi in AGE_BINS:
        if lo <= age <= hi:
            return label
    return None


def forestry_record_class(row: pd.Series) -> tuple[str, str, str, str]:
    for field, desc_field in [("origine", "origine_desc"), ("perturb", "perturb_desc"), ("reb_ess1", "reb_ess1_desc")]:
        code = str(row.get(field, "") or "").strip()
        if not code:
            continue
        desc = str(row.get(desc_field, "") or row.get("classe", "") or "").strip()
        if code in SALVAGE_AFTER_FIRE_CODES:
            return field, code, desc, "salvage_after_fire"
        if code in HARVEST_TOTAL_CODES and field == "origine":
            return field, code, desc, "harvest_total"
        if code in HARVEST_PARTIAL_CODES:
            return field, code, desc, "harvest_partial"
        if code in REFORESTATION_CODES:
            return field, code, desc, "reforestation"
        if code in SITE_PREP_CODES:
            return field, code, desc, "site_preparation"
        if code in THINNING_CODES:
            return field, code, desc, "thinning"
        if code in OTHER_SILVICULTURE_CODES:
            return field, code, desc, "other_silviculture"
        return field, code, desc, "unknown"
    return "", "", "", "unknown"


def build_crosswalk() -> pd.DataFrame:
    candidates = pd.read_csv(FORESTRY_CANDIDATES)
    rows = []
    for (source, field, code), (desc, project_class, primary) in FIRE_CODE_RULES.items():
        rows.append(
            {
                "source": source,
                "field": field,
                "code": code,
                "official_description": desc,
                "project_class": project_class,
                "primary_analysis": bool(primary),
                "notes": "Official fire-layer total/partial disturbance role documented in the disturbance-data audit.",
            }
        )
    for _, r in candidates[candidates["field"].isin(["origine", "perturb", "reb_ess1"])].iterrows():
        code = str(r["code"])
        if code == "<NULL>":
            continue
        desc = str(r["official_description"])
        if code in SALVAGE_AFTER_FIRE_CODES:
            cls, primary, notes = "salvage_after_fire", False, "Compound pathway: fire followed by total salvage logging; excluded from pure fire and pure harvest."
        elif code in HARVEST_TOTAL_CODES and r["field"] == "origine":
            cls, primary, notes = "harvest_total", True, "Official description denotes total/regeneration harvest or major canopy-removing harvest."
        elif code in HARVEST_PARTIAL_CODES:
            cls, primary, notes = "harvest_partial", False, "Official description denotes partial/commercial thinning/partial cut; kept outside primary comparison."
        elif code in REFORESTATION_CODES:
            cls, primary, notes = "reforestation", False, "Regeneration/replanting treatment; not an initial harvest disturbance."
        elif code in SITE_PREP_CODES:
            cls, primary, notes = "site_preparation", False, "Site-preparation treatment; not an initial harvest disturbance."
        elif code in THINNING_CODES:
            cls, primary, notes = "thinning", False, "Thinning/precommercial thinning; not a primary canopy-removal event."
        elif code in OTHER_SILVICULTURE_CODES:
            cls, primary, notes = "other_silviculture", False, "Other treatment or non-primary intervention; requires pathway review if used later."
        else:
            cls, primary, notes = "unknown", False, "Not assigned to primary analysis without additional review."
        rows.append(
            {
                "source": "forestry_intervention",
                "field": r["field"],
                "code": code,
                "official_description": desc,
                "project_class": cls,
                "primary_analysis": bool(primary),
                "notes": notes,
            }
        )
    crosswalk = pd.DataFrame(rows).drop_duplicates(["source", "field", "code", "project_class"])
    crosswalk.to_csv(CONFIG_DIR / "disturbance_code_crosswalk.csv", index=False)
    return crosswalk


def attach_fire_events(fire: gpd.GeoDataFrame, meta: gpd.GeoDataFrame, study: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    meta_keys = meta.drop(columns="geometry")[["geoc_fmj", "exercice", "nopert_pee"]]
    gdf = fire.merge(meta_keys, on=["geoc_fmj", "exercice"], how="left")
    gdf = gdf[gdf.intersects(study.geometry.iloc[0])].copy()
    gdf["geometry"] = gdf.geometry.intersection(study.geometry.iloc[0])
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["disturbance_year"] = coalesce_year(gdf).astype("Int64")
    gdf = gdf[(gdf["disturbance_year"] >= START_YEAR) & (gdf["disturbance_year"] <= END_YEAR)].copy()
    gdf["project_class"] = np.where(gdf["origine"].fillna("").eq("BR"), "fire_total", np.where(gdf["perturb"].fillna("").eq("BRP"), "fire_partial", "unknown"))
    gdf["event_id"] = "fire_" + gdf["nopert_pee"].astype(str) + "_" + gdf["disturbance_year"].astype(str)
    gdf["polygon_id"] = "firepoly_" + gdf.index.astype(str)
    gdf["area_ha"] = gdf.geometry.area / 10_000
    return gdf


def aggregate_fire_events(fire_records: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records = []
    for (event_id, year, cls), part in fire_records.groupby(["event_id", "disturbance_year", "project_class"], dropna=False):
        geom = part.geometry.union_all()
        area = geom.area / 10_000
        records.append(
            {
                "source": "fire",
                "event_id": event_id,
                "site_id": str(event_id),
                "disturbance_type": cls,
                "disturbance_year": int(year),
                "years_since_disturbance": OBSERVATION_YEAR - int(year),
                "age_bin": age_bin(OBSERVATION_YEAR - int(year)),
                "polygon_count": int(len(part)),
                "area_ha": float(area),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)


def classify_forestry_records(forestry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = forestry.copy()
    classes = gdf.apply(forestry_record_class, axis=1, result_type="expand")
    classes.columns = ["class_field", "class_code", "class_description", "project_class"]
    gdf = pd.concat([gdf, classes], axis=1)
    gdf["disturbance_year"] = coalesce_year(gdf).astype("Int64")
    gdf["polygon_id"] = "forestrypoly_" + gdf["ogc_fid"].astype(str)
    gdf["site_id"] = gdf["geoc_ifm"].astype(str)
    gdf["area_ha"] = gdf.geometry.area / 10_000
    return gdf


def aggregate_harvest_events(forestry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    harvest = forestry[
        (forestry["project_class"] == "harvest_total")
        & (forestry["disturbance_year"] >= START_YEAR)
        & (forestry["disturbance_year"] <= END_YEAR)
    ].copy()
    records = []
    for year, part in harvest.groupby("disturbance_year"):
        part = part.reset_index(drop=True)
        if part.empty:
            continue
        parent = list(range(len(part)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        sindex = part.sindex
        for i, geom in enumerate(part.geometry):
            for j in sindex.query(geom, predicate="intersects"):
                j = int(j)
                if j > i:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(len(part)):
            groups.setdefault(find(i), []).append(i)

        for component_idx, indices in enumerate(groups.values()):
            component = part.iloc[indices]
            geom = component.geometry.union_all()
            event_id = f"harvest_{int(year)}_{component_idx:05d}"
            records.append(
                {
                    "source": "forestry_intervention",
                    "event_id": event_id,
                    "site_id": event_id,
                    "disturbance_type": "harvest_total",
                    "disturbance_year": int(year),
                    "years_since_disturbance": OBSERVATION_YEAR - int(year),
                    "age_bin": age_bin(OBSERVATION_YEAR - int(year)),
                    "polygon_count": int(len(component)),
                    "area_ha": float(geom.area / 10_000),
                    "grouping_method": "provisional contiguous/intersecting harvest_total polygons by year",
                    "source_site_ids": ";".join(sorted(set(component["site_id"].dropna().astype(str)))),
                    "source_polygon_ids": ";".join(sorted(set(component["polygon_id"].dropna().astype(str)))),
                    "geometry": geom,
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)


def treatment_histories(forestry: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = [
        "site_id",
        "polygon_id",
        "ogc_fid",
        "geoc_ifm",
        "disturbance_year",
        "project_class",
        "class_field",
        "class_code",
        "class_description",
        "symbologie",
        "classe",
        "area_ha",
    ]
    existing = [c for c in cols if c in forestry.columns]
    hist = forestry[existing].sort_values(["site_id", "disturbance_year", "project_class", "class_code"])
    hist.to_csv(OUT_DIR / "forestry_treatment_histories.csv", index=False)
    return hist


def fire_histories(fire_records: gpd.GeoDataFrame) -> pd.DataFrame:
    cols = ["event_id", "polygon_id", "nopert_pee", "disturbance_year", "project_class", "origine", "perturb", "area_ha"]
    hist = fire_records[[c for c in cols if c in fire_records.columns]].sort_values(["event_id", "disturbance_year"])
    hist.to_csv(OUT_DIR / "fire_event_histories.csv", index=False)
    return hist


def spatial_pairs(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame()
    l = left[["event_id", "disturbance_year", "disturbance_type", "geometry"]].rename(
        columns={"event_id": f"{left_name}_event_id", "disturbance_year": f"{left_name}_year", "disturbance_type": f"{left_name}_type"}
    )
    r = right[["event_id", "disturbance_year", "disturbance_type", "geometry"]].rename(
        columns={"event_id": f"{right_name}_event_id", "disturbance_year": f"{right_name}_year", "disturbance_type": f"{right_name}_type"}
    )
    joined = gpd.sjoin(l, r, predicate="intersects", how="inner")
    if joined.empty:
        return pd.DataFrame()
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def spatial_pairs_indexed(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    """Return intersecting event pairs using the right-hand spatial index.

    This avoids materializing very large treatment-event joins. The inputs here
    are already aggregated events, so a Python loop over events is tractable and
    easier to bound than an all-to-all overlay.
    """
    if left.empty or right.empty:
        return pd.DataFrame()
    right_sindex = right.sindex
    rows = []
    left_cols = ["event_id", "disturbance_year", "disturbance_type"]
    right_cols = ["event_id", "disturbance_year", "disturbance_type"]
    for _, lrow in left.iterrows():
        candidates = list(right_sindex.query(lrow.geometry, predicate="intersects"))
        if not candidates:
            continue
        for ridx in candidates:
            rrow = right.iloc[int(ridx)]
            rows.append(
                {
                    f"{left_name}_event_id": lrow["event_id"],
                    f"{left_name}_year": int(lrow["disturbance_year"]),
                    f"{left_name}_type": lrow["disturbance_type"],
                    f"{right_name}_event_id": rrow["event_id"],
                    f"{right_name}_year": int(rrow["disturbance_year"]),
                    f"{right_name}_type": rrow["disturbance_type"],
                }
            )
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def compound_histories(fire_events: gpd.GeoDataFrame, harvest_events: gpd.GeoDataFrame, forestry_records: gpd.GeoDataFrame) -> pd.DataFrame:
    salvage_records = forestry_records[forestry_records["project_class"] == "salvage_after_fire"].copy()
    salvage_events = aggregate_simple_forestry_events(salvage_records, "salvage_after_fire")
    rows = []
    for pairs, label in [
        (spatial_pairs_indexed(fire_events[fire_events["disturbance_type"] == "fire_total"], salvage_events, "fire", "forestry"), "fire_then_salvage"),
        (spatial_pairs_indexed(fire_events[fire_events["disturbance_type"] == "fire_total"], harvest_events, "fire", "forestry"), "fire_harvest_overlap"),
    ]:
        if pairs.empty:
            continue
        for _, r in pairs.iterrows():
            fire_year = int(r["fire_year"])
            forestry_year = int(r["forestry_year"])
            if fire_year == forestry_year:
                pathway = "ambiguous_same_year"
            elif label == "fire_then_salvage" and forestry_year >= fire_year:
                pathway = "fire_then_salvage"
            elif label == "fire_harvest_overlap" and forestry_year > fire_year:
                pathway = "fire_then_harvest"
            elif label == "fire_harvest_overlap" and forestry_year < fire_year:
                pathway = "harvest_then_fire"
            else:
                pathway = label
            item = r.to_dict()
            if label == "fire_harvest_overlap":
                item["harvest_event_id"] = item["forestry_event_id"]
                item["harvest_year"] = item["forestry_year"]
                item["harvest_type"] = item["forestry_type"]
            item["pathway"] = pathway
            rows.append(item)
    out = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
    out.to_csv(OUT_DIR / "compound_disturbance_histories.csv", index=False)
    return out


def aggregate_simple_forestry_events(records: gpd.GeoDataFrame, cls: str) -> gpd.GeoDataFrame:
    if records.empty:
        return gpd.GeoDataFrame(columns=["event_id", "disturbance_year", "disturbance_type", "geometry"], geometry="geometry", crs=CRS_PROJECTED)
    items = []
    for (year, code), part in records.groupby(["disturbance_year", "class_code"], dropna=False):
        geom = part.geometry.union_all()
        items.append(
            {
                "source": "forestry_intervention",
                "event_id": f"{cls}_{int(year)}_{code}",
                "site_id": f"{cls}_{int(year)}_{code}",
                "disturbance_type": cls,
                "disturbance_year": int(year),
                "years_since_disturbance": OBSERVATION_YEAR - int(year),
                "age_bin": age_bin(OBSERVATION_YEAR - int(year)),
                "polygon_count": int(len(part)),
                "area_ha": float(geom.area / 10_000),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(items, geometry="geometry", crs=CRS_PROJECTED)


def management_pathways(harvest_events: gpd.GeoDataFrame, forestry_records: gpd.GeoDataFrame, fire_events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    harvest = harvest_events.copy()
    for col in [
        "replanted_after_harvest",
        "site_preparation_after_harvest",
        "thinning_after_harvest",
        "second_harvest",
    ]:
        harvest[col] = False
    for col in ["planting_year", "site_preparation_year", "thinning_year", "second_harvest_year", "next_major_disturbance_year", "censor_year"]:
        harvest[col] = pd.NA

    if harvest.empty:
        return harvest

    treatments = forestry_records[["site_id", "project_class", "disturbance_year"]].dropna(subset=["site_id", "disturbance_year"]).copy()
    treatments["disturbance_year"] = treatments["disturbance_year"].astype(int)
    grouped = {
        str(site): part[["project_class", "disturbance_year"]].copy()
        for site, part in treatments.groupby("site_id", sort=False)
    }

    for idx, row in harvest.iterrows():
        source_sites = [s for s in str(row.get("source_site_ids", "")).split(";") if s]
        if not source_sites:
            continue
        later_parts = []
        harvest_year = int(row["disturbance_year"])
        for site in source_sites:
            part = grouped.get(site)
            if part is not None:
                later_parts.append(part[part["disturbance_year"] > harvest_year])
        if not later_parts:
            continue
        part = pd.concat(later_parts, ignore_index=True)
        for cls, flag, year_col in [
            ("reforestation", "replanted_after_harvest", "planting_year"),
            ("site_preparation", "site_preparation_after_harvest", "site_preparation_year"),
            ("thinning", "thinning_after_harvest", "thinning_year"),
            ("harvest_total", "second_harvest", "second_harvest_year"),
        ]:
            yrs = part.loc[part["project_class"] == cls, "disturbance_year"].dropna()
            if not yrs.empty:
                harvest.at[idx, flag] = True
                harvest.at[idx, year_col] = int(yrs.min())

    for idx, row in harvest.iterrows():
        if row["second_harvest"] and (pd.isna(row["next_major_disturbance_year"]) or int(row["second_harvest_year"]) < int(row["next_major_disturbance_year"])):
            harvest.at[idx, "next_major_disturbance_year"] = int(row["second_harvest_year"])
            harvest.at[idx, "censor_year"] = int(row["second_harvest_year"])
    return harvest


def fire_censoring(fire_events: gpd.GeoDataFrame, harvest_events: gpd.GeoDataFrame, forestry_records: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fire = fire_events.copy()
    fire["next_major_disturbance_year"] = pd.NA
    fire["censor_year"] = pd.NA
    later_harvest = spatial_pairs_indexed(fire, harvest_events, "fire", "harvest")
    later_harvest = later_harvest[later_harvest["harvest_year"] > later_harvest["fire_year"]] if not later_harvest.empty else later_harvest
    for event_id, part in later_harvest.groupby("fire_event_id") if not later_harvest.empty else []:
        idx = fire.index[fire["event_id"] == event_id][0]
        yr = int(part["harvest_year"].min())
        fire.at[idx, "next_major_disturbance_year"] = yr
        fire.at[idx, "censor_year"] = yr
    salvage = forestry_records[forestry_records["project_class"] == "salvage_after_fire"].copy()
    if not salvage.empty:
        joined = gpd.sjoin(salvage[["disturbance_year", "geometry"]], fire[["event_id", "disturbance_year", "geometry"]].rename(columns={"disturbance_year": "fire_year"}), predicate="intersects", how="inner")
        joined = joined[joined["disturbance_year"] >= joined["fire_year"]]
        for event_id, part in joined.groupby("event_id"):
            idx = fire.index[fire["event_id"] == event_id][0]
            yr = int(part["disturbance_year"].min())
            current = fire.at[idx, "next_major_disturbance_year"]
            if pd.isna(current) or yr < int(current):
                fire.at[idx, "next_major_disturbance_year"] = yr
                fire.at[idx, "censor_year"] = yr
    return fire


def clean_primary(fire_events: gpd.GeoDataFrame, harvest_events: gpd.GeoDataFrame, compounds: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    fire = fire_events[fire_events["disturbance_type"] == "fire_total"].copy()
    harvest = harvest_events.copy()
    if not compounds.empty:
        contaminated_fire = set(compounds.loc[compounds["pathway"].isin(["fire_then_salvage", "ambiguous_same_year"]), "fire_event_id"].dropna().astype(str))
        contaminated_harvest = set(compounds.loc[compounds["pathway"].isin(["harvest_then_fire", "ambiguous_same_year", "fire_then_harvest"]), "harvest_event_id"].dropna().astype(str))
        fire["primary_cohort"] = ~fire["event_id"].astype(str).isin(contaminated_fire)
        harvest["primary_cohort"] = ~harvest["event_id"].astype(str).isin(contaminated_harvest)
    else:
        fire["primary_cohort"] = True
        harvest["primary_cohort"] = True
    return fire[fire["primary_cohort"]].copy(), harvest[harvest["primary_cohort"]].copy()


def replication_summary(events: gpd.GeoDataFrame, label: str) -> dict:
    if events.empty:
        return {
            "cohort": label,
            "polygons": 0,
            "sites": 0,
            "events": 0,
            "distinct_disturbance_years": 0,
            "area_ha": 0,
            "median_event_area_ha": np.nan,
        }
    return {
        "cohort": label,
        "polygons": int(events["polygon_count"].sum()) if "polygon_count" in events else int(len(events)),
        "sites": int(events["site_id"].nunique()) if "site_id" in events else int(events["event_id"].nunique()),
        "events": int(events["event_id"].nunique()),
        "distinct_disturbance_years": int(events["disturbance_year"].nunique()),
        "area_ha": float(events["area_ha"].sum()),
        "median_event_area_ha": float(events["area_ha"].median()),
        "earliest_year": int(events["disturbance_year"].min()),
        "latest_year": int(events["disturbance_year"].max()),
    }


def age_bin_summary(events: gpd.GeoDataFrame, label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["cohort", "age_bin", "event_count", "distinct_years", "area_ha"])
    out = (
        events.groupby("age_bin", dropna=False)
        .agg(event_count=("event_id", "nunique"), distinct_years=("disturbance_year", "nunique"), area_ha=("area_ha", "sum"))
        .reset_index()
    )
    out.insert(0, "cohort", label)
    return out


def matchability(fire_events: gpd.GeoDataFrame, harvest_events: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    fire_cent = fire_events.geometry.centroid
    harvest_cent = harvest_events.geometry.centroid
    harvest_years = harvest_events["disturbance_year"].astype(int).to_numpy()
    for idx, fire in fire_events.reset_index(drop=True).iterrows():
        year = int(fire["disturbance_year"])
        counts = {}
        for window in [0, 1, 2]:
            mask = np.abs(harvest_years - year) <= window
            counts[f"harvest_candidates_pm{window}_years"] = int(mask.sum())
        mask2 = np.abs(harvest_years - year) <= 2
        nearest_m = np.nan
        if mask2.any():
            distances = harvest_cent[mask2].distance(fire_cent.iloc[idx])
            nearest_m = float(distances.min())
        rows.append(
            {
                "fire_event_id": fire["event_id"],
                "fire_year": year,
                "same_year_harvest_candidates": counts["harvest_candidates_pm0_years"],
                "pm1_year_harvest_candidates": counts["harvest_candidates_pm1_years"],
                "pm2_year_harvest_candidates": counts["harvest_candidates_pm2_years"],
                "nearest_harvest_distance_m_within_pm2_years": nearest_m,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fire_harvest_matchability.csv", index=False)
    return out


def write_events_gpkg(fire: gpd.GeoDataFrame, harvest: gpd.GeoDataFrame, compounds: pd.DataFrame) -> None:
    path = OUT_DIR / "disturbance_events.gpkg"
    if path.exists():
        path.unlink()
    fire.to_file(path, layer="fire_events", driver="GPKG")
    harvest.to_file(path, layer="harvest_events", driver="GPKG")
    if not compounds.empty:
        compounds_gdf = gpd.GeoDataFrame(compounds.copy())
        compounds_gdf.to_csv(OUT_DIR / "compound_disturbance_histories.csv", index=False)


def make_figures(clean_fire: gpd.GeoDataFrame, clean_harvest: gpd.GeoDataFrame, compounds: pd.DataFrame, replication_age: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    if not clean_harvest.empty:
        clean_harvest.plot(ax=ax, color="#2c7fb8", alpha=0.25, linewidth=0)
    if not clean_fire.empty:
        clean_fire.plot(ax=ax, color="#d95f0e", alpha=0.45, linewidth=0)
    ax.set_title("Clean candidate primary events: fire_total vs harvest_total")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clean_fire_vs_harvest_events_map.png", dpi=220)
    plt.close(fig)

    year_df = pd.concat(
        [
            clean_fire.assign(cohort="clean_fire_total")[["cohort", "disturbance_year", "event_id"]],
            clean_harvest.assign(cohort="clean_harvest_total")[["cohort", "disturbance_year", "event_id"]],
        ]
    )
    if not year_df.empty:
        pivot = year_df.groupby(["disturbance_year", "cohort"])["event_id"].nunique().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot.plot(kind="bar", ax=ax, width=0.85)
        ax.set_title("Independent/provisional events by disturbance year")
        ax.set_xlabel("Disturbance year")
        ax.set_ylabel("Event count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "events_by_disturbance_year.png", dpi=220)
        plt.close(fig)

    if not replication_age.empty:
        pivot = replication_age.pivot(index="age_bin", columns="cohort", values="event_count").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Event counts by recovery-age bin")
        ax.set_xlabel("Age bin in 2024")
        ax.set_ylabel("Event count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "event_counts_by_age_bin.png", dpi=220)
        plt.close(fig)

    if not compounds.empty and "pathway" in compounds.columns:
        counts = compounds["pathway"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", ax=ax, color="#756bb1")
        ax.set_title("Compound / ambiguous disturbance pathway counts")
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "compound_pathway_counts.png", dpi=220)
        plt.close(fig)


def write_readme(summary: dict) -> None:
    lines = [
        "# Disturbance event histories",
        "",
        "This dataset prepares candidate disturbance cohorts for a later recovery analysis.",
        "It uses only local official Quebec fire and forestry-intervention vector data.",
        "",
        "## Primary comparison",
        "",
        "- `fire_total`: official fire polygons with `ORIGINE = BR` grouped by transferred `NOPERT_PEE + year`.",
        "- `harvest_total`: official forestry interventions whose `ORIGINE` descriptions denote total/regeneration harvest or major canopy-removing harvest, grouped provisionally by year and contiguous/intersecting geometry.",
        "",
        "Partial fire (`BRP`), partial harvest, reforestation, thinning, and other silvicultural treatments are retained as pathway variables or sensitivity classes, not mixed into the primary comparison.",
        "",
        "## Harvest grouping",
        "",
        "`GEOC_IFM` is documented in the code dictionary as metric X/Y coordinates of a point inside the polygon, not as an event identifier.",
        "Because the WFS subset does not expose `NO_SEC_INT`, harvest events are provisional spatial groups: contiguous/intersecting `harvest_total` polygons in the same year.",
        "",
        "## Key counts",
        "",
        json.dumps(summary, indent=2, default=str),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("Building disturbance code crosswalk...", flush=True)
    crosswalk = build_crosswalk()
    print("Reading fire, metadata, study extent, and forestry intervention inputs...", flush=True)
    study, fire_raw, meta, forestry_raw = read_inputs()
    print("Aggregating official fire events...", flush=True)
    fire_records = attach_fire_events(fire_raw, meta, study)
    fire_events_all = aggregate_fire_events(fire_records)
    print("Classifying forestry intervention records...", flush=True)
    forestry_records = classify_forestry_records(forestry_raw)
    print("Aggregating harvest_total events by year and connected geometry...", flush=True)
    harvest_events_all = aggregate_harvest_events(forestry_records)

    print("Writing treatment and fire histories...", flush=True)
    treatment_histories(forestry_records)
    fire_histories(fire_records)
    print("Building compound disturbance histories...", flush=True)
    compounds = compound_histories(fire_events_all, harvest_events_all, forestry_records)
    print("Summarizing subsequent treatment pathways...", flush=True)
    harvest_events_all = management_pathways(harvest_events_all, forestry_records, fire_events_all)
    print("Summarizing fire-event censoring...", flush=True)
    fire_events_all = fire_censoring(fire_events_all, harvest_events_all, forestry_records)
    print("Selecting clean primary cohorts...", flush=True)
    clean_fire, clean_harvest = clean_primary(fire_events_all, harvest_events_all, compounds)

    clean_fire.drop(columns="geometry").to_csv(OUT_DIR / "primary_fire_events.csv", index=False)
    clean_harvest.drop(columns="geometry").to_csv(OUT_DIR / "primary_harvest_events.csv", index=False)

    rep_rows = [
        replication_summary(fire_events_all[fire_events_all["disturbance_type"] == "fire_total"], "all_fire_total"),
        replication_summary(harvest_events_all, "all_harvest_total"),
        replication_summary(clean_fire, "clean_fire_total"),
        replication_summary(clean_harvest, "clean_harvest_total"),
    ]
    replication = pd.DataFrame(rep_rows)
    replication.to_csv(OUT_DIR / "event_replication_summary.csv", index=False)
    age_summary = pd.concat(
        [
            age_bin_summary(clean_fire, "clean_fire_total"),
            age_bin_summary(clean_harvest, "clean_harvest_total"),
        ],
        ignore_index=True,
    )
    age_summary.to_csv(OUT_DIR / "event_age_bin_summary.csv", index=False)
    match = matchability(clean_fire, clean_harvest)
    print("Writing geospatial outputs and figures...", flush=True)
    write_events_gpkg(fire_events_all, harvest_events_all, compounds)
    make_figures(clean_fire, clean_harvest, compounds, age_summary)

    pathway_counts = compounds["pathway"].value_counts().to_dict() if not compounds.empty and "pathway" in compounds else {}
    summary = {
        "crosswalk_path": str(CONFIG_DIR / "disturbance_code_crosswalk.csv"),
        "fire_total_codes": crosswalk[(crosswalk["project_class"] == "fire_total") & (crosswalk["primary_analysis"])]["code"].tolist(),
        "harvest_total_codes": crosswalk[(crosswalk["project_class"] == "harvest_total") & (crosswalk["primary_analysis"])]["code"].drop_duplicates().tolist(),
        "clean_fire_events": int(clean_fire["event_id"].nunique()),
        "clean_harvest_events": int(clean_harvest["event_id"].nunique()),
        "compound_or_ambiguous_rows": int(len(compounds)),
        "pathway_counts": pathway_counts,
        "fire_to_salvage_cases": int(pathway_counts.get("fire_then_salvage", 0)),
        "matchability": {
            "fire_events": int(len(match)),
            "pct_with_same_year_harvest": float((match["same_year_harvest_candidates"] > 0).mean() * 100) if not match.empty else 0,
            "pct_with_pm1_year_harvest": float((match["pm1_year_harvest_candidates"] > 0).mean() * 100) if not match.empty else 0,
            "pct_with_pm2_year_harvest": float((match["pm2_year_harvest_candidates"] > 0).mean() * 100) if not match.empty else 0,
        },
        "harvest_grouping": "Provisional contiguous/intersecting harvest_total polygons by year; GEOC_IFM is only an interior point coordinate, not an event ID.",
        "uncertainties": [
            "WFS subset does not expose NO_SEC_INT from META_INTERV_FORES_PROV.",
            "Harvest event grouping is provisional and spatial, not an official sector/event ID.",
            "Subsequent silvicultural treatments after harvest are summarized from repeated GEOC_IFM site coordinates; fire/harvest overlap pathways are retained separately in the compound table.",
            "Compound same-year and fire-salvage cases are excluded from clean primary cohorts but retained.",
            "No remote-sensing recovery metric has been computed in this phase.",
        ],
    }
    (OUT_DIR / "event_history_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_readme(summary)

    print("OFFICIAL CODE CROSSWALK")
    print(crosswalk[["source", "field", "code", "official_description", "project_class", "primary_analysis"]].to_string(index=False))
    print("\nFINAL CODES INCLUDED AS CANDIDATE fire_total")
    print(", ".join(summary["fire_total_codes"]))
    print("\nFINAL CODES INCLUDED AS CANDIDATE harvest_total")
    print(", ".join(summary["harvest_total_codes"]))
    print("\nCODES EXCLUDED FROM PRIMARY ANALYSIS")
    excluded = crosswalk[~crosswalk["primary_analysis"]][["source", "field", "code", "project_class", "notes"]]
    print(excluded.to_string(index=False))
    print("\nCLEAN EVENT COUNTS")
    print(replication.to_string(index=False, formatters={"area_ha": "{:.1f}".format, "median_event_area_ha": "{:.2f}".format}))
    print("\nDISTRIBUTION BY AGE BIN")
    print(age_summary.to_string(index=False, formatters={"area_ha": "{:.1f}".format}))
    print("\nFIRE-HARVEST TEMPORAL MATCHABILITY")
    print(json.dumps(summary["matchability"], indent=2))
    print("\nHARVEST EVENT GROUPING")
    print(summary["harvest_grouping"])
    print("\nSUBSEQUENT MANAGEMENT VARIABLES")
    print("replanted_after_harvest / planting_year; site_preparation_after_harvest / site_preparation_year; thinning_after_harvest / thinning_year; second_harvest / second_harvest_year; next_major_disturbance_year / censor_year")
    print("\nFIRE->SALVAGE CASES")
    print(summary["fire_to_salvage_cases"])
    print("\nIMPORTANT UNCERTAINTIES")
    for item in summary["uncertainties"]:
        print(f"- {item}")
    print("\nFILES CREATED/MODIFIED")
    for path in [
        CONFIG_DIR / "disturbance_code_crosswalk.csv",
        Path(__file__),
        OUT_DIR / "disturbance_events.gpkg",
        OUT_DIR / "forestry_treatment_histories.csv",
        OUT_DIR / "fire_event_histories.csv",
        OUT_DIR / "compound_disturbance_histories.csv",
        OUT_DIR / "primary_fire_events.csv",
        OUT_DIR / "primary_harvest_events.csv",
        OUT_DIR / "event_replication_summary.csv",
        OUT_DIR / "event_age_bin_summary.csv",
        OUT_DIR / "fire_harvest_matchability.csv",
        OUT_DIR / "event_history_summary.json",
        OUT_DIR / "README.md",
        FIG_DIR / "clean_fire_vs_harvest_events_map.png",
        FIG_DIR / "events_by_disturbance_year.png",
        FIG_DIR / "event_counts_by_age_bin.png",
        FIG_DIR / "compound_pathway_counts.png",
    ]:
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
