#!/usr/bin/env python3
"""Acquire and audit the lightest official forestry-intervention subset.

This script does not download the provincial 2.05 GB archive. It probes official
Quebec/MRNF access routes, downloads the small official code dictionary, and
uses the official WFS feature layer when available to retrieve only the selected
100 km Lebel-sur-Quevillon study-region subset.
"""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_GEOJSON = REPO_ROOT / "outputs" / "study_extent_selection" / "recommended_study_extent.geojson"
FIRE_GPKG = REPO_ROOT / "data" / "raw" / "fire" / "FEUX_PROV.gpkg"
RAW_DIR = REPO_ROOT / "data" / "raw" / "forestry_interventions"
OUT_DIR = REPO_ROOT / "outputs" / "forestry_subset_audit"
FIG_DIR = REPO_ROOT / "figures" / "forestry_subset_audit"
MPL_DIR = OUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

CRS_PROJECTED = "EPSG:32198"
CRS_GEOGRAPHIC = "EPSG:4269"
YEAR_MIN = 2001
YEAR_MAX = 2024

DATASET_PAGE = "https://www.donneesquebec.ca/recherche/dataset/recolte-et-reboisement"
CKAN_API = "https://www.donneesquebec.ca/recherche/api/3/action/package_show?id=recolte-et-reboisement"
DATA_DIRECTORY = "https://diffusion.mffp.gouv.qc.ca/Diffusion/DonneeGratuite/Foret/INTERVENTIONS_FORESTIERES/Recolte_et_reboisement/"
PROV_DIRECTORY = DATA_DIRECTORY + "02-Donnees/PROV/"
PROV_GPKG = PROV_DIRECTORY + "INTERV_FORES_PROV_GPKG.zip"
PROV_FGDB = PROV_DIRECTORY + "INTERV_FORES_PROV_GDB.zip"
WMS_URL = "https://geoegl.msp.gouv.qc.ca/ws/mffpecofor.fcgi?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
WFS_BASE = "https://geoegl.msp.gouv.qc.ca/ws/mffpecofor.fcgi"
WFS_TYPENAME = "ms:ca_interv_for_close_scale"
DICTIONARY_URL = DATA_DIRECTORY + "01-Documentation/DICTIONNAIRE_RECOLTE_INT.xlsx"
DICTIONARY_PATH = RAW_DIR / "DICTIONNAIRE_RECOLTE_INT.xlsx"
SUBSET_BBOX_PATH = RAW_DIR / "interv_fores_lebel_100km_bbox.geojson"
SUBSET_CIRCLE_PATH = RAW_DIR / "interv_fores_lebel_100km_circle.gpkg"

HARVEST_TERMS = ["coupe", "récolte", "recolte", "récupération", "recuperation"]
TOTAL_HARVEST_TERMS = ["coupe avec protection", "coupe totale", "coupe finale", "coupe sans protection"]
PARTIAL_HARVEST_TERMS = ["coupe partielle", "jardinage", "éclaircie commerciale", "eclaircie commerciale"]
THINNING_TERMS = ["éclaircie", "eclaircie"]
SITE_PREP_TERMS = ["préparation", "scarifi", "deblaiement", "débroussaillement", "brûlage dirigé", "brulage dirige"]
REFORESTATION_TERMS = ["plantation", "reboisement", "regarni", "ensemencement"]
SALVAGE_TERMS = ["récupération", "recuperation"]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def get_url(url: str, *, timeout: int = 60, stream: bool = False) -> requests.Response:
    response = requests.get(url, timeout=timeout, stream=stream)
    response.raise_for_status()
    return response


def audit_url(url: str, method: str, note: str, timeout: int = 60) -> dict:
    row = {"method": method, "url": url, "status_code": None, "content_type": None, "content_length": None, "viability": "unknown", "notes": note}
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code >= 400 or not response.headers:
            response = requests.get(url, timeout=timeout, stream=True)
        row["status_code"] = int(response.status_code)
        row["content_type"] = response.headers.get("content-type")
        row["content_length"] = response.headers.get("content-length")
        row["viability"] = "reachable" if response.ok else "not_reachable"
    except Exception as exc:
        row["viability"] = "error"
        row["notes"] += f" Error: {exc}"
    return row


def official_access_audit() -> list[dict]:
    rows = [
        audit_url(CKAN_API, "Données Québec CKAN package metadata", "Catalog metadata; lists downloadable resources but does not provide spatial feature filtering."),
        audit_url(PROV_DIRECTORY, "MRNF data directory", "Official directory. It contains only PROV-level ZIP downloads for this dataset."),
        audit_url(PROV_GPKG, "Provincial GPKG download", "Official analysis-ready vector data, but full provincial archive; no spatial subset parameter."),
        audit_url(PROV_FGDB, "Provincial FGDB download", "Official analysis-ready vector data, but full provincial archive; no spatial subset parameter."),
        audit_url(WMS_URL, "Official WMS", "Visualization/discovery only; not acceptable as analysis geometry."),
        audit_url(f"{WFS_BASE}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities", "Official WFS", "Queryable vector service; tested for ca_interv_for_close_scale."),
        audit_url(f"{WFS_BASE}?SERVICE=WFS&VERSION=2.0.0&REQUEST=DescribeFeatureType&TYPENAMES={WFS_TYPENAME}", "WFS DescribeFeatureType", "Schema endpoint for intervention feature layer."),
    ]
    return rows


def download_dictionary() -> int:
    if DICTIONARY_PATH.exists() and DICTIONARY_PATH.stat().st_size > 0:
        return DICTIONARY_PATH.stat().st_size
    response = get_url(DICTIONARY_URL, timeout=60, stream=True)
    total = 0
    with DICTIONARY_PATH.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                total += len(chunk)
                fh.write(chunk)
    return total


def xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for si in root.findall("a:si", ns):
        text = "".join(t.text or "" for t in si.findall(".//a:t", ns))
        strings.append(text)
    return strings


def xlsx_sheet_paths(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    ns_main = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("r:Relationship", ns_rel)}
    sheets = []
    for sheet in workbook.findall(".//a:sheet", ns_main):
        name = sheet.attrib.get("name", "")
        rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map.get(rid)
        if target:
            path = "xl/" + target.lstrip("/")
            sheets.append((name, path))
    return sheets


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cell_type = cell.attrib.get("t")
    value_el = cell.find("a:v", ns)
    if value_el is None:
        inline = cell.find(".//a:t", ns)
        return inline.text if inline is not None and inline.text else ""
    raw = value_el.text or ""
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except Exception:
            return raw
    return raw


def parse_dictionary() -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(DICTIONARY_PATH) as zf:
        shared = xlsx_shared_strings(zf)
        for sheet_name, path in xlsx_sheet_paths(zf):
            root = ET.fromstring(zf.read(path))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for ridx, row in enumerate(root.findall(".//a:sheetData/a:row", ns), start=1):
                vals = [cell_value(c, shared).strip() for c in row.findall("a:c", ns)]
                if any(vals):
                    rows.append({"sheet": sheet_name, "row": ridx, "values": " | ".join(vals), "values_lower": " | ".join(vals).lower()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dictionary_text_extract.csv", index=False)
    return df


def dictionary_relevant_rows(dictionary: pd.DataFrame) -> pd.DataFrame:
    terms = [
        "origine",
        "perturb",
        "coupe",
        "récolte",
        "recolte",
        "éclaircie",
        "eclaircie",
        "plantation",
        "reboisement",
        "scarifi",
        "préparation",
        "preparation",
        "récupération",
        "recuperation",
    ]
    mask = dictionary["values_lower"].apply(lambda s: any(term in s for term in terms))
    out = dictionary.loc[mask, ["sheet", "row", "values"]].copy()
    out.to_csv(OUT_DIR / "dictionary_relevant_rows.csv", index=False)
    return out


def study_geometry() -> gpd.GeoDataFrame:
    study = gpd.read_file(STUDY_GEOJSON, engine="pyogrio").to_crs(CRS_PROJECTED)
    return study


def fetch_wfs_bbox(study: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    bbox = study.to_crs(CRS_GEOGRAPHIC).total_bounds
    # The MRNF WFS 2.0 endpoint responds correctly with EPSG:4326 axis order
    # encoded as lat,lon for BBOX.
    bbox_param = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]},EPSG:4326"
    page_size = 1000
    start = 0
    features = []
    byte_count = 0
    requests_made = 0
    while True:
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": WFS_TYPENAME,
            "BBOX": bbox_param,
            "COUNT": str(page_size),
            "STARTINDEX": str(start),
            "OUTPUTFORMAT": "geojson",
        }
        response = requests.get(WFS_BASE, params=params, timeout=120)
        response.encoding = "utf-8"
        response.raise_for_status()
        requests_made += 1
        byte_count += len(response.content)
        payload = response.json()
        batch = payload.get("features", [])
        print(f"WFS page start={start}: {len(batch)} features", flush=True)
        features.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
        time.sleep(0.2)

    if not features:
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS_GEOGRAPHIC)
    else:
        records = []
        for feature in features:
            props = feature.get("properties", {}).copy()
            props["geometry"] = shape(feature["geometry"])
            records.append(props)
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_GEOGRAPHIC)
    gdf.to_file(SUBSET_BBOX_PATH, driver="GeoJSON")
    return gdf, {"bbox_param": bbox_param, "requests_made": requests_made, "features": len(features), "bytes_downloaded": byte_count}


def clip_to_circle(bbox_gdf: gpd.GeoDataFrame, study: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    if bbox_gdf.empty:
        return bbox_gdf.to_crs(CRS_PROJECTED), 0
    projected = bbox_gdf.to_crs(CRS_PROJECTED)
    intersects = projected[projected.intersects(study.geometry.iloc[0])].copy()
    count_intersects = len(intersects)
    clipped = gpd.overlay(intersects, study[["geometry"]], how="intersection", keep_geom_type=True)
    if not clipped.empty:
        clipped["area_ha_clipped"] = clipped.geometry.area / 10_000
        clipped.to_file(SUBSET_CIRCLE_PATH, layer="interv_fores_lebel_100km_circle", driver="GPKG")
    return clipped, count_intersects


def choose_year(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for col in ["an_origine", "an_perturb", "exercice"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
            out = out.fillna(vals)
    return out


def source_area(df: gpd.GeoDataFrame) -> pd.Series:
    if "superficie" in df.columns:
        vals = pd.to_numeric(df["superficie"], errors="coerce")
        return vals.fillna(df.geometry.area / 10_000)
    return df.geometry.area / 10_000


def frequency_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for field, desc_field in [("origine", "origine_desc"), ("perturb", "perturb_desc"), ("reb_ess1", "reb_ess1_desc"), ("symbologie", "classe")]:
        if field not in gdf.columns:
            continue
        years = choose_year(gdf)
        areas = source_area(gdf)
        tmp = pd.DataFrame(
            {
                "field": field,
                "code": gdf[field].replace("", pd.NA).fillna("<NULL>").astype(str),
                "official_description": gdf[desc_field].replace("", pd.NA).fillna("<NULL>").astype(str) if desc_field in gdf.columns else "<NULL>",
                "area_ha": areas,
                "year": years,
            }
        )
        grouped = (
            tmp.groupby(["field", "code", "official_description"], dropna=False)
            .agg(feature_count=("code", "size"), area_ha=("area_ha", "sum"), first_year=("year", "min"), last_year=("year", "max"))
            .reset_index()
        )
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["field", "code", "official_description", "feature_count", "area_ha", "first_year", "last_year"])
    return pd.concat(rows, ignore_index=True).sort_values(["field", "area_ha"], ascending=[True, False])


def potential_role(description: str, field: str) -> str:
    text = (description or "").lower()
    if text in ["", "<null>"]:
        return "requires_review"
    if any(term in text for term in REFORESTATION_TERMS):
        return "reforestation"
    if any(term in text for term in SITE_PREP_TERMS):
        return "site_preparation"
    if any(term in text for term in SALVAGE_TERMS):
        return "silvicultural_treatment"
    if any(term in text for term in TOTAL_HARVEST_TERMS):
        return "candidate_total_harvest"
    if any(term in text for term in PARTIAL_HARVEST_TERMS):
        return "candidate_partial_harvest"
    if any(term in text for term in THINNING_TERMS):
        return "silvicultural_treatment"
    if any(term in text for term in HARVEST_TERMS):
        return "candidate_total_harvest" if field == "origine" else "candidate_partial_harvest"
    return "requires_review"


def year_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if gdf.empty:
        return pd.DataFrame(columns=["year", "feature_count", "area_ha", "median_polygon_area_ha"])
    tmp = pd.DataFrame({"year": choose_year(gdf), "area_ha": source_area(gdf)})
    tmp = tmp.dropna(subset=["year"])
    tmp = tmp[(tmp["year"] >= YEAR_MIN) & (tmp["year"] <= YEAR_MAX)]
    if tmp.empty:
        return pd.DataFrame(columns=["year", "feature_count", "area_ha", "median_polygon_area_ha"])
    return (
        tmp.groupby("year")
        .agg(feature_count=("year", "size"), area_ha=("area_ha", "sum"), median_polygon_area_ha=("area_ha", "median"))
        .reset_index()
        .sort_values("year")
    )


def repeated_interventions(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if gdf.empty:
        return pd.DataFrame([{"metric": "status", "value": "no vector subset"}])
    years = choose_year(gdf)
    rows = [{"metric": "polygon_count", "value": len(gdf)}]
    for key in ["geoc_ifm", "no_sec_int", "territoire", "ogc_fid"]:
        if key in gdf.columns:
            tmp = pd.DataFrame({"key": gdf[key].astype(str), "year": years, "code": gdf.get("symbologie", pd.Series(index=gdf.index, dtype=object)).astype(str)})
            grouped = tmp.groupby("key").agg(records=("key", "size"), distinct_years=("year", "nunique"), distinct_codes=("code", "nunique"))
            rows.append({"metric": f"{key}_unique_values", "value": len(grouped)})
            rows.append({"metric": f"{key}_values_with_multiple_records", "value": int((grouped["records"] > 1).sum())})
            rows.append({"metric": f"{key}_values_with_multiple_years", "value": int((grouped["distinct_years"] > 1).sum())})
            rows.append({"metric": f"{key}_values_with_multiple_codes", "value": int((grouped["distinct_codes"] > 1).sum())})
    rows.append({"metric": "limitation", "value": "This is evidence of repeated or grouped records only, not a final treatment-history reconstruction."})
    return pd.DataFrame(rows)


def make_diagnostic_map(clipped: gpd.GeoDataFrame, study: gpd.GeoDataFrame) -> Path | None:
    if clipped.empty or not FIRE_GPKG.exists():
        return None
    fires = gpd.read_file(FIRE_GPKG, layer="feux_prov", engine="pyogrio")
    fires.columns = [str(c).lower() for c in fires.columns]
    fires = fires.to_crs(CRS_PROJECTED)
    fire_year = pd.to_numeric(fires["an_origine"].replace("", pd.NA), errors="coerce").fillna(pd.to_numeric(fires["an_perturb"].replace("", pd.NA), errors="coerce")).fillna(pd.to_numeric(fires["exercice"], errors="coerce"))
    fires = fires[(fire_year >= YEAR_MIN) & (fire_year <= 2023)]
    fires = fires.iloc[list(fires.sindex.query(study.geometry.iloc[0], predicate="intersects"))].copy()
    fig, ax = plt.subplots(figsize=(10, 10))
    study.boundary.plot(ax=ax, color="#111111", linewidth=2, label="100 km study boundary")
    if not fires.empty:
        fires.plot(ax=ax, color="#d95f02", alpha=0.35, linewidth=0, label="Fire polygons")
    clipped.plot(ax=ax, color="#1b9e77", alpha=0.25, linewidth=0, label="Forestry interventions")
    ax.set_title("Diagnostic subset map: official MRNF WFS interventions and fire polygons")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    path = FIG_DIR / "forestry_subset_diagnostic_map.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_access_md(access_rows: list[dict], dictionary_size: int, wfs_info: dict, subset_status: str) -> None:
    lines = [
        "# Forestry-intervention subset data-access audit",
        "",
        "Dataset: Récolte et autres interventions sylvicoles (depuis 1976), MRNF / Données Québec.",
        "",
        "## Access methods tested",
        "",
        "| Method | URL | Status | Size/content | Viability | Notes |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in access_rows:
        lines.append(
            f"| {row['method']} | {row['url']} | {row['status_code']} | {row['content_length'] or row['content_type'] or ''} | {row['viability']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Subset result",
            "",
            f"- Official WFS typename tested: `{WFS_TYPENAME}`.",
            f"- Dictionary downloaded: `{DICTIONARY_PATH}` ({dictionary_size} bytes).",
            f"- WFS subset status: `{subset_status}`.",
            f"- WFS requests made: `{wfs_info.get('requests_made')}`.",
            f"- Approximate bytes downloaded from WFS: `{wfs_info.get('bytes_downloaded')}`.",
            f"- Features returned by BBOX query: `{wfs_info.get('features')}`.",
            "",
            "The WMS is kept as a discovery/reference endpoint only and is not used as analysis geometry.",
            "The full provincial GPKG/FGDB downloads are official analysis-ready options but were not downloaded here.",
        ]
    )
    (OUT_DIR / "data_access_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    study = study_geometry()
    access_rows = official_access_audit()
    dictionary_size = download_dictionary()
    dictionary = parse_dictionary()
    dictionary_relevant_rows(dictionary)

    bbox_gdf, wfs_info = fetch_wfs_bbox(study)
    clipped, intersects_count = clip_to_circle(bbox_gdf, study)
    subset_status = "obtained" if not clipped.empty else "empty_or_failed"

    freq = frequency_table(clipped)
    freq.to_csv(OUT_DIR / "intervention_frequency_table.csv", index=False)
    candidates = freq.copy()
    if not candidates.empty:
        candidates["potential_role"] = candidates.apply(lambda r: potential_role(r["official_description"], r["field"]), axis=1)
    else:
        candidates["potential_role"] = []
    candidates.to_csv(OUT_DIR / "intervention_code_candidates.csv", index=False)
    ysum = year_summary(clipped)
    ysum.to_csv(OUT_DIR / "intervention_year_summary.csv", index=False)
    repeated = repeated_interventions(clipped)
    repeated.to_csv(OUT_DIR / "repeated_intervention_summary.csv", index=False)
    map_path = make_diagnostic_map(clipped, study)

    summary = {
        "dataset": "Récolte et autres interventions sylvicoles (depuis 1976)",
        "official_sources": {
            "dataset_page": DATASET_PAGE,
            "ckan_api": CKAN_API,
            "wfs_base": WFS_BASE,
            "wfs_typename": WFS_TYPENAME,
            "dictionary_url": DICTIONARY_URL,
            "provincial_gpkg_url": PROV_GPKG,
        },
        "vector_subset_obtained": bool(not clipped.empty),
        "download_size_used_bytes": dictionary_size + wfs_info.get("bytes_downloaded", 0),
        "dictionary_size_bytes": dictionary_size,
        "wfs_info": wfs_info,
        "bbox_feature_count": int(len(bbox_gdf)),
        "circle_intersecting_feature_count": int(intersects_count),
        "circle_clipped_feature_count": int(len(clipped)),
        "crs": str(clipped.crs) if not clipped.empty else CRS_PROJECTED,
        "valid_geometry_count": int(clipped.geometry.is_valid.sum()) if not clipped.empty else 0,
        "invalid_geometry_count": int((~clipped.geometry.is_valid).sum()) if not clipped.empty else 0,
        "total_clipped_area_ha": float(clipped.geometry.area.sum() / 10_000) if not clipped.empty else 0.0,
        "source_area_sum_ha": float(source_area(clipped).sum()) if not clipped.empty else 0.0,
        "years": {
            "first": int(choose_year(clipped).min()) if not clipped.empty and choose_year(clipped).notna().any() else None,
            "last": int(choose_year(clipped).max()) if not clipped.empty and choose_year(clipped).notna().any() else None,
        },
        "potential_event_identifier": "geoc_ifm is present in WFS features; no NO_SEC_INT field was exposed by the WFS subset.",
        "diagnostic_map": str(map_path) if map_path else None,
        "raw_subset_bbox": str(SUBSET_BBOX_PATH),
        "raw_subset_circle": str(SUBSET_CIRCLE_PATH) if SUBSET_CIRCLE_PATH.exists() else None,
    }
    (OUT_DIR / "forestry_subset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_access_md(access_rows, dictionary_size, wfs_info, subset_status)

    print("OFFICIAL DATA-ACCESS METHODS FOUND")
    for row in access_rows:
        print(f"- {row['method']}: {row['viability']} | {row['url']}")
    print("\nREAL VECTOR SUBSET OBTAINED")
    print(bool(not clipped.empty))
    print(f"WFS typename: {WFS_TYPENAME}")
    print(f"Download size used: {summary['download_size_used_bytes']} bytes (dictionary + WFS responses; provincial ZIP not downloaded)")
    print("\nFORESTRY POLYGONS IN 100 KM STUDY REGION")
    print(f"BBOX features returned: {len(bbox_gdf)}")
    print(f"Intersect circular study geometry: {intersects_count}")
    print(f"Clipped features written: {len(clipped)}")
    print(f"Total clipped geometry area: {summary['total_clipped_area_ha']:.1f} ha")
    print("\nOFFICIAL MEANINGS OF ORIGINE AND PERTURB CODES")
    display_cols = ["field", "code", "official_description", "feature_count", "area_ha", "first_year", "last_year"]
    print(freq[freq["field"].isin(["origine", "perturb"])][display_cols].head(40).to_string(index=False) if not freq.empty else "No frequency table")
    print("\nCANDIDATE HARVEST-RELATED CODES")
    print(candidates[candidates["potential_role"].isin(["candidate_total_harvest", "candidate_partial_harvest", "silvicultural_treatment", "reforestation", "site_preparation"])][display_cols + ["potential_role"]].head(80).to_string(index=False) if not candidates.empty else "No candidates")
    print("\nEVIDENCE OF REPEATED INTERVENTIONS")
    print(repeated.to_string(index=False))
    print("\nPOTENTIAL INTERVENTION/EVENT IDENTIFIER")
    print(summary["potential_event_identifier"])
    print("\nUNRESOLVED ISSUES")
    print("- The WFS layer exposes useful descriptions but not necessarily every field from the full provincial GeoPackage.")
    print("- `geoc_ifm` requires official documentation before being treated as an event or treatment-history key.")
    print("- Potential roles are candidates only; no final harvest class was defined.")
    print("\nFILES CREATED/MODIFIED")
    for path in [
        Path(__file__),
        OUT_DIR / "data_access_audit.md",
        OUT_DIR / "intervention_code_candidates.csv",
        OUT_DIR / "intervention_year_summary.csv",
        OUT_DIR / "intervention_frequency_table.csv",
        OUT_DIR / "repeated_intervention_summary.csv",
        OUT_DIR / "forestry_subset_summary.json",
        OUT_DIR / "dictionary_text_extract.csv",
        OUT_DIR / "dictionary_relevant_rows.csv",
        DICTIONARY_PATH,
        SUBSET_BBOX_PATH,
        SUBSET_CIRCLE_PATH,
        map_path,
    ]:
        if path:
            print(f"- {path}")


if __name__ == "__main__":
    main()
